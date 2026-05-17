"""
modal_train.py

Bangladesh legal assistant SFT on Modal.

Why this exists: Colab disconnects mid-train and silently drops HF pushes.
Modal runs to completion on a real GPU, pushes the adapter directly, and
returns a dict you can read in your terminal.

Setup (one-time, ~3 minutes):
    pip install modal
    modal token new
    modal secret create huggingface HF_TOKEN=hf_xxx

Run (one shot):
    modal run modal_train.py

You can override anything with --flag=value, e.g.:
    modal run modal_train.py --base-model=Qwen/Qwen2.5-7B-Instruct --subset-train=0

Cost on Modal's L4 (24 GB): roughly $0.50 per run at the defaults (Qwen2.5-3B,
20K stratified subset, 1 epoch). Free $30/month credit covers it.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "transformers>=4.45.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.0",
        "huggingface_hub>=0.24.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "einops>=0.8.0",
        # Pin torch wheel that matches Modal's CUDA 12.1 base
        "torch==2.4.0",
    )
)

app = modal.App("bd-legal-sft", image=image)

# Persistent volume for HF cache so re-runs skip the dataset re-download.
volume = modal.Volume.from_name("bd-legal-hf-cache", create_if_missing=True)

# Pull the HF token from a Modal secret you create with:
#   modal secret create huggingface HF_TOKEN=hf_xxx
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


# ---------------------------------------------------------------------------
# The training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L4",                   # 24 GB; bump to "A100" for 7B at full corpus
    timeout=60 * 60 * 3,        # 3 hours, plenty for L4 + 20K subset
    secrets=[hf_secret],
    volumes={"/root/.cache/huggingface": volume},
)
def train(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    data_repo: str = "tanziro/bd-legal-sft",
    output_repo: str = "tanziro/bd-legal-qwen25-3b-lora",
    max_len: int = 512,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 2e-4,
    epochs: float = 1.0,
    warmup_ratio: float = 0.03,
    save_steps: int = 200,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    subset_train: int = 20000,   # 0 = use full train split
    subset_val: int = 1000,      # 0 = use full validation split
) -> dict:
    import os, random, time, math, json, inspect
    from collections import Counter
    import torch
    from huggingface_hub import login, create_repo, upload_folder, whoami
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    print("=" * 60)
    print("Bangladesh Legal Assistant SFT on Modal")
    print("=" * 60)
    print(f"base_model   = {base_model}")
    print(f"data_repo    = {data_repo}")
    print(f"output_repo  = {output_repo}")
    print(f"GPU          = {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    token = os.environ["HF_TOKEN"]
    login(token=token)
    print(f"logged in as: {whoami(token=token)['name']}")
    create_repo(output_repo, repo_type="model", private=True, exist_ok=True, token=token)

    # -- dataset --
    print("\nloading dataset ...")
    ds = load_dataset(data_repo, token=token)
    print(f"  full splits: {dict((k, v.num_rows) for k,v in ds.items())}")

    def stratified(d, n, key="task_type"):
        if not n or len(d) <= n: return d
        buckets = {}
        for i, t in enumerate(d[key]):
            buckets.setdefault(t, []).append(i)
        rng = random.Random(seed)
        per = max(1, n // len(buckets))
        picks = []
        for t, idxs in buckets.items():
            rng.shuffle(idxs)
            picks.extend(idxs[:per])
        rng.shuffle(picks)
        return d.select(picks[:n])

    ds["train"] = stratified(ds["train"], subset_train)
    ds["validation"] = stratified(ds["validation"], subset_val)
    print(f"  subset:      train={len(ds['train'])} val={len(ds['validation'])}")
    print(f"  train tasks: {dict(Counter(ds['train']['task_type']))}")

    # -- tokenizer + prompt rendering --
    SYSTEM_PROMPT = (
        "You are a Bangladesh legal research assistant. You are not a lawyer. "
        "Cite the official Laws of Bangladesh portal. "
        "Refuse if the retrieved evidence is insufficient."
    )

    def render_prompt(row):
        return (f"<SYSTEM>{SYSTEM_PROMPT}</SYSTEM>\n"
                f"<INSTRUCTION>{row.get('instruction','')}</INSTRUCTION>\n"
                f"<CONTEXT>{row.get('context','')}</CONTEXT>\n"
                f"<RESPONSE>")

    def render_full(row):
        return render_prompt(row) + f"{row.get('response','')}</RESPONSE>"

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(row):
        prompt = render_prompt(row); full = render_full(row)
        enc_full   = tok(full, truncation=True, max_length=max_len, padding="max_length")
        enc_prompt = tok(prompt, truncation=True, max_length=max_len, add_special_tokens=False)
        labels = list(enc_full["input_ids"])
        plen = min(len(enc_prompt["input_ids"]), len(labels))
        for i in range(plen): labels[i] = -100
        pad = tok.pad_token_id
        for i, t in enumerate(enc_full["input_ids"]):
            if t == pad: labels[i] = -100
        enc_full["labels"] = labels
        return enc_full

    ds_tok = ds.map(tokenize, remove_columns=ds["train"].column_names,
                    desc="tokenize", num_proc=2)

    # -- base model in 4-bit + LoRA --
    print("\nloading base model in 4-bit ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # CRITICAL: fixes "element 0 of tensors does not require grad" on the
    # combination of gradient_checkpointing + k-bit base + LoRA.
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    ))
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError("no trainable LoRA parameters; check target_modules/model architecture")

    # -- training --
    print("\nstarting training ...")
    args = TrainingArguments(
        output_dir="/root/legal-sft-out",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=max(1, int(math.ceil(math.ceil(len(ds_tok["train"]) / (batch_size * grad_accum)) * epochs) * warmup_ratio)) if warmup_ratio else 0,
        weight_decay=0.0,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        seed=seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # CRITICAL for newer torch
        optim="paged_adamw_8bit",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = Trainer(**trainer_kwargs)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    trainer.save_model("/root/legal-sft-out")
    tok.save_pretrained("/root/legal-sft-out")
    final_metrics = trainer.evaluate()

    # -- forced final push with retries --
    print("\npushing adapter to the Hub ...")
    import traceback
    for attempt in range(1, 5):
        try:
            r = upload_folder(
                repo_id=output_repo, repo_type="model",
                folder_path="/root/legal-sft-out",
                token=token,
                commit_message=f"SFT adapter ({base_model.split('/')[-1]}, subset={subset_train})",
                ignore_patterns=["checkpoint-*/optimizer.pt",
                                 "checkpoint-*/scheduler.pt",
                                 "checkpoint-*/training_args.bin"],
            )
            print(f"pushed (attempt {attempt}): {r}")
            break
        except Exception as e:
            print(f"attempt {attempt} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(5 * attempt)
    else:
        raise RuntimeError("all 4 push attempts failed")

    # -- citation sanity check (reload adapter from Hub, run a known prompt) --
    print("\nsanity check ...")
    del model, trainer
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    infer = PeftModel.from_pretrained(base, output_repo, token=token).eval()

    test_row = {
        "instruction": "What does section 302 of the Penal Code, 1860 provide? "
                       "Quote the operative text and cite the source.",
        "context": "",
    }
    enc = tok(render_prompt(test_row), return_tensors="pt").to(infer.device)
    out = infer.generate(
        **enc, max_new_tokens=320, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id)
    sample = tok.decode(out[0], skip_special_tokens=True)
    citation_ok = ("bdlaws.minlaw.gov.bd" in sample) or ("http" in sample)
    print("--- sample output ---")
    print(sample[-1200:])
    print("--- end ---")

    eval_report = {
        "base_model": base_model,
        "data_repo": data_repo,
        "output_repo": output_repo,
        "train_seconds": round(train_seconds, 1),
        "rows_train": len(ds["train"]),
        "rows_val": len(ds["validation"]),
        "final_metrics": {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
        "citation_check_passed": bool(citation_ok),
        "sample_generation": sample[-2000:],
    }

    # push the eval report alongside the adapter
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
        rp = f.name
    from huggingface_hub import upload_file
    upload_file(
        path_or_fileobj=rp, path_in_repo="eval_report.json",
        repo_id=output_repo, repo_type="model", token=token,
        commit_message="Add eval_report.json",
    )
    print(f"\ncitation_check_passed = {citation_ok}")
    print(f"adapter: https://huggingface.co/{output_repo}")
    return eval_report


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    data_repo: str = "tanziro/bd-legal-sft",
    output_repo: str = "tanziro/bd-legal-qwen25-3b-lora",
    subset_train: int = 20000,
    subset_val: int = 1000,
    max_len: int = 512,
    epochs: float = 1.0,
):
    """Run with: modal run modal_train.py"""
    result = train.remote(
        base_model=base_model,
        data_repo=data_repo,
        output_repo=output_repo,
        subset_train=subset_train,
        subset_val=subset_val,
        max_len=max_len,
        epochs=epochs,
    )
    import json
    print("\n========== RESULT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
