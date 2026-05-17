"""
train_qlora.py

Local QLoRA supervised fine-tuning for Advocore on a 12GB RTX 3060.

The script loads an instruct base model in 4-bit NF4, applies LoRA adapters to
the Llama-style linear projection modules, and saves only the adapter weights to
models/advocore-adapters/.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


PROJECT_NAME = "Advocore"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_DIR = "models/advocore-adapters"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 5.0
DEFAULT_WARMUP_RATIO = 0.10
DEFAULT_EVAL_STEPS = 100
DEFAULT_RUN_NAME = "advocore-qlora-llama31-8b"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SYSTEM_PROMPT = (
    "You are Advocore, a Bangladesh legal research assistant. You are not a lawyer. "
    "Cite the official Laws of Bangladesh portal. "
    "Refuse if the retrieved evidence is insufficient."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s train_qlora %(message)s",
)
log = logging.getLogger("train_qlora")


def render_prompt(row: dict[str, Any]) -> str:
    return (
        f"<SYSTEM>{SYSTEM_PROMPT}</SYSTEM>\n"
        f"<INSTRUCTION>{row.get('instruction', '')}</INSTRUCTION>\n"
        f"<CONTEXT>{row.get('context', '')}</CONTEXT>\n"
        f"<RESPONSE>"
    )


def render_full(row: dict[str, Any]) -> str:
    reasoning = (row.get("reasoning") or "").strip()
    response = row.get("response", "")
    if reasoning:
        return (
            render_prompt(row)
            + f"<REASONING>{reasoning}</REASONING>\n"
            + f"<FINAL>{response}</FINAL></RESPONSE>"
        )
    return render_prompt(row) + f"{response}</RESPONSE>"


@dataclass(frozen=True)
class TokenizeConfig:
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH


def build_tokenize_fn(tokenizer: AutoTokenizer, cfg: TokenizeConfig):
    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        prompt = render_prompt(row)
        full = render_full(row)

        encoded = tokenizer(
            full,
            truncation=True,
            max_length=cfg.max_seq_length,
            add_special_tokens=True,
        )
        prompt_encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=cfg.max_seq_length,
            add_special_tokens=False,
        )

        labels = list(encoded["input_ids"])
        prompt_len = min(len(prompt_encoded["input_ids"]), len(labels))
        for idx in range(prompt_len):
            labels[idx] = -100
        encoded["labels"] = labels
        return encoded

    return tokenize


def parse_report_targets(value: str) -> list[str]:
    targets = [v.strip().lower() for v in value.split(",") if v.strip()]
    return targets or ["tensorboard"]


def configure_tracking(project_name: str, run_name: str) -> None:
    os.environ.setdefault("WANDB_PROJECT", project_name)
    os.environ.setdefault("WANDB_NAME", run_name)
    os.environ.setdefault("WANDB_WATCH", "false")
    os.environ.setdefault("WANDB_LOG_MODEL", "false")


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local 4-bit QLoRA training.")
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024**3
    log.info("CUDA device: %s (%.1f GiB VRAM)", props.name, total_gib)
    if total_gib < 11:
        log.warning("Detected less than 11 GiB VRAM; reduce max_seq_length if OOM occurs.")


def citation_targets(row: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for citation in row.get("citations") or []:
        url = citation.get("source_url")
        if url:
            targets.append(str(url))
        act_id = citation.get("act_id")
        section_id = citation.get("section_id")
        if act_id and section_id:
            targets.append(f"section {section_id}")
            targets.append(f"#{section_id}")
    source_url = row.get("source_url")
    if source_url:
        targets.append(str(source_url))
    return [t.lower() for t in targets if t]


def citation_hit(generated_text: str, row: dict[str, Any]) -> bool:
    text = generated_text.lower()
    targets = citation_targets(row)
    if not targets:
        return True
    return any(target in text for target in targets)


class CitationAccuracyTrainer(Trainer):
    def __init__(
        self,
        *,
        citation_rows: list[dict[str, Any]] | None = None,
        citation_sample_size: int = 0,
        citation_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        citation_max_new_tokens: int = 192,
        citation_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.citation_rows = citation_rows or []
        self.citation_sample_size = citation_sample_size
        self.citation_max_seq_length = citation_max_seq_length
        self.citation_max_new_tokens = citation_max_new_tokens
        self.citation_rng = random.Random(citation_seed)

    def evaluate(self, *args, **kwargs):  # noqa: ANN002, ANN003
        metrics = super().evaluate(*args, **kwargs)
        if not self.citation_rows or self.citation_sample_size <= 0:
            return metrics
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Trainer has neither processing_class nor tokenizer.")
        sample_n = min(self.citation_sample_size, len(self.citation_rows))
        sample = self.citation_rng.sample(self.citation_rows, sample_n)
        hits = 0
        model = self.model
        model.eval()
        device = model.device
        with torch.no_grad():
            for row in sample:
                prompt = render_prompt(row)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.citation_max_seq_length,
                ).to(device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.citation_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                )
                hits += int(citation_hit(generated, row))
        accuracy = hits / sample_n if sample_n else 0.0
        citation_metrics = {
            "eval_citation_accuracy": accuracy,
            "eval_citation_accuracy_sample_size": sample_n,
        }
        metrics.update(citation_metrics)
        self.log(citation_metrics)
        model.train()
        return metrics


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--val-file", default="data/val.jsonl")
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="explicit warmup steps; defaults to ceil(total_steps * warmup_ratio)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.3)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    p.add_argument("--max-eval-samples", type=int, default=512,
                   help="validation rows used at each 100-step eval; 0 uses the full validation set")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--citation-eval-samples", type=int, default=24)
    p.add_argument("--citation-max-new-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project-name", default=PROJECT_NAME)
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--report-to", default="tensorboard",
                   help="comma-separated Trainer integrations, e.g. tensorboard,wandb")
    p.add_argument("--logging-dir", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.set_defaults(gradient_checkpointing=True)
    args = p.parse_args()

    require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_tracking(args.project_name, args.run_name)

    log.info("loading model=%s in 4-bit NF4 for %s", args.base_model, PROJECT_NAME)
    model, tokenizer = build_model_and_tokenizer(args)

    log.info("loading dataset: train=%s validation=%s", args.train_file, args.val_file)
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )
    tokenize_fn = build_tokenize_fn(tokenizer, TokenizeConfig(args.max_seq_length))
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset["train"].column_names,
        desc="tokenize Advocore SFT rows",
    )
    train_rows = len(tokenized["train"])
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(train_rows / effective_batch)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    log.info(
        "training rows=%d effective_batch=%d steps_per_epoch=%d total_steps=%d validation_rows=%d",
        train_rows,
        effective_batch,
        steps_per_epoch,
        total_steps,
        min(len(tokenized["validation"]), args.max_eval_samples or len(tokenized["validation"])),
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    logging_dir = args.logging_dir or os.path.join("runs", args.project_name, args.run_name)
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, math.ceil(total_steps * args.warmup_ratio)) if args.warmup_ratio else 0
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        logging_dir=logging_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=True,
        bf16=False,
        tf32=True,
        seed=args.seed,
        report_to=parse_report_targets(args.report_to),
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    if args.max_eval_samples and len(tokenized["validation"]) > args.max_eval_samples:
        eval_indices = list(range(args.max_eval_samples))
        eval_dataset = tokenized["validation"].shuffle(seed=args.seed).select(eval_indices)
        raw_eval_rows = list(dataset["validation"].shuffle(seed=args.seed).select(eval_indices))
    else:
        eval_dataset = tokenized["validation"]
        raw_eval_rows = list(dataset["validation"])
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
        citation_rows=raw_eval_rows,
        citation_sample_size=args.citation_eval_samples,
        citation_max_seq_length=args.max_seq_length,
        citation_max_new_tokens=args.citation_max_new_tokens,
        citation_seed=args.seed,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = CitationAccuracyTrainer(**trainer_kwargs)

    log.info("starting Advocore QLoRA training")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    log.info("saving LoRA adapters only to %s", args.output_dir)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "advocore_training_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "project_name": args.project_name,
                "run_name": args.run_name,
                "base_model": args.base_model,
                "max_seq_length": args.max_seq_length,
                "per_device_train_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "warmup_steps": warmup_steps,
                "optimizer": "paged_adamw_32bit",
                "quantization": "4bit-nf4",
                "target_modules": TARGET_MODULES,
                "adapter_dir": args.output_dir,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
