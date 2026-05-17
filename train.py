"""
train.py

Standard supervised fine-tuning over data/train.jsonl using Hugging Face
datasets + transformers. The base model is configurable via a single variable
(BASE_MODEL) at the top of this file.

Defaults are tuned to be runnable on modest hardware:
- A small open causal LM as the first pass.
- Short sequence length.
- Few epochs.

Each training example is rendered as:

    <SYSTEM>
    You are a Bangladesh legal research assistant. You are not a lawyer.
    Cite the official Laws of Bangladesh portal. Refuse if evidence is insufficient.
    </SYSTEM>
    <INSTRUCTION>{instruction}</INSTRUCTION>
    <CONTEXT>{context}</CONTEXT>
    <RESPONSE>{response}</RESPONSE>

Only the <RESPONSE> tokens contribute to the loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# === Configurable base model ===
BASE_MODEL = os.environ.get("BASE_MODEL", "sshleifer/tiny-gpt2")
# Use e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0" or "Qwen/Qwen2.5-0.5B" for real runs.

SYSTEM_PROMPT = (
    "You are a Bangladesh legal research assistant. You are not a lawyer. "
    "Cite the official Laws of Bangladesh portal. "
    "Refuse if the retrieved evidence is insufficient."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s train %(message)s",
)
log = logging.getLogger("train")


def render_prompt(row: dict[str, Any]) -> str:
    return (
        f"<SYSTEM>{SYSTEM_PROMPT}</SYSTEM>\n"
        f"<INSTRUCTION>{row.get('instruction','')}</INSTRUCTION>\n"
        f"<CONTEXT>{row.get('context','')}</CONTEXT>\n"
        f"<RESPONSE>"
    )


def render_full(row: dict[str, Any]) -> str:
    return render_prompt(row) + f"{row.get('response','')}</RESPONSE>"


@dataclass
class TokenizeConfig:
    max_len: int


def build_tokenize_fn(tokenizer, cfg: TokenizeConfig):
    def fn(row):
        prompt = render_prompt(row)
        full = render_full(row)

        enc_full = tokenizer(
            full,
            truncation=True,
            max_length=cfg.max_len,
            padding="max_length",
        )
        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=cfg.max_len,
            add_special_tokens=False,
        )
        labels = list(enc_full["input_ids"])
        prompt_len = min(len(enc_prompt["input_ids"]), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            for i, t in enumerate(enc_full["input_ids"]):
                if t == pad_id:
                    labels[i] = -100
        enc_full["labels"] = labels
        return enc_full
    return fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--val-file", default="data/val.jsonl")
    p.add_argument("--output-dir", default="checkpoints/legal-sft")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    log.info("loading base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "<|pad|>"
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    log.info("loading dataset: train=%s val=%s", args.train_file, args.val_file)
    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    tok_fn = build_tokenize_fn(tokenizer, TokenizeConfig(max_len=args.max_len))
    ds_tok = ds.map(
        tok_fn,
        remove_columns=ds["train"].column_names,
        desc="tokenize",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    log.info("starting training")
    trainer.train()
    log.info("saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "train_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("final eval metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
