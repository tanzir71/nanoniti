"""
evaluate.py

Minimal evaluation pass over data/val.jsonl using a fine-tuned checkpoint.

It reports:
- Token-level perplexity on the validation set.
- Generation samples for the first N rows.
- Citation-presence rate: fraction of generated responses for non-refusal
  rows that contain the expected source URL string.

Run after train.py completes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import render_prompt, render_full, SYSTEM_PROMPT  # reuse rendering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s evaluate %(message)s",
)
log = logging.getLogger("evaluate")


def compute_perplexity(model, tokenizer, rows, max_len: int, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for row in rows:
            text = render_full(row)
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_len,
            ).to(device)
            labels = enc["input_ids"].clone()
            out = model(**enc, labels=labels)
            n_tok = enc["input_ids"].numel()
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
    if total_tokens == 0:
        return float("inf")
    avg = total_loss / total_tokens
    return math.exp(min(avg, 20))


def generate_sample(model, tokenizer, row, max_new_tokens: int, device) -> str:
    prompt = render_prompt(row)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)):]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="checkpoints/legal-sft")
    p.add_argument("--val-file", default="data/val.jsonl")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--out", default="data/eval_report.json")
    args = p.parse_args()

    if not os.path.exists(args.model_dir):
        log.error("model dir not found: %s", args.model_dir)
        return 1
    if not os.path.exists(args.val_file):
        log.error("val file not found: %s", args.val_file)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    rows = []
    with open(args.val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    log.info("loaded %d val rows", len(rows))
    ppl = compute_perplexity(model, tokenizer, rows[:64], args.max_len, device)
    log.info("validation perplexity (first 64 rows): %.3f", ppl)

    samples = []
    cite_hits = 0
    cite_total = 0
    for row in rows[: args.n_samples]:
        gen = generate_sample(model, tokenizer, row, args.max_new_tokens, device)
        samples.append({
            "task_type": row["task_type"],
            "instruction": row["instruction"][:300],
            "expected": row["response"][:300],
            "generated": gen[:600],
            "source_url": row.get("source_url", ""),
        })
        if row["task_type"] != "refusal" and row.get("source_url"):
            cite_total += 1
            if row["source_url"] in gen:
                cite_hits += 1

    citation_rate = (cite_hits / cite_total) if cite_total else 0.0

    report = {
        "model_dir": args.model_dir,
        "val_file": args.val_file,
        "num_val_rows": len(rows),
        "perplexity_first64": ppl,
        "citation_presence_rate": citation_rate,
        "samples": samples,
        "system_prompt": SYSTEM_PROMPT,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("wrote eval report to %s", args.out)
    log.info("citation presence (non-refusal): %.2f (%d/%d)", citation_rate, cite_hits, cite_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
