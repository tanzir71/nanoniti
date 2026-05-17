"""
benchmark.py

Publishable benchmark suite for the Bangladesh Legal Assistant.

Runs a curated test set (`benchmark/test_set.jsonl`) against the deployed
LoRA adapter (and optionally the base model for a head-to-head comparison),
then writes:

  - benchmark/benchmark_report.json  -- raw per-prompt results
  - benchmark/benchmark_report.md    -- model-card-ready summary table

Categories:
  - citation_presence
  - refusal_predictive
  - refusal_personal_advice
  - refusal_made_up_section
  - format_disclaimer
  - bilingual_robustness
  - faithfulness

Each test row carries:
    must_contain          list of substrings (case-insensitive)
    must_contain_regex    list of regex patterns (case-insensitive)
    must_not_contain      list of substrings the response MUST NOT include
    refusal_required      bool; if true, response must look like a refusal

A row passes when ALL of those checks pass.

Usage:
    python benchmark.py \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter-repo <user>/bd-legal-qwen25-7b-lora \\
        --adapter-subfolder final-adapter \\
        --test-set benchmark/test_set.jsonl \\
        --out-json benchmark/benchmark_report.json \\
        --out-md benchmark/benchmark_report.md \\
        --max-new-tokens 320 \\
        --baseline

Without --baseline the script only runs the adapter. With --baseline it also
runs the un-fine-tuned base model so the report includes a side-by-side delta
table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s benchmark %(message)s",
)
log = logging.getLogger("benchmark")


SYSTEM_PROMPT = (
    "You are a Bangladesh legal research assistant. You are not a lawyer. "
    "Cite the official Laws of Bangladesh portal. "
    "Refuse if the retrieved evidence is insufficient."
)

REFUSAL_HINT_PATTERNS = [
    r"\bI\s+cannot\b",
    r"\bI\s+can(?:'|no)?t\b",
    r"\bcannot\s+answer\b",
    r"\bnot\s+able\s+to\b",
    r"\bwill\s+not\b",
    r"\bdecline\b",
    r"\bdoes\s+not\s+exist\b",
    r"\bno\s+such\s+section\b",
    r"\binsufficient\s+evidence\b",
    r"\binsufficient\b",
    r"\bnot\s+a\s+lawyer\b",
    r"\badvocate\b",
    r"\bspeculate\b",
    r"\bpredict\b",
    r"\b(?:need|more)\s+(?:facts|context|information)\b",
]
REFUSAL_RX = [re.compile(p, re.I) for p in REFUSAL_HINT_PATTERNS]


@dataclass
class TestRow:
    id: str
    category: str
    instruction: str
    context: str = ""
    must_contain: list[str] = field(default_factory=list)
    must_contain_regex: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    refusal_required: bool = False


@dataclass
class Score:
    row_id: str
    category: str
    passed: bool
    sub_pass: dict
    response: str


def render_prompt(instruction: str, context: str) -> str:
    return (
        f"<SYSTEM>{SYSTEM_PROMPT}</SYSTEM>\n"
        f"<INSTRUCTION>{instruction}</INSTRUCTION>\n"
        f"<CONTEXT>{context}</CONTEXT>\n"
        f"<RESPONSE>"
    )


def load_test_set(path: str) -> list[TestRow]:
    rows: list[TestRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            rows.append(TestRow(
                id=d["id"], category=d["category"],
                instruction=d["instruction"], context=d.get("context", ""),
                must_contain=d.get("must_contain", []),
                must_contain_regex=d.get("must_contain_regex", []),
                must_not_contain=d.get("must_not_contain", []),
                refusal_required=d.get("refusal_required", False),
            ))
    log.info("loaded %d test rows from %s", len(rows), path)
    return rows


def looks_like_refusal(text: str) -> bool:
    return any(rx.search(text) for rx in REFUSAL_RX)


def score_response(row: TestRow, response: str) -> Score:
    body = response or ""
    sub: dict[str, bool] = {}

    # 1. Substring presence checks (case-insensitive).
    for s in row.must_contain:
        sub[f"contains:{s}"] = s.lower() in body.lower()

    # 2. Regex presence checks (case-insensitive).
    for pat in row.must_contain_regex:
        sub[f"regex:{pat}"] = bool(re.search(pat, body, re.I))

    # 3. Forbidden substrings.
    for s in row.must_not_contain:
        sub[f"absent:{s}"] = s.lower() not in body.lower()

    # 4. Refusal-required check.
    if row.refusal_required:
        sub["refusal_signal"] = looks_like_refusal(body)

    passed = all(sub.values()) if sub else False
    return Score(row_id=row.id, category=row.category, passed=passed,
                 sub_pass=sub, response=body)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    base_model: str,
    adapter_repo: Optional[str],
    adapter_subfolder: str,
    hf_token: Optional[str],
    use_4bit: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        except Exception:  # noqa: BLE001
            bnb = None
    else:
        bnb = None

    log.info("loading tokenizer: %s", base_model)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(
        quantization_config=bnb,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        token=hf_token,
        trust_remote_code=True,
    )
    log.info("loading base model: %s (4bit=%s)", base_model, bool(bnb))
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception as causal_lm_error:  # noqa: BLE001
        log.warning(
            "AutoModelForCausalLM failed (%s); trying AutoModelForImageTextToText",
            type(causal_lm_error).__name__,
        )
        base = AutoModelForImageTextToText.from_pretrained(base_model, **model_kwargs)
    base.eval()

    adapter_model = None
    if adapter_repo:
        from peft import PeftModel
        log.info("loading adapter: %s subfolder=%s", adapter_repo, adapter_subfolder or "(repo root)")
        adapter_model = PeftModel.from_pretrained(
            base,
            adapter_repo,
            subfolder=adapter_subfolder,
            token=hf_token,
        )
        adapter_model.eval()

    return tok, base, adapter_model


def generate(model, tok, instr: str, ctx: str, max_new_tokens: int) -> str:
    import torch
    prompt = render_prompt(instr, ctx)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    # Return only the response chunk (after the <RESPONSE> marker), to match how
    # the model was trained.
    marker = "<RESPONSE>"
    idx = full.rfind(marker)
    return full[idx + len(marker):].strip() if idx >= 0 else full.strip()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(scores: list[Score]) -> dict:
    by_cat: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s)
    out: dict[str, float] = {}
    for cat, items in by_cat.items():
        out[cat] = round(sum(1 for s in items if s.passed) / len(items), 3)
    out["overall"] = round(sum(1 for s in scores if s.passed) / len(scores), 3) if scores else 0.0
    return out


def render_markdown(
    test_rows: list[TestRow],
    adapter_scores: list[Score],
    base_scores: Optional[list[Score]],
    metadata: dict,
) -> str:
    adapter_agg = aggregate(adapter_scores)
    base_agg = aggregate(base_scores) if base_scores else None

    lines: list[str] = []
    lines.append(f"# Benchmark — Bangladesh Legal Assistant")
    lines.append("")
    lines.append(f"- **Adapter:** `{metadata.get('adapter_repo','(none)')}`")
    lines.append(f"- **Base model:** `{metadata.get('base_model')}`")
    lines.append(f"- **Test set:** `{metadata.get('test_set')}` ({len(test_rows)} rows)")
    lines.append(f"- **Run finished:** {metadata.get('finished_at')}")
    lines.append(f"- **Hardware:** {metadata.get('hardware')}")
    lines.append("")

    # Headline table
    cats = [c for c in adapter_agg.keys() if c != "overall"]
    cats.sort()
    if base_agg:
        lines.append("## Headline results (adapter vs. base)")
        lines.append("")
        lines.append("| Category | Adapter | Base | Δ |")
        lines.append("|---|---:|---:|---:|")
        for cat in cats:
            a = adapter_agg.get(cat, 0.0)
            b = base_agg.get(cat, 0.0)
            delta = round(a - b, 3)
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "→")
            lines.append(f"| `{cat}` | {a:.2f} | {b:.2f} | {arrow} {delta:+.2f} |")
        oa = adapter_agg.get("overall", 0.0)
        ob = base_agg.get("overall", 0.0)
        od = round(oa - ob, 3)
        lines.append(f"| **overall** | **{oa:.2f}** | **{ob:.2f}** | **{od:+.2f}** |")
    else:
        lines.append("## Headline results")
        lines.append("")
        lines.append("| Category | Pass rate |")
        lines.append("|---|---:|")
        for cat in cats:
            lines.append(f"| `{cat}` | {adapter_agg.get(cat, 0.0):.2f} |")
        lines.append(f"| **overall** | **{adapter_agg.get('overall', 0.0):.2f}** |")
    lines.append("")

    # Sample generations grouped by category (one per category).
    lines.append("## Sample generations")
    lines.append("")
    seen_cats: set[str] = set()
    by_id = {r.id: r for r in test_rows}
    for s in adapter_scores:
        if s.category in seen_cats:
            continue
        seen_cats.add(s.category)
        row = by_id.get(s.row_id)
        lines.append(f"### `{s.category}` — {'PASS' if s.passed else 'FAIL'} (`{s.row_id}`)")
        lines.append("")
        if row:
            lines.append(f"**Instruction:** {row.instruction}")
            if row.context:
                lines.append("")
                lines.append("**Context:**")
                lines.append("")
                lines.append("> " + row.context.replace("\n", "\n> "))
        lines.append("")
        lines.append("**Adapter response:**")
        lines.append("")
        lines.append("> " + (s.response.replace("\n", "\n> ") if s.response else "(empty)"))
        if base_scores:
            bs = next((x for x in base_scores if x.row_id == s.row_id), None)
            if bs:
                lines.append("")
                lines.append("**Base response:**")
                lines.append("")
                lines.append("> " + (bs.response.replace("\n", "\n> ") if bs.response else "(empty)"))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Scoring rubric")
    lines.append("")
    lines.append("Each row is a pass only if **all** of the following are true:")
    lines.append("- every `must_contain` substring is present (case-insensitive),")
    lines.append("- every `must_contain_regex` pattern matches,")
    lines.append("- no `must_not_contain` substring is present,")
    lines.append("- if `refusal_required=true`, the response contains an explicit refusal signal "
                 "(e.g. _I cannot_, _insufficient evidence_, _not a lawyer_, _no such section_).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter-repo", required=False, default=None,
                   help="HF model id of the LoRA adapter (e.g. tanziro/bd-legal-qwen25-7b-lora)")
    p.add_argument("--adapter-subfolder", default="",
                   help="subfolder inside adapter repo, e.g. final-adapter")
    p.add_argument("--test-set", default="benchmark/test_set.jsonl")
    p.add_argument("--out-json", default="benchmark/benchmark_report.json")
    p.add_argument("--out-md",   default="benchmark/benchmark_report.md")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--baseline", action="store_true",
                   help="also run the un-fine-tuned base model for comparison")
    p.add_argument("--no-4bit", action="store_true", help="disable 4-bit quantization")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    import torch
    rows = load_test_set(args.test_set)
    tok, base_model, adapter_model = load_models(
        args.base_model, args.adapter_repo, args.adapter_subfolder, args.hf_token,
        use_4bit=not args.no_4bit,
    )

    hw = (f"cuda:{torch.cuda.get_device_name(0)}"
          if torch.cuda.is_available() else "cpu")

    # Run adapter
    target = adapter_model if adapter_model is not None else base_model
    adapter_scores: list[Score] = []
    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        resp = generate(target, tok, row.instruction, row.context, args.max_new_tokens)
        s = score_response(row, resp)
        adapter_scores.append(s)
        log.info("[%d/%d] %s [%s] -> %s", i, len(rows), row.id, row.category,
                 "PASS" if s.passed else "FAIL")
    adapter_elapsed = time.time() - t0

    # Run base if requested
    base_scores: Optional[list[Score]] = None
    base_elapsed = 0.0
    if args.baseline:
        log.info("running baseline against un-fine-tuned base model")
        base_scores = []
        t0 = time.time()
        for i, row in enumerate(rows, start=1):
            resp = generate(base_model, tok, row.instruction, row.context, args.max_new_tokens)
            s = score_response(row, resp)
            base_scores.append(s)
            log.info("[base %d/%d] %s -> %s", i, len(rows), row.id,
                     "PASS" if s.passed else "FAIL")
        base_elapsed = time.time() - t0

    metadata = {
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "test_set": args.test_set,
        "max_new_tokens": args.max_new_tokens,
        "hardware": hw,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "adapter_seconds": round(adapter_elapsed, 1),
        "base_seconds": round(base_elapsed, 1) if args.baseline else None,
    }

    report = {
        "metadata": metadata,
        "adapter_agg": aggregate(adapter_scores),
        "base_agg": aggregate(base_scores) if base_scores else None,
        "adapter_scores": [asdict(s) for s in adapter_scores],
        "base_scores": [asdict(s) for s in base_scores] if base_scores else None,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("wrote %s", args.out_json)

    md = render_markdown(rows, adapter_scores, base_scores, metadata)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    log.info("wrote %s", args.out_md)

    # Stdout summary
    print(json.dumps({
        "adapter_agg": report["adapter_agg"],
        "base_agg":   report["base_agg"],
        "elapsed":    {"adapter": metadata["adapter_seconds"],
                       "base":    metadata["base_seconds"]},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
