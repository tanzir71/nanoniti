"""
build_dataset.py

Convert cleaned documents in data/clean/ into a supervised fine-tuning dataset
written to data/dataset.jsonl. Each row follows the schema:

    instruction, context, response, citations,
    reasoning, source_title, source_url, source_type, jurisdiction,
    topic, task_type, confidence, refusal_reason

The generator emits three dense task types per legal section, all grounded in
retrieved source text:

  1. plain_language_explanation - explain the operative rule plainly.
  2. legal_issue_spotting       - list issues and elements from the text.
  3. comparative_analysis       - relate this section to nearby sections.

Rules:
- Every non-refusal row carries at least one citation tied to source metadata.
- Responses are conservative; they restate the source rather than invent law.
- Complex acts include a `reasoning` field so the model learns to organize
  legal logic before producing a final cited answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from glob import glob
from typing import Iterable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s build_dataset %(message)s",
)
log = logging.getLogger("build_dataset")

DISCLAIMER = (
    "This is automated legal research support, not legal advice; "
    "verify all citations against the official Laws of Bangladesh portal "
    "and consult a qualified Bangladeshi advocate before acting."
)

COMPLEX_ACT_PATTERNS = (
    "penal code",
    "criminal procedure",
    "civil procedure",
    "evidence act",
    "information and communication technology",
    "ict act",
    "digital security",
    "cyber security",
    "cyber safety",
    "anti-terrorism",
    "money laundering",
    "constitution",
)


def first_sentences(text: str, n: int = 3) -> str:
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return " ".join(parts[:n]).strip()


def short_topic(title: str) -> str:
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\s+", " ", t).strip(" -")
    return t[:80]


def make_citation(doc: dict) -> dict:
    return {
        "source_title": doc.get("source_title", ""),
        "source_url": doc.get("url", ""),
        "act_id": doc.get("act_id"),
        "section_id": doc.get("section_id"),
        "retrieved_at": doc.get("retrieved_at"),
    }


def base_row(doc: dict, task: str) -> dict:
    return {
        "instruction": "",
        "context": "",
        "reasoning": "",
        "response": "",
        "citations": [],
        "source_title": doc.get("source_title", ""),
        "source_url": doc.get("url", ""),
        "source_type": doc.get("source_type", ""),
        "jurisdiction": doc.get("jurisdiction", "Bangladesh"),
        "topic": short_topic(doc.get("source_title", "")),
        "task_type": task,
        "confidence": "medium",
        "refusal_reason": "",
    }


def is_complex_doc(doc: dict) -> bool:
    blob = f"{doc.get('source_title', '')} {doc.get('topic', '')}".lower()
    return any(pattern in blob for pattern in COMPLEX_ACT_PATTERNS)


def reasoning_for(doc: dict, focus: str, related: list[dict] | None = None) -> str:
    if not is_complex_doc(doc):
        return ""
    section_id = doc.get("section_id") or "?"
    title = short_topic(doc.get("source_title", ""))
    related_bits = ""
    if related:
        related_bits = " Compare only against the cited neighboring sections: " + ", ".join(
            f"section {r.get('section_id')}" for r in related[:2]
        ) + "."
    return (
        f"Identify the operative words of section {section_id} of {title}. "
        f"Separate legal elements from consequences. "
        f"Use the provided text to address {focus}; do not infer facts outside the source."
        f"{related_bits} End with the exact section citation."
    )


def section_sort_key(doc: dict) -> tuple[int, str]:
    section = str(doc.get("section_id") or "")
    m = re.match(r"(\d+)", section)
    return (int(m.group(1)) if m else 10**9, section)


def nearest_siblings(doc: dict, by_act: dict[str, list[dict]], limit: int = 2) -> list[dict]:
    act_id = str(doc.get("act_id") or "")
    candidates = by_act.get(act_id, [])
    if len(candidates) <= 1:
        return []
    ordered = sorted(candidates, key=section_sort_key)
    key = doc.get("sha1") or doc.get("url")
    idx = 0
    for i, item in enumerate(ordered):
        if (item.get("sha1") or item.get("url")) == key:
            idx = i
            break
    neighbors: list[dict] = []
    for offset in range(1, len(ordered)):
        for pos in (idx - offset, idx + offset):
            if 0 <= pos < len(ordered):
                item = ordered[pos]
                if (item.get("sha1") or item.get("url")) != key and item not in neighbors:
                    neighbors.append(item)
                    if len(neighbors) >= limit:
                        return neighbors
    return neighbors


def build_act_index(docs: list[dict]) -> dict[str, list[dict]]:
    by_act: dict[str, list[dict]] = {}
    for doc in docs:
        if doc.get("source_type") != "section_page":
            continue
        by_act.setdefault(str(doc.get("act_id") or ""), []).append(doc)
    return by_act


def row_plain_language_explanation(doc: dict) -> Optional[dict]:
    body = doc.get("body", "")
    if len(body) < 80:
        return None
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    r = base_row(doc, "plain_language_explanation")
    r["instruction"] = (
        f"Explain section {section_id} of '{short_topic(title)}' in plain language. "
        "Keep the answer grounded in the cited text and include the official source."
    )
    r["context"] = body[:1800]
    r["reasoning"] = reasoning_for(doc, "a plain-language explanation")
    r["response"] = (
        f"Plain-language explanation of section {section_id}:\n"
        f"{first_sentences(body, 3)}\n\n"
        f"In practical terms, this section should be read as a rule stated by the source text, "
        f"not as advice about any specific facts.\n\n"
        f"Source: {doc.get('url', '')}\n\n{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    r["confidence"] = "high" if len(body) > 300 else "medium"
    return r


def row_legal_issue_spotting(doc: dict) -> Optional[dict]:
    body = doc.get("body", "")
    clauses = doc.get("clauses") or []
    if not clauses and len(body) < 120:
        return None
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    bullets: list[str] = []
    for clause in clauses[:6]:
        marker = clause.get("marker")
        text = clause.get("text", "")
        if text:
            bullets.append(f"- Clause ({marker}) issue: {first_sentences(text, 1)}")
    if not bullets:
        bullets.append(f"- Core issue under section {section_id}: {first_sentences(body, 1)}")
    r = base_row(doc, "legal_issue_spotting")
    r["instruction"] = (
        f"Spot the legal issues and elements raised by section {section_id} of "
        f"'{short_topic(title)}'. Use only the provided text."
    )
    r["context"] = body[:2000]
    r["reasoning"] = reasoning_for(doc, "issue spotting")
    r["response"] = (
        f"Legal issues grounded in section {section_id}:\n"
        + "\n".join(bullets)
        + f"\n\nEach issue above is tied to the cited section text, not external facts. "
        f"Source: {doc.get('url', '')}\n\n{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    return r


def row_comparative_analysis(doc: dict, siblings: list[dict]) -> Optional[dict]:
    body = doc.get("body", "")
    if len(body) < 80:
        return None
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    related = siblings[:2]
    related_context = "\n\n".join(
        f"Related section {s.get('section_id')}: {s.get('body', '')[:700]}"
        for s in related
        if s.get("body")
    )
    r = base_row(doc, "comparative_analysis")
    r["instruction"] = (
        f"Compare section {section_id} of '{short_topic(title)}' with nearby or related "
        "sections in the same Act. Explain how they fit together and cite each source used."
    )
    r["context"] = (
        f"Current section {section_id}: {body[:1400]}"
        + (f"\n\n{related_context}" if related_context else "\n\nNo neighboring section text was available.")
    )
    r["reasoning"] = reasoning_for(doc, "comparative analysis", related)
    if related:
        comparisons = []
        for s in related:
            comparisons.append(
                f"- Section {section_id} states the current rule, while section "
                f"{s.get('section_id')} provides nearby context: {first_sentences(s.get('body', ''), 1)}"
            )
        citations = [make_citation(doc)] + [make_citation(s) for s in related]
        response = "\n".join(comparisons)
    else:
        citations = [make_citation(doc)]
        response = (
            f"- Section {section_id} is analyzed on its own because no neighboring section "
            "text was available in the cleaned corpus for comparison."
        )
    r["response"] = (
        f"Comparative analysis for section {section_id}:\n"
        f"{response}\n\n"
        f"The comparison is limited to the cited source text and should be verified against "
        f"the official portal.\n\n{DISCLAIMER}"
    )
    r["citations"] = citations
    return r


def make_rows_for_doc(doc: dict, siblings: list[dict]) -> list[dict]:
    rows = [
        row_plain_language_explanation(doc),
        row_legal_issue_spotting(doc),
        row_comparative_analysis(doc, siblings),
    ]
    return [row for row in rows if row is not None]


def row_statute_lookup(doc: dict) -> Optional[dict]:
    body = doc.get("body", "")
    if not body:
        return None
    section_id = doc.get("section_id") or "?"
    title = doc.get("source_title", "the cited act")
    snippet = body[:1200]
    r = base_row(doc, "statute_lookup")
    r["instruction"] = (
        f"What does section {section_id} of '{short_topic(title)}' provide? "
        "Quote the operative text and cite the source."
    )
    r["context"] = snippet
    r["response"] = (
        f"Section {section_id} of {short_topic(title)} provides:\n\n"
        f"\"{first_sentences(snippet, 4)}\"\n\n"
        f"Citation: {doc.get('url','')}\n\n{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    r["confidence"] = "high" if len(body) > 400 else "medium"
    return r


def row_section_summarization(doc: dict) -> Optional[dict]:
    body = doc.get("body", "")
    if not body or len(body) < 120:
        return None
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    r = base_row(doc, "section_summarization")
    r["instruction"] = (
        f"Summarize section {section_id} of '{short_topic(title)}' in three "
        "neutral sentences. Do not introduce facts not present in the source."
    )
    r["context"] = body[:1800]
    r["response"] = (
        f"Summary of section {section_id} ({short_topic(title)}):\n"
        f"{first_sentences(body, 3)}\n\n"
        f"Source: {doc.get('url','')}\n\n{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    return r


def row_issue_spotting(doc: dict) -> Optional[dict]:
    body = doc.get("body", "")
    clauses = doc.get("clauses") or []
    if not clauses and len(body) < 300:
        return None
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    bullets = []
    for c in clauses[:5]:
        m = c.get("marker")
        t = c.get("text", "")
        if t:
            bullets.append(f"- Issue raised by sub-clause ({m}): {first_sentences(t, 1)}")
    if not bullets:
        bullets.append(f"- General issue framed by section {section_id}: {first_sentences(body, 1)}")
    r = base_row(doc, "issue_spotting")
    r["instruction"] = (
        f"Identify the legal issues raised by section {section_id} of "
        f"'{short_topic(title)}'. Base each issue strictly on the provided text."
    )
    r["context"] = body[:1800]
    r["response"] = (
        "Issues grounded in the cited section:\n"
        + "\n".join(bullets)
        + f"\n\nSource: {doc.get('url','')}\n\n{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    return r


def row_legal_memo(doc: dict) -> Optional[dict]:
    body = doc.get("body", "")
    if len(body) < 200:
        return None
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    r = base_row(doc, "legal_memo")
    r["instruction"] = (
        f"Draft a short legal research memo on section {section_id} of "
        f"'{short_topic(title)}'. Use only the provided source text. "
        "Structure: Issue, Rule, Application, Conclusion."
    )
    r["context"] = body[:1800]
    summary = first_sentences(body, 2)
    r["response"] = (
        f"MEMORANDUM\nRe: Section {section_id}, {short_topic(title)}\n\n"
        f"Issue: What does section {section_id} provide and how is it framed?\n\n"
        f"Rule: {summary}\n\n"
        f"Application: The cited text governs the matters described above; "
        f"no facts beyond the source were assumed.\n\n"
        f"Conclusion: For any matter touching this section, consult the full "
        f"text at {doc.get('url','')} and a qualified Bangladeshi advocate.\n\n"
        f"{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    return r


def row_citation_extraction(doc: dict) -> Optional[dict]:
    title = doc.get("source_title", "")
    section_id = doc.get("section_id")
    if not section_id:
        return None
    body = doc.get("body", "")
    r = base_row(doc, "citation_extraction")
    r["instruction"] = (
        "Extract the formal citation for the following Bangladesh legal text "
        "as 'Act Title, Section N (URL)'."
    )
    r["context"] = (body[:600] + ("..." if len(body) > 600 else ""))
    r["response"] = f"{short_topic(title)}, Section {section_id} ({doc.get('url','')})"
    r["citations"] = [make_citation(doc)]
    r["confidence"] = "high"
    return r


def row_clarification(doc: dict) -> Optional[dict]:
    title = doc.get("source_title", "")
    section_id = doc.get("section_id") or "?"
    if not doc.get("body"):
        return None
    r = base_row(doc, "clarification")
    r["instruction"] = (
        f"A user asks: 'Will section {section_id} of {short_topic(title)} apply "
        f"to my situation?' Respond by asking only for the facts that are "
        "missing and necessary to answer."
    )
    r["context"] = doc.get("body", "")[:1200]
    r["response"] = (
        "I cannot apply this section without more facts. Please provide:\n"
        "- The parties involved and their roles.\n"
        "- The date and place where the conduct occurred.\n"
        "- The specific act or omission at issue.\n"
        "- Any prior proceedings or notices.\n"
        f"Once provided, I can compare your facts to the language of section "
        f"{section_id} ({doc.get('url','')}).\n\n{DISCLAIMER}"
    )
    r["citations"] = [make_citation(doc)]
    r["confidence"] = "medium"
    return r


def row_refusal(doc: dict) -> Optional[dict]:
    title = doc.get("source_title", "")
    r = base_row(doc, "refusal")
    r["instruction"] = (
        f"Based on the snippet below alone, predict the outcome of a future "
        f"Supreme Court of Bangladesh judgment interpreting '{short_topic(title)}'."
    )
    r["context"] = (doc.get("body", "")[:400] + "...") if doc.get("body") else ""
    r["response"] = (
        "I cannot answer this safely. The retrieved snippet does not contain "
        "case law, judicial reasoning, or a verifiable prediction basis, and "
        "I will not speculate about future judgments. Please supply specific "
        "decided cases or appellate authorities and I will analyze those.\n\n"
        + DISCLAIMER
    )
    r["citations"] = []
    r["confidence"] = "high"
    r["refusal_reason"] = "insufficient_evidence_predictive_question"
    return r


def iter_docs(in_dir: str) -> Iterable[dict]:
    for fp in sorted(glob(os.path.join(in_dir, "*.json"))):
        with open(fp, "r", encoding="utf-8") as f:
            yield json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="data/clean")
    p.add_argument("--out", default="data/dataset.jsonl")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    random.seed(args.seed)
    rows: list[dict] = []
    docs = list(iter_docs(args.in_dir))
    if not docs:
        log.error("no cleaned documents in %s", args.in_dir)
        return 1

    by_act = build_act_index(docs)
    for doc in docs:
        if doc.get("source_type") != "section_page":
            continue
        for row in make_rows_for_doc(doc, nearest_siblings(doc, by_act)):
            if not row["citations"]:
                continue
            rows.append(row)

    # Light shuffle for downstream split.
    random.shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote %d rows to %s", len(rows), args.out)

    # Quick task-type summary.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["task_type"]] = counts.get(r["task_type"], 0) + 1
    log.info("task distribution: %s", json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
