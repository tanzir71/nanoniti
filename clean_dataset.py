"""
clean_dataset.py

Read parsed JSON documents from data/parsed/, normalize text, deduplicate by
content hash, and write cleaned documents to data/clean/<sha1>.json.

- Strip boilerplate, normalize whitespace.
- Reject documents with empty body or too-short content.
- Hash on normalized body for dedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s clean_dataset %(message)s",
)
log = logging.getLogger("clean_dataset")

BOILERPLATE_PATTERNS = [
    r"Copyright\s*©.*",
    r"All rights reserved\.?",
    r"Ministry of Law,\s*Justice and Parliamentary Affairs",
    r"Government of the People'?s Republic of Bangladesh",
    r"Last Updated.*",
    r"Print\s+View",
    r"Home\s+>\s+.+",
]


def normalize(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def body_hash(doc: dict) -> str:
    body = doc.get("body") or " ".join(doc.get("preamble", []))
    body = normalize(body)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def is_keepable(doc: dict) -> bool:
    if doc.get("source_type") == "section_page":
        return len(doc.get("body", "")) >= 80
    if doc.get("source_type") == "act_page":
        return bool(doc.get("title")) and (
            len(doc.get("sections") or []) > 0 or len(doc.get("preamble") or []) > 0
        )
    return False


def clean_doc(doc: dict) -> dict:
    if "body" in doc:
        doc["body"] = normalize(doc["body"])
    if "preamble" in doc:
        doc["preamble"] = [normalize(p) for p in doc["preamble"] if normalize(p)]
    if "section_heading" in doc:
        doc["section_heading"] = normalize(doc["section_heading"])
    if "clauses" in doc:
        doc["clauses"] = [
            {"marker": c["marker"], "text": normalize(c["text"])}
            for c in doc["clauses"]
            if normalize(c.get("text", ""))
        ]
    return doc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="data/parsed")
    p.add_argument("--out-dir", default="data/clean")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(args.in_dir, "*.json")))
    if not files:
        log.error("no parsed files in %s", args.in_dir)
        return 1

    seen_hashes: set[str] = set()
    kept = 0
    dropped_dup = 0
    dropped_empty = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc = clean_doc(doc)
        if not is_keepable(doc):
            dropped_empty += 1
            continue
        h = body_hash(doc)
        if h in seen_hashes:
            dropped_dup += 1
            continue
        seen_hashes.add(h)
        doc["content_hash"] = h
        out_path = os.path.join(args.out_dir, f"{doc['sha1']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        kept += 1
    log.info("clean kept=%d dropped_dup=%d dropped_empty=%d", kept, dropped_dup, dropped_empty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
