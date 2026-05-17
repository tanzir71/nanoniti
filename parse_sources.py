"""
parse_sources.py

Parse raw HTML files referenced in data/manifest.json into structured JSON
documents under data/parsed/<sha1>.json.

For each record:
- Extract title, act metadata, section number, headings, and clause text.
- Preserve structural numbering (chapters, parts, sections, sub-sections).
- Skip and log records that cannot be parsed reliably.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s parse_sources %(message)s",
)
log = logging.getLogger("parse_sources")


def text_of(node: Tag) -> str:
    return re.sub(r"\s+", " ", node.get_text(separator=" ", strip=True)).strip()


def parse_act_page(soup: BeautifulSoup) -> dict:
    out: dict = {"chapters": [], "sections": []}
    title_el = soup.find(["h1", "h2"])
    if title_el:
        out["title"] = text_of(title_el)

    # Act metadata table (preamble / enactment date) varies, capture nearby paragraphs.
    paragraphs = []
    for p in soup.find_all("p"):
        t = text_of(p)
        if t:
            paragraphs.append(t)
    out["preamble"] = paragraphs[:8]

    # Capture chapter / part headings if present.
    for h in soup.find_all(["h3", "h4"]):
        t = text_of(h)
        if t and re.search(r"\b(CHAPTER|PART)\b", t, flags=re.IGNORECASE):
            out["chapters"].append(t)

    # Capture section index from anchor list.
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "section-" in href:
            label = text_of(a)
            if label:
                out["sections"].append({"label": label, "href": href})
    return out


def parse_section_page(soup: BeautifulSoup) -> dict:
    out: dict = {}
    title_el = soup.find(["h1", "h2", "h3"])
    if title_el:
        out["section_heading"] = text_of(title_el)

    # The portal renders the section body inside a main container.
    candidates = soup.find_all(["div", "article", "section"])
    best_text = ""
    for c in candidates:
        t = text_of(c)
        if len(t) > len(best_text):
            best_text = t

    # Fallback: collect <p> blocks.
    if len(best_text) < 200:
        ps = [text_of(p) for p in soup.find_all("p")]
        best_text = "\n".join([p for p in ps if p])

    out["body"] = best_text

    # Extract sub-clause numbering, e.g. (1), (2), (a), (b).
    clauses: list[dict] = []
    for m in re.finditer(
        r"\(([0-9]{1,3}|[a-zA-Z]{1,3})\)\s+([^()]{2,}?)(?=\([0-9a-zA-Z]{1,3}\)|$)",
        best_text,
    ):
        clauses.append({"marker": m.group(1), "text": m.group(2).strip()[:1200]})
    out["clauses"] = clauses[:50]
    return out


def parse_record(record: dict) -> Optional[dict]:
    raw_path = record["raw_path"]
    if not os.path.exists(raw_path):
        log.warning("raw missing for %s", record["url"])
        return None
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception as e:
        log.warning("read failed %s: %s", raw_path, e)
        return None

    soup = BeautifulSoup(html, "lxml")
    parsed: dict = {
        "sha1": record["sha1"],
        "url": record["url"],
        "source_title": record["source_title"],
        "source_type": record["source_type"],
        "jurisdiction": record["jurisdiction"],
        "act_id": record.get("act_id"),
        "section_id": record.get("section_id"),
        "retrieved_at": record["retrieved_at"],
    }

    if record["source_type"] == "act_page":
        parsed.update(parse_act_page(soup))
    elif record["source_type"] == "section_page":
        parsed.update(parse_section_page(soup))
    elif record["source_type"] == "volume_index":
        # Volume indexes are navigational only; we keep act enumeration.
        acts = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "act-details-" in href:
                acts.append({"href": href, "label": text_of(a)})
        parsed["act_links"] = acts
    else:
        return None

    # Reliability check: must have at least some content.
    text_blob = json.dumps(parsed, ensure_ascii=False)
    if len(text_blob) < 300 and record["source_type"] != "volume_index":
        log.warning("parsed content too small for %s, skipping", record["url"])
        return None

    return parsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--out-dir", default="data/parsed")
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        log.error("manifest not found: %s", args.manifest)
        return 1
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = 0
    n_fail = 0
    for rec in manifest["records"]:
        parsed = parse_record(rec)
        if not parsed:
            n_fail += 1
            continue
        out_path = os.path.join(args.out_dir, f"{rec['sha1']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        n_ok += 1
    log.info("parsed ok=%d failed=%d", n_ok, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
