"""
collect_sources.py

Multi-source recursive harvester for Bangladesh legal sources.

Sources (each connector is bounded and independently toggleable):
  - bdlaws:       Primary + secondary legislation on http://bdlaws.minlaw.gov.bd
                  (acts, repealed acts, rules, regulations, SROs, amendments).
  - supremecourt: Public bulletins / cause lists on supremecourt.gov.bd
                  (judgment summaries when discoverable).
  - chancery:     Headnotes on chancerylawchronicles.com (when reachable).
  - gazettes:     Bangladesh Gazette notifications linked from bdlaws acts.

Capabilities:
  - HTML and PDF fetch with rate limiting and exponential-backoff retries.
  - PDF text extraction with column-aware parsing; OCR fallback for image-only
    documents (see pdf_processor.py).
  - Relationship mapping: amendments, repeals, parent_act linkage, and detected
    cross-references between acts.
  - Recursive link-follower bounded by --max-depth.
  - Per-record completeness score over {title, commencement_date, preamble,
    sections}.
  - Failures appended to data/failed_sources.log.

Re-runs are safe: previously fetched URLs are skipped, and amendments / refs
are merged into the manifest in place.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from pdf_processor import download_pdf, extract_pdf

DEFAULT_HF_BASELINE = "sakhadib/Bangladesh-Legal-Acts-Dataset"
DEFAULT_DELTA_AFTER = "2025-07-31"
DEFAULT_DELTA_MAX_MISSES = 5

# Top 20 critical Bangladesh acts targeted for SRO enrichment. Names are
# matched case-insensitively against bdlaws act titles.
CRITICAL_ACTS: list[tuple[str, str]] = [
    ("Penal Code", "1860"),
    ("Code of Criminal Procedure", "1898"),
    ("Code of Civil Procedure", "1908"),
    ("Evidence Act", "1872"),
    ("Contract Act", "1872"),
    ("Transfer of Property Act", "1882"),
    ("Specific Relief Act", "1877"),
    ("Limitation Act", "1908"),
    ("Constitution of the People's Republic of Bangladesh", "1972"),
    ("Companies Act", "1994"),
    ("Income Tax Ordinance", "1984"),
    ("Value Added Tax and Supplementary Duty Act", "2012"),
    ("Customs Act", "1969"),
    ("Bangladesh Labour Act", "2006"),
    ("Negotiable Instruments Act", "1881"),
    ("Registration Act", "1908"),
    ("Stamp Act", "1899"),
    ("Arbitration Act", "2001"),
    ("Anti-Corruption Commission Act", "2004"),
    ("Money Laundering Prevention Act", "2012"),
]

# Several newer/high-value acts are titled in Bengali on bdlaws, so English
# title matching is not enough. These ids are the portal's act-print ids.
CRITICAL_ACT_ID_OVERRIDES: dict[tuple[str, str], str] = {
    ("Companies Act", "1994"): "788",
    ("Value Added Tax and Supplementary Duty Act", "2012"): "1106",
    ("Bangladesh Labour Act", "2006"): "952",
    ("Arbitration Act", "2001"): "850",
    ("Anti-Corruption Commission Act", "2004"): "914",
    ("Money Laundering Prevention Act", "2012"): "1088",
}

BASE_BDLAWS = "http://bdlaws.minlaw.gov.bd/"
VOLUME_URL_FMT = "http://bdlaws.minlaw.gov.bd/volume-{volume}.html"
ACT_URL_FMT = "http://bdlaws.minlaw.gov.bd/act-details-{act_id}.html"
SECTION_URL_FMT = "http://bdlaws.minlaw.gov.bd/act-{act_id}/section-{section_id}.html"

SUPREME_COURT_HOME = "http://www.supremecourt.gov.bd/"
CHANCERY_HOME = "https://www.chancerylawchronicles.com/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "BangladeshLegalAssistantCollector/0.2 "
        "(+research-only; contact: tanzir71@gmail.com)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s collect_sources %(message)s",
)
log = logging.getLogger("collect_sources")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now_utc().isoformat()


def retrieval_date_from_iso(ts: str) -> str:
    return (ts or now_iso())[:10]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    sha1: str
    url: str
    source_title: str
    source_type: str        # volume_index | act_page | section_page | rule | sro |
                            # amendment | gazette | judgment | headnote | pdf_doc
    jurisdiction: str
    act_id: Optional[str]
    section_id: Optional[str]
    retrieved_at: str
    raw_path: str
    http_status: int
    content_type: str = "html"
    ocr_used: bool = False
    parent_id: Optional[str] = None       # sha1 of parent record (e.g., the Act
                                          # that this amendment / SRO / rule
                                          # belongs to).
    relationship_type: Optional[str] = None  # amendment_of | rule_of | sro_of |
                                             # repeal_of | gazette_of | references
    completeness_score: float = 0.0
    references: list[dict] = field(default_factory=list)  # cross-act references
    depth: int = 0
    source_origin: str = "live_portal"   # hf_baseline | live_portal | supreme_court
                                         # | chancery | gazette_portal
    section_text_sha256: Optional[str] = None  # global dedup key for sections
    retrieval_date: Optional[str] = None
    publication_date: Optional[str] = None


class FetchError(Exception):
    pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha1_of(s: str | bytes) -> str:
    b = s.encode("utf-8") if isinstance(s, str) else s
    return hashlib.sha1(b).hexdigest()


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((requests.RequestException, FetchError)),
)
def http_get(session: requests.Session, url: str, timeout: int = 30) -> requests.Response:
    resp = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    if resp.status_code >= 500:
        raise FetchError(f"server error {resp.status_code} for {url}")
    return resp


def response_text(resp: requests.Response) -> str:
    """Decode bdlaws pages from bytes so Bengali text is not mojibake."""
    ctype = resp.headers.get("Content-Type", "").lower()
    if "charset=utf-8" in ctype or "charset=UTF-8".lower() in ctype:
        return resp.content.decode("utf-8", errors="replace")
    apparent = (resp.apparent_encoding or "").lower()
    if apparent in {"utf-8", "utf_8"}:
        return resp.content.decode("utf-8", errors="replace")
    return resp.text


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": "1.1",
        "description": "Source manifest for the Bangladesh legal assistant pipeline.",
        "primary_source": BASE_BDLAWS,
        "records": [],
        "relationships": [],
    }


def save_manifest(manifest: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def find_record(manifest: dict, url: str) -> Optional[dict]:
    for r in manifest["records"]:
        if r["url"] == url:
            return r
    return None


def already_fetched(manifest: dict, url: str) -> bool:
    return find_record(manifest, url) is not None


def log_failure(failed_log_path: str, url: str, reason: str) -> None:
    os.makedirs(os.path.dirname(failed_log_path) or ".", exist_ok=True)
    ts = now_iso()
    with open(failed_log_path, "a", encoding="utf-8") as f:
        f.write(f"{ts} | {url} | {reason}\n")
    log.warning("failed: %s -> %s", url, reason)


def write_raw_bytes(content: bytes, raw_dir: str, ext: str) -> tuple[str, str]:
    digest = sha1_of(content)
    path = os.path.join(raw_dir, f"{digest}.{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)
    return digest, path


def html_doc(title: str, body: str, *, heading: str = "h1") -> str:
    title = html_lib.escape(title or "Bangladesh legal source")
    body = html_lib.escape(body or "").replace("\n", "<br>\n")
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{title}</title></head><body>"
        f"<{heading}>{title}</{heading}>"
        f"<div class=\"txt-details\" id=\"sec-dec\">{body}</div>"
        "</body></html>"
    )


def write_raw_text(content: str, raw_dir: str, ext: str = "txt") -> tuple[str, str]:
    return write_raw_bytes(content.encode("utf-8"), raw_dir, ext)


def extract_title_html(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h = soup.find(["h1", "h2"])
    if h and h.get_text(strip=True):
        return h.get_text(strip=True)
    return fallback


# ---------------------------------------------------------------------------
# Completeness score
# ---------------------------------------------------------------------------

COMMENCEMENT_PAT = re.compile(
    r"(?:date\s+of\s+commencement|commenced\s+on|came\s+into\s+force|"
    r"shall\s+come\s+into\s+force)",
    re.IGNORECASE,
)
PREAMBLE_PAT = re.compile(r"\bpreamble\b|\bwhereas\b", re.IGNORECASE)
SECTION_PAT = re.compile(r"(?:^|\W)section\s+\d+|\(\s*\d+\s*\)", re.IGNORECASE)


def completeness_score(text: str, title: str | None) -> float:
    score = 0.0
    if title and len(title.strip()) > 5:
        score += 0.25
    if COMMENCEMENT_PAT.search(text or ""):
        score += 0.25
    if PREAMBLE_PAT.search(text or ""):
        score += 0.25
    if SECTION_PAT.search(text or ""):
        score += 0.25
    return round(score, 2)


# ---------------------------------------------------------------------------
# Cross-reference detection
# ---------------------------------------------------------------------------

# Captures patterns like:
#   "the Transfer of Property Act, 1882"
#   "Penal Code, 1860"
#   "Code of Civil Procedure, 1908"
ACT_REF_PAT = re.compile(
    r"\b((?:the\s+)?(?:[A-Z][A-Za-z]+(?:\s+(?:of|and|&)\s+|\s+)){1,6}"
    r"(?:Act|Code|Ordinance|Rules|Regulations|Order))\s*,?\s*(\d{4})",
    re.IGNORECASE,
)


def detect_references(text: str, exclude_title: str = "") -> list[dict]:
    refs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for m in ACT_REF_PAT.finditer(text or ""):
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,.")
        year = m.group(2)
        if exclude_title and name.lower() in exclude_title.lower():
            continue
        key = (name.lower(), year)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"name": name, "year": year, "raw": m.group(0).strip()})
    return refs[:25]


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def parse_act_id(url: str) -> Optional[str]:
    m = re.search(r"act-details-(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"/act-(\d+)/", url)
    if m:
        return m.group(1)
    return None


def parse_section_id(url: str) -> Optional[str]:
    m = re.search(r"section-(\d+)", url)
    return m.group(1) if m else None


def looks_like_pdf(url: str, content_type: str = "") -> bool:
    return url.lower().endswith(".pdf") or "pdf" in content_type.lower()


# ---------------------------------------------------------------------------
# Core fetch + record
# ---------------------------------------------------------------------------

def record_html(
    manifest: dict,
    *,
    url: str,
    html: str,
    raw_dir: str,
    source_type: str,
    act_id: Optional[str],
    section_id: Optional[str],
    http_status: int,
    title_hint: str,
    parent_id: Optional[str],
    relationship_type: Optional[str],
    depth: int,
) -> SourceRecord:
    digest, raw_path = write_raw_bytes(html.encode("utf-8"), raw_dir, "html")
    soup = BeautifulSoup(html, "lxml")
    body_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    title = extract_title_html(html, title_hint)
    refs = detect_references(body_text, exclude_title=title)
    retrieved_at = now_iso()
    rec = SourceRecord(
        sha1=digest,
        url=url,
        source_title=title,
        source_type=source_type,
        jurisdiction="Bangladesh",
        act_id=act_id,
        section_id=section_id,
        retrieved_at=retrieved_at,
        raw_path=raw_path,
        http_status=http_status,
        content_type="html",
        ocr_used=False,
        parent_id=parent_id,
        relationship_type=relationship_type,
        completeness_score=completeness_score(body_text, title),
        references=refs,
        depth=depth,
        retrieval_date=retrieval_date_from_iso(retrieved_at),
    )
    manifest["records"].append(asdict(rec))
    for r in refs:
        manifest["relationships"].append({
            "from": digest,
            "to_name": r["name"],
            "to_year": r["year"],
            "type": "references",
        })
    if parent_id and relationship_type:
        manifest["relationships"].append({
            "from": digest,
            "to": parent_id,
            "type": relationship_type,
        })
    return rec


def record_pdf(
    manifest: dict,
    *,
    url: str,
    pdf_path: str,
    raw_dir: str,
    source_type: str,
    act_id: Optional[str],
    section_id: Optional[str],
    http_status: int,
    title_hint: str,
    parent_id: Optional[str],
    relationship_type: Optional[str],
    depth: int,
) -> Optional[SourceRecord]:
    with open(pdf_path, "rb") as f:
        content = f.read()
    digest = sha1_of(content)
    # Move into raw_dir under canonical name if not already there.
    canonical = os.path.join(raw_dir, f"{digest}.pdf")
    if os.path.abspath(canonical) != os.path.abspath(pdf_path):
        if not os.path.exists(canonical):
            with open(canonical, "wb") as f:
                f.write(content)
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    extraction = extract_pdf(canonical)
    if not extraction.text:
        return None
    text_path = canonical.replace(".pdf", ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(extraction.text)

    refs = detect_references(extraction.text, exclude_title=title_hint)
    retrieved_at = now_iso()
    rec = SourceRecord(
        sha1=digest,
        url=url,
        source_title=title_hint,
        source_type=source_type,
        jurisdiction="Bangladesh",
        act_id=act_id,
        section_id=section_id,
        retrieved_at=retrieved_at,
        raw_path=canonical,
        http_status=http_status,
        content_type="pdf",
        ocr_used=extraction.ocr_used,
        parent_id=parent_id,
        relationship_type=relationship_type,
        completeness_score=completeness_score(extraction.text, title_hint),
        references=refs,
        depth=depth,
        retrieval_date=retrieval_date_from_iso(retrieved_at),
    )
    manifest["records"].append(asdict(rec))
    for r in refs:
        manifest["relationships"].append({
            "from": digest,
            "to_name": r["name"],
            "to_year": r["year"],
            "type": "references",
        })
    if parent_id and relationship_type:
        manifest["relationships"].append({
            "from": digest,
            "to": parent_id,
            "type": relationship_type,
        })
    return rec


# ---------------------------------------------------------------------------
# bdlaws connector (primary + secondary legislation)
# ---------------------------------------------------------------------------

AMENDMENT_KEYWORDS = ("amendment", "amending", "amended")
RULES_KEYWORDS = ("rules", "regulation", "regulations", "by-law", "bye-law", "bylaw")
SRO_KEYWORDS = ("sro", "s.r.o", "statutory rules and orders", "notification")
GAZETTE_KEYWORDS = ("gazette",)
REPEAL_KEYWORDS = ("repealed", "repeal")


def classify_link(text: str) -> tuple[str, Optional[str]]:
    """Return (source_type, relationship_type) for a child link off an act page."""
    t = (text or "").lower()
    if any(k in t for k in AMENDMENT_KEYWORDS):
        return "amendment", "amendment_of"
    if any(k in t for k in SRO_KEYWORDS):
        return "sro", "sro_of"
    if any(k in t for k in RULES_KEYWORDS):
        return "rule", "rule_of"
    if any(k in t for k in GAZETTE_KEYWORDS):
        return "gazette", "gazette_of"
    if any(k in t for k in REPEAL_KEYWORDS):
        return "amendment", "repeal_of"
    return "section_page", None


def enumerate_act_links(volume_html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(volume_html, "lxml")
    out: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        if "act-details-" in a["href"]:
            full = urljoin(BASE_BDLAWS, a["href"])
            out.append((full, a.get_text(strip=True) or full))
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for url, title in out:
        if url not in seen:
            seen.add(url)
            uniq.append((url, title))
    return uniq


def enumerate_child_links(act_html: str, act_id: str) -> list[tuple[str, str]]:
    """Section links and any other linked subsidiary docs (PDFs, gazettes, etc)."""
    soup = BeautifulSoup(act_html, "lxml")
    out: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(BASE_BDLAWS, href)
        label = a.get_text(strip=True)
        if not full.startswith("http"):
            continue
        if f"act-{act_id}/section-" in href:
            out.append((full, label))
        elif looks_like_pdf(full):
            out.append((full, label))
        elif any(k in label.lower() for k in
                 AMENDMENT_KEYWORDS + RULES_KEYWORDS + SRO_KEYWORDS + GAZETTE_KEYWORDS):
            out.append((full, label))
    # Dedupe.
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for u, t in out:
        if u not in seen:
            seen.add(u)
            uniq.append((u, t))
    return uniq


def harvest_bdlaws(
    session: requests.Session,
    manifest: dict,
    *,
    raw_dir: str,
    failed_log: str,
    volumes: Iterable[int],
    max_acts: int,
    max_children_per_act: int,
    delay: float,
    max_depth: int,
    manifest_path: str,
) -> None:
    acts_seen = 0
    for vol in volumes:
        vol_url = VOLUME_URL_FMT.format(volume=vol)
        vol_html = _fetch_html_or_cached(session, manifest, vol_url, raw_dir, failed_log, delay,
                                         source_type="volume_index",
                                         title_hint=f"Volume {vol}", depth=0)
        if not vol_html:
            continue
        save_manifest(manifest, manifest_path)

        for act_url, act_title in enumerate_act_links(vol_html):
            if max_acts and acts_seen >= max_acts:
                log.info("max_acts reached (%d)", max_acts)
                return
            act_id = parse_act_id(act_url)
            if not act_id:
                continue
            act_html = _fetch_html_or_cached(session, manifest, act_url, raw_dir, failed_log, delay,
                                             source_type="act_page",
                                             title_hint=act_title, act_id=act_id, depth=0)
            if not act_html:
                continue
            acts_seen += 1
            parent_rec = find_record(manifest, act_url)
            parent_sha = parent_rec["sha1"] if parent_rec else None
            save_manifest(manifest, manifest_path)

            for child_url, child_label in enumerate_child_links(act_html, act_id)[:max_children_per_act]:
                stype, rel = classify_link(child_label)
                # Sections go to the standard path.
                if "section-" in child_url:
                    sec_id = parse_section_id(child_url)
                    _fetch_html_or_cached(
                        session, manifest, child_url, raw_dir, failed_log, delay,
                        source_type="section_page",
                        title_hint=f"{act_title} - section {sec_id}",
                        act_id=act_id, section_id=sec_id,
                        parent_id=parent_sha, relationship_type="section_of",
                        depth=1,
                    )
                elif looks_like_pdf(child_url):
                    _fetch_pdf(
                        session, manifest, child_url, raw_dir, failed_log, delay,
                        source_type=stype if stype != "section_page" else "pdf_doc",
                        title_hint=child_label or f"{act_title} - linked PDF",
                        act_id=act_id,
                        parent_id=parent_sha, relationship_type=rel,
                        depth=1,
                    )
                else:
                    # HTML amendment / rules / gazette landing.
                    _fetch_html_or_cached(
                        session, manifest, child_url, raw_dir, failed_log, delay,
                        source_type=stype,
                        title_hint=child_label or f"{act_title} - {stype}",
                        act_id=act_id,
                        parent_id=parent_sha, relationship_type=rel,
                        depth=1,
                    )
                save_manifest(manifest, manifest_path)

            # Recursive cross-reference following (HTML only, bounded by depth).
            if max_depth > 1 and parent_rec:
                _follow_references(session, manifest, parent_rec, raw_dir, failed_log,
                                   delay, max_depth, manifest_path)


def _follow_references(
    session: requests.Session,
    manifest: dict,
    parent_rec: dict,
    raw_dir: str,
    failed_log: str,
    delay: float,
    max_depth: int,
    manifest_path: str,
) -> None:
    """Best-effort: walk references and attempt to locate matching act pages
    inside already-collected acts. We only record the link, we do not search
    the public web from arbitrary text."""
    refs = parent_rec.get("references") or []
    if not refs:
        return
    # Build a name->record index from already-collected acts.
    index: dict[str, dict] = {}
    for r in manifest["records"]:
        if r["source_type"] == "act_page":
            index[r["source_title"].lower()] = r
    for ref in refs:
        target = None
        for title, rec in index.items():
            if ref["name"].lower() in title and ref["year"] in title:
                target = rec
                break
        if target:
            manifest["relationships"].append({
                "from": parent_rec["sha1"],
                "to": target["sha1"],
                "type": "references_resolved",
                "ref_name": ref["name"],
                "ref_year": ref["year"],
            })
    save_manifest(manifest, manifest_path)


def _fetch_html_or_cached(
    session: requests.Session,
    manifest: dict,
    url: str,
    raw_dir: str,
    failed_log: str,
    delay: float,
    *,
    source_type: str,
    title_hint: str,
    act_id: Optional[str] = None,
    section_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    depth: int = 0,
) -> Optional[str]:
    existing = find_record(manifest, url)
    if existing:
        if os.path.exists(existing["raw_path"]) and existing["content_type"] == "html":
            with open(existing["raw_path"], "r", encoding="utf-8") as f:
                return f.read()
        return None
    try:
        log.info("fetch html: %s", url)
        resp = http_get(session, url)
        time.sleep(delay)
    except Exception as e:  # noqa: BLE001
        log_failure(failed_log, url, f"fetch_error:{e.__class__.__name__}:{e}")
        return None
    if resp.status_code != 200:
        log_failure(failed_log, url, f"http_{resp.status_code}")
        return None
    ctype = resp.headers.get("Content-Type", "")
    if looks_like_pdf(url, ctype):
        # Was advertised as HTML but is actually a PDF; redirect to PDF path.
        pdf_path = os.path.join(raw_dir, f"_tmp_{sha1_of(url)}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        record_pdf(manifest,
                   url=url, pdf_path=pdf_path, raw_dir=raw_dir,
                   source_type=source_type if source_type != "section_page" else "pdf_doc",
                   act_id=act_id, section_id=section_id,
                   http_status=resp.status_code,
                   title_hint=title_hint, parent_id=parent_id,
                   relationship_type=relationship_type, depth=depth)
        return None
    try:
        html = response_text(resp)
        if "html" not in (ctype.lower() or "html") and "<html" not in html.lower():
            log_failure(failed_log, url, f"unexpected_content_type:{ctype}")
            return None
        record_html(manifest,
                    url=url, html=html, raw_dir=raw_dir,
                    source_type=source_type, act_id=act_id, section_id=section_id,
                    http_status=resp.status_code, title_hint=title_hint,
                    parent_id=parent_id, relationship_type=relationship_type,
                    depth=depth)
        return html
    except Exception as e:  # noqa: BLE001
        log_failure(failed_log, url, f"parse_error:{e.__class__.__name__}:{e}")
        return None


def _fetch_pdf(
    session: requests.Session,
    manifest: dict,
    url: str,
    raw_dir: str,
    failed_log: str,
    delay: float,
    *,
    source_type: str,
    title_hint: str,
    act_id: Optional[str] = None,
    section_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    depth: int = 0,
) -> None:
    if already_fetched(manifest, url):
        return
    tmp = os.path.join(raw_dir, f"_tmp_{sha1_of(url)}.pdf")
    try:
        log.info("fetch pdf: %s", url)
        download_pdf(session, url, tmp)
        time.sleep(delay)
    except Exception as e:  # noqa: BLE001
        log_failure(failed_log, url, f"pdf_fetch_error:{e.__class__.__name__}:{e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return
    try:
        rec = record_pdf(
            manifest, url=url, pdf_path=tmp, raw_dir=raw_dir,
            source_type=source_type, act_id=act_id, section_id=section_id,
            http_status=200, title_hint=title_hint,
            parent_id=parent_id, relationship_type=relationship_type, depth=depth,
        )
        if rec is None:
            log_failure(failed_log, url, "pdf_extraction_empty")
    except Exception as e:  # noqa: BLE001
        log_failure(failed_log, url, f"pdf_record_error:{e.__class__.__name__}:{e}")


# ---------------------------------------------------------------------------
# Supreme Court of Bangladesh connector
# ---------------------------------------------------------------------------

def harvest_supreme_court(
    session: requests.Session,
    manifest: dict,
    *,
    raw_dir: str,
    failed_log: str,
    delay: float,
    max_pages: int,
    manifest_path: str,
) -> None:
    """Discover and fetch judgment summaries / cause lists when reachable."""
    home_html = _fetch_html_or_cached(
        session, manifest, SUPREME_COURT_HOME, raw_dir, failed_log, delay,
        source_type="court_index", title_hint="Supreme Court of Bangladesh", depth=0,
    )
    if not home_html:
        log.warning("supreme court home unreachable; skipping connector")
        return
    soup = BeautifulSoup(home_html, "lxml")
    candidates: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        label = a.get_text(strip=True).lower()
        href = a["href"]
        if any(k in label for k in ("judgment", "bulletin", "cause list", "verdict", "headnote")):
            candidates.append((urljoin(SUPREME_COURT_HOME, href), a.get_text(strip=True)))
    log.info("supreme court: %d candidate links", len(candidates))
    for url, label in candidates[:max_pages]:
        if looks_like_pdf(url):
            _fetch_pdf(session, manifest, url, raw_dir, failed_log, delay,
                       source_type="judgment", title_hint=label, depth=1)
        else:
            _fetch_html_or_cached(session, manifest, url, raw_dir, failed_log, delay,
                                  source_type="judgment", title_hint=label, depth=1)
        save_manifest(manifest, manifest_path)


# ---------------------------------------------------------------------------
# Chancery Law Chronicles connector
# ---------------------------------------------------------------------------

def harvest_chancery(
    session: requests.Session,
    manifest: dict,
    *,
    raw_dir: str,
    failed_log: str,
    delay: float,
    max_pages: int,
    manifest_path: str,
) -> None:
    home_html = _fetch_html_or_cached(
        session, manifest, CHANCERY_HOME, raw_dir, failed_log, delay,
        source_type="commentary_index", title_hint="Chancery Law Chronicles", depth=0,
    )
    if not home_html:
        log.warning("chancery home unreachable; skipping connector")
        return
    soup = BeautifulSoup(home_html, "lxml")
    candidates: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        label = a.get_text(strip=True).lower()
        href = a["href"]
        if any(k in label for k in ("headnote", "case", "judgment", "article", "summary")):
            candidates.append((urljoin(CHANCERY_HOME, href), a.get_text(strip=True)))
    log.info("chancery: %d candidate links", len(candidates))
    for url, label in candidates[:max_pages]:
        if looks_like_pdf(url):
            _fetch_pdf(session, manifest, url, raw_dir, failed_log, delay,
                       source_type="headnote", title_hint=label, depth=1)
        else:
            _fetch_html_or_cached(session, manifest, url, raw_dir, failed_log, delay,
                                  source_type="headnote", title_hint=label, depth=1)
        save_manifest(manifest, manifest_path)


# ---------------------------------------------------------------------------
# Global section dedup (SHA-256)
# ---------------------------------------------------------------------------

def normalize_section_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip().lower()
    return t


def sha256_of(s: str | bytes) -> str:
    b = s.encode("utf-8") if isinstance(s, str) else s
    return hashlib.sha256(b).hexdigest()


_BN_DIGIT_MAP = str.maketrans({
    "\u09e6": "0",
    "\u09e7": "1",
    "\u09e8": "2",
    "\u09e9": "3",
    "\u09ea": "4",
    "\u09eb": "5",
    "\u09ec": "6",
    "\u09ed": "7",
    "\u09ee": "8",
    "\u09ef": "9",
})

_MONTHS = {
    "january": "01",
    "jan": "01",
    "\u099c\u09be\u09a8\u09c1\u09df\u09be\u09b0\u09bf": "01",
    "\u099c\u09be\u09a8\u09c1\u09af\u09bc\u09be\u09b0\u09bf": "01",
    "february": "02",
    "feb": "02",
    "\u09ab\u09c7\u09ac\u09cd\u09b0\u09c1\u09df\u09be\u09b0\u09bf": "02",
    "\u09ab\u09c7\u09ac\u09cd\u09b0\u09c1\u09af\u09bc\u09be\u09b0\u09bf": "02",
    "march": "03",
    "mar": "03",
    "\u09ae\u09be\u09b0\u09cd\u099a": "03",
    "april": "04",
    "apr": "04",
    "\u098f\u09aa\u09cd\u09b0\u09bf\u09b2": "04",
    "may": "05",
    "\u09ae\u09c7": "05",
    "june": "06",
    "jun": "06",
    "\u099c\u09c1\u09a8": "06",
    "july": "07",
    "jul": "07",
    "\u099c\u09c1\u09b2\u09be\u0987": "07",
    "august": "08",
    "aug": "08",
    "\u0986\u0997\u09b8\u09cd\u099f": "08",
    "september": "09",
    "sep": "09",
    "\u09b8\u09c7\u09aa\u09cd\u099f\u09c7\u09ae\u09cd\u09ac\u09b0": "09",
    "october": "10",
    "oct": "10",
    "\u0985\u0995\u09cd\u099f\u09cb\u09ac\u09b0": "10",
    "november": "11",
    "nov": "11",
    "\u09a8\u09ad\u09c7\u09ae\u09cd\u09ac\u09b0": "11",
    "december": "12",
    "dec": "12",
    "\u09a1\u09bf\u09b8\u09c7\u09ae\u09cd\u09ac\u09b0": "12",
}


def ascii_digits(text: str) -> str:
    return (text or "").translate(_BN_DIGIT_MAP)


def parse_any_date(text: str) -> Optional[str]:
    """Return YYYY-MM-DD for common bdlaws/HF date forms."""
    if not text:
        return None
    t = ascii_digits(text)
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", t)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = re.search(r"(\d{1,2})\s+([^,\]\[]+?)\s*,?\s+(\d{4})", t, flags=re.IGNORECASE)
    if m:
        d, month_name, y = m.groups()
        key = re.sub(r"\s+", " ", month_name.strip().lower())
        mo = _MONTHS.get(key)
        if mo:
            return f"{int(y):04d}-{mo}-{int(d):02d}"
    m = re.search(r"\b(\d{4})\b", t)
    if m:
        return f"{int(m.group(1)):04d}-01-01"
    return None


def existing_section_hashes(manifest: dict) -> set[str]:
    return {
        r["section_text_sha256"]
        for r in manifest["records"]
        if r.get("section_text_sha256")
    }


def dedup_global(manifest: dict) -> dict:
    """Drop duplicate section records by section_text_sha256, keeping the
    earliest retrieved copy. Returns a summary dict."""
    seen: set[str] = set()
    kept: list[dict] = []
    dropped = 0
    for r in sorted(manifest["records"], key=lambda x: x.get("retrieved_at", "")):
        h = r.get("section_text_sha256")
        if h and r.get("source_type") == "section_page":
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
        kept.append(r)
    manifest["records"] = kept
    return {"unique_section_hashes": len(seen), "duplicates_removed": dropped}


def max_manifest_act_print_id(manifest: dict, *, origin: Optional[str] = None) -> int:
    ids: list[int] = []
    for r in manifest.get("records", []):
        if origin and r.get("source_origin") != origin:
            continue
        url = r.get("url", "")
        m = re.search(r"act-print-(\d+)\.html", url)
        if m:
            ids.append(int(m.group(1)))
        elif r.get("source_type") == "act_page":
            act_id = str(r.get("act_id") or "")
            if act_id.isdigit():
                ids.append(int(act_id))
    return max(ids) if ids else 0


def is_missing_act_print(html: str) -> bool:
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    return "404 Not Found" in text or "Requested page not found" in text


def parse_section_number(section_text: str, fallback: int) -> str:
    converted = ascii_digits(section_text or "")
    m = re.match(r"\s*(\d+[A-Za-z]?)\s*[\.\)\u0964]", converted)
    return m.group(1) if m else str(fallback)


def parse_bdlaws_act_print(html: str, url: str, act_id: str) -> Optional[dict]:
    if is_missing_act_print(html):
        return None
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find(id="printheader") or soup.find("h3") or soup.title
    title = re.sub(r"\s+", " ", title_el.get_text(" ", strip=True)).strip() if title_el else url
    pub_el = soup.find("p", class_="publish-date")
    pub_raw = pub_el.get_text(" ", strip=True) if pub_el else ""
    publication_date = parse_any_date(pub_raw)

    body_container = soup.find(id="hide") or soup.body or soup
    full_text = re.sub(r"\s+", " ", body_container.get_text(" ", strip=True)).strip()
    sections: list[dict] = []
    for idx, row in enumerate(soup.select("div.row.lineremoves, div.row.lineremove"), start=1):
        details = row.select_one(".txt-details")
        if not details:
            continue
        text = re.sub(r"\s+", " ", details.get_text(" ", strip=True)).strip()
        if not text:
            continue
        heading_el = row.select_one(".txt-head")
        heading = re.sub(r"\s+", " ", heading_el.get_text(" ", strip=True)).strip() if heading_el else ""
        sections.append({
            "section_id": parse_section_number(text, len(sections) + 1),
            "heading": heading,
            "text": text,
        })

    return {
        "act_id": act_id,
        "url": url,
        "title": title,
        "publication_date": publication_date,
        "publication_date_raw": pub_raw,
        "body": full_text,
        "sections": sections,
    }


def add_text_record(
    manifest: dict,
    *,
    raw_dir: str,
    url: str,
    title: str,
    body: str,
    source_type: str,
    act_id: Optional[str],
    section_id: Optional[str],
    source_origin: str,
    parent_id: Optional[str] = None,
    relationship_type: Optional[str] = None,
    section_text_sha256: Optional[str] = None,
    publication_date: Optional[str] = None,
    depth: int = 0,
) -> SourceRecord:
    raw_html = html_doc(title, body, heading="h3" if source_type == "section_page" else "h1")
    digest, raw_path = write_raw_text(raw_html, raw_dir, "html")
    retrieved_at = now_iso()
    refs = detect_references(body, exclude_title=title)
    rec = SourceRecord(
        sha1=digest,
        url=url,
        source_title=title,
        source_type=source_type,
        jurisdiction="Bangladesh",
        act_id=act_id,
        section_id=section_id,
        retrieved_at=retrieved_at,
        raw_path=raw_path,
        http_status=200,
        content_type="html",
        ocr_used=False,
        parent_id=parent_id,
        relationship_type=relationship_type,
        completeness_score=completeness_score(body, title),
        references=refs,
        depth=depth,
        source_origin=source_origin,
        section_text_sha256=section_text_sha256,
        retrieval_date=retrieval_date_from_iso(retrieved_at),
        publication_date=publication_date,
    )
    manifest["records"].append(asdict(rec))
    if parent_id and relationship_type:
        manifest["relationships"].append({
            "from": digest,
            "to": parent_id,
            "type": relationship_type,
        })
    for r in refs:
        manifest["relationships"].append({
            "from": digest,
            "to_name": r["name"],
            "to_year": r["year"],
            "type": "references",
        })
    return rec


def ensure_manifest_provenance(manifest: dict) -> None:
    for r in manifest.get("records", []):
        r.setdefault("source_origin", "unknown")
        if not r.get("retrieval_date"):
            r["retrieval_date"] = retrieval_date_from_iso(r.get("retrieved_at", ""))
        if "publication_date" not in r:
            r["publication_date"] = None


def normalize_title_key(text: str) -> str:
    t = ascii_digits(text or "").lower()
    t = re.sub(r"[\u200c\u200d\u2018\u2019'`]", "", t)
    t = re.sub(r"[-_/(),.:;]+", " ", t)
    t = re.sub(r"\bthe\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Hugging Face baseline loader
# ---------------------------------------------------------------------------

# Common field names we may encounter across HF Bangladesh legal datasets.
_HF_TEXT_FIELDS = ("section_text", "text", "content", "body", "section_body")
_HF_TITLE_FIELDS = ("act_name", "act_title", "title", "name")
_HF_ACT_ID_FIELDS = ("act_id", "act_number", "act_no", "act")
_HF_SECTION_FIELDS = ("section_id", "section_no", "section_number", "section")
_HF_URL_FIELDS = ("url", "source_url", "link", "act_url")
_HF_CHAPTER_FIELDS = ("chapter", "part", "chapter_title")
_HF_DATE_FIELDS = ("date_of_commencement", "commencement_date", "enacted", "year")


def _first(d: dict, keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return str(d[k])
    return None


def harvest_hf_baseline(
    manifest: dict,
    *,
    raw_dir: str,
    failed_log: str,
    dataset_id: str,
    manifest_path: str,
    max_rows: Optional[int] = None,
    local_files_only: bool = False,
    min_files: int = 1400,
) -> dict:
    """Pull the baseline acts dataset from the Hugging Face Hub and
    materialize each act + section as a manifest record. Sections are
    deduplicated globally by SHA-256 of normalized section text.

    The default dataset `sakhadib/Bangladesh-Legal-Acts-Dataset` ships as a
    directory of per-act JSON files (`acts/act-print-<n>.json`); each file
    contains `act_title`, `act_no`, `act_year`, `publication_date`,
    `source_url`, and a `sections` list of `{section_content: str}` blobs.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:  # noqa: BLE001
        log_failure(failed_log, f"hf://{dataset_id}", f"hf_hub_import_error:{e}")
        return {"loaded": 0, "skipped_dup": 0, "acts": 0, "sections": 0}

    log.info("snapshot_download HF baseline: %s", dataset_id)
    try:
        local_dir = snapshot_download(
            repo_id=dataset_id, repo_type="dataset",
            allow_patterns=["acts/*.json"],
            local_files_only=local_files_only,
        )
    except Exception as e:  # noqa: BLE001
        log_failure(failed_log, f"hf://{dataset_id}", f"hf_snapshot_error:{e}")
        return {"loaded": 0, "skipped_dup": 0, "acts": 0, "sections": 0}

    acts_dir = os.path.join(local_dir, "acts")
    if not os.path.isdir(acts_dir):
        log_failure(failed_log, f"hf://{dataset_id}", "acts_dir_missing")
        return {"loaded": 0, "skipped_dup": 0, "acts": 0, "sections": 0}

    files = sorted(
        f for f in os.listdir(acts_dir)
        if f.startswith("act-print-") and f.endswith(".json")
    )
    log.info("HF baseline: %d act files discovered", len(files))
    if min_files and len(files) < min_files:
        reason = f"hf_baseline_incomplete:{len(files)}_files_lt_{min_files}"
        log_failure(failed_log, f"hf://{dataset_id}", reason)
        return {
            "loaded": 0,
            "skipped_dup": 0,
            "acts": 0,
            "sections": 0,
            "files_seen": len(files),
            "incomplete": True,
            "reason": reason,
        }

    seen_hashes = existing_section_hashes(manifest)
    seen_act_urls = {r["url"] for r in manifest["records"] if r.get("source_type") == "act_page"}
    n_loaded_sections = 0
    n_skipped = 0
    n_acts = 0
    cap = max_rows if max_rows else len(files)

    for i, fn in enumerate(files):
        if i >= cap:
            break
        path = os.path.join(acts_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:  # noqa: BLE001
            log_failure(failed_log, f"hf://{dataset_id}/{fn}", f"json_read_error:{e}")
            continue

        title = (doc.get("act_title") or "Unknown Bangladesh Act").strip()
        title = re.sub(r"^\d+", "", title).strip()  # leading numeric junk in some titles
        act_year = str(doc.get("act_year") or "").strip()
        act_no = str(doc.get("act_no") or "").strip()
        source_url = doc.get("source_url") or f"hf://{dataset_id}/{fn}"
        publication_date = doc.get("publication_date") or doc.get("fetch_timestamp") or ""
        # Stable act_id: number embedded in filename (act-print-N.json -> N).
        m = re.search(r"act-print-(\d+)\.json$", fn)
        act_id = m.group(1) if m else sha1_of(source_url)[:8]
        sections = doc.get("sections") or []

        # Synthesize / register the parent act_page.
        if source_url not in seen_act_urls:
            stub = (
                f"{title}\nAct No: {act_no}\nAct Year: {act_year}\n"
                f"Publication Date: {publication_date}\n"
                f"Source: {dataset_id}/{fn}\nLive URL: {source_url}\n"
                f"Preamble: see source_url.\nSections: {len(sections)}\n"
            )
            act_digest, act_path = write_raw_text(html_doc(title, stub), raw_dir, "html")
            retrieved_at = now_iso()
            act_rec = SourceRecord(
                sha1=act_digest,
                url=source_url,
                source_title=title,
                source_type="act_page",
                jurisdiction="Bangladesh",
                act_id=act_id,
                section_id=None,
                retrieved_at=retrieved_at,
                raw_path=act_path,
                http_status=200,
                content_type="html",
                source_origin="hf_baseline",
                completeness_score=completeness_score(stub, title),
                references=[],
                depth=0,
                retrieval_date=retrieval_date_from_iso(retrieved_at),
                publication_date=parse_any_date(publication_date),
            )
            manifest["records"].append(asdict(act_rec))
            seen_act_urls.add(source_url)
            n_acts += 1
        else:
            act_digest = next(
                r["sha1"] for r in manifest["records"]
                if r.get("source_type") == "act_page" and r["url"] == source_url
            )

        for idx, section in enumerate(sections, start=1):
            text = (section.get("section_content") or "").strip() if isinstance(section, dict) else ""
            if not text:
                continue
            sec_hash = sha256_of(normalize_section_text(text))
            if sec_hash in seen_hashes:
                n_skipped += 1
                continue
            seen_hashes.add(sec_hash)

            section_id = str(section.get("section_number") or idx)
            sec_title = f"{title} - section {section_id}"
            sec_digest, sec_path = write_raw_text(
                html_doc(sec_title, text, heading="h3"), raw_dir, "html"
            )
            sec_url = f"{source_url}#section={section_id}"
            refs = detect_references(text, exclude_title=title)
            retrieved_at = now_iso()
            rec = SourceRecord(
                sha1=sec_digest,
                url=sec_url,
                source_title=sec_title,
                source_type="section_page",
                jurisdiction="Bangladesh",
                act_id=act_id,
                section_id=section_id,
                retrieved_at=retrieved_at,
                raw_path=sec_path,
                http_status=200,
                content_type="html",
                source_origin="hf_baseline",
                parent_id=act_digest,
                relationship_type="section_of",
                completeness_score=completeness_score(text, title),
                references=refs,
                section_text_sha256=sec_hash,
                depth=1,
                retrieval_date=retrieval_date_from_iso(retrieved_at),
                publication_date=parse_any_date(publication_date),
            )
            manifest["records"].append(asdict(rec))
            manifest["relationships"].append({
                "from": sec_digest, "to": act_digest, "type": "section_of",
            })
            for r in refs:
                manifest["relationships"].append({
                    "from": sec_digest,
                    "to_name": r["name"],
                    "to_year": r["year"],
                    "type": "references",
                })
            n_loaded_sections += 1

        if (i + 1) % 100 == 0:
            save_manifest(manifest, manifest_path)
            log.info("hf baseline progress: files=%d acts=%d sections=%d skipped_dup=%d",
                     i + 1, n_acts, n_loaded_sections, n_skipped)

    save_manifest(manifest, manifest_path)
    summary = {
        "loaded": n_loaded_sections,
        "skipped_dup": n_skipped,
        "acts": n_acts,
        "sections": n_loaded_sections,
        "files_seen": min(cap, len(files)),
    }
    log.info("hf baseline done: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Delta crawl from bdlaws (live portal), filtered by date
# ---------------------------------------------------------------------------

def harvest_bdlaws_delta(
    session: requests.Session,
    manifest: dict,
    *,
    raw_dir: str,
    failed_log: str,
    volumes: Iterable[int],
    max_acts: int,
    max_children_per_act: int,
    delay: float,
    max_depth: int,
    manifest_path: str,
    after_date: str,
    start_act_id: Optional[int] = None,
    end_act_id: Optional[int] = None,
    max_misses: int = DEFAULT_DELTA_MAX_MISSES,
) -> dict:
    """Probe live `act-print-N.html` pages newer than the HF baseline.

    The Laws of Bangladesh portal exposes newly published Acts/Ordinances in
    monotonically increasing `act-print` ids. Starting from the maximum
    baseline act id avoids re-crawling all historical volumes and gives a
    deterministic delta path.
    """
    del volumes, max_children_per_act, max_depth  # retained for CLI compatibility
    baseline_max = max_manifest_act_print_id(manifest, origin="hf_baseline")
    if baseline_max == 0:
        baseline_max = max_manifest_act_print_id(manifest)
    if baseline_max == 0 and start_act_id is None:
        reason = "delta_start_missing_baseline_max_act_id"
        log_failure(failed_log, "bdlaws://delta", reason)
        return {
            "baseline_max_act_id": 0,
            "start_act_id": None,
            "end_act_id": end_act_id,
            "cutoff": after_date,
            "acts_added": 0,
            "sections_added": 0,
            "section_duplicates_skipped": 0,
            "missing_pages": 0,
            "date_filtered": 0,
            "checked": 0,
            "error": reason,
        }
    current = start_act_id or (baseline_max + 1 if baseline_max else 1)
    seen_hashes = existing_section_hashes(manifest)
    existing_urls = {r.get("url") for r in manifest.get("records", [])}
    misses = 0
    checked = 0
    summary = {
        "baseline_max_act_id": baseline_max,
        "start_act_id": current,
        "end_act_id": end_act_id,
        "cutoff": after_date,
        "acts_added": 0,
        "sections_added": 0,
        "section_duplicates_skipped": 0,
        "missing_pages": 0,
        "date_filtered": 0,
    }

    while True:
        if end_act_id is not None and current > end_act_id:
            break
        if end_act_id is None and misses >= max_misses:
            break
        if max_acts and checked >= max_acts:
            break

        url = f"http://bdlaws.minlaw.gov.bd/act-print-{current}.html"
        checked += 1
        if url in existing_urls:
            current += 1
            continue

        try:
            log.info("probe live delta act-print-%s", current)
            resp = http_get(session, url)
            time.sleep(delay)
            html = response_text(resp)
        except Exception as e:  # noqa: BLE001
            log_failure(failed_log, url, f"delta_fetch_error:{e.__class__.__name__}:{e}")
            misses += 1
            summary["missing_pages"] += 1
            current += 1
            continue

        parsed = parse_bdlaws_act_print(html, url, str(current))
        if not parsed:
            misses += 1
            summary["missing_pages"] += 1
            current += 1
            continue
        misses = 0

        publication_date = parsed.get("publication_date")
        if publication_date and publication_date <= after_date and current <= baseline_max:
            summary["date_filtered"] += 1
            current += 1
            continue

        act_body = (
            f"{parsed['title']}\n"
            f"Live URL: {url}\n"
            f"Publication Date: {publication_date or parsed.get('publication_date_raw') or ''}\n"
            f"Fetched: {now_iso()}\n"
            f"Sections: {len(parsed['sections'])}\n\n"
            f"{parsed['body']}"
        )
        act_rec = add_text_record(
            manifest,
            raw_dir=raw_dir,
            url=url,
            title=parsed["title"],
            body=act_body,
            source_type="act_page",
            act_id=str(current),
            section_id=None,
            source_origin="live_portal",
            publication_date=publication_date,
            depth=0,
        )
        existing_urls.add(url)
        summary["acts_added"] += 1

        for section in parsed["sections"]:
            text = section["text"]
            sec_hash = sha256_of(normalize_section_text(text))
            if sec_hash in seen_hashes:
                summary["section_duplicates_skipped"] += 1
                continue
            seen_hashes.add(sec_hash)
            sec_id = str(section["section_id"])
            title = f"{parsed['title']} - section {sec_id}"
            if section.get("heading"):
                text = f"{section['heading']}\n{text}"
            add_text_record(
                manifest,
                raw_dir=raw_dir,
                url=f"{url}#section={sec_id}",
                title=title,
                body=text,
                source_type="section_page",
                act_id=str(current),
                section_id=sec_id,
                source_origin="live_portal",
                parent_id=act_rec.sha1,
                relationship_type="section_of",
                section_text_sha256=sec_hash,
                publication_date=publication_date,
                depth=1,
            )
            summary["sections_added"] += 1

        save_manifest(manifest, manifest_path)
        current += 1

    summary["checked"] = checked
    summary["last_checked_act_id"] = current - 1
    save_manifest(manifest, manifest_path)
    return summary


# ---------------------------------------------------------------------------
# Critical-act SRO targeting (top 20)
# ---------------------------------------------------------------------------

def harvest_critical_act_sros(
    session: requests.Session,
    manifest: dict,
    *,
    raw_dir: str,
    failed_log: str,
    delay: float,
    manifest_path: str,
    max_sros_per_act: int = 25,
) -> dict:
    """For each act in CRITICAL_ACTS, locate its act_page record (whether
    sourced from HF baseline or the live portal), then visit the live act
    page on bdlaws.minlaw.gov.bd to discover linked SROs and download them
    as PDFs."""
    summary = {"acts_targeted": 0, "sros_downloaded": 0, "failed": 0}

    # Build a lookup from act_page records.
    act_records = [
        r for r in manifest["records"]
        if r["source_type"] == "act_page"
    ]

    for name, year in CRITICAL_ACTS:
        match = None
        name_key = normalize_title_key(name)
        for r in act_records:
            title = r.get("source_title") or ""
            title_key = normalize_title_key(title)
            year_text = ascii_digits(title)
            if name_key in title_key and (year in year_text or title_key == name_key):
                match = r
                break
        if not match:
            override_id = CRITICAL_ACT_ID_OVERRIDES.get((name, year))
            if override_id:
                for r in act_records:
                    if str(r.get("act_id") or "") == override_id:
                        match = r
                        break
        if not match:
            log.info("critical act not present in manifest: %s, %s", name, year)
            continue
        summary["acts_targeted"] += 1

        # Resolve the live bdlaws URL.
        act_id = match.get("act_id")
        live_url = (
            ACT_URL_FMT.format(act_id=act_id) if act_id and str(act_id).isdigit()
            else match.get("url", "")
        )
        if not live_url.startswith("http"):
            continue

        # Re-fetch (cached if seen) the act page so we can scan child links
        # for SROs even if the manifest entry came from HF baseline.
        act_html = _fetch_html_or_cached(
            session, manifest, live_url, raw_dir, failed_log, delay,
            source_type="act_page", title_hint=match["source_title"],
            act_id=str(act_id) if act_id else None,
            depth=0,
        )
        if not act_html:
            summary["failed"] += 1
            continue

        n_sros = 0
        for child_url, child_label in enumerate_child_links(act_html, str(act_id or "")):
            label_low = (child_label or "").lower()
            if not any(k in label_low for k in SRO_KEYWORDS):
                continue
            if n_sros >= max_sros_per_act:
                break
            if looks_like_pdf(child_url):
                before = len(manifest["records"])
                _fetch_pdf(
                    session, manifest, child_url, raw_dir, failed_log, delay,
                    source_type="sro", title_hint=child_label,
                    act_id=str(act_id) if act_id else None,
                    parent_id=match["sha1"], relationship_type="sro_of",
                    depth=1,
                )
                after = len(manifest["records"])
                if after > before:
                    manifest["records"][-1]["source_origin"] = "live_portal"
                    summary["sros_downloaded"] += 1
                    n_sros += 1
            else:
                before = len(manifest["records"])
                _fetch_html_or_cached(
                    session, manifest, child_url, raw_dir, failed_log, delay,
                    source_type="sro", title_hint=child_label,
                    act_id=str(act_id) if act_id else None,
                    parent_id=match["sha1"], relationship_type="sro_of",
                    depth=1,
                )
                after = len(manifest["records"])
                if after > before:
                    manifest["records"][-1]["source_origin"] = "live_portal"
                    summary["sros_downloaded"] += 1
                    n_sros += 1
            save_manifest(manifest, manifest_path)

    save_manifest(manifest, manifest_path)
    log.info("critical-act SRO targeting: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data")
    p.add_argument("--mode", default="hybrid",
                   choices=["hybrid", "baseline", "delta", "live"],
                   help="hybrid = HF baseline + delta crawl + critical SROs; "
                        "baseline = HF only; delta = live portal delta only; "
                        "live = full live crawl with no HF baseline")
    p.add_argument("--sources", default="bdlaws",
                   help="comma-separated subset of: bdlaws,supremecourt,chancery "
                        "(applies to live/delta modes)")
    p.add_argument("--hf-dataset", default=DEFAULT_HF_BASELINE,
                   help="HF dataset id for the baseline load")
    p.add_argument("--hf-max-rows", type=int, default=0,
                   help="cap rows pulled from the HF baseline (0 = no cap)")
    p.add_argument("--hf-local-files-only", action="store_true",
                   help="load the HF baseline from the local cache only")
    p.add_argument("--hf-min-files", type=int, default=1400,
                   help="minimum HF act files required before hybrid proceeds")
    p.add_argument("--delta-after", default=DEFAULT_DELTA_AFTER,
                   help="YYYY-MM-DD cutoff for the delta crawl")
    p.add_argument("--delta-start-act-id", type=int, default=None,
                   help="override live delta start act-print id")
    p.add_argument("--delta-end-act-id", type=int, default=None,
                   help="optional live delta end act-print id")
    p.add_argument("--delta-max-misses", type=int, default=DEFAULT_DELTA_MAX_MISSES,
                   help="stop after this many consecutive missing act-print pages")
    p.add_argument("--sro-critical", action="store_true", default=True,
                   help="enable critical-act SRO targeting (default on)")
    p.add_argument("--no-sro-critical", dest="sro_critical", action="store_false")
    p.add_argument("--volumes", default="1,2,3,4,5",
                   help="bdlaws volume numbers")
    p.add_argument("--max-acts", type=int, default=0,
                   help="cap live/delta acts checked (0 = no cap)")
    p.add_argument("--max-children-per-act", type=int, default=40,
                   help="cap on sections + amendments + rules + SROs per act")
    p.add_argument("--max-depth", type=int, default=2,
                   help="recursion depth for following cross-references")
    p.add_argument("--max-court-pages", type=int, default=30)
    p.add_argument("--max-chancery-pages", type=int, default=30)
    p.add_argument("--delay", type=float, default=2.0,
                   help="seconds between requests (rate limit)")
    p.add_argument("--rebuild-manifest", action="store_true",
                   help="backup and rebuild manifest.json from selected sources")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    failed_log = os.path.join(args.out_dir, "failed_sources.log")
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if args.rebuild_manifest and os.path.exists(manifest_path):
        backup = manifest_path + f".bak-{now_utc().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(manifest_path, backup)
        log.info("backed up existing manifest to %s", backup)
        os.remove(manifest_path)
    manifest = load_manifest(manifest_path)
    manifest["schema_version"] = "1.3"

    session = requests.Session()
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    summary: dict = {"mode": args.mode}

    # 1. Baseline load from Hugging Face.
    if args.mode in ("hybrid", "baseline"):
        summary["hf_baseline"] = harvest_hf_baseline(
            manifest,
            raw_dir=raw_dir, failed_log=failed_log,
            dataset_id=args.hf_dataset,
            manifest_path=manifest_path,
            max_rows=(args.hf_max_rows or None),
            local_files_only=args.hf_local_files_only,
            min_files=args.hf_min_files,
        )
        if summary["hf_baseline"].get("incomplete"):
            manifest["last_run"] = {
                "finished_at": now_iso(),
                "summary": summary,
                "error": "HF baseline incomplete; hybrid crawl aborted before delta.",
            }
            save_manifest(manifest, manifest_path)
            print(json.dumps(summary, indent=2))
            return 2

    # 2. Delta / live crawl.
    if args.mode in ("hybrid", "delta", "live") and "bdlaws" in sources:
        volumes = [int(v.strip()) for v in args.volumes.split(",") if v.strip()]
        if args.mode == "live":
            harvest_bdlaws(
                session, manifest,
                raw_dir=raw_dir, failed_log=failed_log,
                volumes=volumes, max_acts=args.max_acts,
                max_children_per_act=args.max_children_per_act,
                delay=args.delay, max_depth=args.max_depth,
                manifest_path=manifest_path,
            )
            summary["live"] = {"records": len(manifest["records"])}
        else:
            summary["delta"] = harvest_bdlaws_delta(
                session, manifest,
                raw_dir=raw_dir, failed_log=failed_log,
                volumes=volumes, max_acts=args.max_acts,
                max_children_per_act=args.max_children_per_act,
                delay=args.delay, max_depth=args.max_depth,
                manifest_path=manifest_path,
                after_date=args.delta_after,
                start_act_id=args.delta_start_act_id,
                end_act_id=args.delta_end_act_id,
                max_misses=args.delta_max_misses,
            )

    if "supremecourt" in sources and args.mode in ("live", "delta", "hybrid"):
        harvest_supreme_court(
            session, manifest,
            raw_dir=raw_dir, failed_log=failed_log,
            delay=args.delay, max_pages=args.max_court_pages,
            manifest_path=manifest_path,
        )

    if "chancery" in sources and args.mode in ("live", "delta", "hybrid"):
        harvest_chancery(
            session, manifest,
            raw_dir=raw_dir, failed_log=failed_log,
            delay=args.delay, max_pages=args.max_chancery_pages,
            manifest_path=manifest_path,
        )

    # 3. Critical-act SRO enrichment.
    if args.mode in ("hybrid", "delta") and args.sro_critical:
        summary["critical_sros"] = harvest_critical_act_sros(
            session, manifest,
            raw_dir=raw_dir, failed_log=failed_log,
            delay=args.delay, manifest_path=manifest_path,
        )

    # 4. Global SHA-256 dedup pass.
    summary["dedup"] = dedup_global(manifest)
    ensure_manifest_provenance(manifest)
    save_manifest(manifest, manifest_path)

    # 5. Counts by origin and type.
    counts_by_origin: dict[str, int] = {}
    counts_by_type: dict[str, int] = {}
    section_count = 0
    for r in manifest["records"]:
        counts_by_origin[r.get("source_origin", "?")] = counts_by_origin.get(
            r.get("source_origin", "?"), 0) + 1
        counts_by_type[r.get("source_type", "?")] = counts_by_type.get(
            r.get("source_type", "?"), 0) + 1
        if r.get("source_type") == "section_page":
            section_count += 1
    summary["counts_by_origin"] = counts_by_origin
    summary["counts_by_type"] = counts_by_type
    summary["total_records"] = len(manifest["records"])
    summary["total_sections"] = section_count
    summary["total_relationships"] = len(manifest.get("relationships", []))

    # Persist a stamp of the run.
    manifest["last_run"] = {
        "finished_at": now_iso(),
        "summary": summary,
    }
    save_manifest(manifest, manifest_path)

    log.info("HARVEST SUMMARY: %s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
