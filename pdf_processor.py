"""
pdf_processor.py

PDF download and text extraction with OCR fallback.

Strategy per PDF:
1. Try PyMuPDF text extraction.
2. If extracted text is empty or below a minimum threshold, try pdfplumber's
   column-aware extraction.
3. If text is still sparse, treat the PDF as image-only and render each page
   with PyMuPDF (fitz), then OCR each page image with pytesseract.

Both pdfplumber and pytesseract are optional at install-time; if a backend is
missing the corresponding path is skipped and the failure is reported.

Tesseract binary must be installed system-side for OCR to work
(e.g. `apt-get install tesseract-ocr` or `brew install tesseract`).
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

log = logging.getLogger("pdf_processor")

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except Exception:  # noqa: BLE001
    _HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF  # type: ignore
    _HAS_FITZ = True
except Exception:  # noqa: BLE001
    _HAS_FITZ = False

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _HAS_OCR = True
except Exception:  # noqa: BLE001
    _HAS_OCR = False


@dataclass
class PdfExtraction:
    text: str
    method: str          # "pdfplumber", "ocr", or "none"
    pages: int
    ocr_used: bool
    notes: str = ""


MIN_TEXT_CHARS = 200


def download_pdf(session: requests.Session, url: str, dest_path: str, timeout: int = 60) -> None:
    """Stream a PDF to disk. Raises on non-200 or non-PDF content-type."""
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf"):
            raise ValueError(f"not a PDF (content-type={ctype}) at {url}")
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)


def _extract_with_pdfplumber(path: str) -> tuple[str, int]:
    if not _HAS_PDFPLUMBER:
        return "", 0
    chunks: list[str] = []
    n_pages = 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            n_pages += 1
            # Column-aware extraction: extract words, sort by (x0, top), then group.
            try:
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            except Exception:  # noqa: BLE001
                words = []
            if words:
                # Heuristic: split page into left/right halves by median x0 when
                # the spread suggests a two-column layout.
                xs = sorted(w["x0"] for w in words)
                if xs and (xs[-1] - xs[0]) > 300:
                    mid = (xs[0] + xs[-1]) / 2
                    left = [w for w in words if w["x0"] < mid]
                    right = [w for w in words if w["x0"] >= mid]
                    left.sort(key=lambda w: (w["top"], w["x0"]))
                    right.sort(key=lambda w: (w["top"], w["x0"]))
                    chunks.append(" ".join(w["text"] for w in left))
                    chunks.append(" ".join(w["text"] for w in right))
                    continue
            txt = page.extract_text() or ""
            chunks.append(txt)
    return ("\n\n".join(c for c in chunks if c).strip(), n_pages)


def _extract_with_pymupdf(path: str) -> tuple[str, int]:
    if not _HAS_FITZ:
        return "", 0
    chunks: list[str] = []
    doc = fitz.open(path)
    try:
        for page in doc:
            chunks.append(page.get_text("text") or "")
        return ("\n\n".join(c for c in chunks if c).strip(), len(doc))
    finally:
        doc.close()


def _extract_with_ocr(path: str, dpi: int = 200, lang: str = "eng+ben") -> tuple[str, int, str]:
    if not (_HAS_FITZ and _HAS_OCR):
        return "", 0, "ocr_backend_missing"
    chunks: list[str] = []
    notes = ""
    doc = fitz.open(path)
    n_pages = len(doc)
    for page in doc:
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                txt = pytesseract.image_to_string(img, lang=lang)
            except pytesseract.TesseractError:
                # Fall back to English-only if Bengali traineddata is unavailable.
                notes = "ocr_lang_fallback_eng"
                txt = pytesseract.image_to_string(img, lang="eng")
            chunks.append(txt)
        except Exception as e:  # noqa: BLE001
            log.warning("ocr page failed: %s", e)
            chunks.append("")
    doc.close()
    return ("\n\n".join(c for c in chunks if c).strip(), n_pages, notes)


def extract_pdf(path: str) -> PdfExtraction:
    """Extract text from a PDF, falling back to OCR for image-only docs."""
    text = ""
    pages = 0
    notes = ""
    method = "none"

    if _HAS_FITZ:
        try:
            text, pages = _extract_with_pymupdf(path)
            method = "pymupdf"
        except Exception as e:  # noqa: BLE001
            notes = f"pymupdf_error:{e.__class__.__name__}"
            log.warning("PyMuPDF failed on %s: %s", path, e)

    if len(text) < MIN_TEXT_CHARS and _HAS_PDFPLUMBER:
        try:
            plumber_text, plumber_pages = _extract_with_pdfplumber(path)
            if len(plumber_text) > len(text):
                text = plumber_text
                pages = plumber_pages or pages
                method = "pdfplumber"
        except Exception as e:  # noqa: BLE001
            notes = (notes + f"|pdfplumber_error:{e.__class__.__name__}").strip("|")
            log.warning("pdfplumber failed on %s: %s", path, e)

    if len(text) < MIN_TEXT_CHARS:
        if _HAS_FITZ and _HAS_OCR:
            ocr_text, ocr_pages, ocr_notes = _extract_with_ocr(path)
            if len(ocr_text) > len(text):
                text = ocr_text
                pages = ocr_pages or pages
                method = "ocr"
                notes = (notes + "|" + ocr_notes).strip("|") if ocr_notes else notes
        else:
            notes = (notes + "|ocr_backend_missing").strip("|")

    return PdfExtraction(
        text=text,
        method=method,
        pages=pages,
        ocr_used=(method == "ocr"),
        notes=notes,
    )
