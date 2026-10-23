"""The document archive of the plant and its ground truth.

data/docs/            what the plant has: born-digital PDFs, scans, a drawing, a photo
data/documents.csv    registry of these files
data/ground_truth/    page texts of the born-digital twins, tables, clauses, questions, P&ID tags, nameplate
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from . import DATA_DIR, DOCS_DIR, GT_DIR
from .jsonl import read_json_records

SCANNED_FILES = ["R-12_rev3_scan.pdf", "PS-101_scan.pdf", "DS-P-205_scan.pdf"]


def load_registry() -> list[dict]:
    with open(DATA_DIR / "documents.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_gt_pages() -> dict[tuple[str, int], str]:
    """(file, page) -> text of the same page in the born-digital original."""
    return {(r["file"], r["page"]): r["text"] for r in read_json_records(GT_DIR / "pages.jsonl")}


def load_questions() -> list[dict]:
    return list(read_json_records(GT_DIR / "questions.jsonl"))


def load_json(name: str):
    return json.loads((GT_DIR / name).read_text(encoding="utf-8"))


def page_count(pdf: str | Path) -> int:
    with pymupdf.open(pdf) as doc:
        return len(doc)


# ---------------------------------------------------------------------------------------------
# Triage: born-digital or scan?
# ---------------------------------------------------------------------------------------------
@dataclass
class Triage:
    file: str
    pages: int
    chars_per_page: float
    images: int
    kind: str  # "digital", "scan" or "image"


def triage(path: str | Path, min_chars: int = 200) -> Triage:
    """A PDF page is "digital" when its text layer has real text, "scan" when it is one big picture.

    Two signals are enough for our archive: characters in the text layer and pictures on the page.
    A real archive also has mixed PDFs (a scan with a bad OCR layer, a digital PDF with scanned
    attachments): there the decision is per page, not per file.
    """
    path = Path(path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return Triage(path.name, 1, 0.0, 1, "image")
    with pymupdf.open(path) as doc:
        chars = [len(p.get_text("text").strip()) for p in doc]
        images = sum(len(p.get_images()) for p in doc)
        n = len(doc)
    per_page = sum(chars) / n
    return Triage(path.name, n, round(per_page, 1), images, "digital" if per_page >= min_chars else "scan")


def triage_archive() -> list[Triage]:
    return [triage(DOCS_DIR / r["file"]) for r in load_registry()]
