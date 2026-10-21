"""Six ways to read a PDF page, one interface: pdf -> list of page texts.

| Reader       | What it is                                   | Runs on                  |
|--------------|----------------------------------------------|--------------------------|
| pymupdf      | text layer of the PDF, no recognition at all | CPU, milliseconds        |
| tesseract    | classic OCR engine, page by page             | CPU, needs the binary    |
| easyocr      | classic OCR with a neural detector           | CPU or GPU, pip only     |
| docling      | document parser: layout + tables + OCR       | CPU or GPU, model weights|
| vlm:dots-ocr | document VLM: layout boxes + text, one model | GPU server (lecturer key)|
| vlm:<profile>| a general VLM reads the page image by prompt | course server or Ollama  |

pymupdf+tables (text layer plus PyMuPDF's table finder) reads the born-digital PDFs for the store.
Docling runs with its default settings; forcing full-page OCR did not change its scores on our scans.

Saved outputs live in data/parsed/<reader>.jsonl, one record per page:
{"reader", "file", "page", "text", "seconds", "prompt_tokens", "completion_tokens", "cost_usd", "finish_reason",
 "version"}. The notebook reads them, so heavy readers do not have to run during the session.
"""

from __future__ import annotations

import json
import re
import time
from importlib import metadata
from pathlib import Path
from typing import Callable

import pymupdf

from . import PARSED_DIR
from .images import render_page, vision_message
from .jsonl import read_json_records

OCR_DPI = 300  # the usual advice for classic OCR engines

TRANSCRIBE_PROMPT = """Transcribe this scanned page exactly as written.
- Keep the original language, spelling, numbers, units and punctuation (for example, keep a decimal comma).
- Do not translate, correct, summarise or add anything.
- Keep the reading order: running header, headings, paragraphs, tables, notes, footer.
- Write headings and paragraphs as Markdown. Write every table as an HTML <table>, with rowspan and colspan for merged cells.
- If a part cannot be read, write [unreadable] instead of guessing.
Output only the transcription."""


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def page_record(reader: str, file: str, page: int, text: str, seconds: float, version: str, **extra) -> dict:
    return {"reader": reader, "file": file, "page": page, "text": text, "seconds": round(seconds, 3),
            "version": version, **extra}


# ---------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------
def read_pymupdf(pdf: str | Path) -> list[dict]:
    """Text layer only. On a scan there is no text layer, and the result is empty pages."""
    out = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            t0 = time.perf_counter()
            text = page.get_text("text")
            out.append(page_record("pymupdf", Path(pdf).name, i, text, time.perf_counter() - t0,
                                   f"pymupdf {pymupdf.__version__}"))
    return out


def read_pymupdf_tables(pdf: str | Path) -> list[dict]:
    """Born-digital PDF: text blocks of the text layer plus PyMuPDF's table finder, tables as Markdown.

    The table finder also takes a two-column text for a table. A domain rule fixes it: in a GOST-style
    document every table has a caption "Таблица N —" right above it.
    """
    out = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            t0 = time.perf_counter()
            tables = []
            for t in page.find_tables().tables:
                above = pymupdf.Rect(0, t.bbox[1] - 40, page.rect.width, t.bbox[1])
                if "Таблица" in page.get_text("text", clip=above):
                    tables.append(t)
            boxes = [pymupdf.Rect(t.bbox) for t in tables]
            # a text block is a paragraph: join its wrapped lines into one line
            items = [(b[3], " ".join(b[4].split())) for b in page.get_text("blocks")
                     if b[6] == 0 and not any(pymupdf.Rect(b[:4]).intersects(r) for r in boxes)]
            for t in sorted(tables, key=lambda t: t.bbox[1]):
                # right after the last block (in stream order) that ends above the table: its caption
                above_ids = [j for j, (bottom, _) in enumerate(items) if bottom <= t.bbox[1] + 2]
                k = max(above_ids) + 1 if above_ids else 0
                items.insert(k, (t.bbox[3], t.to_markdown().strip()))
            out.append(page_record("pymupdf+tables", Path(pdf).name, i, "\n".join(x for _, x in items),
                                   time.perf_counter() - t0, f"pymupdf {pymupdf.__version__} find_tables"))
    return out


def read_tesseract(pdf: str | Path, lang: str = "rus+eng") -> list[dict]:
    """Tesseract on a 300 DPI render. The language packs must be installed (rus, eng)."""
    import pytesseract

    version = f"tesseract {pytesseract.get_tesseract_version()} lang={lang}"
    out = []
    with pymupdf.open(pdf) as doc:
        n = len(doc)
    for i in range(1, n + 1):
        img = render_page(pdf, i, OCR_DPI)
        t0 = time.perf_counter()
        text = pytesseract.image_to_string(img, lang=lang)
        out.append(page_record("tesseract", Path(pdf).name, i, text, time.perf_counter() - t0, version))
    return out


_EASYOCR = {}


def _lines_from_boxes(results) -> str:
    """EasyOCR returns boxes with text. Group boxes into lines by their vertical centre, left to right."""
    items = []
    for box, text, _conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(xs), text))
    items.sort()
    lines: list[list[tuple]] = []
    for it in items:
        if lines and abs(it[0] - lines[-1][-1][0]) < 0.5 * max(it[1], lines[-1][-1][1]):
            lines[-1].append(it)
        else:
            lines.append([it])
    return "\n".join(" ".join(t for *_, t in sorted(line, key=lambda x: x[2])) for line in lines)


def read_easyocr(pdf: str | Path, langs: tuple[str, ...] = ("ru", "en")) -> list[dict]:
    import easyocr
    import numpy as np

    key = tuple(langs)
    if key not in _EASYOCR:
        _EASYOCR[key] = easyocr.Reader(list(langs), verbose=False)
    reader = _EASYOCR[key]
    version = f"easyocr {_version('easyocr')} lang={'+'.join(langs)}"
    out = []
    with pymupdf.open(pdf) as doc:
        n = len(doc)
    for i in range(1, n + 1):
        img = np.asarray(render_page(pdf, i, OCR_DPI))
        t0 = time.perf_counter()
        text = _lines_from_boxes(reader.readtext(img))
        out.append(page_record("easyocr", Path(pdf).name, i, text, time.perf_counter() - t0, version))
    return out


_DOCLING = {}


def read_docling(pdf: str | Path, ocr_langs: tuple[str, ...] = ("ru", "en")) -> list[dict]:
    """Docling: layout model, TableFormer for tables, EasyOCR inside for scans. Markdown per page.

    The OCR language is set explicitly: a parser reads a scan only as well as its OCR engine does.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    key = tuple(ocr_langs)
    if key not in _DOCLING:
        options = PdfPipelineOptions(do_ocr=True, do_table_structure=True,
                                     ocr_options=EasyOcrOptions(lang=list(ocr_langs)))
        _DOCLING[key] = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    t0 = time.perf_counter()
    doc = _DOCLING[key].convert(str(pdf)).document
    seconds = time.perf_counter() - t0
    n = len(doc.pages)
    version = f"docling {_version('docling')} ocr=easyocr lang={'+'.join(ocr_langs)}"
    return [page_record("docling", Path(pdf).name, i, doc.export_to_markdown(page_no=i), seconds / n, version)
            for i in range(1, n + 1)]


FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.S)


def strip_fence(text: str) -> str:
    """Small models often wrap the whole answer in ```markdown ... ```: remove the wrapper."""
    m = FENCE.match(text)
    return m.group(1) if m else text


def read_vlm(pdf: str | Path, client, pages: list[int] | None = None, reader_name: str | None = None,
             concurrency: int = 1) -> list[dict]:
    """A VLM reads each page image with the TRANSCRIBE_PROMPT. `client` is an LLMClient or ReplayClient."""
    from concurrent.futures import ThreadPoolExecutor

    name = reader_name or f"vlm:{client.profile_name}"
    with pymupdf.open(pdf) as doc:
        pages = pages or list(range(1, len(doc) + 1))

    def one(i: int) -> dict:
        msg = vision_message(TRANSCRIBE_PROMPT, render_page(pdf, i))
        r = client.chat([msg], meta={"task": "transcribe", "file": Path(pdf).name, "page": i})
        if not r.ok:
            raise RuntimeError(f"{name} failed on {Path(pdf).name} p.{i}: {r.error}")
        return page_record(name, Path(pdf).name, i, strip_fence(r.content), r.latency_s, client.model,
                           prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                           cost_usd=r.cost_usd, finish_reason=r.finish_reason)

    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        return list(pool.map(one, pages))


# The layout prompt of dots.ocr (rednote hi lab), as in its repository: the model was trained on it.
DOTS_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
_ELEMENT = re.compile(r'\{"bbox":\s*\[[^\]]*\],\s*"category":\s*"[^"]*"(?:,\s*"text":\s*"(?:[^"\\]|\\.)*")?\}')


def parse_layout(content: str) -> list[dict]:
    """dots.ocr answers with a JSON list of {bbox, category, text}. If the answer was cut by the token limit,
    keep the elements that are complete."""
    try:
        data = json.loads(strip_fence(content))
        return data if isinstance(data, list) else data.get("layout", [])
    except (json.JSONDecodeError, AttributeError):
        return [json.loads(m.group(0)) for m in _ELEMENT.finditer(content)]


def read_dots_ocr(pdf: str | Path, client, pages: list[int] | None = None, reader_name: str = "vlm:dots-ocr",
                  concurrency: int = 1) -> list[dict]:
    """A document VLM: one model returns the layout (blocks with a category and a box) and the text of each block.

    The page text is the blocks in reading order (tables as HTML). The boxes are kept in "layout", converted
    from pixels of the 200 DPI render to PDF points, so a chunk can point to its place on the page.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .images import VLM_DPI, image_part

    with pymupdf.open(pdf) as doc:
        pages = pages or list(range(1, len(doc) + 1))
    scale = 72 / VLM_DPI

    def one(i: int) -> dict:
        # dots.ocr expects the image first, then the prompt
        msg = {"role": "user", "content": [image_part(render_page(pdf, i)), {"type": "text", "text": DOTS_LAYOUT_PROMPT}]}
        r = client.chat([msg], meta={"task": "layout", "file": Path(pdf).name, "page": i})
        if not r.ok:
            raise RuntimeError(f"{reader_name} failed on {Path(pdf).name} p.{i}: {r.error}")
        blocks = parse_layout(r.content)
        text = "\n\n".join(b["text"] for b in blocks if b.get("text"))
        layout = [{"category": b["category"], "bbox_pt": [round(v * scale, 1) for v in b["bbox"]]} for b in blocks]
        return page_record(reader_name, Path(pdf).name, i, text, r.latency_s, client.model,
                           prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                           cost_usd=r.cost_usd, finish_reason=r.finish_reason, layout=layout)

    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        return list(pool.map(one, pages))


# ---------------------------------------------------------------------------------------------
# Saved outputs
# ---------------------------------------------------------------------------------------------
def parsed_path(reader: str) -> Path:
    return PARSED_DIR / f"{reader.replace(':', '_')}.jsonl"


def save_outputs(reader: str, records: list[dict]) -> Path:
    path = parsed_path(reader)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda r: (r["file"], r["page"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_outputs(reader: str) -> dict[tuple[str, int], dict]:
    """(file, page) -> saved record of a reader."""
    return {(r["file"], r["page"]): r for r in read_json_records(parsed_path(reader))}


def available_readers() -> list[str]:
    return sorted(p.stem.replace("vlm_", "vlm:", 1) for p in PARSED_DIR.glob("*.jsonl"))


LOCAL_READERS: dict[str, Callable[[str | Path], list[dict]]] = {
    "pymupdf": read_pymupdf,
    "pymupdf+tables": read_pymupdf_tables,
    "tesseract": read_tesseract,
    "easyocr": read_easyocr,
    "docling": read_docling,
}
