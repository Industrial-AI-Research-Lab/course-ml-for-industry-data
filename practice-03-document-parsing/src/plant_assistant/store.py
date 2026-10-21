"""Chunk store: structural chunks with metadata. The input of practice 4 (RAG).

Steps for a regulation (a list of page texts from any reader):
1. metadata of the document from the title page, by rules (number, revision, date, what it replaces, scope);
2. hygiene: running headers and footers are removed from the text;
3. structure: clause numbers at the start of a line rebuild the tree (section -> subsection -> clause);
   a number is accepted only if it continues the numbering, so "720 часов" at the start of a wrapped line
   is not a clause;
4. chunks: one chunk per clause, a table is one chunk as a whole with its caption, headings give "breadcrumbs"
   and parent chunks (parent-child);
5. versions: every revision stays in the store, older ones get status "superseded" and superseded_by.

A chunk keeps where it came from (file, page, clause, parser): without this, topic 4 cannot cite the source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import STORE_DIR

TAG_RE = re.compile(r"\b[A-Z]{1,3}-\d{3}\b")
CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(\S.*)$")
APPENDIX_RE = re.compile(r"^Приложение\s+([А-ЯA-Z])\b")
CAPTION_RE = re.compile(r"^Таблица\s+[\dА-ЯA-Z.]+\s*[—–-]")
RUNNING = [re.compile(p) for p in (r"^R-12\s*«.*»$", r"^Редакция\s+\d+$", r"^Стр\.\s*\d+\s*из\s*\d+$",
                                   r"^R-12\s*«.*»\s*Редакция\s+\d+$")]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    revision: int | None
    chunk_type: str  # section, clause, table, datasheet
    clause: str | None
    section_path: list[str]
    page: int
    text: str
    parent_id: str | None = None
    equipment_tags: list[str] = field(default_factory=list)
    status: str = ""
    effective_date: str | None = None
    source_file: str = ""
    parser: str = ""
    parser_version: str = ""

    def breadcrumbs(self) -> str:
        return " › ".join(self.section_path)


# ---------------------------------------------------------------------------------------------
# Metadata by rules
# ---------------------------------------------------------------------------------------------
def _clean(line: str) -> str:
    """A line without Markdown markup, list bullets and HTML tags."""
    line = re.sub(r"<[^>]+>", " ", line)
    line = re.sub(r"^\s*[-*+•]\s+", "", line)  # parsers often turn numbered clauses into list items
    return re.sub(r"\s+", " ", re.sub(r"[*#_`]", "", line)).strip()


def title_metadata(page1: str) -> dict:
    """Number, title, revision, date, replaced revision and scope from the title page."""
    text = "\n".join(_clean(x) for x in page1.splitlines())

    def find(pattern: str):
        m = re.search(pattern, text, re.M)
        return m.group(1).strip() if m else None

    approved = find(r"Утвержд[её]н\s+(\d{2}\.\d{2}\.\d{4})")
    iso = "-".join(reversed(approved.split("."))) if approved else None
    rev = find(r"^Редакция\s+(\d+)")
    supersedes = find(r"Взамен редакции\s+(\d+)")
    m = re.search(r"РЕГЛАМЕНТ\s+R-\d+\s*\n(.+?)\n\s*Редакция", text, re.S)  # the title can wrap
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    return {"doc_id": find(r"РЕГЛАМЕНТ\s+(R-\d+)"), "title": title,
            "revision": int(rev) if rev else None, "effective_date": iso,
            "supersedes": int(supersedes) if supersedes else None, "scope": find(r"Область действия:\s*(.+?)(?:\s+Владелец документа:.*)?$")}


# ---------------------------------------------------------------------------------------------
# Blocks: tables are cut out of the text first, so that they stay whole
# ---------------------------------------------------------------------------------------------
def blocks(page_text: str) -> list[tuple[str, str]]:
    """A page as ("line", text) and ("table", html_or_markdown) blocks, in reading order."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in re.finditer(r"<table.*?</table>", page_text, re.S | re.I):
        out += [("line", x) for x in page_text[pos:m.start()].splitlines()]
        out.append(("table", m.group(0)))
        pos = m.end()
    out += [("line", x) for x in page_text[pos:].splitlines()]
    merged: list[tuple[str, str]] = []
    for kind, text in out:  # Markdown tables: consecutive |...| lines
        if kind == "line" and text.strip().startswith("|"):
            if merged and merged[-1][0] == "table" and merged[-1][1].lstrip().startswith("|"):
                merged[-1] = ("table", merged[-1][1] + "\n" + text.strip())
                continue
            merged.append(("table", text.strip()))
        else:
            merged.append((kind, text))
    return merged


def _key(number: str) -> tuple[int, ...]:
    return tuple(int(x) for x in number.split("."))


def _continues(number: str, current: str | None) -> bool:
    """A clause number is accepted only if it goes after the current one and stays in this or the next section."""
    if current is None:
        return _key(number)[0] <= 2
    new, cur = _key(number), _key(current)
    return new > cur and new[0] in (cur[0], cur[0] + 1)


def chunk_regulation(pages: list[str], source_file: str, parser: str = "", parser_version: str = "") -> tuple[dict, list[Chunk]]:
    meta = title_metadata(pages[0])
    doc_id, rev = meta["doc_id"] or Path(source_file).stem, meta["revision"]
    root = f"{doc_id} ред. {rev}" if rev else doc_id
    prefix = f"{doc_id}-r{rev}"
    chunks: list[Chunk] = []
    titles: dict[str, str] = {}  # heading number -> "4.3 Нормы вибрации"
    current: str | None = None  # last accepted number: the numbering must continue from it
    owner: str | None = None  # the section a table belongs to: subsection of the last clause, or an appendix
    open_chunk: Chunk | None = None
    caption: str | None = None

    def path_for(number: str) -> list[str]:
        parts = number.split(".")
        return [root] + [titles[".".join(parts[:i])] for i in range(1, len(parts)) if ".".join(parts[:i]) in titles]

    def parent_of(number: str) -> str | None:
        parts = number.split(".")
        for i in range(len(parts) - 1, 0, -1):
            p = ".".join(parts[:i])
            if p in titles:
                return f"{prefix}-{p}"
        return None

    def new_chunk(kind: str, number: str | None, text: str, page: int, path: list[str], cid: str) -> Chunk:
        return Chunk(chunk_id=cid, doc_id=doc_id, revision=rev, chunk_type=kind, clause=number, section_path=path,
                     page=page, text=text, parent_id=parent_of(number) if number else None,
                     effective_date=meta["effective_date"], source_file=source_file, parser=parser,
                     parser_version=parser_version)

    n_tables = 0
    for page_no, page_text in enumerate(pages[1:], start=2):  # page 1 is the title page
        for kind, raw in blocks(page_text):
            if kind == "table":
                n_tables += 1
                number = owner
                head = f"{caption}\n" if caption else ""
                cid = f"{prefix}-table{n_tables}"
                ch = new_chunk("table", number, head + raw.strip(), page_no, path_for(number or "") + (
                    [titles[number]] if number in titles else []), cid)
                ch.parent_id = f"{prefix}-{number}" if number in titles else ch.parent_id
                chunks.append(ch)
                caption, open_chunk = None, None
                continue
            line = _clean(raw)
            if not line or any(p.match(line) for p in RUNNING):
                continue
            if CAPTION_RE.match(line):
                caption = line
                continue
            if caption and open_chunk is None:
                # a caption without a detected table: the table went into the text (plain OCR)
                line, caption = f"{caption} {line}", None
            app = APPENDIX_RE.match(line)
            m = CLAUSE_RE.match(line)
            if app:
                number = app.group(1)
                titles[number] = line
                open_chunk = new_chunk("section", number, line, page_no, [root], f"{prefix}-{number}")
                chunks.append(open_chunk)
                current, owner = "99", number  # no numbered clauses after an appendix starts
            elif m and _continues(m.group(1), current):
                number, rest = m.group(1), m.group(2)
                current = number
                owner = number.rsplit(".", 1)[0] if number.count(".") == 2 else number
                # "4.3 Нормы вибрации" is a heading; "1.1 Настоящий регламент устанавливает..." is a clause,
                # even when OCR cut its line short: headings are short and have no punctuation
                is_heading = number.count(".") < 2 and len(rest.split()) <= 5 and not re.search(r"[.,:;]", rest)
                if is_heading:
                    titles[number] = f"{number} {rest}"
                    open_chunk = new_chunk("section", number, f"{number} {rest}", page_no, path_for(number),
                                           f"{prefix}-{number}")
                else:
                    open_chunk = new_chunk("clause", number, line, page_no, path_for(number), f"{prefix}-{number}")
                chunks.append(open_chunk)
            elif open_chunk is not None:
                open_chunk.text += ("\n" if open_chunk.chunk_type == "section" else " ") + line
    for ch in chunks:
        ch.equipment_tags = sorted(set(TAG_RE.findall(ch.text)))
    return meta, chunks


def chunk_datasheet(pages: list[str], doc_id: str, source_file: str, parser: str = "",
                    parser_version: str = "") -> tuple[dict, list[Chunk]]:
    """A datasheet is short: one chunk per page, kept whole (it is a table of parameters)."""
    chunks = []
    for i, text in enumerate(pages, start=1):
        clean = "\n".join(x for x in (_clean(y) for y in text.splitlines()) if x) if "<table" not in text else text
        chunks.append(Chunk(chunk_id=f"{doc_id}-p{i}", doc_id=doc_id, revision=None, chunk_type="datasheet",
                            clause=None, section_path=[doc_id], page=i, text=clean.strip(),
                            equipment_tags=sorted(set(TAG_RE.findall(text))), source_file=source_file,
                            parser=parser, parser_version=parser_version))
    return {"doc_id": doc_id, "title": None, "revision": None, "effective_date": None, "supersedes": None,
            "scope": None}, chunks


# ---------------------------------------------------------------------------------------------
# Versions and writing
# ---------------------------------------------------------------------------------------------
def link_revisions(documents: list[dict], chunks: list[Chunk]) -> None:
    """The newest revision of a document is in force; older ones are superseded by it."""
    latest: dict[str, int] = {}
    for d in documents:
        if d.get("revision") is not None:
            latest[d["doc_id"]] = max(latest.get(d["doc_id"], 0), d["revision"])
    for d in documents:
        rev, last = d.get("revision"), latest.get(d["doc_id"])
        d["status"] = "in force" if rev is None or rev == last else "superseded"
        d["superseded_by"] = None if d["status"] == "in force" else last
    status = {(d["doc_id"], d.get("revision")): d["status"] for d in documents}
    for ch in chunks:
        ch.status = status.get((ch.doc_id, ch.revision), "")


def content_hash(path: str | Path) -> str:
    """Fingerprint of the file contents: finds the same document under different names (deduplication)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_store(documents: list[dict], chunks: list[Chunk], folder: str | Path = STORE_DIR) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
    return folder


def structure_score(chunks: list[Chunk], gt_clauses: list[str]) -> dict:
    """Share of the clause and section numbers of the ground truth that became chunks."""
    found = {c.clause for c in chunks if c.chunk_type in ("clause", "section")}
    missed = [c for c in gt_clauses if c not in found]
    return {"clauses": len(gt_clauses), "found": len(gt_clauses) - len(missed),
            "share": 1 - len(missed) / len(gt_clauses), "missed": missed}


def build_store(scan_reader: str = "vlm:course", folder: str | Path = STORE_DIR) -> dict:
    """The store of the demo project: regulations from the born-digital PDFs (text layer + table finder),
    datasheets from the scans as read by `scan_reader`. Returns a short summary."""
    from collections import Counter

    from . import DOCS_DIR
    from .readers import load_outputs, read_pymupdf_tables

    documents, chunks = [], []
    for file in ("R-12_rev2.pdf", "R-12_rev3.pdf"):
        recs = read_pymupdf_tables(DOCS_DIR / file)
        meta, ch = chunk_regulation([r["text"] for r in recs], file, "pymupdf+tables", recs[0]["version"])
        documents.append({**meta, "source_file": file, "pages": len(recs),
                          "content_hash": content_hash(DOCS_DIR / file), "parser": "pymupdf+tables"})
        chunks += ch
    outs = load_outputs(scan_reader)
    for file, doc_id in (("PS-101_scan.pdf", "PS-101"), ("DS-P-205_scan.pdf", "DS-P-205")):
        rec = outs[(file, 1)]
        meta, ch = chunk_datasheet([rec["text"]], doc_id, file, scan_reader, rec["version"])
        documents.append({**meta, "source_file": file, "pages": 1, "content_hash": content_hash(DOCS_DIR / file),
                          "parser": scan_reader})
        chunks += ch
    link_revisions(documents, chunks)
    write_store(documents, chunks, folder)
    return {"folder": str(Path(folder).name), "documents": len(documents), "chunks": len(chunks),
            "by_type": dict(Counter(c.chunk_type for c in chunks)),
            "clause_4_3_rev3": [c.chunk_id for c in chunks if c.revision == 3 and (c.clause or "").startswith("4.3")]}
