"""The plant archive v2 as a chunk store: the practice 3 store plus the new documents of practice 4.

Steps (all by code, no model):
1. take the chunks of practice 3 (R-12 rev. 2 and rev. 3, PS-101, DS-P-205) as they are;
2. read the new born-digital PDFs with PyMuPDF (text layer + table finder) and chunk them with the
   chunker of practice 3; document metadata (dates, scope, running headers) comes from the document
   registry `data/documents.csv`, as it would come from the plant's document management system;
3. add the new fields: dates of validity, scope, language, table number;
4. an amendment replaces single clauses: the base chunk gets `effective_to` and `amended_by`;
5. references in the text ("см. п. 4.3.3", "по таблице 3", "п. 6.2 R-12") become links in both directions;
6. a long table is also cut into rows: one chunk per row, with the caption, the column names and units;
7. `index_version`: a fingerprint of the chunks, the parser and the embedding model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pymupdf

from . import DATA_DIR, DOCS_DIR, INPUTS_DIR, STORE_DIR
from .normalize import find_tags
from .store import TABLE_NO_RE, Chunk, _clean, chunk_datasheet, chunk_regulation, load_store, write_store

PRACTICE3_STORE = INPUTS_DIR / "practice-03" / "store"
REGISTRY_FILE = DATA_DIR / "documents.csv"
PARSER = "pymupdf+tables"
LONG_TABLE_ROWS = 10  # a table with more data rows than this is also stored row by row


# ---------------------------------------------------------------------------------------------
# Reading the new PDFs (born-digital): practice 3 reader, captions in Russian and English
# ---------------------------------------------------------------------------------------------
CAPTION_ABOVE = re.compile(r"[Тт]аблиц|Table")


def read_pdf(pdf: str | Path) -> list[str]:
    """Page texts: text blocks of the text layer, tables as Markdown in their place."""
    out = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            tables = []
            for t in page.find_tables().tables:
                above = pymupdf.Rect(0, t.bbox[1] - 40, page.rect.width, t.bbox[1])
                if CAPTION_ABOVE.search(page.get_text("text", clip=above)):
                    tables.append(t)
            boxes = [pymupdf.Rect(t.bbox) for t in tables]
            items = [(b[3], " ".join(b[4].split())) for b in page.get_text("blocks")
                     if b[6] == 0 and not any(pymupdf.Rect(b[:4]).intersects(r) for r in boxes)]
            for t in sorted(tables, key=lambda t: t.bbox[1]):
                above_ids = [j for j, (bottom, _) in enumerate(items) if bottom <= t.bbox[1] + 2]
                k = max(above_ids) + 1 if above_ids else 0
                items.insert(k, (t.bbox[3], t.to_markdown().strip()))
            out.append("\n".join(x for _, x in items))
    return out


def parser_version() -> str:
    return f"pymupdf {pymupdf.__version__} find_tables"


# ---------------------------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------------------------
def load_registry(path: str | Path = REGISTRY_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["revision"] = int(r["revision"]) if r["revision"] else None
        r["amendment"] = int(r["amendment"]) if r["amendment"] else None
        r["effective_to"] = r["effective_to"] or None
        r["pages"] = int(r["pages"])
        for k in ("scope_stations", "scope_equipment_types", "scope_tags"):
            r[k] = [x for x in r[k].split(";") if x]
    return rows


def _doc_key(doc_id: str, revision, amendment=None) -> tuple:
    return doc_id, revision, amendment


# ---------------------------------------------------------------------------------------------
# Chunking the new documents
# ---------------------------------------------------------------------------------------------
AMEND_RE = re.compile(r"^(?:Пункт|п\.)\s+(\d+(?:\.\d+)*)\s+изложить в новой редакции")


def chunk_amendment(pages: list[str], doc: dict) -> list[Chunk]:
    """An amendment is a list of numbered items. An item that sets a new wording of a clause becomes a chunk
    with that clause number (cited as "R-12 rev. 3, Amendment No. 1, cl. 4.4.1")."""
    prefix = f"{doc['doc_id']}-r{doc['revision']}-a{doc['amendment']}"
    root = f"{doc['doc_id']} ред. {doc['revision']}, изменение № {doc['amendment']}"
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        for line in text.splitlines():
            line = _clean(line)
            m = re.match(r"^(\d)\s+(\S.*)$", line)
            if not m:
                continue
            item, body = m.groups()
            am = AMEND_RE.match(body)
            clause = am.group(1) if am else None
            chunks.append(Chunk(
                chunk_id=f"{prefix}-{clause}" if clause else f"{prefix}-item{item}", doc_id=doc["doc_id"],
                revision=doc["revision"], chunk_type="clause" if clause else "note", clause=clause,
                section_path=[root], page=page_no,
                text=f"Изменение № {doc['amendment']} к {doc['doc_id']} ред. {doc['revision']}, позиция {item}. {body}",
                source_file=doc["file"], parser=PARSER, parser_version=parser_version(),
                amendment=doc["amendment"]))
    return chunks


def chunk_document(doc: dict) -> list[Chunk]:
    pages = read_pdf(DOCS_DIR / doc["file"])
    if doc["kind"] == "amendment":
        return chunk_amendment(pages, doc)
    if doc["kind"] == "datasheet":
        _, chunks = chunk_datasheet(pages, doc["doc_id"], doc["file"], PARSER, parser_version())
        return chunks
    meta = {"doc_id": doc["doc_id"], "revision": doc["revision"], "effective_date": doc["effective_from"]}
    if doc["language"] == "en":
        meta["root"] = f"{doc['doc_id']} edition {doc['revision']}"
    running = {doc["header_left"], doc["header_right"], f"{doc['header_left']} {doc['header_right']}"}
    _, chunks = chunk_regulation(pages, doc["file"], PARSER, parser_version(), meta=meta, running=running)
    return chunks


# ---------------------------------------------------------------------------------------------
# Long tables: one chunk per row
# ---------------------------------------------------------------------------------------------
def markdown_rows(text: str) -> tuple[list[str], list[list[str]]]:
    """Header and data rows of the Markdown table in a table chunk."""
    lines = [x.strip() for x in text.splitlines() if x.strip().startswith("|")]

    def cells(line: str) -> list[str]:
        return [re.sub(r"\s+,", ",", re.sub(r"\s+", " ", c.replace("<br>", " ").replace("**", ""))).strip()
                for c in line.strip().strip("|").split("|")]

    if len(lines) < 2:
        return [], []
    header = cells(lines[0])
    rows = [cells(x) for x in lines[2:] if not re.fullmatch(r"\|?[\s|:-]+\|?", x)]
    rows = [r for r in rows if r != header]  # a repeated header row
    return header, rows


def split_long_tables(chunks: list[Chunk]) -> list[Chunk]:
    """Rows of long tables as chunks: the caption, the column names and the units go into every row."""
    groups: dict[tuple, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.chunk_type == "table" and c.table_no:
            groups[(c.doc_id, c.revision, c.table_no)].append(c)
    out = []
    for (doc_id, rev, no), parts in groups.items():
        parsed = [(p, *markdown_rows(p.text)) for p in parts]
        if sum(len(rows) for _, _, rows in parsed) <= LONG_TABLE_ROWS:
            continue
        caption = parts[0].text.splitlines()[0]
        for page_chunk, header, rows in parsed:
            page_chunk.split_into_rows = True
            for row in rows:
                key = row[0]
                fields = "; ".join(f"{h}: {v}" for h, v in zip(header, row))
                out.append(Chunk(
                    chunk_id=f"{doc_id}-r{rev}-t{no}-{key}", doc_id=doc_id, revision=rev, chunk_type="table_row",
                    clause=page_chunk.clause, section_path=page_chunk.section_path, page=page_chunk.page,
                    text=f"{caption}. Строка {key}. {fields}", parent_id=page_chunk.chunk_id,
                    source_file=page_chunk.source_file, parser=page_chunk.parser,
                    parser_version=page_chunk.parser_version, table_no=no, row_key=key))
    return out


# ---------------------------------------------------------------------------------------------
# References in both directions
# ---------------------------------------------------------------------------------------------
DOC_ID_RE = r"(?:R-\d+|PS-\d+|VL-\d+|OM-P-\d+)"
REF_RE = re.compile(
    r"(?:(?P<cl>п\.|пп\.|[Пп]ункт[а-я]*|section|clause)\s*(?P<clause>\d+(?:\.\d+)*(?:\s*(?:и|and|,)\s*(?:п\.\s*)?\d+(?:\.\d+)+)*)"
    r"|(?P<tb>[Тт]аблиц[а-я]*|Table)\s+(?P<table>[\dА-ЯA-Z]+(?:\.\d+)?)"
    r"|(?P<ap>[Пп]риложени[а-я]*)\s+(?P<app>[А-Я])\b)"
    rf"(?P<doc>\s+{DOC_ID_RE})?")


def find_references(text: str) -> list[tuple[str | None, str, str]]:
    """(target document or None for "this document", kind "clause"/"table"/"appendix", number)."""
    out = []
    for m in REF_RE.finditer(text):
        doc = m.group("doc").strip() if m.group("doc") else None
        if m.group("clause"):
            for num in re.findall(r"\d+(?:\.\d+)*", m.group("clause")):
                out.append((doc, "clause", num))
        elif m.group("table"):
            out.append((doc, "table", m.group("table")))
        elif m.group("app"):
            out.append((doc, "appendix", m.group("app")))
    return out


def link_references(chunks: list[Chunk]) -> None:
    """Resolve references to chunk ids. Inside a document: the same revision; to another document: every
    revision of it (the date filter keeps the right one)."""
    by_clause: dict[tuple, list[str]] = defaultdict(list)
    by_table: dict[tuple, list[str]] = defaultdict(list)
    for c in chunks:
        if c.chunk_type == "table_row":
            continue
        if c.chunk_type == "table" and c.table_no:
            by_table[(c.doc_id, c.revision, c.table_no)].append(c.chunk_id)
        elif c.clause and c.amendment is None:
            by_clause[(c.doc_id, c.revision, c.clause)].append(c.chunk_id)
    revisions: dict[str, set] = defaultdict(set)
    for c in chunks:
        revisions[c.doc_id].add(c.revision)
    index = {c.chunk_id: c for c in chunks}
    for c in chunks:
        refs: list[str] = []
        if c.chunk_type == "table_row":
            c.references = []
            continue
        text = c.text.split("\n", 1)[-1] if c.chunk_type == "table" else c.text  # not the caption itself
        for doc, kind, num in find_references(text):
            targets = [(c.doc_id, c.revision)] if doc is None or doc == c.doc_id else \
                [(doc, r) for r in sorted(revisions.get(doc, []), key=lambda r: r or 0)]
            for d, r in targets:
                key = (d, r, num)
                refs += by_table.get(key, []) if kind == "table" else by_clause.get(key, [])
        c.references = [r for r in dict.fromkeys(refs) if r != c.chunk_id]
    for c in chunks:
        c.referenced_by = []
    for c in chunks:
        for r in c.references:
            index[r].referenced_by.append(c.chunk_id)


# ---------------------------------------------------------------------------------------------
# Building the store
# ---------------------------------------------------------------------------------------------
def apply_metadata(chunks: list[Chunk], registry: list[dict]) -> None:
    meta = {_doc_key(d["doc_id"], d["revision"], d["amendment"]): d for d in registry}
    for c in chunks:
        d = meta[_doc_key(c.doc_id, c.revision, c.amendment)]
        c.effective_from, c.effective_to = d["effective_from"], d["effective_to"]
        c.scope_stations = d["scope_stations"]
        c.scope_equipment_types = d["scope_equipment_types"]
        c.scope_tags = d["scope_tags"]
        c.doc_language, c.doc_title, c.doc_kind = d["language"], d["title"], d["kind"]
        c.status = c.status or "in force"
        c.effective_date = c.effective_date or d["effective_from"]
        c.equipment_tags = find_tags(c.text)
        if c.chunk_type == "table" and not c.table_no:  # practice 3 chunks: the number is in the caption
            m = TABLE_NO_RE.search(c.text.splitlines()[0])
            c.table_no = m.group(1) if m else None


def apply_amendments(chunks: list[Chunk]) -> None:
    """A clause with a new wording in an amendment: the base chunk is valid until the amendment comes into force."""
    base = {(c.doc_id, c.revision, c.clause): c for c in chunks if c.amendment is None and c.clause}
    for c in chunks:
        if c.amendment is not None and c.clause:
            old = base[(c.doc_id, c.revision, c.clause)]
            old.effective_to = c.effective_from
            old.amended_by = c.chunk_id
            # base + amendments = the text in force: the new wording sits in the section of the old one
            c.parent_id, c.section_path = old.parent_id, old.section_path


def index_version(chunks: list[Chunk], embedder: str | None = None) -> str:
    """Corpus fingerprint + parser versions + embedding model: changes whenever the index must be rebuilt."""
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda c: c.chunk_id):
        h.update(f"{c.chunk_id}\x00{c.text}\x00".encode("utf-8"))
    parsers = sorted({c.parser_version for c in chunks if c.parser_version})
    hp = hashlib.sha256("|".join(parsers).encode()).hexdigest()[:6]
    return f"corpus-{h.hexdigest()[:10]}.parser-{hp}.emb-{embedder or 'none'}"


def build_corpus(folder: str | Path = STORE_DIR) -> dict:
    """Build the store of practice 4 and write it to `folder`. Returns a short summary."""
    t0 = time.perf_counter()
    registry = load_registry()
    _, chunks = load_store(PRACTICE3_STORE)
    for doc in registry:
        if doc["source"] == "practice 4":
            chunks += chunk_document(doc)
    apply_metadata(chunks, registry)
    apply_amendments(chunks)
    rows = split_long_tables(chunks)
    apply_metadata(rows, registry)
    chunks += rows
    link_references(chunks)
    documents = []
    for d in registry:
        n = sum(1 for c in chunks if (c.doc_id, c.revision, c.amendment) == (d["doc_id"], d["revision"], d["amendment"]))
        documents.append({k: d[k] for k in ("doc_id", "revision", "amendment", "title", "kind", "language", "approved",
                                            "effective_from", "effective_to", "scope_stations", "scope_equipment_types",
                                            "scope_tags", "pages", "source")} | {"source_file": d["file"], "chunks": n})
    write_store(documents, chunks, folder)
    version = index_version(chunks)
    summary = {"documents": len(documents), "pages": sum(d["pages"] for d in documents), "chunks": len(chunks),
               "chunks_without_rows": sum(c.chunk_type != "table_row" for c in chunks),
               "table_rows": len(rows), "index_version": version, "seconds": round(time.perf_counter() - t0, 1)}
    (Path(folder) / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def chunk_table(chunks: list[Chunk]) -> list[dict]:
    """Chunks as rows for a DataFrame (the long text cut)."""
    return [{**{k: v for k, v in asdict(c).items() if k not in ("text", "section_path")},
             "text": c.text[:120]} for c in chunks]


if __name__ == "__main__":
    print(build_corpus())
