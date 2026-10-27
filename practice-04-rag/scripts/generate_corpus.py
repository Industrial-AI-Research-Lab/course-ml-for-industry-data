"""Generate the new documents of the plant archive v2 (practice 4) and their ground truth.

Same trick as practice 3: the structure (scripts/corpus_content.py) is the ground truth, the PDF is rendered
from it. All new documents are born-digital: reading is topic 3 and must not get in the way here.
R-12 rev. 2 and rev. 3, PS-101 and DS-P-205 are not rendered again: their chunks come from practice 3.

Output (all in data/):
  docs/*.pdf                     the new documents
  ground_truth/pages.jsonl       page texts of the new documents (the text layer)
  ground_truth/structure.json    clause and table pages of every new document
  documents.csv                  registry of the whole archive v2 with dates and scope (the plant's document system)
  equipment_registry.csv         the equipment registry of practice 1, aligned with R-12, with power and class

Run:  uv run --extra generate python scripts/generate_corpus.py
The output is deterministic. It needs the DejaVu fonts that ship with matplotlib.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
import pymupdf
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import BalancedColumns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_content as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = DATA / "docs"
GT = DATA / "ground_truth"
TTF_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
rl_config.invariant = 1  # no creation date and random ids in the PDF: the same content gives the same bytes


def register_fonts() -> None:
    for name, file in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                       ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Sans", "DejaVuSans.ttf"),
                       ("Sans-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(TTF_DIR / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                                  boldItalic="Serif-Bold")


S = {
    "body": ParagraphStyle("body", fontName="Serif", fontSize=11, leading=15, alignment=TA_LEFT, spaceAfter=5),
    "h1": ParagraphStyle("h1", fontName="Serif-Bold", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=6,
                         keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Serif-Bold", fontSize=11, leading=15, spaceBefore=8, spaceAfter=4,
                         keepWithNext=1),
    "caption": ParagraphStyle("caption", fontName="Serif", fontSize=10.5, leading=14, spaceBefore=6,
                              spaceAfter=4, keepWithNext=1),
    "note": ParagraphStyle("note", fontName="Serif", fontSize=9.5, leading=12, spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Serif", fontSize=10, leading=12.5, alignment=TA_CENTER),
    "cell_left": ParagraphStyle("cell_left", fontName="Serif", fontSize=10, leading=12.5, alignment=TA_LEFT),
    "cell_head": ParagraphStyle("cell_head", fontName="Serif-Bold", fontSize=10, leading=12.5,
                                alignment=TA_CENTER),
    "small": ParagraphStyle("small", fontName="Serif", fontSize=9, leading=11, alignment=TA_LEFT),
    "small_head": ParagraphStyle("small_head", fontName="Serif-Bold", fontSize=9, leading=11, alignment=TA_CENTER),
    "col": ParagraphStyle("col", fontName="Serif", fontSize=10.5, leading=14, alignment=TA_LEFT, spaceAfter=5),
    "t_org": ParagraphStyle("t_org", fontName="Serif", fontSize=12, leading=16, alignment=TA_CENTER),
    "t_kind": ParagraphStyle("t_kind", fontName="Serif-Bold", fontSize=20, leading=26, alignment=TA_CENTER,
                             spaceBefore=90),
    "t_title": ParagraphStyle("t_title", fontName="Serif-Bold", fontSize=16, leading=22, alignment=TA_CENTER,
                              spaceBefore=10),
    "t_line": ParagraphStyle("t_line", fontName="Serif", fontSize=12, leading=18, alignment=TA_CENTER),
    "t_approve": ParagraphStyle("t_approve", fontName="Serif", fontSize=11, leading=15, alignment=TA_RIGHT),
    "t_small": ParagraphStyle("t_small", fontName="Serif-Italic", fontSize=9.5, leading=12, alignment=TA_CENTER),
}

LABELS = {
    "ru": {"revision": "Редакция", "approved": "Утверждён", "scope": "Область действия", "owner": "Владелец документа",
           "page": "Стр. {page} из {total}", "continued": "Продолжение таблицы {no}", "appendix": "Приложение",
           "approve_block": "УТВЕРЖДАЮ<br/>Главный инженер<br/>_____________ А. В. Учебный<br/>{date}",
           "effective": "Дата введения в действие: {date}"},
    "en": {"revision": "Edition", "approved": "Issued", "scope": "Applies to", "owner": "Manufacturer",
           "page": "Page {page} of {total}", "continued": "Table {no} (continued)", "appendix": "Appendix",
           "approve_block": "{date}", "effective": "Valid from: {date}"},
}
EDITION_LETTER = {1: "A", 2: "B", 3: "C"}


def revision_label(doc: dict) -> str:
    lab = LABELS[doc["language"]]
    rev = EDITION_LETTER[doc["revision"]] if doc["language"] == "en" else doc["revision"]
    return f"{lab['revision']} {rev}"


def header_left(doc: dict) -> str:
    return f"{doc['doc_id']} «{doc['title']}»"


def table_number(caption: str) -> str:
    m = re.match(r"(?:Таблица|Table)\s+([\dА-ЯA-Z]+(?:\.\d+)?)", caption)
    return m.group(1) if m else ""


def make_table(spec: dict, rows: list[list] | None = None, small: bool = False) -> Table:
    """A table from cells (row, col, rowspan, colspan, text). `rows` renders only these data rows (long tables)."""
    cells = spec["cells"]
    hr = spec["header_rows"]
    if rows is not None:  # header rows + the given data rows, renumbered
        head = [c for c in cells if c[0] < hr]
        cells = head + [(hr + i, c, 1, 1, v) for i, row in enumerate(rows) for c, v in enumerate(row)]
    n_rows = max(r + rs for r, _, rs, _, _ in cells)
    grid = [["" for _ in range(spec["n_cols"])] for _ in range(n_rows)]
    style = [("GRID", (0, 0), (-1, -1), 0.7, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("BACKGROUND", (0, 0), (-1, hr - 1), colors.HexColor("#e8e8e8"))]
    if small:
        style += [("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    for r, c, rs, cs, text in cells:
        head = r < hr
        if small:
            st = S["small_head"] if head else S["small"]
        else:
            st = S["cell_head"] if head else (S["cell_left"] if c == 0 else S["cell"])
        grid[r][c] = Paragraph(text, st)
        if rs > 1 or cs > 1:
            style.append(("SPAN", (c, r), (c + cs - 1, r + rs - 1)))
    return Table(grid, colWidths=[w * mm for w in spec["widths"]], style=TableStyle(style), repeatRows=hr)


def table_flow(doc: dict, table_id: str) -> list:
    spec = doc["tables"][table_id]
    if table_id in doc.get("long_tables", set()):
        # A long table: page by page, the header repeats and every next page starts with "continued"
        data = [[c[4] for c in spec["cells"] if c[0] == r] for r in range(1, max(c[0] for c in spec["cells"]) + 1)]
        parts, first, nxt = [], 33, 33
        parts.append(data[:first])
        rest = data[first:]
        while rest:
            parts.append(rest[:nxt])
            rest = rest[nxt:]
        flow = [PageBreak()]
        for i, part in enumerate(parts):
            if i:
                flow.append(PageBreak())
                cap = LABELS[doc["language"]]["continued"].format(no=table_number(spec["caption"]))
            else:
                cap = spec["caption"]
            flow += [Paragraph(cap, S["caption"]), make_table(spec, part, small=True)]
        return flow
    parts = [Paragraph(spec["caption"], S["caption"]), make_table(spec)]
    parts.append(Paragraph(spec["note"], S["note"]) if spec.get("note") else Spacer(1, 6))
    return [KeepTogether(parts)]


def regulation_story(doc: dict) -> list:
    lab = LABELS[doc["language"]]
    story = [
        Paragraph(doc.get("manufacturer", C.ORG), S["t_org"]),
        Spacer(1, 28),
        Paragraph(lab["approve_block"].format(date=doc["approved"]), S["t_approve"]),
        Paragraph(f"{doc['kind_label']} {doc['doc_id']}", S["t_kind"]),
        Paragraph(doc["title"], S["t_title"]),
        Spacer(1, 26),
        Paragraph(revision_label(doc), S["t_line"]),
        Paragraph(f"{lab['approved']} {doc['approved']}", S["t_line"]),
        Paragraph(lab["effective"].format(date=".".join(reversed(doc["effective_from"].split("-")))), S["t_line"]),
        Paragraph(doc["replaces"], S["t_line"]),
        Spacer(1, 26),
        Paragraph(f"{lab['scope']}: {doc['scope_text']}", S["t_line"]),
    ]
    if doc["language"] == "ru":
        story.append(Paragraph(f"{lab['owner']}: {C.OWNER}", S["t_line"]))
    story += [Spacer(1, 120), Paragraph(C.DISCLAIMER if doc["language"] == "ru" else
                                        "Teaching document of the ML in Industry 2026 course. Not a real manual.",
                                        S["t_small"]),
              NextPageTemplate("body"), PageBreak()]
    two_col = doc.get("two_column", set())
    for num, title, items in doc["sections"]:
        flow = []
        for item in items:
            kind = item[0]
            if kind == "clause":
                flow.append(Paragraph(f"{item[1]} {item[2]}", S["col"] if num in two_col else S["body"]))
            elif kind == "para":
                flow.append(Paragraph(item[1], S["body"]))
            elif kind == "subsection":
                flow.append(Paragraph(f"{item[1]} {item[2]}", S["h2"]))
            elif kind == "table":
                flow += table_flow(doc, item[1])
        story.append(Paragraph(f"{num} {title}", S["h1"]))
        if num in two_col:
            intro = [f for f in flow if getattr(f, "style", None) is S["body"]]
            story.extend(intro)
            story.append(BalancedColumns([f for f in flow if f not in intro], nCols=2, spaceBefore=2,
                                         innerPadding=8))
        else:
            story.extend(flow)
    if doc.get("appendix"):
        letter, title, items = doc["appendix"]
        story += [PageBreak(), Paragraph(f"{lab['appendix']} {letter}",
                                         ParagraphStyle("app", parent=S["h1"], alignment=TA_CENTER)),
                  Paragraph(title, ParagraphStyle("app2", parent=S["body"], alignment=TA_CENTER))]
        for kind, *rest in items:
            if kind == "para":
                story.append(Paragraph(rest[0], S["body"]))
            else:
                story += table_flow(doc, rest[0])
    return story


def render(story_fn, path: Path, header: tuple[str, str] | None, page_label: str, title: str,
           first_template: str = "title") -> int:
    def once(total: int) -> tuple[bytes, int]:
        buf = io.BytesIO()
        d = BaseDocTemplate(buf, pagesize=A4, leftMargin=25 * mm, rightMargin=15 * mm, topMargin=22 * mm,
                            bottomMargin=20 * mm, title=title, author=C.OWNER, subject="Учебный документ")
        frame = Frame(d.leftMargin, d.bottomMargin, d.width, d.height, id="f")

        def header_footer(canvas, dd):
            canvas.saveState()
            canvas.setFont("Sans", 8.5)
            top = A4[1] - 14 * mm
            if header:
                canvas.drawString(dd.leftMargin, top, header[0])
                canvas.drawRightString(A4[0] - dd.rightMargin, top, header[1])
                canvas.setLineWidth(0.5)
                canvas.line(dd.leftMargin, top - 2.5 * mm, A4[0] - dd.rightMargin, top - 2.5 * mm)
            canvas.drawCentredString(A4[0] / 2, 11 * mm, page_label.format(page=dd.page, total=total))
            canvas.restoreState()

        templates = [PageTemplate(id="title", frames=[frame]),
                     PageTemplate(id="body", frames=[frame], onPage=header_footer)]
        if first_template == "body":
            templates.reverse()
        d.addPageTemplates(templates)
        d.build(story_fn())
        return buf.getvalue(), d.page

    _, pages = once(0)
    data, pages2 = once(pages)
    assert pages == pages2
    path.write_bytes(data)
    return pages


def build_regulation(doc: dict) -> int:
    lab = LABELS[doc["language"]]
    return render(lambda: regulation_story(doc), DOCS / doc["file"], (header_left(doc), revision_label(doc)),
                  lab["page"], f"{doc['doc_id']} {doc['title']}")


def build_amendment(doc: dict) -> int:
    def story():
        out = [Paragraph(C.ORG, S["t_org"]), Spacer(1, 16),
               Paragraph(LABELS["ru"]["approve_block"].format(date=doc["approved"]), S["t_approve"]),
               Spacer(1, 20),
               Paragraph(f"ИЗМЕНЕНИЕ № {doc['amendment']}", ParagraphStyle("k", parent=S["t_kind"], spaceBefore=10)),
               Paragraph(f"к регламенту {doc['doc_id']} «{doc['title']}», редакция {doc['revision']}", S["t_line"]),
               Paragraph(f"Утверждено {doc['approved']}", S["t_line"]),
               Paragraph(LABELS["ru"]["effective"].format(date=".".join(reversed(doc["effective_from"].split("-")))),
                         S["t_line"]),
               Paragraph(f"Область действия: {doc['scope_text']}", S["t_line"]), Spacer(1, 24)]
        out += [Paragraph(f"{num} {text}", S["body"]) for num, _, text in doc["items"]]
        out += [Spacer(1, 40), Paragraph(C.DISCLAIMER, S["t_small"])]
        return out

    return render(story, DOCS / doc["file"], None, LABELS["ru"]["page"],
                  f"{doc['doc_id']} ред. {doc['revision']}, изменение № {doc['amendment']}", first_template="body")


def build_datasheet(spec: dict, path: Path) -> None:
    """Born-digital datasheet in the layout of PS-101 (practice 3)."""
    font, bold = "Sans", "Sans-Bold"
    st = ParagraphStyle("ds", fontName=font, fontSize=10.5, leading=14)
    st_b = ParagraphStyle("dsb", parent=st, fontName=bold)
    st_c = ParagraphStyle("dsc", parent=st, alignment=TA_CENTER)
    st_title = ParagraphStyle("dst", fontName=bold, fontSize=17, leading=24, alignment=TA_CENTER)
    st_sub = ParagraphStyle("dss", fontName=bold, fontSize=12, leading=16, spaceBefore=14, spaceAfter=6)
    st_foot = ParagraphStyle("dsf", fontName=font, fontSize=8.5, leading=11, alignment=TA_CENTER)
    grid = [("GRID", (0, 0), (-1, -1), 0.7, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    head = Table([[Paragraph(k, st_b), Paragraph(v, st)] for k, v in spec["head"]],
                 colWidths=[70 * mm, 95 * mm], style=TableStyle(grid))
    params = Table([[Paragraph("Параметр / Parameter", st_b), Paragraph("Ед. / Unit", st_b),
                     Paragraph("Значение / Value", st_b)]]
                   + [[Paragraph(p, st), Paragraph(u, st_c), Paragraph(v, st_c)] for p, u, v in spec["params"]],
                   colWidths=[85 * mm, 35 * mm, 45 * mm],
                   style=TableStyle(grid + [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8"))]))
    materials = Table([[Paragraph(k, st), Paragraph(v, st)] for k, v in spec["materials"]],
                      colWidths=[70 * mm, 95 * mm], style=TableStyle(grid))
    story = [Paragraph(spec["title_ru"], st_title), Paragraph(spec["title_en"], st_title), Spacer(1, 16), head,
             Paragraph(spec["params_caption"], st_sub), params,
             Paragraph(spec["materials_caption"], st_sub), materials, Spacer(1, 60),
             Paragraph(spec["footer"], st_foot)]
    d = BaseDocTemplate(str(path), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=25 * mm,
                        bottomMargin=20 * mm, title=f"{spec['doc_id']} datasheet")
    d.addPageTemplates([PageTemplate(id="p", frames=[Frame(d.leftMargin, d.bottomMargin, d.width, d.height)])])
    d.build(story)


# ---------------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------------
def page_texts(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as d:
        return [p.get_text("text") for p in d]


CLAUSE_LINE = re.compile(r"^(\d+(?:\.\d+){0,2}) \S", re.M)


def structure_of(doc: dict, texts: list[str]) -> dict:
    """Pages of every clause and of every table (a long table spans several pages)."""
    clauses: dict[str, int] = {}
    for i, t in enumerate(texts[1:], start=2):
        for m in CLAUSE_LINE.finditer(t):
            clauses.setdefault(m.group(1), i)
    expected = []
    for num, _, items in doc["sections"]:
        expected.append(num)
        expected += [it[1] for it in items if it[0] in ("clause", "subsection")]
    missing = [n for n in expected if n not in clauses]
    if missing:
        raise ValueError(f"{doc['key']}: clauses not found on pages: {missing}")
    tables = {}
    for tid, spec in doc["tables"].items():
        no = table_number(spec["caption"])
        cont = LABELS[doc["language"]]["continued"].format(no=no)
        pages = [i for i, t in enumerate(texts, start=1) if spec["caption"][:14] in t or cont in t]
        if not pages:
            raise ValueError(f"{doc['key']}: table {tid} not found")
        tables[no] = {"table_id": tid, "pages": pages, "caption": spec["caption"]}
    return {"clauses": {n: clauses[n] for n in expected}, "tables": tables, "pages": len(texts)}


def write_registry() -> None:
    src = ROOT / "inputs" / "practice-01" / "equipment_registry.csv"
    with open(src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_tag = {r["equipment_id"]: r for r in rows}
    for tag, area in C.REGISTRY_FIX.items():
        by_tag[tag]["area"] = area
    for tag, (etype, area, name, power, cls, ds) in C.REGISTRY_EXTRA.items():
        by_tag.setdefault(tag, {"equipment_id": tag})
        by_tag[tag].update({"equipment_type": etype, "area": area, "name": name, "power_kw": power,
                            "machine_class": cls, "datasheet": ds})
    with open(DATA / "equipment_registry.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["equipment_id", "equipment_type", "area", "station", "name", "power_kw",
                               "machine_class", "datasheet", "documents"], lineterminator="\n")
        w.writeheader()
        for r in by_tag.values():
            m = re.match(r"Pump station (\d+)", r["area"])
            w.writerow({"equipment_id": r["equipment_id"], "equipment_type": r["equipment_type"], "area": r["area"],
                        "station": m.group(1) if m else "", "name": r["name"], "power_kw": r.get("power_kw", ""),
                        "machine_class": r.get("machine_class", ""), "datasheet": r.get("datasheet", ""),
                        "documents": ";".join(C.REGISTRY_DOCUMENTS.get(r["equipment_id"], []))})


def main() -> None:
    register_fonts()
    DOCS.mkdir(parents=True, exist_ok=True)
    GT.mkdir(parents=True, exist_ok=True)
    registry = []
    structure = {}
    pages_out = []

    def add(doc: dict, n_pages: int, header: tuple[str, str] | None, kind: str):
        texts = page_texts(DOCS / doc["file"])
        for i, t in enumerate(texts, start=1):
            pages_out.append({"file": doc["file"], "doc_id": doc["doc_id"], "revision": doc["revision"],
                              "amendment": doc.get("amendment"), "page": i, "text": t})
        registry.append({
            "file": doc["file"], "doc_id": doc["doc_id"], "revision": doc["revision"],
            "amendment": doc.get("amendment") or "", "kind": kind, "title": doc.get("title", ""),
            "language": doc.get("language", "ru"), "approved": doc.get("approved", ""),
            "effective_from": doc["effective_from"], "effective_to": doc.get("effective_to") or "",
            "scope_stations": ";".join(doc["scope_stations"]),
            "scope_equipment_types": ";".join(doc["scope_equipment_types"]),
            "scope_tags": ";".join(doc.get("scope_tags", [])), "pages": n_pages,
            "header_left": header[0] if header else "", "header_right": header[1] if header else "",
            "source": "practice 4"})
        return texts

    for doc in C.REGULATIONS:
        n = build_regulation(doc)
        texts = add(doc, n, (header_left(doc), revision_label(doc)), doc["kind"])
        structure[doc["key"]] = structure_of(doc, texts)
    a = C.AMENDMENT_R12
    n = build_amendment(a)
    add(a, n, None, "amendment")
    for ds in C.DATASHEETS:
        file = f"{ds['doc_id']}.pdf"
        build_datasheet(ds, DOCS / file)
        meta = {"file": file, "doc_id": ds["doc_id"], "revision": None, "language": "ru", "title": "Паспорт насоса",
                "effective_from": f"{ds['year']}-01-01", "scope_stations": [],
                "scope_equipment_types": ["pump"], "scope_tags": [ds["tag"]]}
        station = C.REGISTRY_EXTRA[ds["tag"]][1].split()[-1]
        meta["scope_stations"] = [station]
        add(meta, 1, None, "datasheet")
    # documents of practice 3: their chunks are reused, the registry adds dates and scope
    p3_files = {("R-12", 2): ("R-12_rev2.pdf", 6), ("R-12", 3): ("R-12_rev3.pdf", 6),
                ("PS-101", None): ("PS-101_scan.pdf", 1), ("DS-P-205", None): ("DS-P-205_scan.pdf", 1)}
    for (doc_id, rev), m in C.PRACTICE3_DOCS.items():
        file, pages = p3_files[(doc_id, rev)]
        registry.append({"file": file, "doc_id": doc_id, "revision": rev if rev is not None else "", "amendment": "",
                         "kind": m["kind"], "title": "Вибрационный контроль насосного оборудования" if rev else
                         "Паспорт насоса", "language": m["language"],
                         "approved": {2: "01.06.2022", 3: "14.03.2025"}.get(rev, ""),
                         "effective_from": m["effective_from"], "effective_to": m["effective_to"] or "",
                         "scope_stations": ";".join(m["scope_stations"]),
                         "scope_equipment_types": ";".join(m["scope_equipment_types"]),
                         "scope_tags": ";".join(m["scope_tags"]), "pages": pages, "header_left": "",
                         "header_right": "", "source": "practice 3"})
    registry.sort(key=lambda r: (r["source"] != "practice 3", r["doc_id"], str(r["revision"]), str(r["amendment"])))
    with open(DATA / "documents.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(registry[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(registry)
    with open(GT / "pages.jsonl", "w", encoding="utf-8") as f:
        for p in pages_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (GT / "structure.json").write_text(json.dumps(structure, ensure_ascii=False, indent=1), encoding="utf-8")
    write_registry()
    total = sum(int(r["pages"]) for r in registry)
    print(f"{len(registry)} documents, {total} pages")
    for r in registry:
        print(f"  {r['file']:32s} {r['doc_id']:9s} rev {str(r['revision']):2s} {r['pages']:>2} p.  {r['source']}")


if __name__ == "__main__":
    main()
