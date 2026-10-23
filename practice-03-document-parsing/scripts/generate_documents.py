"""Generate the synthetic documents of practice 3 and their ground truth.

Same trick as the maintenance log of practice 1: first the structure (it is the ground truth), then
the "damage". The regulation is rendered to a born-digital PDF; the scan is a picture of the same pages
with rotation, noise, blur, a stamp and a shadow, saved without a text layer. So the ground truth text
of every scanned page is the text layer of its born-digital twin.

Output (all in data/):
  docs/                 what the "plant" has: PDFs (digital and scans), a P&ID image, a nameplate photo
  drawings/             two hard P&ID samples for the tag reader: a dense sheet and its scanned paper copy
  ground_truth/twins/   born-digital twins of the scanned datasheets (not part of the plant archive)
  ground_truth/*.json*  page texts, tables, clauses, questions, P&ID tags, nameplate fields
  documents.csv         registry of the files in docs/

Run:  uv run --extra generate python scripts/generate_documents.py
      ... generate_documents.py --only-drawings   # the P&ID samples and pid.json only
The output is deterministic (fixed seeds). It needs the fonts that ship with matplotlib (DejaVu) and the
CID font STSong-Light that ships with ReportLab (Chinese datasheet).
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import BalancedColumns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import document_content as C  # noqa: E402
from questions_content import QUESTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = DATA / "docs"
GT = DATA / "ground_truth"
TWINS = GT / "twins"
DRAWINGS = DATA / "drawings"  # more P&ID sheets of the plant: test samples of the tag reader, not in the archive
SCAN_DPI = 150  # a cheap office scanner setting; 300 DPI is the usual advice for OCR

# ---------------------------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------------------------
TTF_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def register_fonts() -> None:
    for name, file in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                       ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Sans", "DejaVuSans.ttf"),
                       ("Sans-Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(TTF_DIR / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                                  boldItalic="Serif-Bold")
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(TTF_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


# ---------------------------------------------------------------------------------------------
# Regulation R-12
# ---------------------------------------------------------------------------------------------
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


def make_table(table_id: str, rev: int, widths_mm: list[float]) -> Table:
    spec = C.TABLES[table_id]
    cells = spec["cells"](rev)
    n_rows = max(r + rs for r, _, rs, _, _ in cells)
    grid = [["" for _ in range(spec["n_cols"])] for _ in range(n_rows)]
    style = [("GRID", (0, 0), (-1, -1), 0.7, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("BACKGROUND", (0, 0), (-1, spec["header_rows"] - 1), colors.HexColor("#e8e8e8"))]
    for r, c, rs, cs, text in cells:
        head = r < spec["header_rows"]
        st = S["cell_head"] if head else (S["cell_left"] if c == 0 else S["cell"])
        grid[r][c] = Paragraph(text, st)
        if rs > 1 or cs > 1:
            style.append(("SPAN", (c, r), (c + cs - 1, r + rs - 1)))
    return Table(grid, colWidths=[w * mm for w in widths_mm], style=TableStyle(style),
                 repeatRows=spec["header_rows"])


TABLE_WIDTHS = {"table3": [55, 32, 38, 38], "table2": [60, 50, 30], "tableA1": [20, 62, 30, 20, 28]}


def table_block(table_id: str, rev: int) -> KeepTogether:
    spec = C.TABLES[table_id]
    parts = [Paragraph(spec["caption"], S["caption"]), make_table(table_id, rev, TABLE_WIDTHS[table_id])]
    if spec["note"]:
        parts.append(Paragraph(spec["note"], S["note"]))
    else:
        parts.append(Spacer(1, 6))
    return KeepTogether(parts)


def r12_story(rev: int) -> list:
    R = C.REVISIONS[rev]
    story = [
        Paragraph(C.ORG, S["t_org"]),
        Spacer(1, 28),
        Paragraph("УТВЕРЖДАЮ<br/>Главный инженер<br/>_____________ А. В. Учебный<br/>"
                  f"{R['approved']}", S["t_approve"]),
        Paragraph(f"РЕГЛАМЕНТ {C.DOC_ID}", S["t_kind"]),
        Paragraph(C.DOC_TITLE, S["t_title"]),
        Spacer(1, 26),
        Paragraph(f"Редакция {rev}", S["t_line"]),
        Paragraph(f"Утверждён {R['approved']}", S["t_line"]),
        Paragraph(R["replaces"], S["t_line"]),
        Spacer(1, 26),
        Paragraph(f"Область действия: {C.SCOPE}", S["t_line"]),
        Paragraph(f"Владелец документа: {C.OWNER}", S["t_line"]),
        Spacer(1, 150),
        Paragraph(C.DISCLAIMER, S["t_small"]),
        NextPageTemplate("body"),
        PageBreak(),
    ]
    for num, title, items in C.SECTIONS:
        flow = []
        for item in items:
            kind = item[0]
            if kind == "clause":
                flow.append(Paragraph(f"{item[1]} {item[2].format(**R)}",
                                      S["col"] if num in C.TWO_COLUMN_SECTIONS else S["body"]))
            elif kind == "para":
                flow.append(Paragraph(item[1].format(**R), S["body"]))
            elif kind == "subsection":
                flow.append(Paragraph(f"{item[1]} {item[2]}", S["h2"]))
            elif kind == "table":
                flow.append(table_block(item[1], rev))
        story.append(Paragraph(f"{num} {title}", S["h1"]))
        if num in C.TWO_COLUMN_SECTIONS:
            intro = [f for f in flow if f.style is S["body"]]
            story.extend(intro)
            story.append(BalancedColumns([f for f in flow if f not in intro], nCols=2, spaceBefore=2,
                                         innerPadding=8))
        else:
            story.extend(flow)
    letter, title, items = C.APPENDIX
    story += [PageBreak(), Paragraph(f"Приложение {letter}", ParagraphStyle("app", parent=S["h1"],
                                                                             alignment=TA_CENTER)),
              Paragraph(title, ParagraphStyle("app2", parent=S["body"], alignment=TA_CENTER))]
    for kind, *rest in items:
        story.append(Paragraph(rest[0], S["body"]) if kind == "para" else table_block(rest[0], rev))
    return story


def build_r12(rev: int, path: Path) -> int:
    def render(total: int) -> tuple[bytes, int]:
        buf = io.BytesIO()
        doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=25 * mm, rightMargin=15 * mm, topMargin=22 * mm,
                              bottomMargin=20 * mm, title=f"{C.DOC_ID} {C.DOC_TITLE}, ред. {rev}",
                              author=C.OWNER, subject="Учебный документ")
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

        def header_footer(canvas, d):
            canvas.saveState()
            canvas.setFont("Sans", 8.5)
            top = A4[1] - 14 * mm
            canvas.drawString(d.leftMargin, top, f"{C.DOC_ID} «{C.DOC_TITLE}»")
            canvas.drawRightString(A4[0] - d.rightMargin, top, f"Редакция {rev}")
            canvas.setLineWidth(0.5)
            canvas.line(d.leftMargin, top - 2.5 * mm, A4[0] - d.rightMargin, top - 2.5 * mm)
            canvas.drawCentredString(A4[0] / 2, 11 * mm, f"Стр. {d.page} из {total}")
            canvas.restoreState()

        doc.addPageTemplates([PageTemplate(id="title", frames=[frame]),
                              PageTemplate(id="body", frames=[frame], onPage=header_footer)])
        doc.build(r12_story(rev))
        return buf.getvalue(), doc.page

    _, pages = render(0)
    data, pages2 = render(pages)
    assert pages == pages2
    path.write_bytes(data)
    return pages


# ---------------------------------------------------------------------------------------------
# Datasheets
# ---------------------------------------------------------------------------------------------
CJK_RUN = re.compile("([　-鿿＀-￯]+)")


def build_datasheet(spec: dict, path: Path) -> None:
    cjk = spec["font"] == "cjk"
    font, bold = "Sans", "Sans-Bold"

    def P(text: str, style: ParagraphStyle) -> Paragraph:
        """Chinese characters in the CID font, everything else (Latin, digits, ³) in DejaVu Sans."""
        return Paragraph(CJK_RUN.sub(r'<font name="STSong-Light">\1</font>', text), style)

    st = ParagraphStyle("ds", fontName=font, fontSize=10.5, leading=14)
    st_b = ParagraphStyle("dsb", parent=st, fontName=bold)
    st_c = ParagraphStyle("dsc", parent=st, alignment=TA_CENTER)
    st_title = ParagraphStyle("dst", fontName=bold, fontSize=17, leading=24, alignment=TA_CENTER)
    st_sub = ParagraphStyle("dss", fontName=bold, fontSize=12, leading=16, spaceBefore=14, spaceAfter=6)
    st_foot = ParagraphStyle("dsf", fontName=font, fontSize=8.5, leading=11, alignment=TA_CENTER)
    grid = [("GRID", (0, 0), (-1, -1), 0.7, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    head = Table([[P(k, st_b), P(v, st)] for k, v in spec["head"]],
                 colWidths=[70 * mm, 95 * mm], style=TableStyle(grid))
    unit_head = "单位" if cjk else "Ед. / Unit"
    value_head = "数值" if cjk else "Значение / Value"
    param_head = "参数" if cjk else "Параметр / Parameter"
    params = Table([[P(param_head, st_b), P(unit_head, st_b), P(value_head, st_b)]]
                   + [[P(p, st), P(u, st_c), P(v, st_c)] for p, u, v in spec["params"]],
                   colWidths=[85 * mm, 35 * mm, 45 * mm],
                   style=TableStyle(grid + [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8"))]))
    materials = Table([[P(k, st), P(v, st)] for k, v in spec["materials"]],
                      colWidths=[70 * mm, 95 * mm], style=TableStyle(grid))
    story = [P(spec["title_ru"], st_title)] + ([P(spec["title_en"], st_title)] if spec["title_en"] else [])
    story += [Spacer(1, 16), head,
             P(spec["params_caption"], st_sub), params,
             P(spec["materials_caption"], st_sub), materials, Spacer(1, 60),
             P(spec["footer"], st_foot)]
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=25 * mm,
                          bottomMargin=20 * mm, title=f"{spec['doc_id']} datasheet")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width,
                                                             doc.height)])])
    doc.build(story)


# ---------------------------------------------------------------------------------------------
# Scans: render the born-digital page and damage it
# ---------------------------------------------------------------------------------------------
def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return Image.frombytes("L", (pix.width, pix.height), pix.samples)


def stamp(img: Image.Image, center: tuple[int, int], lines: list[str], radius: int) -> None:
    layer = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(layer)
    x, y = center
    for r, w in ((radius, 7), (radius - 22, 3)):
        d.ellipse([x - r, y - r, x + r, y + r], outline=255, width=w)
    font = pil_font(int(radius * 0.22), bold=True)
    for i, text in enumerate(lines):
        tw = d.textlength(text, font=font)
        d.text((x - tw / 2, y - radius * 0.35 + i * radius * 0.33), text, fill=255, font=font)
    layer = layer.rotate(-14, center=center).filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * (1 - 0.55 * np.asarray(layer, dtype=np.float32) / 255) + 0
    img.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))


def damage(img: Image.Image, rng: np.random.Generator, angle: float, blur: float, noise: float,
           ink: int = 25, quality: int = 55, shadow: bool = True) -> Image.Image:
    """Rotation, blur, a binding shadow, noise, specks and JPEG artefacts. Higher `ink` = fainter print."""
    paper = 236
    arr = np.asarray(img, dtype=np.float32) / 255
    arr = ink + arr * (paper - ink)  # grey paper, not quite black ink
    out = Image.fromarray(arr.astype(np.uint8)).rotate(angle, resample=Image.BICUBIC, fillcolor=paper)
    out = out.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, dtype=np.float32)
    h, w = arr.shape
    if shadow:  # binding shadow on the left edge
        ramp = np.clip(np.linspace(0, 1, w) / 0.07, 0, 1) * 0.3 + 0.7
        arr = arr * ramp[None, :]
    arr += rng.normal(0, noise, arr.shape)
    specks = rng.random(arr.shape) < 0.0008
    arr[specks] = rng.uniform(0, 90, specks.sum())
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue()))


def make_scan(src: Path, dst: Path, seed: int, page_settings: dict[int, dict] | None = None,
              dpi: int = SCAN_DPI) -> None:
    rng = np.random.default_rng(seed)
    page_settings = page_settings or {}
    out = pymupdf.open()
    with pymupdf.open(src) as doc:
        for i, page in enumerate(doc, start=1):
            img = render_page(page, dpi)
            s = page_settings.get(i, {})
            for st in s.get("stamps", []):
                stamp(img, (int(st[0] * img.width), int(st[1] * img.height)), st[2], int(st[3] * dpi / 200))
            angle = s.get("angle", float(rng.uniform(-1.2, 1.2)))
            img = damage(img, rng, angle=angle, blur=s.get("blur", 0.8), noise=s.get("noise", 11),
                         ink=s.get("ink", 25), quality=s.get("quality", 50))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=55)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=buf.getvalue())
    out.set_metadata({"title": f"Scan of {src.stem}", "producer": "scanner (synthetic)"})
    out.save(dst, garbage=4, deflate=True)
    out.close()


# ---------------------------------------------------------------------------------------------
# P&ID fragment and nameplate photo
# ---------------------------------------------------------------------------------------------
def draw_pid(path: Path) -> None:
    W, H = 2000, 1300
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    f_tag, f_small, f_line = pil_font(30, bold=True), pil_font(24), pil_font(22)
    lw = 4

    def line(pts, width=lw, dash=False):
        if not dash:
            d.line(pts, fill=0, width=width)
            return
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            n = int(max(abs(x2 - x1), abs(y2 - y1)) // 18)
            for k in range(0, n, 2):
                a, b = k / n, min((k + 1) / n, 1)
                d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)],
                       fill=0, width=2)

    def arrow(x, y, direction="right"):
        s = 16
        pts = {"right": [(x, y), (x - s * 1.6, y - s), (x - s * 1.6, y + s)],
               "down": [(x, y), (x - s, y - s * 1.6), (x + s, y - s * 1.6)]}[direction]
        d.polygon(pts, fill=0)

    def label(x, y, text, font=f_tag, anchor="mm"):
        d.text((x, y), text, fill=0, font=font, anchor=anchor)

    def gate_valve(x, y, tag, vertical=False, tag_above=True):
        s = 26
        if vertical:
            d.polygon([(x - s, y - s), (x + s, y - s), (x, y)], outline=0, fill=255, width=3)
            d.polygon([(x - s, y + s), (x + s, y + s), (x, y)], outline=0, fill=255, width=3)
        else:
            d.polygon([(x - s, y - s), (x - s, y + s), (x, y)], outline=0, fill=255, width=3)
            d.polygon([(x + s, y - s), (x + s, y + s), (x, y)], outline=0, fill=255, width=3)
        label(x, y - 58 if tag_above else y + 58, tag)

    def bubble(x, y, letters, number, to=None):
        if to:
            line([(x, y), to], dash=True)
        r = 46
        d.ellipse([x - r, y - r, x + r, y + r], outline=0, fill=255, width=3)
        label(x, y - 16, letters, f_small)
        label(x, y + 18, number, f_small)

    def pump(x, y, tag):
        r = 55
        d.ellipse([x - r, y - r, x + r, y + r], outline=0, fill=255, width=lw)
        d.polygon([(x - 28, y - 35), (x - 28, y + 35), (x + 40, y)], outline=0, fill=255, width=3)
        d.line([(x - 60, y + r + 8), (x + 60, y + r + 8)], fill=0, width=lw)
        label(x, y + r + 45, tag)

    # tank T-201 with level transmitter
    d.rounded_rectangle([140, 230, 340, 780], radius=70, outline=0, width=lw, fill=255)
    label(240, 820, "T-201")
    bubble(240, 150, "LT", "201", to=(240, 230))
    # suction line and header
    line([(340, 700), (520, 700)])
    arrow(470, 700)
    label(400, 670, "100-PL-001", f_line)
    line([(520, 500), (520, 900)])
    rows = [(500, "P-101", "XV-101"), (900, "P-102", "XV-102")]
    for y, ptag, vtag in rows:
        line([(520, y), (745, y)])
        gate_valve(630, y, vtag, tag_above=(y == 500))
        pump(800, y, ptag)
        line([(855, y), (1150, y)])
        arrow(1100, y)
    # discharge header and outlet with flow control
    line([(1150, 500), (1150, 900)])
    line([(1150, 500), (1900, 500)])
    arrow(1880, 500)
    label(1700, 470, "80-PL-002", f_line)
    # control valve FV-305 with actuator
    x = 1500
    gate_valve(x, 500, "", tag_above=True)
    line([(x, 474), (x, 420)], width=3)
    d.chord([x - 38, 380, x + 38, 456], 180, 360, outline=0, fill=255, width=3)
    d.line([(x - 38, 418), (x + 38, 418)], fill=0, width=3)
    label(x + 100, 400, "FV-305")
    bubble(1320, 360, "FT", "305", to=(1320, 500))
    line([(1366, 360), (1462, 400)], dash=True)
    # relief valve on P-101 discharge
    line([(1000, 500), (1000, 330)])
    d.polygon([(976, 330), (1024, 330), (1000, 290)], outline=0, fill=255, width=3)
    d.polygon([(1000, 290), (1040, 266), (1040, 314)], outline=0, fill=255, width=3)
    label(1000, 240, "PSV-101")
    # instruments on P-101
    bubble(930, 680, "PT", "101", to=(930, 500))
    bubble(700, 300, "VT", "101", to=(785, 445))
    bubble(860, 300, "TT", "101", to=(815, 445))
    # title block
    x0, y0, x1, y1 = 1360, 1030, 1970, 1270
    d.rectangle([x0, y0, x1, y1], outline=0, width=4)
    for yy in (1090, 1150, 1210):
        d.line([(x0, yy), (x1, yy)], fill=0, width=2)
    d.line([(1660, 1210), (1660, 1270)], fill=0, width=2)
    label(x0 + 15, 1060, "АО «Учебный НПЗ» · учебный чертёж", f_line, "lm")
    label(x0 + 15, 1120, C.PID_TITLE["title"], f_small, "lm")
    label(x0 + 15, 1180, f"Чертёж {C.PID_TITLE['drawing']}", f_tag, "lm")
    label(x0 + 15, 1240, f"Ред. {C.PID_TITLE['revision']}", f_small, "lm")
    label(1675, 1240, C.PID_TITLE["date"], f_small, "lm")
    img.save(path, optimize=True)


def draw_pid_sheet2(path: Path) -> None:
    """A full sheet in small print: what a real P&ID looks like next to the clean fragment of sheet 1."""
    W, H = 3300, 2300
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    f_tag, f_small, f_line = pil_font(22, bold=True), pil_font(17), pil_font(16)
    lw = 3

    def line(pts, width=lw, dash=False):
        if not dash:
            d.line(pts, fill=0, width=width)
            return
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            n = max(int(max(abs(x2 - x1), abs(y2 - y1)) // 12), 2)
            for k in range(0, n, 2):
                a, b = k / n, min((k + 1) / n, 1)
                d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a), (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)],
                       fill=0, width=2)

    def arrow(x, y, direction="right"):
        s = 11
        pts = {"right": [(x, y), (x - s * 1.6, y - s), (x - s * 1.6, y + s)],
               "down": [(x, y), (x - s, y - s * 1.6), (x + s, y - s * 1.6)],
               "up": [(x, y), (x - s, y + s * 1.6), (x + s, y + s * 1.6)]}[direction]
        d.polygon(pts, fill=0)

    def label(x, y, text, font=f_tag, anchor="mm"):
        d.text((x, y), text, fill=0, font=font, anchor=anchor)

    def vlabel(x, y, text, font=f_tag):
        """A tag written along a vertical line, read bottom to top."""
        w = int(d.textlength(text, font=font)) + 8
        layer = Image.new("L", (w, 32), 255)
        ImageDraw.Draw(layer).text((w / 2, 16), text, fill=0, font=font, anchor="mm")
        layer = layer.rotate(90, expand=True)
        img.paste(layer, (int(x - layer.width / 2), int(y - layer.height / 2)))

    def gate_valve(x, y, tag="", vertical=False, at=None):
        s = 17
        if vertical:
            d.polygon([(x - s, y - s), (x + s, y - s), (x, y)], outline=0, fill=255, width=2)
            d.polygon([(x - s, y + s), (x + s, y + s), (x, y)], outline=0, fill=255, width=2)
        else:
            d.polygon([(x - s, y - s), (x - s, y + s), (x, y)], outline=0, fill=255, width=2)
            d.polygon([(x + s, y - s), (x + s, y + s), (x, y)], outline=0, fill=255, width=2)
        if tag:
            label(*(at or (x, y - 40)), tag)

    def control_valve(x, y, tag, at, vertical=False):
        gate_valve(x, y, vertical=vertical)
        if vertical:
            line([(x, y), (x + 40, y)], width=2)
            d.chord([x + 40, y - 25, x + 90, y + 25], 270, 90, outline=0, fill=255, width=2)
            d.line([(x + 65, y - 25), (x + 65, y + 25)], fill=0, width=2)
        else:
            line([(x, y - 17), (x, y - 40)], width=2)
            d.chord([x - 25, y - 65, x + 25, y - 15], 180, 360, outline=0, fill=255, width=2)
            d.line([(x - 25, y - 40), (x + 25, y - 40)], fill=0, width=2)
        label(*at, tag)

    def bubble(x, y, letters, number, to=(), panel=False):
        for t in to:
            line([(x, y), t], dash=True)
        r = 30
        d.ellipse([x - r, y - r, x + r, y + r], outline=0, fill=255, width=2)
        if panel:  # a line across the bubble: the instrument is on the control room panel
            d.line([(x - r, y), (x + r, y)], fill=0, width=2)
        label(x, y - 11, letters, f_line)
        label(x, y + 12, number, f_line)

    def pump(x, y, tag):
        r = 38
        d.ellipse([x - r, y - r, x + r, y + r], outline=0, fill=255, width=lw)
        d.polygon([(x - 19, y - 24), (x - 19, y + 24), (x + 27, y)], outline=0, fill=255, width=2)
        d.line([(x - 42, y + r + 6), (x + 42, y + r + 6)], fill=0, width=lw)
        label(x, y + r + 32, tag)

    def offsheet(x, y, text, left=False):
        """Off-sheet connector: a pointed box with the number of the other drawing. Not a tag."""
        w, h = 190, 44
        pts = ([(x, y - h / 2), (x + w, y - h / 2), (x + w, y + h / 2), (x, y + h / 2), (x - 26, y)] if left else
               [(x - w, y - h / 2), (x, y - h / 2), (x + 26, y), (x, y + h / 2), (x - w, y + h / 2)])
        d.polygon(pts, outline=0, fill=255, width=2)
        label(x + w / 2 if left else x - w / 2, y, text, f_line)

    # drain vessel V-211 with level instruments and a vent valve
    d.rounded_rectangle([200, 520, 420, 1300], radius=60, outline=0, width=lw, fill=255)
    label(310, 1340, "V-211")
    offsheet(66, 300, "PID-R12-001", left=True)
    line([(256, 300), (290, 300), (290, 520)])
    arrow(290, 515, "down")
    vlabel(262, 420, "100-DR-011", f_line)
    line([(380, 520), (380, 360)], width=2)
    gate_valve(380, 420, vertical=True)
    vlabel(425, 420, "XV-219")
    bubble(560, 620, "LT", "211", to=[(420, 620)])
    bubble(560, 780, "LSHH", "211", to=[(420, 780)], panel=True)
    bubble(560, 1180, "LSLL", "211", to=[(420, 1180)], panel=True)
    # suction header and the spare pair P-212A/B
    line([(310, 1300), (310, 1500), (800, 1500)])
    line([(800, 1500), (800, 2120)])
    for y, sfx in ((1500, "A"), (1900, "B")):
        line([(800, y), (1062, y)])
        gate_valve(920, y, f"XV-212{sfx}")
        pump(1100, y, f"P-212{sfx}")
        line([(1138, y), (1400, y)])
        arrow(1360, y)
    bubble(1260, 1370, "PI", "212A", to=[(1260, 1500)])
    bubble(1260, 2030, "PI", "212B", to=[(1260, 1900)])
    line([(1400, 1500), (1400, 1900)])
    bubble(1300, 1700, "PT", "221", to=[(1400, 1700)], panel=True)
    gate_valve(800, 2050, vertical=True)
    vlabel(760, 2050, "HV-217")
    label(870, 2160, "25-DR-012", f_line)
    # duplex filter F-215A/B with a differential pressure transmitter
    line([(1400, 1700), (1500, 1700), (1500, 1600)])
    line([(1500, 1700), (1500, 1800)])
    for y, sfx in ((1600, "A"), (1800, "B")):
        line([(1500, y), (1560, y)])
        d.rectangle([1560, y - 40, 1680, y + 40], outline=0, width=lw, fill=255)
        d.line([(1560, y - 40), (1680, y + 40)], fill=0, width=2)
        label(1620, y + 62 if sfx == "B" else y - 62, f"F-215{sfx}")
        line([(1680, y), (1760, y)])
    line([(1760, 1600), (1760, 1800)])
    bubble(1620, 1440, "PDIT", "215", to=[(1530, 1600), (1730, 1600)])
    # cooler E-213 on cooling water with temperature control
    line([(1760, 1700), (2080, 1700)])
    label(1900, 1675, "80-PL-023", f_line)
    d.ellipse([2080, 1630, 2220, 1770], outline=0, width=lw, fill=255)
    line([(2100, 1700), (2125, 1670), (2150, 1730), (2175, 1670), (2200, 1700)], width=2)
    label(2150, 1810, "E-213")
    bubble(2010, 1540, "TT", "213", to=[(2010, 1700)])
    line([(2150, 1260), (2150, 1630)])
    arrow(2150, 1625, "down")
    vlabel(2115, 1330, "50-CW-021", f_line)
    control_valve(2150, 1450, "TV-214", at=(2300, 1440), vertical=True)
    line([(2150, 1770), (2150, 2000), (2300, 2000)])
    line([(2420, 2000), (2490, 2000)])
    offsheet(2490, 2000, "PID-R12-003")
    label(2220, 1975, "50-CW-022", f_line)
    line([(2220, 1700), (2610, 1700)])
    bubble(2300, 1560, "TT", "214", to=[(2300, 1700)])
    bubble(2480, 1380, "TIC", "214", to=[(2300, 1530)], panel=True)
    line([(2450, 1380), (2240, 1440)], dash=True)  # the signal line runs through the tag TV-214
    line([(2400, 1700), (2400, 1600)])
    d.polygon([(2384, 1600), (2416, 1600), (2400, 1572)], outline=0, fill=255, width=2)
    d.polygon([(2400, 1572), (2428, 1556), (2428, 1588)], outline=0, fill=255, width=2)
    label(2400, 1530, "PSV-218")
    # flow control and return to sheet 1
    control_valve(2650, 1700, "FV-216", at=(2650, 1600))
    bubble(2530, 1570, "FT", "216", to=[(2530, 1700)])
    bubble(2830, 1480, "FIC", "216", to=[(2560, 1570)], panel=True)
    line([(2800, 1480), (2675, 1640)], dash=True)
    line([(2667, 1700), (3050, 1700)])
    label(2900, 1675, "80-PL-024", f_line)
    offsheet(3240, 1700, "PID-R12-001")
    # seal water tank T-222 and pump P-223
    d.rectangle([1400, 350, 1650, 750], outline=0, width=lw, fill=255)
    label(1525, 330, "T-222")
    bubble(1760, 470, "LG", "222", to=[(1650, 470)])
    line([(1525, 750), (1525, 900), (1862, 900)])
    gate_valve(1680, 900, "XV-223")
    pump(1900, 900, "P-223")
    line([(1938, 900), (2600, 900)])
    arrow(2600, 900)
    label(2400, 875, "25-SW-031", f_line)
    label(2620, 900, "к уплотнениям насосов", f_line, "lm")
    bubble(2100, 760, "PI", "223", to=[(2100, 900)])
    # notes
    label(2550, 200, "Примечания", f_small, "lm")
    label(2550, 235, "1. XV-219 нормально закрыта.", f_line, "lm")
    label(2550, 265, "2. Трубопроводы CW: сталь 20, DN50, PN16.", f_line, "lm")
    label(2550, 295, "3. Продолжение см. PID-R12-001, PID-R12-003.", f_line, "lm")
    # revision table above the title block: three revisions, only the last one is current
    x0, x1 = 2550, 3250
    d.rectangle([x0, 1790, x1, 1950], outline=0, width=2)
    for k, row in enumerate([("Ред.", "Дата", "Изменение")] + C.PID2_REVISIONS):
        y = 1790 + k * 40
        if k:
            d.line([(x0, y), (x1, y)], fill=0, width=1)
        for x, text in zip((x0 + 12, x0 + 100, x0 + 260), row):
            label(x, y + 20, text, f_line, "lm")
    for x in (x0 + 88, x0 + 248):
        d.line([(x, 1790), (x, 1950)], fill=0, width=1)
    # title block
    y0, y1 = 1960, 2250
    d.rectangle([x0, y0, x1, y1], outline=0, width=3)
    for yy in (2030, 2100, 2170):
        d.line([(x0, yy), (x1, yy)], fill=0, width=2)
    d.line([(2850, 2170), (2850, 2250)], fill=0, width=2)
    label(x0 + 12, 1995, "АО «Учебный НПЗ» · учебный чертёж", f_line, "lm")
    label(x0 + 12, 2065, C.PID2_TITLE["title"], f_small, "lm")
    label(x0 + 12, 2135, f"Чертёж {C.PID2_TITLE['drawing']}", f_tag, "lm")
    label(x0 + 12, 2210, f"Ред. {C.PID2_TITLE['revision']}", f_small, "lm")
    label(2865, 2210, C.PID2_TITLE["date"], f_small, "lm")
    img.save(path, optimize=True)


def make_drawing_scan(src: Path, dst: Path, seed: int) -> None:
    """A paper copy of a sheet scanned at a lower resolution: folds, a stamp, rotation, noise, JPEG."""
    rng = np.random.default_rng(seed)
    img = Image.open(src).convert("L")
    img = img.resize((int(img.width * 0.6), int(img.height * 0.6)), Image.LANCZOS)
    stamp(img, (int(img.width * 0.40), int(img.height * 0.86)), ["КОПИЯ", "архив", "ОГМ"], 120)
    arr = np.asarray(img, dtype=np.float32)
    for x in (img.width // 2, img.width // 4):  # folds of the paper copy
        arr[:, x - 2:x + 2] *= 0.62
        arr[:, x + 2:x + 14] *= np.linspace(0.85, 1, 12)[None, :]
    y = img.height // 2
    arr[y - 2:y + 2, :] *= 0.62
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = damage(img, rng, angle=1.3, blur=0.9, noise=14, ink=60, quality=40, shadow=False)
    img.save(dst, "JPEG", quality=60)


def perspective_coeffs(src, dst):
    """Coefficients for Image.transform(PERSPECTIVE) that map dst points back to src points."""
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src).reshape(8)
    return np.linalg.solve(np.array(a, dtype=float), b)


def draw_nameplate(path: Path, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    W, H = 1200, 760
    base = rng.normal(178, 10, (H, W)).astype(np.float32)
    base = np.asarray(Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).filter(
        ImageFilter.BoxBlur(1)).resize((W, H)), dtype=np.float32)
    streaks = np.repeat(rng.normal(0, 7, (1, W)), H, axis=0)  # brushed metal
    plate = Image.fromarray(np.clip(base + streaks, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(plate)
    d.rounded_rectangle([6, 6, W - 6, H - 6], radius=28, outline=(70, 70, 70), width=6)
    for cx, cy in ((40, 40), (W - 40, 40), (40, H - 40), (W - 40, H - 40)):
        d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(120, 120, 120), outline=(60, 60, 60), width=3)
    n = C.NAMEPLATE
    ink = (35, 35, 40)
    rows = [
        (70, "АО «УЧЕБНЫЙ НАСОСНЫЙ ЗАВОД»", 40, True),
        (150, f"НАСОС ЦЕНТРОБЕЖНЫЙ  {n['model']}", 42, True),
        (250, f"Зав. №  {n['serial_number']}        Год  {n['year']}", 38, False),
        (340, f"Q  {n['flow_m3h']} м³/ч            H  {n['head_m']} м", 40, False),
        (430, f"n  {n['speed_rpm']} об/мин       N  {n['power_kw']} кВт", 40, False),
        (520, f"Масса  {n['mass_kg']} кг            IP54", 38, False),
        (640, "Сделано в России · учебный образец", 28, False),
    ]
    for y, text, size, bold in rows:
        d.text((W / 2, y), text, fill=ink, font=pil_font(size, bold), anchor="mm")
    # photo: plate on a dark background, perspective, glare, blur, noise
    BW, BH = 1600, 1100
    photo = Image.new("RGB", (BW, BH), (52, 58, 66))
    dst = [(250, 210), (1380, 150), (1430, 960), (200, 900)]
    coeffs = perspective_coeffs([(0, 0), (W, 0), (W, H), (0, H)], dst)
    warped = plate.transform((BW, BH), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    mask = Image.new("L", (W, H), 255).transform((BW, BH), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    photo.paste(warped, (0, 0), mask)
    arr = np.asarray(photo, dtype=np.float32)
    yy, xx = np.mgrid[0:BH, 0:BW]
    glare = np.exp(-(((xx - 1050) / 260) ** 2 + ((yy - 420) / 170) ** 2)) * 150
    arr = arr + glare[..., None]
    arr += rng.normal(0, 6, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.3))
    out.save(path, "JPEG", quality=72)


# ---------------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------------
def page_texts(pdf: Path) -> list[str]:
    with pymupdf.open(pdf) as doc:
        return [p.get_text("text") for p in doc]


def find_page(texts: list[str], anchor: str) -> int:
    hits = [i for i, t in enumerate(texts, start=1) if anchor in t]
    if not hits:
        raise ValueError(f"anchor not found: {anchor!r}")
    return hits[0]


CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+){0,2}) \S", re.M)


def clause_pages(texts: list[str]) -> dict[str, int]:
    """Clause and section numbers as they start a line on a born-digital page."""
    found: dict[str, int] = {}
    for i, t in enumerate(texts[1:], start=2):  # skip the title page
        for m in CLAUSE_RE.finditer(t):
            found.setdefault(m.group(1), i)
    return found


def build_drawings() -> None:
    """The P&ID samples and their tags: the clean fragment of the archive plus a dense sheet and its scan."""
    DRAWINGS.mkdir(parents=True, exist_ok=True)
    draw_pid(DOCS / "PID-R12-001.png")
    draw_pid_sheet2(DRAWINGS / "PID-R12-002.png")
    make_drawing_scan(DRAWINGS / "PID-R12-002.png", DRAWINGS / "PID-R12-002_scan.jpg", seed=2)
    sheet1 = {"tags": C.PID_TAGS, **C.PID_TITLE}
    sheet2 = {"tags": C.PID2_TAGS, **C.PID2_TITLE}
    samples = {
        "clean": {"file": "docs/PID-R12-001.png", "about": "fragment of sheet 1, large print", **sheet1},
        "dense": {"file": "drawings/PID-R12-002.png", "about": "full sheet 2, small print", **sheet2},
        "dense_scan": {"file": "drawings/PID-R12-002_scan.jpg", "about": "paper copy of sheet 2, scanned",
                       **sheet2},
    }
    (GT / "pid.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    if "--only-drawings" in sys.argv:
        build_drawings()
        return
    register_fonts()
    for d in (DOCS, TWINS):
        d.mkdir(parents=True, exist_ok=True)

    # born-digital originals
    pages = {}
    for rev in (2, 3):
        pages[rev] = build_r12(rev, DOCS / f"R-12_rev{rev}.pdf")
    build_datasheet(C.DATASHEET_P101, TWINS / "PS-101.pdf")
    build_datasheet(C.DATASHEET_P205, TWINS / "DS-P-205.pdf")

    # scans
    r12_texts = page_texts(DOCS / "R-12_rev3.pdf")
    t3_page = find_page(r12_texts, "Таблица 3 —")
    make_scan(DOCS / "R-12_rev3.pdf", DOCS / "R-12_rev3_scan.pdf", seed=12, page_settings={
        1: {"stamps": [(0.70, 0.25, ["КОПИЯ ВЕРНА", "служба", "гл. механика"], 210)], "angle": -0.8},
        t3_page: {"angle": 1.4, "noise": 12},
        5: {"angle": 2.3, "blur": 1.2, "noise": 18, "ink": 85, "quality": 40},  # a copy of a copy
    })
    make_scan(TWINS / "PS-101.pdf", DOCS / "PS-101_scan.pdf", seed=101,
              page_settings={1: {"angle": -1.6, "blur": 0.9, "noise": 12,
                                 "stamps": [(0.70, 0.60, ["ОТК", "принято", "2019"], 170)]}})
    make_scan(TWINS / "DS-P-205.pdf", DOCS / "DS-P-205_scan.pdf", seed=205,
              page_settings={1: {"angle": 0.9, "blur": 0.8, "noise": 10}})
    build_drawings()
    draw_nameplate(DOCS / "nameplate_P-101.jpg")

    # page texts of every born-digital original: the ground truth of every reader
    twins = {
        "R-12_rev2.pdf": ("R-12", 2, DOCS / "R-12_rev2.pdf"),
        "R-12_rev3.pdf": ("R-12", 3, DOCS / "R-12_rev3.pdf"),
        "R-12_rev3_scan.pdf": ("R-12", 3, DOCS / "R-12_rev3.pdf"),
        "PS-101_scan.pdf": ("PS-101", None, TWINS / "PS-101.pdf"),
        "DS-P-205_scan.pdf": ("DS-P-205", None, TWINS / "DS-P-205.pdf"),
    }
    with open(GT / "pages.jsonl", "w", encoding="utf-8") as f:
        for file, (doc_id, rev, twin) in twins.items():
            for i, text in enumerate(page_texts(twin), start=1):
                f.write(json.dumps({"file": file, "doc_id": doc_id, "revision": rev, "page": i, "text": text},
                                   ensure_ascii=False) + "\n")

    # tables with their pages
    tables = {}
    for rev in (2, 3):
        texts = page_texts(DOCS / f"R-12_rev{rev}.pdf")
        for tid in ("table3", "tableA1"):
            spec = C.TABLES[tid]
            tables[f"R-12_rev{rev}/{tid}"] = {
                "doc_id": "R-12", "revision": rev, "table": tid, "caption": spec["caption"],
                "clause": spec["clause"], "page": find_page(texts, spec["caption"][:12]),
                "header_rows": spec["header_rows"], "n_cols": spec["n_cols"],
                "cells": [list(c) for c in spec["cells"](rev)], "rows": [list(r) for r in spec["rows"](rev)],
            }
    (GT / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")

    # clause tree per revision: number, title or text start, page
    clauses = {}
    for rev in (2, 3):
        texts = page_texts(DOCS / f"R-12_rev{rev}.pdf")
        cp = clause_pages(texts)
        numbers = []
        for num, title, items in C.SECTIONS:
            numbers.append(num)
            numbers += [it[1] for it in items if it[0] in ("clause", "subsection")]
        missing = [n for n in numbers if n not in cp]
        if missing:
            raise ValueError(f"clauses not found on pages of rev {rev}: {missing}")
        clauses[f"R-12_rev{rev}"] = [{"clause": n, "page": cp[n]} for n in numbers]
    (GT / "clauses.json").write_text(json.dumps(clauses, ensure_ascii=False, indent=1), encoding="utf-8")

    # questions with the page that holds the answer
    files = {"R-12": "R-12_rev3_scan.pdf", "PS-101": "PS-101_scan.pdf", "DS-P-205": "DS-P-205_scan.pdf"}
    gt_pages = {}
    with open(GT / "pages.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            gt_pages.setdefault(p["file"], []).append(p["text"])
    with open(GT / "questions.jsonl", "w", encoding="utf-8") as f:
        for qid, doc, question, value, unit, where, slice_, anchor in QUESTIONS:
            file = files[doc]
            page = find_page(gt_pages[file], anchor)
            if value is not None:
                assert value in gt_pages[file][page - 1].replace("\n", " "), (qid, value, page)
            f.write(json.dumps({"id": qid, "doc_id": doc, "file": file, "page": page, "question": question,
                                "value": value, "unit": unit, "where": where, "slice": slice_},
                               ensure_ascii=False) + "\n")

    (GT / "nameplate.json").write_text(json.dumps(C.NAMEPLATE, ensure_ascii=False, indent=1), encoding="utf-8")

    # registry of the plant archive
    rows = [
        ("R-12_rev2.pdf", "R-12", "regulation", 2, "superseded", pages[2]),
        ("R-12_rev3.pdf", "R-12", "regulation", 3, "in force", pages[3]),
        ("R-12_rev3_scan.pdf", "R-12", "regulation", 3, "in force", pages[3]),
        ("PS-101_scan.pdf", "PS-101", "datasheet", "", "", 1),
        ("DS-P-205_scan.pdf", "DS-P-205", "datasheet", "", "", 1),
        ("PID-R12-001.png", "PID-R12-001", "drawing", 3, "", 1),
        ("nameplate_P-101.jpg", "NP-P-101", "photo", "", "", 1),
    ]
    with open(DATA / "documents.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "doc_id", "kind", "revision", "status", "pages"])
        w.writerows(rows)
    print(f"R-12 rev 2: {pages[2]} pages, rev 3: {pages[3]} pages; Table 3 on page {t3_page} of rev 3")
    for p in sorted(DOCS.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
