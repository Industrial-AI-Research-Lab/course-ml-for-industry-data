"""Tables for the notebook and the reference: reading metrics, tables, structure, QA, images."""

from __future__ import annotations

import pandas as pd

from .docs import SCANNED_FILES, load_gt_pages, load_json
from .metrics import page_scores
from .qa import attribute, paired, wilson
from .readers import load_outputs
from .store import chunk_regulation, structure_score
from .tables import table_score

READER_NAMES = {
    "digital": "Born-digital original (ceiling)",
    "pymupdf": "PyMuPDF, text layer",
    "tesseract": "Tesseract",
    "easyocr": "EasyOCR",
    "docling": "Docling (+EasyOCR)",
    "vlm:dots-ocr": "dots.ocr 1.5, document VLM (replay)",
    "vlm:local": "Qwen3.5 4B, local VLM",
    "vlm:course": "Qwen 3.8 27B, course VLM",
    "vlm:gpt-6.1-sol": "gpt-6.1-sol (replay)",
}
DOC_SHORT = {"R-12_rev3_scan.pdf": "R-12", "PS-101_scan.pdf": "PS-101", "DS-P-205_scan.pdf": "P-205 (zh)"}


def reader_pages(reader: str) -> dict[tuple[str, int], str]:
    """(file, page) -> text a reader produced. "digital" is the text layer of the born-digital original."""
    if reader == "digital":
        gt = load_gt_pages()
        return {k: v for k, v in gt.items() if k[0] in SCANNED_FILES}
    return {k: r["text"] for k, r in load_outputs(reader).items()}


def page_table(readers: list[str]) -> pd.DataFrame:
    """One row per reader and page: CER, numbers, length, repetitions, seconds."""
    gt = load_gt_pages()
    rows = []
    for reader in readers:
        outs = load_outputs(reader)
        for (file, page), rec in sorted(outs.items()):
            if file not in SCANNED_FILES:
                continue
            s = page_scores(gt[(file, page)], rec["text"])
            rows.append({"reader": reader, "doc": DOC_SHORT[file], "file": file, "page": page, **s,
                         "seconds": rec["seconds"], "cost_usd": rec.get("cost_usd") or 0.0})
    return pd.DataFrame(rows)


def reading_summary(pages: pd.DataFrame) -> pd.DataFrame:
    """Per reader: mean CER per document, numbers kept, length ratio, repeated lines, seconds per page."""
    cer = pages.pivot_table(index="reader", columns="doc", values="cer", aggfunc="mean")
    cer.columns = [f"CER {c}" for c in cer.columns]
    agg = pages.groupby("reader").agg(numbers_exact=("numbers_exact", "mean"), numbers_value=("numbers_value", "mean"),
                                      length_ratio=("length_ratio", "mean"), repeated=("repeated", "mean"),
                                      sec_per_page=("seconds", "mean"), usd_per_page=("cost_usd", "mean"))
    out = cer.join(agg)
    out = out.loc[[r for r in pages["reader"].unique()]]  # the order in which readers were given
    out.index = [READER_NAMES.get(r, r) for r in out.index]
    return out


def table_summary(readers: list[str]) -> pd.DataFrame:
    """Rows of Table 3 and Table A.1 that each reader recovered as a row (key + numbers in order)."""
    tables = load_json("tables.json")
    rows = []
    for reader in readers:
        pages = reader_pages(reader)
        for tid in ("table3", "tableA1"):
            t = tables[f"R-12_rev3/{tid}"]
            s = table_score(pages.get(("R-12_rev3_scan.pdf", t["page"]), ""), t["rows"])
            name = "Table 3" if tid == "table3" else "Table A.1"
            rows.append({"reader": READER_NAMES.get(reader, reader), "table": name,
                         "recovered": f"{s['recovered']}/{s['rows']}", "lost": ", ".join(s["lost"]) or "—"})
    df = pd.DataFrame(rows)
    out = df.pivot(index="reader", columns="table", values=["recovered", "lost"])
    out.columns = [f"{t}: {k}" for k, t in out.columns]
    out = out.loc[[READER_NAMES.get(r, r) for r in readers]]
    return out[["Table 3: recovered", "Table 3: lost", "Table A.1: recovered", "Table A.1: lost"]]


def structure_summary(readers: list[str]) -> pd.DataFrame:
    """Clause tree rebuilt from each reader's text of the scanned regulation."""
    gt = [c["clause"] for c in load_json("clauses.json")["R-12_rev3"]]
    n_pages = sum(1 for f, _ in load_gt_pages() if f == "R-12_rev3_scan.pdf")
    rows = []
    for reader in readers:
        pages = reader_pages(reader)
        texts = [pages.get(("R-12_rev3_scan.pdf", i), "") for i in range(1, n_pages + 1)]
        _, chunks = chunk_regulation(texts, "R-12_rev3_scan.pdf", parser=reader)
        s = structure_score(chunks, gt)
        tables = sum(c.chunk_type == "table" for c in chunks)
        rows.append({"reader": READER_NAMES.get(reader, reader), "clauses found": f"{s['found']}/{s['clauses']}",
                     "table chunks": tables, "missed (first 6)": ", ".join(s["missed"][:6]) or "—"})
    return pd.DataFrame(rows).set_index("reader")


def qa_summary(results: list[dict]) -> pd.DataFrame:
    """Accuracy with a 95% interval per reader, counts per slice, and where wrong answers were lost."""
    df = pd.DataFrame(attribute([dict(r) for r in results]))
    rows = []
    for reader, g in df.groupby("reader", sort=False):
        k, n = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, n)
        row = {"reader": READER_NAMES.get(reader, reader), "accuracy": k / n, "95% CI": f"{lo:.0%}–{hi:.0%}"}
        for sl, gs in g.groupby("slice"):
            row[sl] = f"{int(gs['correct'].sum())}/{len(gs)}"
        row["lost in reading"] = int((g["lost_at"] == "reading").sum())
        row["lost in the model"] = int((g["lost_at"] == "model").sum())
        rows.append(row)
    cols = ["reader", "accuracy", "95% CI", "table", "text", "datasheet", "rare_language", "not_stated",
            "lost in reading", "lost in the model"]
    return pd.DataFrame(rows)[cols].set_index("reader")


def qa_pairs(results: list[dict], pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired comparison on the same questions (lecture 2): better / worse / not different."""
    df = pd.DataFrame(results)
    by = {r: g.set_index("id")["correct"] for r, g in df.groupby("reader")}
    rows = []
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        ids = by[a].index.intersection(by[b].index)
        p = paired(list(by[a][ids]), list(by[b][ids]))
        rows.append({"A": READER_NAMES.get(a, a), "B": READER_NAMES.get(b, b), "A − B": f"{p['diff']:+.0%}",
                     "95% CI": f"{p['low']:+.0%} … {p['high']:+.0%}", "only A right": p["a_only"],
                     "only B right": p["b_only"], "verdict": f"A is {p['verdict']}" if p["verdict"] != "not different"
                     else "not different"})
    return pd.DataFrame(rows)
