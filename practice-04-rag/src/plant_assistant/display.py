"""Small tables for the notebook: what search found, what the model answered, what the checks said."""

from __future__ import annotations

import pandas as pd

from .cases import TestCase
from .store import Chunk


def where(c: Chunk) -> str:
    """'R-12 rev. 3 cl. 4.3.3', 'R-12 rev. 3 Amd. 1 cl. 4.4.1', 'VL-2 rev. 4 Table А.2 row V-12', 'PS-101'."""
    rev = f" rev. {c.revision}" if c.revision is not None else ""
    amd = f" Amd. {c.amendment}" if c.amendment else ""
    if c.chunk_type == "table_row":
        place = f" Table {c.table_no} row {c.row_key}"
    elif c.chunk_type == "table":
        place = f" Table {c.table_no}" + (f" (p. {c.page})" if c.split_into_rows else "")
    elif c.clause and c.chunk_type in ("clause", "section"):
        place = f" {'cl.' if c.chunk_type == 'clause' else 'section'} {c.clause}"
    else:
        place = ""
    return f"{c.doc_id}{rev}{amd}{place}"


def hits_frame(rec: dict, case: TestCase, by_id: dict[str, Chunk], k: int = 5) -> pd.DataFrame:
    gold = {cid: e["role"] for e in case.evidence for cid in e["chunk_ids"]}
    rows = []
    for i, cid in enumerate(rec["retrieved"][:k], start=1):
        c = by_id[cid]
        rows.append({"rank": i, "chunk": where(c), "valid": f"{c.effective_from} → {c.effective_to or 'now'}",
                     "gold": gold.get(cid, ""), "text": c.text.replace("\n", " ")[:110]})
    return pd.DataFrame(rows).set_index("rank")


def context_frame(rec: dict, case: TestCase, by_id: dict[str, Chunk]) -> pd.DataFrame:
    gold = {cid: e["role"] for e in case.evidence for cid in e["chunk_ids"]}
    return pd.DataFrame([{"chunk": where(by_id[c]), "gold": gold.get(c, ""), "tokens": len(by_id[c].text) // 3}
                         for c in rec["context"]])


def answer_frame(rec: dict) -> pd.DataFrame:
    a = rec.get("answer") or {}
    checks = (rec.get("checks") or {}).get("citations", [])
    rows = [{"field": k, "value": a.get(k)} for k in ("status", "answer", "value", "unit")]
    df = pd.DataFrame(rows).set_index("field")
    return df, pd.DataFrame(checks)


def gold_frame(case: TestCase) -> pd.DataFrame:
    return pd.DataFrame([{"role": e["role"], "locator (any of)": " | ".join(e["any_of"]),
                          "gold chunk ids": ", ".join(e["chunk_ids"])} for e in case.evidence])


# Readable column labels for the notebook (display only: the CSV files in reference/ keep their headers)
LABELS = {
    "config": "row", "vs": "compared with", "e2e: fixed / broke": "fixed / broke", "e2e verdict": "verdict",
    "abstain ok": "no answer: refused", "conflict ok": "conflicts shown", "right abstentions": "right refusals",
    "invalid": "invalid answers", "leaks": "reasoning in the answer", "fallbacks": "handed to a person",
    "median s / question": "seconds per question", "model calls / question": "model calls",
    "chat calls / question": "chat calls", "prompt tokens / question": "tokens in",
    "completion tokens / question": "tokens out", "cross-encoder tokens / question": "cross-encoder tokens",
    "end-to-end old": "end-to-end before", "end-to-end new": "end-to-end after",
    "new vs old: fixed / broke": "fixed / broke", "s / q": "seconds per question", "tokens out / q": "tokens out",
    # slices
    "base": "base (practice 3)", "language": "English wording", "point_in_time": "past date",
    "what_changed": "what changed", "multi_evidence": "several chunks", "cross_doc": "two documents",
    "table_row": "table row", "tag_variant": "tag spelling", "not_in_documents": "not in documents",
}


def show(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """The columns the text talks about, with readable labels."""
    out = df[cols] if cols else df
    return out.rename(columns=LABELS).fillna("")
