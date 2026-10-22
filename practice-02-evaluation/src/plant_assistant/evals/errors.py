"""Error analysis, the counting step.

Open coding (reading traces and writing notes) is done by a person. For the typed fields we can
then count the categories automatically: for every field, was the value invented, missed,
or simply wrong? For the free-text field we count the categories of the human labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import pandas as pd

from ..metrics import score_record
from ..schema import SCORED_FIELDS
from .cases import TestCase


def field_error_category(field: str, scored: dict[str, Any]) -> str | None:
    if scored["correct"]:
        return None
    if scored["invented"]:
        return "invented"  # the text does not state it, the model filled it in
    if scored["missed"]:
        return "missed"  # the text states it, the model left it empty
    return "wrong value"


def error_table(cases: Iterable[TestCase], results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per wrong field: case, field, category, expected and predicted value, the text."""
    rows = []
    for c in cases:
        if c.expected is None:
            continue
        res = results.get(c.case_id, {})
        pred = res.get("record")
        if pred is None:
            rows.append({"case_id": c.case_id, "field": "<record>", "category": res.get("status", "missing"),
                         "expected": "", "predicted": "", "text": c.text})
            continue
        for f, s in score_record(c.expected, pred).items():
            cat = field_error_category(f, s)
            if cat:
                rows.append({"case_id": c.case_id, "field": f, "category": cat,
                             "expected": c.expected.get(f), "predicted": pred.get(f), "text": c.text})
    return pd.DataFrame(rows, columns=["case_id", "field", "category", "expected", "predicted", "text"])


def error_counts(errors: pd.DataFrame) -> pd.DataFrame:
    """Field × category counts: the table we prioritise from."""
    if errors.empty:
        return pd.DataFrame()
    table = errors.pivot_table(index="field", columns="category", values="case_id", aggfunc="count", fill_value=0)
    table["total"] = table.sum(axis=1)
    order = [f for f in SCORED_FIELDS if f in table.index] + [f for f in table.index if f not in SCORED_FIELDS]
    return table.loc[order].sort_values("total", ascending=False)


def label_counts(labels: Iterable[dict[str, Any]]) -> Counter:
    """Categories of the human FAIL labels on the free-text field."""
    return Counter(d.get("rule") or "?" for d in labels if d["label"] == "FAIL")
