"""Metrics: field accuracy against the ground truth, validity, invented values, latency, cost."""

from __future__ import annotations

import statistics
from typing import Any, Iterable

from .schema import SCORED_FIELDS

DURATION_TOLERANCE_H = 0.05  # 3 minutes


def field_correct(field: str, truth: Any, pred: Any) -> bool:
    if field == "replaced_parts":
        return sorted(truth or []) == sorted(pred or [])
    if truth is None or pred is None:
        return truth is None and pred is None
    if field == "duration_hours":
        return abs(float(truth) - float(pred)) <= DURATION_TOLERANCE_H
    return truth == pred


def score_record(truth: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, Any]:
    """Per-field correctness for one record. pred=None (no valid answer) is wrong on every field."""
    out: dict[str, Any] = {}
    for f in SCORED_FIELDS:
        t = truth.get(f)
        p = pred.get(f) if pred else None
        correct = pred is not None and field_correct(f, t, p)
        empty_truth = t is None or (f == "replaced_parts" and not t)
        empty_pred = p is None or (f == "replaced_parts" and not p)
        out[f] = {
            "correct": correct,
            # the text does not state it, but the model filled it in
            "invented": pred is not None and empty_truth and not empty_pred,
            # the text states it, but the model left it empty
            "missed": pred is not None and not empty_truth and empty_pred,
            "truth_empty": empty_truth,
        }
    return out


def _pct(x: float) -> float:
    return round(100 * x, 1)


def summarize(
    rows: Iterable[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    """rows: dicts with keys truth, pred, valid, first_valid, latency_s, cost_usd, tokens_in, tokens_out."""
    rows = list(rows)
    n = len(rows)
    # Accuracy only where a ground truth exists (some edge cases have none); validity on all rows.
    scored = [score_record(r["truth"], r["pred"]) for r in rows if r["truth"] is not None]
    n_scored = max(len(scored), 1)
    per_field = {f: _pct(sum(s[f]["correct"] for s in scored) / n_scored) for f in SCORED_FIELDS}
    truth_empty = sum(s[f]["truth_empty"] for s in scored for f in SCORED_FIELDS)
    truth_filled = len(scored) * len(SCORED_FIELDS) - truth_empty
    invented = sum(s[f]["invented"] for s in scored for f in SCORED_FIELDS)
    missed = sum(s[f]["missed"] for s in scored for f in SCORED_FIELDS)
    latencies = [r["latency_s"] for r in rows]
    cost = sum(r.get("cost_usd", 0.0) for r in rows)
    return {
        "name": name,
        "n": n,
        "valid_first_pct": _pct(sum(r["first_valid"] for r in rows) / n),
        "valid_final_pct": _pct(sum(r["valid"] for r in rows) / n),
        "field_accuracy_pct": round(statistics.mean(per_field.values()), 1),
        "record_exact_pct": _pct(sum(all(s[f]["correct"] for f in SCORED_FIELDS) for s in scored) / n_scored),
        "invented_pct": _pct(invented / truth_empty) if truth_empty else 0.0,
        "missed_pct": _pct(missed / truth_filled) if truth_filled else 0.0,
        "latency_median_s": round(statistics.median(latencies), 2),
        "latency_p90_s": round(sorted(latencies)[int(0.9 * (n - 1))], 2),
        "tokens_in_avg": round(statistics.mean(r.get("tokens_in", 0) for r in rows)),
        "tokens_out_avg": round(statistics.mean(r.get("tokens_out", 0) for r in rows)),
        "cost_per_1000_usd": round(1000 * cost / n, 3),
        "per_field_pct": per_field,
    }


def coverage(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Share of records where a field was filled at all (for the rules baseline)."""
    rows = list(rows)
    out = {}
    for f in SCORED_FIELDS:
        filled = sum(1 for r in rows if r["pred"] and r["pred"].get(f) not in (None, []))
        out[f] = _pct(filled / len(rows))
    return out
