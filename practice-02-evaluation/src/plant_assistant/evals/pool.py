"""The pool of free-text summaries from practice 1 and the human labels for them.

data/summary_pool.jsonl   213 unique (record, defect_summary) pairs from 14 setups of practice 1
data/summary_labels.jsonl PASS / FAIL by a person, with the rule that was broken and a note

The labelling guideline is evals/guideline_summary.md. The judge prompt "binary" uses the same rules.
"""

from __future__ import annotations

from typing import Any

from .. import DATA_DIR, PROJECT_ROOT
from ..data import load_records, load_registry
from ..jsonl import read_json_records

POOL_FILE = DATA_DIR / "summary_pool.jsonl"
LABELS_FILE = DATA_DIR / "summary_labels.jsonl"
GUIDELINE_FILE = PROJECT_ROOT / "evals" / "guideline_summary.md"


def _registry_line(eid: str | None, registry) -> str:
    return f"Registry: {eid} is a {registry[eid].equipment_type.replace('_', ' ')}." if eid in registry else ""


def load_pool() -> list[dict[str, Any]]:
    """Pool items with the ground truth failure_mode and a registry line for the judge."""
    truth = {r.record_id: r.ground_truth or {} for r in load_records()}
    registry = load_registry()
    items = []
    for d in read_json_records(POOL_FILE):
        gt = truth.get(d["record_id"], {})
        items.append({**d, "failure_mode": gt.get("failure_mode"),
                      "context": _registry_line(gt.get("equipment_id"), registry)})
    return items


def items_from_results(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Judge items from a run: one per record with a non-empty defect_summary. item_id = record id."""
    records = {r.record_id: r for r in load_records()}
    registry = load_registry()
    items = []
    for rid, res in results.items():
        summary = ((res.get("record") or {}).get("defect_summary") or "").strip()
        if not summary or rid not in records:
            continue
        gt = records[rid].ground_truth or {}
        items.append({"item_id": rid, "record_id": rid, "text": records[rid].text, "summary": summary,
                      "failure_mode": gt.get("failure_mode"),
                      "context": _registry_line(gt.get("equipment_id"), registry)})
    return items


def load_labels() -> dict[str, dict[str, Any]]:
    return {d["item_id"]: d for d in read_json_records(LABELS_FILE)}
