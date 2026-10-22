"""Collect the free-text field `defect_summary` from the practice 1 runs into one pool.

Input:  inputs/practice-01/runs/<setup>/results.jsonl (eval_50, every model of practice 1)
Output: data/summary_pool.jsonl, one line per unique (record, summary) pair

The judges grade this pool, and data/summary_labels.jsonl holds the human labels. Item ids are stable.

Usage (from practice-02-evaluation/):  uv run python scripts/build_summary_pool.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_assistant.data import load_records  # noqa: E402

RUNS = ROOT / "inputs" / "practice-01" / "runs"
OUT = ROOT / "data" / "summary_pool.jsonl"
SKIP = {"rules"}


def item_id(record_id: str, summary: str) -> str:
    return "S-" + hashlib.sha256(f"{record_id}|{summary}".encode("utf-8")).hexdigest()[:8]


def main() -> None:
    texts = {r.record_id: r for r in load_records()}
    pool: dict[tuple[str, str], dict] = {}
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name not in SKIP):
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
            res = json.loads(line)
            summary = ((res.get("record") or {}).get("defect_summary") or "").strip()
            if not summary:
                continue
            key = (res["record_id"], summary.rstrip(".").lower())
            if key not in pool:
                rec = texts[res["record_id"]]
                pool[key] = {
                    "item_id": item_id(res["record_id"], summary),
                    "record_id": res["record_id"],
                    "text": rec.text,
                    "summary": summary,
                    "setups": [],
                }
            pool[key]["setups"].append(run_dir.name)
    items = sorted(pool.values(), key=lambda d: d["record_id"])  # stable: record, then first appearance
    with open(OUT, "w", encoding="utf-8") as f:
        for d in items:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(items)} unique summaries from {len({d['record_id'] for d in items})} records -> {OUT}")


if __name__ == "__main__":
    main()
