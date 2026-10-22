"""The harness: run a setup on the test cases, score every case, slice the results.

The unit of evaluation is the case, not the average. Every later step (slices, error analysis,
paired comparison, the gate) works on the per-case table that `score_cases` returns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .. import PROJECT_ROOT, RUNS_DIR
from ..data import load_registry
from ..jsonl import read_json_records
from ..llm import ChatClient
from ..metrics import score_record
from ..runner import run_extraction
from ..schema import SCORED_FIELDS
from .cases import TestCase, fingerprint

PRACTICE1_RUNS = PROJECT_ROOT / "inputs" / "practice-01" / "runs"


def load_results(run: str | Path) -> dict[str, dict[str, Any]]:
    """Results of a run by record id. `run` is a folder, or a setup name in runs/ or in the practice 1 inputs."""
    path = Path(run)
    if not path.exists():
        for base in (RUNS_DIR, PRACTICE1_RUNS):
            if (base / str(run)).exists():
                path = base / str(run)
                break
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results for '{run}' in runs/ or inputs/practice-01/runs/.")
    return {r["record_id"]: r for r in read_json_records(path)}


def run_cases(name: str, client: ChatClient, cases: list[TestCase], mode: str = "structured",
              run_dir: Path | None = None, progress: bool = True) -> dict[str, dict[str, Any]]:
    """Run the extraction pipeline of practice 1 on the cases and return the results by case id."""
    out = run_extraction(name, client, [c.to_record() for c in cases], mode=mode, registry=load_registry(),
                         run_dir=run_dir, progress=progress)
    meta = {"cases_fingerprint": fingerprint(cases), "n_cases": len(cases)}
    (out.run_dir / "cases.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {r.record_id: r.to_dict() for r in out.results}


def score_cases(cases: Iterable[TestCase], results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per case: validity, every field right or wrong, all fields right, tags, the free text."""
    rows = []
    for c in cases:
        res = results.get(c.case_id, {})
        pred = res.get("record")
        row: dict[str, Any] = {
            "case_id": c.case_id,
            "valid": bool(res.get("valid", False)),
            "status": res.get("status", "missing"),
        }
        if c.expected is not None:
            scored = score_record(c.expected, pred)
            for f in SCORED_FIELDS:
                row[f] = scored[f]["correct"]
            row["all_fields_right"] = all(scored[f]["correct"] for f in SCORED_FIELDS)
            row["field_accuracy"] = sum(scored[f]["correct"] for f in SCORED_FIELDS) / len(SCORED_FIELDS)
        row["defect_summary"] = (pred or {}).get("defect_summary")
        row["text"] = c.text
        row.update({f"tag_{k}": v for k, v in c.tags.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(df: pd.DataFrame, name: str) -> dict[str, Any]:
    return {
        "setup": name,
        "n": len(df),
        "valid, %": round(100 * df["valid"].mean(), 1),
        "field accuracy, %": round(100 * df["field_accuracy"].mean(), 1),
        "all fields right, %": round(100 * df["all_fields_right"].mean(), 1),
    }


def slice_table(df: pd.DataFrame, by: str, metric: str = "all_fields_right") -> pd.DataFrame:
    """Metric per slice, with the number of cases: a small slice gives a wide interval."""
    from .stats import wilson

    col = by if by.startswith("tag_") else f"tag_{by}"
    rows = []
    for value, g in df.groupby(col):
        k, n = int(g[metric].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({by: value, "n": n, f"{metric}, %": round(100 * k / n, 1),
                     "95% interval": f"{100 * lo:.0f}–{100 * hi:.0f}"})
    total_k, total_n = int(df[metric].sum()), len(df)
    lo, hi = wilson(total_k, total_n)
    rows.append({by: "ALL", "n": total_n, f"{metric}, %": round(100 * total_k / total_n, 1),
                 "95% interval": f"{100 * lo:.0f}–{100 * hi:.0f}"})
    return pd.DataFrame(rows)
