"""Tables of practice 2, shared by the notebook and the reference script.

Every function reads saved runs (runs/<name>/), so the tables are the same in live and replay mode.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .. import RUNS_DIR
from ..jsonl import read_json_records
from .cases import TestCase
from .harness import load_results, score_cases, summary_row
from .pool import load_labels, load_pool
from .stats import agreement_report, compare, mean_ci, ppi_mean, wilson

EXPERIMENT_RUNS = {"v1": ["course-v1-r1", "course-v1-r2", "course-v1-r3"],
                   "v2": ["course-v2-r1", "course-v2-r2", "course-v2-r3"]}


def load_judgements(name: str) -> dict[str, dict[str, Any]]:
    path = RUNS_DIR / name / "judgements.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No judgements in runs/{name}/. Run the judge first (or the reference script).")
    return {d["item_id"]: d for d in read_json_records(path)}


def labelled_pool() -> pd.DataFrame:
    labels = load_labels()
    rows = [{**d, "human": labels[d["item_id"]]["label"], "rule": labels[d["item_id"]]["rule"],
             "note": labels[d["item_id"]]["note"]} for d in load_pool() if d["item_id"] in labels]
    return pd.DataFrame(rows)


def judge_vs_human(judge_names: list[str]) -> pd.DataFrame:
    """Agreement of each judge with the human labels on the whole pool."""
    pool = labelled_pool()
    rows = []
    for name in judge_names:
        verdicts = load_judgements(name)
        usable = pool[pool["item_id"].map(lambda i: (verdicts.get(i) or {}).get("verdict") is not None)]
        report = agreement_report(list(usable["human"]), [verdicts[i]["verdict"] for i in usable["item_id"]])
        rows.append({"judge": name, "n": len(usable), "unusable answers": len(pool) - len(usable), **report})
    return pd.DataFrame(rows)


def faithful_flags(cases: list[TestCase], run: str, judge: str) -> list[bool]:
    """Per case: True if the summary has no unsupported claim (an empty summary claims nothing)."""
    results = load_results(run)
    verdicts = load_judgements(judge)
    out = []
    for c in cases:
        summary = ((results.get(c.case_id, {}).get("record") or {}).get("defect_summary") or "").strip()
        out.append(True if not summary else (verdicts.get(c.case_id) or {}).get("verdict") != "FAIL")
    return out


def experiment_table(cases: list[TestCase], judge_profile: str) -> pd.DataFrame:
    rows = []
    for version, runs in EXPERIMENT_RUNS.items():
        for run in runs:
            df = score_cases(cases, load_results(run))
            faithful = faithful_flags(cases, run, f"judge-binary-{judge_profile}-on-{run}")
            k, n = sum(faithful), len(faithful)
            lo, hi = wilson(k, n)
            rows.append({**summary_row(df, run), "prompt": version,
                         "summary faithful, %": round(100 * k / n, 1),
                         "faithful 95% interval": f"{100 * lo:.0f}–{100 * hi:.0f}"})
    cols = ["prompt", "setup", "n", "valid, %", "field accuracy, %", "all fields right, %",
            "summary faithful, %", "faithful 95% interval"]
    return pd.DataFrame(rows)[cols]


def experiment_compare(cases: list[TestCase], judge_profile: str, run_index: int = 0) -> dict[str, Any]:
    """Paired comparison v1 vs v2 on the same cases, one run of each."""
    a_run, b_run = EXPERIMENT_RUNS["v1"][run_index], EXPERIMENT_RUNS["v2"][run_index]
    a = faithful_flags(cases, a_run, f"judge-binary-{judge_profile}-on-{a_run}")
    b = faithful_flags(cases, b_run, f"judge-binary-{judge_profile}-on-{b_run}")
    return {"A": a_run, "B": b_run, **compare(a, b)}


def experiment_compare_pooled(cases: list[TestCase], judge_profile: str) -> dict[str, Any]:
    """All runs together: a case counts as faithful for a version if it is faithful in most of its runs."""
    def majority(version: str) -> list[bool]:
        runs = [faithful_flags(cases, r, f"judge-binary-{judge_profile}-on-{r}") for r in EXPERIMENT_RUNS[version]]
        return [sum(flags) * 2 > len(flags) for flags in zip(*runs)]
    return {"A": "v1, majority of 3 runs", "B": "v2, majority of 3 runs", **compare(majority("v1"), majority("v2"))}


def ppi_demo(judge: str, n_labelled: int = 40, seed: int = 7) -> pd.DataFrame:
    """Pretend that the expert labelled only n_labelled random summaries. What do we know about the pool?

    The full human labels are the "truth" we compare with: in real life we would not have them.
    Share measured: summaries with no unsupported claim (human PASS).
    """
    pool = labelled_pool()
    verdicts = load_judgements(judge)
    pool = pool[pool["item_id"].map(lambda i: (verdicts.get(i) or {}).get("verdict") is not None)].reset_index(drop=True)
    human = (pool["human"] == "PASS").astype(float).tolist()
    judged = [1.0 if verdicts[i]["verdict"] == "PASS" else 0.0 for i in pool["item_id"]]
    idx = list(range(len(pool)))
    random.Random(seed).shuffle(idx)
    lab, unlab = idx[:n_labelled], idx[n_labelled:]
    truth = sum(human) / len(human)
    h_mean, h_lo, h_hi = mean_ci([human[i] for i in lab])
    judge_only = sum(judged) / len(judged)
    ppi = ppi_mean([judged[i] for i in unlab], [judged[i] for i in lab], [human[i] for i in lab])
    rows = [
        {"estimate": f"expert on {n_labelled} only", "faithful, %": 100 * h_mean, "low": 100 * h_lo, "high": 100 * h_hi},
        {"estimate": f"judge on all {len(pool)}", "faithful, %": 100 * judge_only, "low": None, "high": None},
        {"estimate": f"PPI, judge on {len(unlab)} and expert on {n_labelled}", "faithful, %": 100 * ppi["estimate"],
         "low": 100 * ppi["low"], "high": 100 * ppi["high"]},
        {"estimate": f"truth, expert on all {len(pool)}", "faithful, %": 100 * truth,
         "low": None, "high": None},
    ]
    return pd.DataFrame(rows).round(1)
