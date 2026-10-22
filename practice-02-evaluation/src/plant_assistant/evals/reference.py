"""reference/results.md: the numbers a student compares with. Written by scripts/build_reference.py."""

from __future__ import annotations

from pathlib import Path

from .. import PROJECT_ROOT
from .analysis import experiment_compare, experiment_compare_pooled, experiment_table, judge_vs_human, ppi_demo
from .cases import load_cases

JUDGES = ["judge-scale-course", "judge-binary-course", "judge-scale-course-thinking", "judge-binary-course-thinking"]


def best_judge_profile() -> str:
    table = judge_vs_human(["judge-binary-course", "judge-binary-course-thinking"])
    return table.sort_values("kappa", ascending=False).iloc[0]["judge"].replace("judge-binary-", "")


def write_reference(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    evals = PROJECT_ROOT / "evals"
    cases = [c for c in load_cases(evals / "cases_v1.jsonl") + load_cases(evals / "holdout_v1.jsonl")
             if c.source == "log"]
    judges = judge_vs_human(JUDGES)
    best = best_judge_profile()
    exp = experiment_table(cases, best)
    paired = [experiment_compare(cases, best, i) for i in range(3)] + [experiment_compare_pooled(cases, best)]
    ppi = ppi_demo(f"judge-binary-{best}")
    lines = [
        "# Reference results of practice 2",
        "",
        "Measured on the synthetic maintenance log.",
        "",
        "## Judges against the human labels (213 summaries from practice 1)",
        "",
        judges.to_markdown(index=False),
        "",
        f"The experiment below uses the binary judge `{best}`, which has the highest kappa.",
        "",
        "## Prompt v1 vs v2, course model, 3 runs each, eval_50 + holdout_50 (100 entries)",
        "",
        exp.to_markdown(index=False),
        "",
        "## Paired comparison of summary faithfulness, v1 vs v2, run by run",
        "",
        "| A | B | n | A right | B right | B fixed | B broke | p-value | decision |",
        "|---|---|---|---|---|---|---|---|---|",
        *[f"| {p['A']} | {p['B']} | {p['n']} | {p['A right']} | {p['B right']} | {p['B fixed']} | {p['B broke']} "
          f"| {p['p_value']} | {p['decision']} |" for p in paired],
        "",
        "## Prediction-powered inference on the pool (expert labels on 40 random summaries)",
        "",
        ppi.to_markdown(index=False),
        "",
    ]
    path = out_dir / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
