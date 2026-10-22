"""Run every live step of practice 2 and write reference/results.md.

Runs that already exist in runs/ are reused. Delete a folder in runs/ to repeat that run.

Usage (from practice-02-evaluation/)
    uv run python scripts/build_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_assistant import REFERENCE_DIR  # noqa: E402
from plant_assistant.evals.cases import load_cases  # noqa: E402
from plant_assistant.evals.reference import best_judge_profile, write_reference  # noqa: E402
from plant_assistant.evals.runs import extraction_runs, pool_judges, run_judges  # noqa: E402


def main() -> None:
    evals = ROOT / "evals"
    cases = [c for c in load_cases(evals / "cases_v1.jsonl") if c.source == "log"] + load_cases(evals / "holdout_v1.jsonl")
    runs = extraction_runs(cases)
    pool_judges()
    run_judges(runs, profile=best_judge_profile())
    print(f"reference written: {write_reference(REFERENCE_DIR)}")


if __name__ == "__main__":
    main()
