"""Build the golden set of practice 2: evals/cases_v1.jsonl and evals/holdout_v1.jsonl.

cases_v1   = eval_50 (labelled log) + the edge cases with a ground truth
holdout_v1 = holdout_50 (records that nobody looks at while tuning prompts)

Usage (from practice-02-evaluation/):  uv run python scripts/build_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_assistant.evals.cases import cases_from_edge, cases_from_split, fingerprint, save_cases  # noqa: E402

EVALS = ROOT / "evals"


def main() -> None:
    golden = cases_from_split("eval_50") + [c for c in cases_from_edge() if c.expected is not None]
    holdout = cases_from_split("holdout_50")
    for name, cases in (("cases_v1", golden), ("holdout_v1", holdout)):
        path = save_cases(cases, EVALS / f"{name}.jsonl")
        print(f"{name}: {len(cases)} cases, fingerprint {fingerprint(cases)} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
