"""The regression gate: is the candidate not worse than the baseline on the same cases?

Topic 6 puts this into CI. The rule is deliberately simple and written down in advance:
- the candidate fails the gate if it is significantly worse on the main metric (paired McNemar test);
- or if it breaks more than `max_broken` cases that the baseline solved.

Usage:
    uv run python -m plant_assistant.evals.gate runs/course-v1-run1 runs/course-v2-run1
    uv run python -m plant_assistant.evals.gate local-structured course-structured --metric all_fields_right
Exit code 0 = pass, 1 = fail: that is all a CI system needs.
"""

from __future__ import annotations

import argparse
import sys

from .cases import cases_from_split
from .harness import load_results, score_cases
from .stats import compare


def gate(baseline: str, candidate: str, metric: str = "all_fields_right", split: str = "eval_50",
         max_broken: int = 2) -> dict:
    cases = cases_from_split(split)
    a = score_cases(cases, load_results(baseline))
    b = score_cases(cases, load_results(candidate))
    result = compare(list(a[metric]), list(b[metric]))
    reasons = []
    if result["decision"] == "B is worse":
        reasons.append("significantly worse than the baseline")
    if result["B broke"] > max_broken:
        reasons.append(f"breaks {result['B broke']} cases that the baseline solved (limit {max_broken})")
    broken = [cid for cid, x, y in zip(a["case_id"], a[metric], b[metric]) if x and not y]
    return {**result, "metric": metric, "passed": not reasons, "reasons": reasons, "broken_cases": broken}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regression gate: candidate vs baseline on the same cases.")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--metric", default="all_fields_right")
    ap.add_argument("--split", default="eval_50")
    ap.add_argument("--max-broken", type=int, default=2)
    args = ap.parse_args(argv)
    r = gate(args.baseline, args.candidate, args.metric, args.split, args.max_broken)
    print(f"{args.candidate} vs {args.baseline} on {args.split}, metric {r['metric']}")
    print(f"  baseline right {r['A right']}/{r['n']}, candidate right {r['B right']}/{r['n']}, "
          f"fixed {r['B fixed']}, broke {r['B broke']}, p = {r['p_value']}  ->  {r['decision']}")
    if r["broken_cases"]:
        print(f"  broken cases: {', '.join(r['broken_cases'])}")
    print("GATE: PASS" if r["passed"] else "GATE: FAIL (" + "; ".join(r["reasons"]) + ")")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
