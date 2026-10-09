"""Lecturer script: run every reference setup, save replay files and the reference results.

Outputs:
- data/replay/<setup>.jsonl   call logs; students replay them instead of calling the models
- runs/<setup>/               results.jsonl and summary.json of each setup
- reference/                  results.md, results.csv, per_field.csv, quality_latency_cost.png

Usage (from practice-01-structured-extraction/):
    uv run python scripts/build_reference.py                 # all setups
    uv run python scripts/build_reference.py --only course-structured rules
    uv run python scripts/build_reference.py --skip-paid     # no paid API calls
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plant_assistant import REFERENCE_DIR, RUNS_DIR  # noqa: E402
from plant_assistant.config import get_profile  # noqa: E402
from plant_assistant.data import load_edge_cases, load_registry, load_split  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient  # noqa: E402
from plant_assistant.report import comparison_table, per_field_table, plot_quality_latency_cost  # noqa: E402
from plant_assistant.runner import REPLAY_DIR, run_extraction, run_rules  # noqa: E402


@dataclass
class Setup:
    name: str
    profile: str | None  # None = rules
    mode: str  # baseline / structured / structured-schema-only / rules
    split: str  # eval_50 / dev_20 / edge
    fallback: str | None = None
    in_table: bool = True  # goes into the main comparison table

    @property
    def paid(self) -> bool:
        if self.profile is None:
            return False
        p = get_profile(self.profile)
        return p.price_in_per_m > 0 or p.price_out_per_m > 0


SETUPS = [
    # main comparison on eval_50
    Setup("rules", None, "rules", "eval_50"),
    Setup("local-baseline", "local", "baseline", "eval_50"),
    Setup("local-structured-schema-only", "local", "structured-schema-only", "eval_50"),
    Setup("local-structured", "local", "structured", "eval_50"),
    Setup("local-small-structured", "local-small", "structured", "eval_50"),
    Setup("course-baseline", "course", "baseline", "eval_50"),
    Setup("course-structured-schema-only", "course", "structured-schema-only", "eval_50"),
    Setup("course-structured", "course", "structured", "eval_50"),
    Setup("course-thinking-structured", "course-thinking", "structured", "eval_50"),
    Setup("gpt-oss-120b-structured", "gpt-oss-120b", "structured", "eval_50"),
    Setup("deepseek-v3.2-structured", "deepseek-v3.2", "structured", "eval_50"),
    Setup("gpt-6-luna-structured", "gpt-6-luna", "structured", "eval_50"),
    Setup("gpt-6.1-sol-structured", "gpt-6.1-sol", "structured", "eval_50"),
    Setup("claude-sonnet-5-structured", "claude-sonnet-5", "structured", "eval_50"),
    Setup("local-structured+course-fallback", "local", "structured", "eval_50", fallback="course"),
    # notebook steps on dev_20 (live for the lecturer, replay for students)
    Setup("local-baseline-dev20", "local", "baseline", "dev_20", in_table=False),
    Setup("local-structured-dev20", "local", "structured", "dev_20", in_table=False),
    Setup("course-baseline-dev20", "course", "baseline", "dev_20", in_table=False),
    Setup("course-structured-dev20", "course", "structured", "dev_20", in_table=False),
    # edge cases
    Setup("local-structured-edge", "local", "structured", "edge", in_table=False),
    Setup("course-structured-edge", "course", "structured", "edge", in_table=False),
    Setup("gpt-6.1-sol-structured-edge", "gpt-6.1-sol", "structured", "edge", in_table=False),
]


def records_for(split: str):
    return load_edge_cases() if split == "edge" else load_split(split)


def run_setup(s: Setup, registry) -> dict:
    records = records_for(s.split)
    run_dir = RUNS_DIR / s.name
    if s.mode == "rules":
        return run_rules(s.name, records, registry, run_dir=run_dir).summary
    calls = run_dir / "calls.jsonl"
    if calls.exists():
        calls.unlink()
    logger = CallLogger(calls)
    client = LLMClient(get_profile(s.profile), logger=logger)
    fallback = LLMClient(get_profile(s.fallback), logger=logger) if s.fallback else None
    out = run_extraction(s.name, client, records, mode=s.mode, registry=registry, fallback=fallback, run_dir=run_dir)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(calls, REPLAY_DIR / f"{s.name}.jsonl")
    return out.summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="setup names to run (others are read from runs/)")
    ap.add_argument("--skip-paid", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=1.5, help="stop paid setups after this spend")
    args = ap.parse_args()

    registry = load_registry()
    spent = 0.0
    for s in SETUPS:
        if args.only and s.name not in args.only:
            continue
        if s.paid and (args.skip_paid or spent >= args.budget_usd):
            print(f"skip {s.name} (paid; spent so far ${spent:.3f})")
            continue
        summary = run_setup(s, registry)
        spent += summary["cost_per_1000_usd"] * summary["n"] / 1000
        print(f"{s.name:36s} valid1={summary['valid_first_pct']:5} final={summary['valid_final_pct']:5} "
              f"acc={summary['field_accuracy_pct']:5} lat={summary['latency_median_s']}s spent=${spent:.3f}")

    # collect whatever is in runs/ for the table
    summaries = []
    for s in SETUPS:
        path = RUNS_DIR / s.name / "summary.json"
        if s.in_table and path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    REFERENCE_DIR.mkdir(exist_ok=True)
    table = comparison_table(summaries)
    fields = per_field_table(summaries)
    table.to_csv(REFERENCE_DIR / "results.csv", index=False)
    fields.to_csv(REFERENCE_DIR / "per_field.csv")
    plot_quality_latency_cost([s for s in summaries if s["name"] != "local-structured+course-fallback"],
                              REFERENCE_DIR / "quality_latency_cost.png")
    md = ["# Reference results: practice 1 (eval_50)", "",
          "Measured by the lecturer on the synthetic maintenance log, split `eval_50` (50 records).",
          "Compare your numbers with these. Small differences are normal: models are not fully deterministic.", "",
          table.to_markdown(index=False), "", "## Field accuracy by field, %", "", fields.to_markdown(), ""]
    (REFERENCE_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
