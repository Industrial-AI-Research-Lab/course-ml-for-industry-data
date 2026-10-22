"""Live runs of practice 2 with a cache in runs/<name>/.

A run is made once with the course model. Later calls find runs/<name>/ and reuse it,
so the notebook can be re-run without new model calls.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .. import RUNS_DIR
from ..config import get_profile
from ..llm import CallLogger, LLMClient
from .cases import TestCase
from .harness import load_results, run_cases
from .judge import judge_items
from .pool import items_from_results, load_pool

PROMPTS = {"v1": "structured", "v2": "structured-v2"}


def _client(profile: str, name: str) -> LLMClient:
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    calls = run_dir / "calls.jsonl"
    if calls.exists():
        calls.unlink()
    return LLMClient(get_profile(profile), logger=CallLogger(calls))


def _write_judgements(name: str, results) -> Path:
    path = RUNS_DIR / name / "judgements.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return path


def extraction_runs(cases: list[TestCase], n_runs: int = 3, profile: str = "course") -> list[str]:
    """Prompt v1 and v2, n_runs each, on the cases. Returns the run names."""
    names = []
    for version, mode in PROMPTS.items():
        for k in range(1, n_runs + 1):
            name = f"course-{version}-r{k}"
            names.append(name)
            if not (RUNS_DIR / name / "results.jsonl").exists():
                run_cases(name, _client(profile, name), cases, mode=mode)
    return names


def pool_judges(profiles=("course", "course-thinking"), prompts=("scale", "binary")) -> list[str]:
    """Every judge prompt with every profile on the pool of practice 1 summaries."""
    names = []
    pool = None
    for profile in profiles:
        for prompt in prompts:
            name = f"judge-{prompt}-{profile}"
            names.append(name)
            if not (RUNS_DIR / name / "judgements.jsonl").exists():
                pool = pool or load_pool()
                _write_judgements(name, judge_items(_client(profile, name), pool, prompt=prompt))
    return names


def run_judges(run_names: list[str], profile: str = "course-thinking") -> list[str]:
    """The binary judge on the summaries of each extraction run."""
    names = []
    for run in run_names:
        name = f"judge-binary-{profile}-on-{run}"
        names.append(name)
        if not (RUNS_DIR / name / "judgements.jsonl").exists():
            items = items_from_results(load_results(RUNS_DIR / run))
            _write_judgements(name, judge_items(_client(profile, name), items, prompt="binary"))
    return names
