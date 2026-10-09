"""Run one extraction setup over a list of records and collect results and metrics.

A "setup" is: which client (live profile or replay file), which mode (baseline, structured,
rules), which records. Every run writes its call log (= a replay file) and its results.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from . import DATA_DIR, RUNS_DIR
from .config import get_profile
from .data import Equipment, LogRecord, load_registry
from .extraction import VALID_FIRST, Attempt, ExtractionResult, extract_baseline, extract_structured
from .llm import CallLogger, ChatClient, LLMClient, ReplayClient
from .metrics import summarize
from .rules_baseline import extract_with_rules

REPLAY_DIR = DATA_DIR / "replay"


@dataclass
class RunOutput:
    name: str
    results: list[ExtractionResult]
    summary: dict[str, Any]
    run_dir: Path


def replay_path(setup_name: str) -> Path:
    return REPLAY_DIR / f"{setup_name}.jsonl"


def open_client(
    profile_or_replay: str,
    logger: CallLogger | None = None,
    prefer_replay: bool = False,
) -> ChatClient:
    """Live client for a profile name, or a replay client for a path / saved setup name."""
    path = Path(profile_or_replay)
    if path.suffix == ".jsonl":
        return ReplayClient(path, logger=logger)
    if prefer_replay:
        return ReplayClient(replay_path(profile_or_replay), logger=logger)
    return LLMClient(get_profile(profile_or_replay), logger=logger)


def result_rows(records: list[LogRecord], results: list[ExtractionResult]) -> list[dict[str, Any]]:
    rows = []
    for rec, res in zip(records, results):
        rows.append({
            "truth": rec.ground_truth,
            "pred": res.record,
            "valid": res.valid,
            "first_valid": res.attempts[0].error_type is None if res.attempts else res.valid,
            "latency_s": res.latency_s,
            "cost_usd": res.cost_usd,
            "tokens_in": res.prompt_tokens,
            "tokens_out": res.completion_tokens,
        })
    return rows


def run_extraction(
    name: str,
    client: ChatClient,
    records: list[LogRecord],
    mode: str = "structured",
    registry: dict[str, Equipment] | None = None,
    max_attempts: int = 3,
    fallback: ChatClient | None = None,
    concurrency: int | None = None,
    run_dir: Path | None = None,
    progress: bool = True,
) -> RunOutput:
    registry = registry if registry is not None else load_registry()
    run_dir = run_dir or (RUNS_DIR / name)
    run_dir.mkdir(parents=True, exist_ok=True)

    def one(rec: LogRecord) -> ExtractionResult:
        if mode == "baseline":
            return extract_baseline(client, rec.record_id, rec.text)
        if mode == "structured":
            return extract_structured(client, rec.record_id, rec.text, registry, max_attempts, fallback)
        if mode == "structured-schema-only":
            return extract_structured(client, rec.record_id, rec.text, registry, max_attempts, fallback,
                                      field_guide=False)
        raise ValueError(f"Unknown mode '{mode}'")

    if concurrency is None:
        concurrency = getattr(getattr(client, "profile", None), "concurrency", 4)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        it = pool.map(one, records)
        results = list(tqdm(it, total=len(records), desc=name, disable=not progress))

    summary = summarize(result_rows(records, results), name)
    summary.update(mode=mode, model=client.model, profile=client.profile_name)
    _write(run_dir, results, summary)
    return RunOutput(name=name, results=results, summary=summary, run_dir=run_dir)


def run_rules(name: str, records: list[LogRecord], registry: dict[str, Equipment] | None = None,
              run_dir: Path | None = None) -> RunOutput:
    registry = registry if registry is not None else load_registry()
    run_dir = run_dir or (RUNS_DIR / name)
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for rec in records:
        start = time.perf_counter()
        pred = extract_with_rules(rec.text, registry)
        elapsed = round(time.perf_counter() - start, 6)
        attempt = Attempt(1, "rules", "", None, [], elapsed, 0, 0, 0, 0.0)
        results.append(ExtractionResult(rec.record_id, "rules", "rules", VALID_FIRST, pred, [attempt]))
    summary = summarize(result_rows(records, results), name)
    summary.update(mode="rules", model="regex + dictionaries + registry", profile="rules")
    _write(run_dir, results, summary)
    return RunOutput(name=name, results=results, summary=summary, run_dir=run_dir)


def _write(run_dir: Path, results: list[ExtractionResult], summary: dict[str, Any]) -> None:
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def new_logger(name: str, run_dir: Path | None = None) -> CallLogger:
    run_dir = run_dir or (RUNS_DIR / name)
    path = run_dir / "calls.jsonl"
    if path.exists():
        path.unlink()
    return CallLogger(path)
