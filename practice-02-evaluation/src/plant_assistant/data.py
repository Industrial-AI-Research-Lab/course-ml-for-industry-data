"""Loading the synthetic maintenance log, the equipment registry and the splits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import DATA_DIR, FIXTURES_DIR
from .jsonl import read_json_records

LOG_FILE = "maintenance_log.jsonl"
REGISTRY_FILE = "equipment_registry.csv"
SPLITS_FILE = "splits.json"
EDGE_CASES_FILE = "edge_cases.jsonl"

DATA_HELP = (
    "Data not found in {path}. The dataset ships with the repository in data/: check that your clone "
    "is complete (git status, git pull), or regenerate it with scripts/generate_dataset.py."
)


@dataclass
class LogRecord:
    record_id: str
    text: str
    language: str
    ground_truth: dict[str, Any] | None
    meta: dict[str, Any]


@dataclass
class Equipment:
    equipment_id: str
    equipment_type: str
    area: str
    name: str


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(DATA_HELP.format(path=path))
    return path


def load_records(data_dir: Path = DATA_DIR) -> list[LogRecord]:
    path = _require(data_dir / LOG_FILE)
    return [
        LogRecord(
            record_id=d["record_id"],
            text=d["text"],
            language=d.get("language", ""),
            ground_truth=d.get("ground_truth"),
            meta=d.get("meta", {}),
        )
        for d in read_json_records(path)
    ]


def load_split(name: str, data_dir: Path = DATA_DIR) -> list[LogRecord]:
    splits = json.loads(_require(data_dir / SPLITS_FILE).read_text(encoding="utf-8"))
    if name not in splits:
        raise KeyError(f"Unknown split '{name}'. Known splits: {', '.join(splits)}")
    by_id = {r.record_id: r for r in load_records(data_dir)}
    return [by_id[i] for i in splits[name]]


def load_registry(data_dir: Path = DATA_DIR) -> dict[str, Equipment]:
    path = _require(data_dir / REGISTRY_FILE)
    with open(path, encoding="utf-8", newline="") as f:
        return {row["equipment_id"]: Equipment(**row) for row in csv.DictReader(f)}


def load_edge_cases(path: Path = FIXTURES_DIR / EDGE_CASES_FILE) -> list[LogRecord]:
    return [
        LogRecord(
            record_id=d["record_id"],
            text=d["text"],
            language=d.get("language", ""),
            ground_truth=d.get("ground_truth"),
            meta={"case": d.get("case", ""), "expected": d.get("expected", "")},
        )
        for d in read_json_records(_require(path))
    ]
