"""Test cases: the golden set of the harness.

A test case is one input with the expected behaviour and tags. Tags let us slice the results
(language, wording, typos, work type) and find a failing group that the average hides.

Three sources, as in the lecture:
- the labelled log (eval_50, holdout_50): realistic text with checked ground truth;
- hand-written edge cases (fixtures/edge_cases.jsonl): what the system must not do;
- call logs of a previous run: failures become draft cases that an expert confirms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..data import LogRecord, load_edge_cases, load_split
from ..jsonl import read_json_records


@dataclass
class TestCase:
    case_id: str
    text: str
    expected: dict[str, Any] | None  # None: no single right answer, the case is graded by its note
    tags: dict[str, Any] = field(default_factory=dict)
    source: str = "log"  # log / edge / from_calls
    note: str = ""  # what a good system should do, in words
    status: str = "confirmed"  # confirmed / draft (waits for an expert)

    __test__ = False  # not a pytest test class

    def to_record(self) -> LogRecord:
        """The extraction pipeline works on LogRecord: convert without losing the id."""
        return LogRecord(self.case_id, self.text, self.tags.get("language", ""), self.expected, {"tags": self.tags})


def _tags_from_log(rec: LogRecord) -> dict[str, Any]:
    truth = rec.ground_truth or {}
    return {
        "language": rec.language,
        "wording": rec.meta.get("wording", ""),  # hint: the text names the category; free: it does not
        "typos": "yes" if rec.meta.get("typos", 0) else "no",
        "work_type": truth.get("work_type", ""),
        "equipment_type": truth.get("equipment_type", ""),
    }


def cases_from_split(split: str) -> list[TestCase]:
    return [
        TestCase(r.record_id, r.text, r.ground_truth, {**_tags_from_log(r), "split": split}, "log")
        for r in load_split(split)
    ]


def cases_from_edge() -> list[TestCase]:
    out = []
    for r in load_edge_cases():
        tags = {"language": r.language, "edge": r.meta["case"], "split": "edge"}
        out.append(TestCase(r.record_id, r.text, r.ground_truth, tags, "edge", note=r.meta["expected"]))
    return out


def cases_from_calls(results_path: str | Path, records: dict[str, LogRecord],
                     only_failures: bool = True) -> list[TestCase]:
    """Draft test cases from a run: every record where the system was wrong.

    `expected` is the ground truth if we have it; otherwise the model output, which an expert must check.
    Drafts get status "draft": they go into the golden set only after review.
    Records that are not in `records` are skipped: pass only the records you may use (never the holdout).
    """
    from ..metrics import score_record

    drafts = []
    for res in read_json_records(results_path):
        rec = records.get(res["record_id"])
        if rec is None:
            continue
        truth = rec.ground_truth
        pred = res.get("record")
        wrong = [f for f, s in score_record(truth, pred).items() if not s["correct"]] if truth else []
        if only_failures and truth and not wrong and res.get("valid", True):
            continue
        note = f"wrong fields in {res.get('profile', '?')}: {', '.join(wrong)}" if wrong else res.get("status", "")
        drafts.append(TestCase(
            case_id=f"{res['record_id']}-from-calls",
            text=rec.text,
            expected=truth if truth else pred,
            tags={**_tags_from_log(rec), "found_by": res.get("profile", "")},
            source="from_calls",
            note=note,
            status="draft",
        ))
    return drafts


def save_cases(cases: Iterable[TestCase], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return path


def load_cases(path: str | Path, status: str | None = "confirmed") -> list[TestCase]:
    cases = [TestCase(**d) for d in read_json_records(path)]
    return [c for c in cases if status is None or c.status == status]


def fingerprint(cases: Iterable[TestCase]) -> str:
    """Short hash of the case set. Store it next to every result: which version of the set was used?"""
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.case_id):
        h.update(json.dumps(asdict(c), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:10]
