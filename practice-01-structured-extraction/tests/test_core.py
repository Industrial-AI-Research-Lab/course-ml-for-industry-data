"""Fast tests without network: schema, business rules, rules baseline, metrics, replay."""

from __future__ import annotations

import json

import pytest

from plant_assistant.data import Equipment
from plant_assistant.extraction import (
    INVALID,
    NEEDS_REVIEW,
    VALID_AFTER_RETRY,
    VALID_FIRST,
    extract_baseline,
    extract_structured,
)
from plant_assistant.llm import CallLogger, ChatResponse, ReplayClient, ReplayMiss
from plant_assistant.metrics import field_correct, summarize
from plant_assistant.rules_baseline import extract_with_rules, find_date, find_duration_hours, find_equipment_id
from plant_assistant.schema import MaintenanceRecord, strict_json_schema
from plant_assistant.validation import check_business_rules

REGISTRY = {
    "P-101": Equipment("P-101", "pump", "Pump station 1", "pump P-101"),
    "P-101A": Equipment("P-101A", "pump", "Pump station 1", "pump P-101A"),
    "M-201": Equipment("M-201", "electric_motor", "Pump station 2", "motor M-201"),
}
TEXT = "Р-101 замена подшипника, гудел, 12.02.2026, 4 ч, простой"
GOOD = {
    "equipment_id": "P-101", "equipment_type": "pump", "work_type": "corrective_repair",
    "failure_mode": "bearing_wear", "defect_summary": "bearing noise", "replaced_parts": ["bearing"],
    "duration_hours": 4, "work_date": "2026-02-12", "downtime": True, "evidence": ["замена подшипника"],
}


# ---------------------------------------------------------------- schema


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_strict_schema_has_no_refs_and_requires_everything():
    schema = strict_json_schema()
    assert "$defs" not in schema
    for node in _walk(schema):
        assert "$ref" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


def test_schema_rejects_non_iso_date_and_unknown_enum():
    with pytest.raises(Exception):
        MaintenanceRecord.model_validate({**GOOD, "work_date": "12.02.2026"})
    with pytest.raises(Exception):
        MaintenanceRecord.model_validate({**GOOD, "failure_mode": "broken"})


# ---------------------------------------------------------------- business rules


def test_business_rules_pass_for_good_record():
    rec = MaintenanceRecord.model_validate(GOOD)
    assert check_business_rules(rec, TEXT, REGISTRY) == []


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"equipment_id": "Р-101"}, "wrong format"),  # Cyrillic Р
        ({"equipment_id": "P-199"}, "not in the equipment registry"),
        ({"equipment_type": "fan"}, "does not match the registry"),
        ({"work_date": "2027-12-12"}, "in the future"),
        ({"duration_hours": 0}, "outside the range"),
        ({"evidence": ["замена муфты"]}, "not an exact quote"),
    ],
)
def test_business_rules_catch_problems(change, expected):
    rec = MaintenanceRecord.model_validate({**GOOD, **change})
    problems = check_business_rules(rec, TEXT, REGISTRY)
    assert any(expected in p for p in problems), problems


# ---------------------------------------------------------------- rules baseline


def test_rules_find_tag_with_cyrillic_twins_and_without_hyphen():
    assert find_equipment_id("Р-101 замена", REGISTRY) == "P-101"
    assert find_equipment_id("нас. P101A течь", REGISTRY) == "P-101A"
    assert find_equipment_id("P-199 замена", REGISTRY) is None


@pytest.mark.parametrize(
    "text, hours",
    [("работы 3 ч", 3.0), ("2,5ч", 2.5), ("45 мин", 0.75), ("1ч30м", 1.5), ("6 hrs", 6.0), ("90min", 1.5)],
)
def test_rules_duration(text, hours):
    assert find_duration_hours(text) == pytest.approx(hours, abs=0.01)


def test_rules_date_formats():
    assert find_date("12.02.2026") == "2026-02-12"
    assert find_date("12.02.26") == "2026-02-12"
    assert find_date("2026-02-12") == "2026-02-12"
    assert find_date("без даты") is None


def test_rules_full_record():
    pred = extract_with_rules(TEXT, REGISTRY)
    assert pred["equipment_id"] == "P-101" and pred["equipment_type"] == "pump"
    assert pred["replaced_parts"] == ["bearing"]
    assert pred["downtime"] is True


# ---------------------------------------------------------------- metrics


def test_field_correct():
    assert field_correct("duration_hours", 0.6667, 0.67)
    assert not field_correct("duration_hours", 0.67, 40)
    assert field_correct("replaced_parts", ["oil", "bearing"], ["bearing", "oil"])
    assert field_correct("work_date", None, None)
    assert not field_correct("work_date", None, "2026-01-01")


def test_summarize_counts_invented_values():
    truth = {**{k: GOOD[k] for k in GOOD if k not in ("defect_summary", "evidence")}, "work_date": None}
    rows = [{"truth": truth, "pred": {**truth, "work_date": "2026-01-01"}, "valid": True, "first_valid": True,
             "latency_s": 1.0, "cost_usd": 0.001, "tokens_in": 100, "tokens_out": 50}]
    s = summarize(rows, "t")
    assert s["per_field_pct"]["work_date"] == 0.0
    assert s["invented_pct"] > 0
    assert s["cost_per_1000_usd"] == pytest.approx(1.0)


# ---------------------------------------------------------------- extraction with a fake client


class FakeClient:
    """Returns prepared answers in order and remembers the requests."""

    def __init__(self, answers, logger=None):
        self.answers = list(answers)
        self.calls = []
        self.profile_name = "fake"
        self.model = "fake-model"
        self.logger = logger

    def chat(self, messages, response_format=None, meta=None):
        from plant_assistant.llm import request_hash

        self.calls.append(messages)
        content = self.answers.pop(0)
        r = ChatResponse(content=content, finish_reason="stop", model=self.model, profile=self.profile_name,
                         prompt_tokens=10, completion_tokens=5, latency_s=0.1,
                         request_hash=request_hash(self.model, messages, response_format))
        if self.logger:
            self.logger.log(r, messages, response_format, {}, meta)
        return r


def test_baseline_markdown_fence_is_invalid():
    client = FakeClient(["```json\n" + json.dumps(GOOD) + "\n```"])
    res = extract_baseline(client, "r1", TEXT)
    assert res.status == INVALID and res.first_error_type == "markdown_fence"


def test_structured_retry_fixes_business_rule():
    bad = {**GOOD, "equipment_id": "Р-101"}
    client = FakeClient([json.dumps(bad, ensure_ascii=False), json.dumps(GOOD, ensure_ascii=False)])
    res = extract_structured(client, "r1", TEXT, REGISTRY)
    assert res.status == VALID_AFTER_RETRY
    assert len(res.attempts) == 2
    assert "wrong format" in client.calls[1][-1]["content"]  # the error text went into the retry prompt


def test_structured_goes_to_review_then_fallback():
    bad = json.dumps({**GOOD, "equipment_id": "P-199"})
    res = extract_structured(FakeClient([bad] * 3), "r1", TEXT, REGISTRY, max_attempts=3)
    assert res.status == NEEDS_REVIEW and res.record is None
    fallback = FakeClient([json.dumps(GOOD, ensure_ascii=False)])
    res2 = extract_structured(FakeClient([bad] * 3), "r1", TEXT, REGISTRY, max_attempts=3, fallback=fallback)
    assert res2.status == "fallback_valid"


# ---------------------------------------------------------------- replay


def test_replay_reproduces_a_logged_run(tmp_path):
    log = CallLogger(tmp_path / "calls.jsonl")
    live = FakeClient([json.dumps(GOOD, ensure_ascii=False)], logger=log)
    first = extract_structured(live, "r1", TEXT, REGISTRY)
    replay = ReplayClient(tmp_path / "calls.jsonl")
    second = extract_structured(replay, "r1", TEXT, REGISTRY)
    assert first.status == second.status == VALID_FIRST
    assert first.record == second.record
    with pytest.raises(ReplayMiss):
        extract_structured(ReplayClient(tmp_path / "calls.jsonl"), "r1", TEXT + " changed", REGISTRY)


def test_local_urls_bypass_proxy():
    from plant_assistant.llm import is_local_url

    assert is_local_url("http://localhost:11434/v1")
    assert is_local_url("http://127.0.0.1:11434/v1")
    assert not is_local_url("https://api.example.com/v1")


def test_reads_jsonl_and_pretty_printed_json(tmp_path):
    from plant_assistant.jsonl import read_json_records

    records = [{"record_id": "A", "text": "x"}, {"record_id": "B", "text": "y\nz"}]
    compact = tmp_path / "compact.jsonl"
    compact.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    pretty = tmp_path / "pretty.jsonl"  # what an editor's "reformat" makes of a JSONL file
    pretty.write_text("\n".join(json.dumps(r, indent=2) for r in records), encoding="utf-8")
    assert list(read_json_records(compact)) == records
    assert list(read_json_records(pretty)) == records
