"""Offline tests of the evaluation harness: no network, no model."""

from __future__ import annotations

import json
import math

import pytest

from plant_assistant import prompts
from plant_assistant.evals import stats
from plant_assistant.evals.cases import TestCase, cases_from_edge, cases_from_split, fingerprint, load_cases, save_cases
from plant_assistant.evals.checks import check_summary
from plant_assistant.evals.errors import error_counts, error_table
from plant_assistant.evals.harness import load_results, score_cases, slice_table
from plant_assistant.evals.judge import BinaryVerdict, judge_messages, judge_one
from plant_assistant.evals.pool import load_labels, load_pool
from plant_assistant.llm import ChatResponse


# ---------------------------------------------------------------- statistics
def test_wilson_matches_lecture_numbers():
    lo, hi = stats.wilson(44, 50)  # 88% on 50 records
    assert round(100 * lo) == 76 and round(100 * hi) == 94
    lo, hi = stats.wilson(46, 50)  # 92%
    assert round(100 * lo) == 81 and round(100 * hi) == 97


def test_sample_sizes_match_lecture_chart():
    assert 890 <= stats.n_independent(0.80, 0.85) <= 910
    assert 300 <= stats.n_paired(0.05, 0.10) <= 320


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 5) == pytest.approx(0.7266, abs=1e-3)
    assert stats.mcnemar_exact(0, 10) < 0.01


def test_compare_decision():
    a = [True] * 41 + [True] * 3 + [False] * 5 + [False]
    b = [True] * 41 + [False] * 3 + [True] * 5 + [False]
    r = stats.compare(a, b)
    assert (r["B fixed"], r["B broke"], r["decision"]) == (5, 3, "no detectable difference")


def test_kappa_of_always_pass_judge_is_zero():
    truth = ["PASS"] * 90 + ["FAIL"] * 10
    assert stats.cohen_kappa(truth, ["PASS"] * 100) == pytest.approx(0.0)
    judge_b = ["PASS"] * 85 + ["FAIL"] * 5 + ["FAIL"] * 7 + ["PASS"] * 3
    assert stats.cohen_kappa(truth, judge_b) == pytest.approx(0.59, abs=0.01)


def test_ppi_recovers_bias():
    # judge says PASS for everything, the expert finds 20% FAIL on the labelled part
    est = stats.ppi_mean([1.0] * 200, [1.0] * 40, [1.0] * 32 + [0.0] * 8)
    assert est["estimate"] == pytest.approx(0.8)
    assert est["low"] < 0.8 < est["high"]


# ---------------------------------------------------------------- cases and harness
def test_cases_from_split_have_tags():
    cases = cases_from_split("eval_50")
    assert len(cases) == 50
    assert {"language", "wording", "typos", "work_type", "split"} <= set(cases[0].tags)


def test_holdout_does_not_overlap():
    hold = {c.case_id for c in cases_from_split("holdout_50")}
    used = {c.case_id for c in cases_from_split("eval_50")} | {c.case_id for c in cases_from_split("dev_20")}
    assert len(hold) == 50 and not hold & used


def test_save_load_fingerprint(tmp_path):
    cases = cases_from_split("eval_50")[:5] + cases_from_edge()[:2]
    path = save_cases(cases, tmp_path / "c.jsonl")
    loaded = load_cases(path, status=None)
    assert fingerprint(loaded) == fingerprint(cases)


def test_golden_set_files_are_built():
    from plant_assistant import PROJECT_ROOT

    golden = load_cases(PROJECT_ROOT / "evals" / "cases_v1.jsonl")
    assert len(golden) >= 50 and all(c.expected for c in golden)


def test_score_cases_on_practice1_run():
    cases = cases_from_split("eval_50")
    df = score_cases(cases, load_results("course-structured"))
    assert len(df) == 50
    assert df["field_accuracy"].mean() > 0.95  # practice 1 reference: 99.8%
    table = slice_table(df, "language")
    assert table.iloc[-1]["language"] == "ALL" and table.iloc[-1]["n"] == 50


def test_error_table_categories():
    cases = cases_from_split("eval_50")
    errors = error_table(cases, load_results("local-structured"))
    assert set(errors["category"]) <= {"invented", "missed", "wrong value", "needs_review", "invalid", "missing"}
    assert not error_counts(errors).empty


# ---------------------------------------------------------------- free text
def test_code_checks():
    assert check_summary(None, "none") == []
    assert check_summary("Seal leak.", "seal_leak") == []
    assert "not in English" in check_summary("износ подшипника", "bearing_wear")
    assert "placeholder text instead of null" in check_summary("null", "none")
    assert any("no defect" in p for p in check_summary("No defect found.", "none"))


def test_pool_and_labels_match():
    pool, labels = load_pool(), load_labels()
    assert len(pool) == 213 and set(labels) == {d["item_id"] for d in pool}
    fails = sum(d["label"] == "FAIL" for d in labels.values())
    assert 20 <= fails <= 45
    assert all(d["rule"] in {"cause", "added_fact", "contradiction", "term"} for d in labels.values() if d["label"] == "FAIL")


class _FakeJudge:
    profile_name = "fake"
    model = "fake"

    def __init__(self, content: str):
        self.content = content
        self.seen = None

    def chat(self, messages, response_format=None, meta=None):
        self.seen = (messages, response_format)
        return ChatResponse(content=self.content, finish_reason="stop", model="fake", profile="fake")


def test_judge_binary_parses_structured_answer():
    item = {"item_id": "x", "text": "P-101 течь по прокладке", "summary": "Seal leak causing stop", "context": ""}
    fake = _FakeJudge(json.dumps({"reason": "The stop is not stated.", "verdict": "FAIL"}))
    res = judge_one(fake, "binary", item)
    assert res.verdict == "FAIL" and res.error is None
    messages, rf = fake.seen
    assert "течь по прокладке" in messages[1]["content"]  # the source is in the prompt
    assert rf["json_schema"]["schema"]["properties"]["verdict"]["enum"] == ["PASS", "FAIL"]


def test_judge_scale_threshold_and_bad_answer():
    item = {"item_id": "x", "text": "t", "summary": "s"}
    assert judge_one(_FakeJudge('{"score": 8}'), "scale", item).verdict == "PASS"
    assert judge_one(_FakeJudge('{"score": 3}'), "scale", item).verdict == "FAIL"
    assert judge_one(_FakeJudge("not json"), "scale", item).verdict is None


def test_scale_prompt_has_no_source():
    messages = judge_messages("scale", "SOURCE TEXT", "a summary")
    assert "SOURCE TEXT" not in json.dumps(messages)


# ---------------------------------------------------------------- prompt versions
def test_prompt_v1_is_unchanged_and_v2_adds_rule():
    v1 = prompts.extraction_messages("x")
    v2 = prompts.extraction_messages("x", summary_rule=True)
    assert prompts.SUMMARY_RULE not in v1[0]["content"]
    assert v2[0]["content"].endswith(prompts.SUMMARY_RULE)
    assert v2[1] == v1[1]


def test_binary_verdict_schema():
    assert set(BinaryVerdict.model_fields) == {"reason", "verdict"}
    assert not math.isnan(stats.Z95)
