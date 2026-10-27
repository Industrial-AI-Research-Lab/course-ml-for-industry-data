"""Tests of practice 4. No network: the store is built from the PDFs in data/docs/ and the practice 3 store,
model answers come from the replay files."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from plant_assistant import REFERENCE_DIR, REPLAY_DIR
from plant_assistant.answer import Answer, Citation, parse
from plant_assistant.applicability import in_scope, linked_chunks, load_registry, scope_for
from plant_assistant.assemble import evidence_pack, header, render
from plant_assistant.cases import SLICES, load_cases, parse_locator, resolve
from plant_assistant.corpus import build_corpus, find_references, index_version, markdown_rows
from plant_assistant.evals.stats import compare, wilson
from plant_assistant.filters import in_force, make_filter, post_filter, status_in_force
from plant_assistant.normalize import find_tags, normalize_tags
from plant_assistant.pipeline import CONFIGS, System
from plant_assistant.query import extract_rules, find_date
from plant_assistant.reval import attribute, grade, retrieval_metrics, score
from plant_assistant.search import BM25Index, DenseIndex, VectorStore, rrf, tokenize, top_k, Hit
from plant_assistant.store import load_store
from plant_assistant.units import canonical_unit, has_value, same_value
from plant_assistant.verify import check_answer

warnings.filterwarnings("ignore")
TODAY = "2026-10-27"


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    folder = tmp_path_factory.mktemp("store")
    summary = build_corpus(folder)
    _, chunks = load_store(folder)
    return summary, chunks


@pytest.fixture(scope="session")
def chunks(store):
    return store[1]


@pytest.fixture(scope="session")
def by_id(chunks):
    return {c.chunk_id: c for c in chunks}


@pytest.fixture(scope="session")
def cases():
    return load_cases()


@pytest.fixture(scope="session")
def system(chunks):
    from plant_assistant.llm import ReplayClient

    return System(chunks, answer_client=ReplayClient(REPLAY_DIR / "answers_course.jsonl"),
                  rerank_client=ReplayClient(REPLAY_DIR / "rerank_course.jsonl"),
                  translate_client=ReplayClient(REPLAY_DIR / "translate_course.jsonl"))


# ---------------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------------
def test_store_is_reproducible(store):
    summary, chunks = store
    manifest = json.loads((REFERENCE_DIR / "store" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["index_version"] == manifest["index_version"]
    assert summary["documents"] == 16 and summary["table_rows"] == 110


def test_core_numbers_unchanged(by_id):
    t3 = by_id["R-12-r3-table1"]
    assert t3.table_no == "3" and "|II|4,5|7,1|" in t3.text
    assert "|II|5,0|7,1|" in by_id["R-12-r2-table1"].text
    assert "24 часов" in by_id["R-12-r3-4.3.3"].text
    assert by_id["R-12-r3-4.3.3"].effective_from == "2025-03-14"


def test_amendment_replaces_one_clause(by_id):
    base, new = by_id["R-12-r3-4.4.1"], by_id["R-12-r3-a1-4.4.1"]
    assert base.amended_by == new.chunk_id and base.effective_to == "2026-09-01" == new.effective_from
    assert "500 часов" in new.text and new.amendment == 1 and new.parent_id == base.parent_id
    assert by_id["R-12-r3-4.3.3"].effective_to is None  # the rest of rev. 3 stays in force


def test_references_both_ways(by_id):
    assert "R-12-r3-4.3.3" in by_id["R-12-r3-6.2"].references
    assert "R-12-r3-6.2" in by_id["R-12-r3-4.3.3"].referenced_by
    assert "R-12-r3-table1" in by_id["R-12-r3-4.3.3"].references
    assert {"R-12-r3-6.2", "R-14-r1-6.2"} <= set(by_id["R-20-r2-3.3"].references)
    assert "R-20-r2-3.3" in by_id["R-12-r3-6.2"].referenced_by
    assert find_references("см. п. 4.3.3 и п. 4.3.4") == [(None, "clause", "4.3.3"), (None, "clause", "4.3.4")]
    assert find_references("по п. 7.1 R-14 для насосов") == [("R-14", "clause", "7.1")]


def test_long_table_rows(by_id, chunks):
    row = by_id["VL-2-r4-tА.2-V-12"]
    assert row.chunk_type == "table_row" and row.row_key == "V-12" and row.table_no == "А.2"
    assert "PN, МПа: 1,6" in row.text and "Таблица А.2" in row.text
    pages = [c for c in chunks if c.chunk_type == "table" and c.table_no == "А.2"]
    assert len(pages) == 4 and all(c.split_into_rows for c in pages)
    header_, rows = markdown_rows(pages[1].text)
    assert header_[0] == "Тег" and len(rows) == 33


def test_index_version_changes_with_text(chunks):
    v1 = index_version(chunks)
    changed = [*chunks[:-1], type(chunks[-1])(**{**chunks[-1].__dict__, "text": chunks[-1].text + "!"})]
    assert index_version(changed) != v1 and index_version(chunks, "bge-m3").endswith("emb-bge-m3")


# ---------------------------------------------------------------------------------------------
# Tags, fields, filters, applicability
# ---------------------------------------------------------------------------------------------
def test_tag_normalization():
    assert normalize_tags("Р-101, P101, p-101, P 101") == "P-101, P-101, P-101, P-101"
    assert normalize_tags("в 12 месяцев, ТО-1, G2.5, R-12") == "в 12 месяцев, ТО-1, G2.5, R-12"
    assert find_tags("кран V37 и насос Р-310") == ["V-37", "P-310"]


def test_query_fields():
    f = extract_rules("10.02.2025 насос P-101 вибрировал 4,8 мм/с", TODAY)
    assert f.as_of == "2025-02-10" and f.tags == ["P-101"] and f.measured_value == "4,8"
    assert extract_rules("Какая периодичность на 15 августа 2026 года?", TODAY).as_of == "2026-08-15"
    assert extract_rules("Как в ред. 3 R-12 изменилась граница?", TODAY).as_of is None
    assert extract_rules("Насос насосной станции № 3", TODAY).stations == ["3"]
    assert find_date("on 1 May 2024") == "2024-05-01"


def test_in_force_by_dates(by_id):
    r2, r3, amd = by_id["R-12-r2-4.4.1"], by_id["R-12-r3-4.4.1"], by_id["R-12-r3-a1-4.4.1"]
    assert [in_force(c, "2024-05-01") for c in (r2, r3, amd)] == [True, False, False]
    assert [in_force(c, "2026-08-15") for c in (r2, r3, amd)] == [False, True, False]
    assert [in_force(c, TODAY) for c in (r2, r3, amd)] == [False, False, True]
    assert status_in_force(r3)  # the flag says "in force": the typical mistake


def test_applicability_near_duplicates(by_id):
    reg = load_registry()
    assert reg["P-101"].station == "2" and reg["P-310"].station == "3"
    q = "Насос P-310 вибрирует 4,0 мм/с"
    scope = scope_for(extract_rules(q, TODAY), q, reg)
    assert not in_scope(by_id["R-12-r3-table1"], scope) and in_scope(by_id["R-14-r1-table1"], scope)
    assert not in_scope(by_id["OM-P-205-r2-table2"], scope)  # the manual of P-205 is not about P-310


def test_registry_links_the_manual(chunks, by_id):
    reg = load_registry()
    assert reg["P-205"].documents == ["DS-P-205", "OM-P-205"]
    assert not by_id["OM-P-205-r2-table2"].equipment_tags  # the manufacturer does not know the plant tag
    q = "Какая граница вибрации у насоса P-205?"
    linked = linked_chunks(scope_for(extract_rules(q, TODAY), q, reg), chunks)
    assert {"DS-P-205-p1", "OM-P-205-r2-table2"} <= set(linked)
    assert all(by_id[c].doc_id in ("DS-P-205", "OM-P-205") for c in linked)


def test_same_language_conflict_in_force(by_id):
    r12, r15 = by_id["R-12-r3-7.1"], by_id["R-15-r1-4.1"]
    assert "2000" in r12.text and "3000" in r15.text
    assert in_force(r12, TODAY) and in_force(r15, TODAY) and set(r12.scope_stations) & set(r15.scope_stations)


def test_separate_rows_collection(system, cases):
    case = next(c for c in cases if c.case_id == "r01")
    r = system.retrieve(case, CONFIGS["S2b"])
    rows = [c for c in r["top"] if system.by_id[c].chunk_type == "table_row"]
    assert 1 <= len(rows) <= 2 and "VL-2-r4-tА.2-V-12" in rows
    assert len(r["top"]) - len(rows) == 5
    z01 = next(c for c in cases if c.case_id == "z01")
    assert "DS-P-205-p1" in system.retrieve(z01, CONFIGS["S2b"])["top"]
    assert "DS-P-205-p1" not in system.retrieve(z01, CONFIGS["S2"])["top"]  # mixed rows push the datasheet out


def test_post_filter_loses_what_pre_filter_keeps(chunks, by_id):
    idx = BM25Index(chunks, normalize=True)
    q = "Какая граница осмотра действовала для насоса P-101 на 01.12.2024?"
    allowed = make_filter(extract_rules(q, TODAY), None, use_scope=False, use_date=True)
    pre = idx.search(q, 5, allowed)
    post = post_filter(idx.search(q, 5), by_id, allowed)
    assert len(pre) == 5 and len(post) <= len(pre)


# ---------------------------------------------------------------------------------------------
# Search and assembly
# ---------------------------------------------------------------------------------------------
def test_tokens_keep_technical_strings():
    assert tokenize("P-101 4,5 мм/с п. 4.3.3") == ["p-101", "4,5", "мм/с", "п", "4.3.3"]


def test_normalization_finds_cyrillic_tag(chunks):
    q = "Какая подача у насоса Р101?"
    raw = [h.chunk_id for h in BM25Index(chunks).search(q, 5)]
    norm = [h.chunk_id for h in BM25Index(chunks, normalize=True).search(q, 5)]
    assert "PS-101-p1" not in raw and "PS-101-p1" in norm


def test_top_k_respects_filter_and_ties():
    from plant_assistant.store import Chunk

    cs = [Chunk(f"c{i}", "D", 1, "clause", str(i), [], 1, "x") for i in range(4)]
    hits = top_k([c.chunk_id for c in cs], cs, np.array([1.0, 2.0, 2.0, 0.5]), 3, lambda c: c.chunk_id != "c1", "t")
    assert [h.chunk_id for h in hits] == ["c2", "c0", "c3"]


def test_rrf_and_dense_from_saved_vectors(tmp_path, chunks):
    ids = [c.chunk_id for c in chunks]
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(len(ids), 8))
    q = vec[ids.index("PS-101-p1")] + 0.01
    np.savez(tmp_path / "toy.npz", chunk_vectors=vec, query_vectors=q[None, :])
    (tmp_path / "toy.json").write_text(json.dumps({"model": "toy", "corpus": "corpus-x", "chunk_ids": ids,
                                                   "queries": ["question"]}), encoding="utf-8")
    vs = VectorStore(tmp_path / "toy")
    assert DenseIndex(chunks, vs).search("question", 1)[0].chunk_id == "PS-101-p1"
    with pytest.raises(ValueError):
        vs.check("corpus-other.parser-1.emb-none")
    fused = rrf([[Hit("a", 1, 1), Hit("b", 1, 2)], [Hit("b", 1, 1), Hit("c", 1, 2)]], k=3)
    assert [h.chunk_id for h in fused][0] == "b"


def test_evidence_pack_for_p101(system, cases):
    case = next(c for c in cases if c.case_id == "m01")
    r = system.retrieve(case, CONFIGS["M3@S6"])  # the reranker brings Table 3 into the top-5, the pack the rest
    assert {"R-12-r3-4.3.3", "R-12-r3-table1", "R-12-r3-6.1", "R-12-r3-6.2"} <= set(r["context"])
    assert "R-12-r2-4.3.3" not in r["context"]  # the date filter holds in the pack too
    r0 = system.retrieve(case, CONFIGS["S0"])
    assert len(r0["context"]) == 5


def test_header_has_citation_fields(by_id):
    h = header(by_id["R-12-r3-a1-4.4.1"], show_dates=True)
    assert "amendment=1" in h and "clause=4.4.1" in h and "in force from 2026-09-01 to now" in h
    text, sources = render(["VL-2-r4-tА.2-V-12"], by_id, False)
    assert sources[0].label == "S1" and "row=V-12" in text


# ---------------------------------------------------------------------------------------------
# Retrieval set, grading, checks
# ---------------------------------------------------------------------------------------------
def test_retrieval_set(cases, chunks):
    assert len(cases) == 143 and {c.slice for c in cases} == set(SLICES)
    ids = {c.chunk_id for c in chunks}
    for c in cases:
        for e in c.evidence:
            assert set(e["chunk_ids"]) <= ids and e["chunk_ids"] == sorted({x for loc in e["any_of"] for x in resolve(loc, chunks)})
    assert parse_locator("R-12 r3a1 cl 4.4.1") == {"doc_id": "R-12", "revision": 3, "amendment": 1, "clause": "4.4.1",
                                                   "table_no": None, "row_key": None}
    lang = [c for c in cases if c.slice == "language"]
    assert all(c.query == c.question_en for c in lang) and len(lang) == 20


def test_units_and_values():
    assert same_value("4,5", "4.5") and same_value("II", "ii") and not same_value("4,5", "5,0")
    assert canonical_unit("мм/с") == "mm/s" and canonical_unit("суток") == "days"
    assert has_value("в течение 24 часов", "24") and not has_value("4.3.3", "4,3")


def test_grade(cases):
    by = {c.case_id: c for c in cases}
    assert grade(by["t03"], {"status": "answered", "value": "4.5", "unit": "mm/s"})[0]
    assert not grade(by["t03"], {"status": "answered", "value": "5,0", "unit": "mm/s"})[0]
    assert grade(by["m01"], {"status": "answered", "answer": "Limit 4.5 mm/s exceeded: inspect within 24 hours.",
                             "value": "4,5"})[0]
    assert grade(by["x01"], {"status": "not_in_documents", "value": None})[0]
    assert not grade(by["f01"], {"status": "answered", "value": "4,5"})[0]
    assert grade(by["f01"], {"status": "conflict", "value": None})[0]
    assert grade(by["x01"], {"status": "needs_human", "value": None})[0]  # no value: accepted


def test_outcomes_split_the_failures(cases):
    """Every question is one of: right, false refusal, wrong, invented, invalid. The counts of the reference run."""
    from plant_assistant.reval import failure_modes, outcome

    by = {c.case_id: c for c in cases}
    assert outcome(by["t03"], {"status": "not_in_documents", "value": None}, False) == "false refusal"
    assert outcome(by["t03"], {"status": "needs_human", "value": None}, False) == "false refusal"
    assert outcome(by["f01"], {"status": "answered", "value": "4,5"}, False) == "wrong"
    assert outcome(by["x01"], {"status": "answered", "value": "16"}, False) == "invented"
    assert outcome(by["t03"], None, False) == "invalid"
    ref = pd.read_csv(REFERENCE_DIR / "per_question.csv")
    got = {name: failure_modes(ref[ref.config == name]) for name in ("S0", "S6", "M3@S0", "M3@S6")}
    assert got["S0"]["false refusals"].startswith("42/123") and got["S0"]["wrong or invented"].startswith("28")
    assert got["S6"]["false refusals"].startswith("12/123") and got["S6"]["wrong or invented"] == "1"
    assert got["M3@S0"]["wrong or invented"] == "27 (3 invented)"
    assert got["M3@S6"]["invalid"] == "3"
    for name in ("S0", "S6", "M3@S0", "M3@S6"):
        assert ref[ref.config == name].outcome.notna().all()


def test_needs_human_set(cases, by_id):
    """A separate set: the right answer is "needs a person"; the registry rows it needs do not touch the 143."""
    nh = load_cases(REPLAY_DIR.parent / "needs_human.jsonl")
    assert 4 <= len(nh) <= 6 and {c.slice for c in nh} == {"needs_human"}
    assert not {c.case_id for c in nh} & {c.case_id for c in cases}
    assert all(e["chunk_ids"] for c in nh for e in c.evidence)
    reg = load_registry()
    assert reg["P-215"].machine_class == "" and reg["P-215"].power_kw == "" and not reg["P-215"].documents
    assert reg["P-216"].station == "" and "P-220" not in reg and {"P-220A", "P-220B"} <= set(reg)
    asked = " ".join(c.question_ru for c in cases)
    assert not any(t in asked for t in ("P-215", "P-216", "P-220"))
    case = nh[0]
    assert grade(case, {"status": "needs_human", "value": None})[0]
    assert not grade(case, {"status": "not_in_documents", "value": None})[0]
    assert grade(case, {"status": "answered", "value": "2,8"}) == (False, "answered without the missing data")


def test_citation_checks_catch_the_neighbour_clause(chunks, by_id):
    ans = Answer(status="answered", answer="Inspect within 24 hours.", value="24", unit="h",
                 citations=[Citation(source="S1", doc_id="R-12", revision=3, clause="4.3.4", page=3)])
    r = check_answer(ans, [], chunks, by_id, TODAY)
    assert r["exists"] and r["in_force"] and not r["supports"] and not r["verified"]
    ans.citations[0].clause = "4.3.3"
    assert check_answer(ans, [], chunks, by_id, TODAY)["verified"]
    ans.citations[0].revision = 2  # rev. 2 is not in force today
    assert not check_answer(ans, [], chunks, by_id, TODAY)["in_force"]
    ans.citations[0].clause = "4.3.9"
    assert not check_answer(ans, [], chunks, by_id, TODAY)["exists"]


def test_parse_contract_and_simple():
    a = parse('{"answer": "x", "value": null, "unit": null, "sources": []}', contract=False)
    assert a.status == "not_in_documents"
    assert parse("not json", contract=True) is None


def test_attribution(cases, by_id):
    case = next(c for c in cases if c.case_id == "m01")
    rec = {"retrieved": ["R-12-r3-4.3.3"], "context": ["R-12-r3-4.3.3", "R-12-r3-table1"]}
    assert attribute(case, rec, False, by_id) == "assembly"
    assert attribute(case, {**rec, "context": ["R-12-r3-6.1"]}, False, by_id) == "search"
    full = {**rec, "context": ["R-12-r3-4.3.3", "R-12-r3-table1", "R-12-r3-6.1", "R-12-r3-6.2"]}
    assert attribute(case, full, False, by_id) == "model" and retrieval_metrics(case, full)["evidence_recall"]


def test_stats_from_practice_2():
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97
    assert compare([True] * 10 + [False] * 10, [True] * 20)["decision"] == "B is better"


# ---------------------------------------------------------------------------------------------
# Replay reproduces the reference
# ---------------------------------------------------------------------------------------------
def test_replay_reproduces_reference(system, cases):
    ref = pd.read_csv(REFERENCE_DIR / "per_question.csv")
    for name in ("S0", "S2b", "S6"):
        records = system.run(cases, CONFIGS[name], progress=False)
        df = score(cases, records, system.by_id)
        mine = df.set_index("case_id")[["correct", "verified", "recall@k"]]
        theirs = ref[ref.config == name].set_index("case_id")[["correct", "verified", "recall@k"]]
        def text(x: pd.DataFrame) -> pd.DataFrame:
            return x.astype(object).where(x.notna(), "-").astype(str)

        assert text(mine).equals(text(theirs.loc[mine.index])), name


def test_p101_answer(system, cases):
    case = next(c for c in cases if c.case_id == "m01")
    lost = system.run_case(case, CONFIGS["S6"])  # BM25 puts Table 3 at rank 6: lost in search
    assert not grade(case, lost["answer"])[0] and "R-12-r3-table1" not in lost["top"]
    rec = system.run_case(case, CONFIGS["M3@S6"])
    assert grade(case, rec["answer"])[0] and rec["checks"]["verified"]
    cited = {c["chunk_id"] for c in rec["checks"]["citations"]}
    assert {"R-12-r3-4.3.3", "R-12-r3-table1"} <= cited and cited & {"R-12-r3-6.1", "R-12-r3-6.2"}


def test_holdout_is_separate_and_graded(cases):
    holdout = load_cases(REPLAY_DIR.parent / "holdout.jsonl")
    assert 25 <= len(holdout) <= 30 and {c.slice for c in holdout} == {"holdout"}
    assert not {c.case_id for c in holdout} & {c.case_id for c in cases}
    assert all(c.note.startswith("role: ") for c in holdout)


def test_needs_human_replay(system):
    nh = load_cases(REPLAY_DIR.parent / "needs_human.jsonl")
    ref = pd.read_csv(REFERENCE_DIR / "needs_human_per_question.csv")
    for name in ("S6", "M3@S6"):
        df = score(nh, system.run(nh, CONFIGS[name], progress=False), system.by_id)
        assert list(df.correct) == list(ref[ref.config == name].correct), name
    assert int(ref[ref.config == "M3@S6"].correct.sum()) == 5


def test_cost_columns(system, cases):
    from plant_assistant.reval import cost

    recs = system.run([c for c in cases if c.slice == "conflict"], CONFIGS["M3@S6"], progress=False)
    c = cost(recs)
    assert c["model calls / question"] == 2 and c["prompt tokens / question"] > 1000
