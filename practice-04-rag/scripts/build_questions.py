"""Build the retrieval set data/questions.jsonl from scripts/questions_content.py and the store.

Every question gets its gold chunk ids from the evidence locators, by code. The script also checks that the
expected value is really in the text of an answer chunk (otherwise the question is wrong, or the reading lost
it) and reports how much of the wording overlaps with the answer chunk (a lexical leak makes search too easy).

Run:  uv run python scripts/build_questions.py        (needs store/: uv run python -m plant_assistant.corpus)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from questions_content import BASE, LANGUAGE_IDS, NEW, TODAY  # noqa: E402

from plant_assistant import INPUTS_DIR, STORE_DIR  # noqa: E402
from plant_assistant.cases import TestCase, counts_by_slice, resolve, write_cases  # noqa: E402
from plant_assistant.jsonl import read_json_records  # noqa: E402
from plant_assistant.store import load_store  # noqa: E402
from plant_assistant.units import has_value  # noqa: E402


def evidence(items, chunks) -> list[dict]:
    return [{"role": role, "any_of": locs, "chunk_ids": sorted({cid for loc in locs for cid in resolve(loc, chunks)})}
            for role, locs in items]


def english(q: dict) -> str:
    text = q["question"]
    if "R-12" not in text and q["doc_id"] == "R-12":
        text = text.rstrip("?") + ", according to R-12?"
    return text


def main() -> None:
    _, chunks = load_store(STORE_DIR)
    by_id = {c.chunk_id: c for c in chunks}
    p3 = {q["id"]: q for q in read_json_records(INPUTS_DIR / "practice-03" / "questions.jsonl")}
    cases: list[TestCase] = []
    for qid, (ru, ev) in BASE.items():
        q = p3[qid]
        if qid == "c06":  # Amendment No. 1 is in force today
            expected = {"value": "500", "unit": "h"}
        elif q["value"] is None:
            expected = {"status": "not_in_documents"}
        else:
            expected = {"value": q["value"], "unit": q["unit"]}
        note = "practice 3 answer 720 h; Amendment No. 1 (from 2026-09-01) makes it 500 h" if qid == "c06" else ""
        cases.append(TestCase(qid, "base", "ru", ru, english(q), TODAY, TODAY, expected, evidence(ev, chunks),
                              "practice 3", note))
    for qid in LANGUAGE_IDS:
        b = next(c for c in cases if c.case_id == qid)
        cases.append(TestCase(f"en-{qid}", "language", "en", b.question_ru, b.question_en, TODAY, TODAY,
                              b.expected, b.evidence, "practice 3", "English wording goes to search as is"))
    for qid, slice_, ru, en, as_of, exp, ev, note in NEW:
        if exp[0] == "status":
            expected = {"status": exp[1]}
        elif exp[0] == "values":
            expected = {"values": exp[1], "unit": exp[2]}
        else:
            expected = {"value": exp[0], "unit": exp[1]}
        as_of = None if slice_ == "what_changed" else (as_of or TODAY)
        cases.append(TestCase(qid, slice_, "ru", ru, en, TODAY, as_of, expected, evidence(ev, chunks),
                              "practice 4", note))

    # check: every expected value is in the text of the evidence
    problems = []
    for c in cases:
        values = c.expected.get("values") or ([c.expected["value"]] if "value" in c.expected else [])
        evidence_text = " ".join(by_id[cid].text for e in c.evidence for cid in e["chunk_ids"])
        for v in values:
            if not has_value(evidence_text, v):
                problems.append(f"{c.case_id}: value {v} not in its evidence chunks")
    write_cases(cases)
    print(f"{len(cases)} questions:", counts_by_slice(cases))
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
