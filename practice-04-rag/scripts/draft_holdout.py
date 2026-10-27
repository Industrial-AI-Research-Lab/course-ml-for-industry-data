"""Holdout questions: written by role, not by slice, by a model that never sees the ladder or the slices.

The retrieval set of practice 4 was written together with the fixes, so it can flatter them. The holdout is a check
against that: a model gets only the documents and short descriptions of four roles, and writes the questions these
people would ask. Then code keeps the questions it can grade: the evidence resolves to chunks, the value is in the
evidence, the wording does not copy the answer chunk (lexical leak), the question is not a near-copy of the set.

Live (lecturer):  uv run python scripts/draft_holdout.py --draft        # calls the `drafter` profile
Rebuild:          uv run python scripts/draft_holdout.py                # from data/holdout_drafts.jsonl
Writes data/holdout.jsonl (slice "holdout", tag "role") and reference/holdout_review.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_content import TODAY  # noqa: E402

from plant_assistant import DATA_DIR, REFERENCE_DIR, STORE_DIR  # noqa: E402
from plant_assistant.cases import TestCase, load_cases, resolve, write_cases  # noqa: E402
from plant_assistant.config import get_profile  # noqa: E402
from plant_assistant.jsonl import read_json_records  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient  # noqa: E402
from plant_assistant.search import tokenize  # noqa: E402
from plant_assistant.store import load_store  # noqa: E402
from plant_assistant.units import has_value  # noqa: E402

# Rejected at the review of the kept drafts (the checks by code passed, a person did not)
REVIEWED_OUT = {
    "h11": "a free-text answer (a fire extinguisher model with its name): code cannot grade it",
    "h24": "not a conflict: R-12 and R-14 apply to different stations; the answer is two values",
    "h42": "repeats h04 in other words",
}
DRAFTS = DATA_DIR / "holdout_drafts.jsonl"
HOLDOUT = DATA_DIR / "holdout.jsonl"
LEAK_MAX = 0.6  # share of the content words of the question found in the answer chunk
NEAR_COPY = 0.7  # Jaccard of word stems with a question of the retrieval set

ROLES = {
    "operator": "Operator (обходчик) of pump stations No. 2 and No. 3: walks the rounds, measures vibration and "
                "temperature with portable instruments, reports to the shift supervisor.",
    "mechanic": "Maintenance mechanic: repairs and services pumps, valves and compressors, does lubrication and "
                "alignment, works with requests in the CMMS.",
    "reliability": "Reliability engineer: sets inspection intervals and limits, analyses failures, compares the "
                   "plant rules with the manufacturers' documents.",
    "auditor": "Inspector or auditor of industrial safety: checks whether the staff followed the rules that were in "
               "force on a given date, reviews incidents and records.",
}

SYSTEM = """You write test questions for a search assistant of an industrial plant. You get the plant documents
(regulations, procedures, datasheets, a manufacturer's manual, lists) and descriptions of people who work there.
Write the questions these people would really ask the assistant in their daily work, in their own words.

Rules:
- Each question is asked in Russian by the person (question_ru), with an English translation (question_en).
- Do not copy phrases from the documents: a person does not know the wording of the clause.
- Each question must have one checkable answer: a number with its unit taken from the documents, or a short text
  value (for example a machine class). If a real person would ask something the documents do not answer, you may
  include such a question: then expected.type is "not_in_documents" and evidence is empty.
- If the documents give different answers to a question and none of them clearly overrides the other, set
  expected.type to "conflict" and list both places as evidence.
- evidence: every place needed for the answer, with doc_id and revision exactly as in the document header, and
  amendment (if the place is in an amendment), and clause, or table (its number from the caption), or table and row
  (the key in the first column). role is "answer" for the place that holds the value, "support" for a place needed to
  choose it (for example the class of the machine), "action" for what to do.
- as_of: the date the question is about as YYYY-MM-DD if the person names a date or an event date; otherwise null
  (the question is asked today, """ + TODAY + """).
- Mix the people evenly. Do not repeat the same question with another tag.
Return JSON: {"questions": [{"role": ..., "question_ru": ..., "question_en": ..., "as_of": ..., "expected":
{"type": "value" | "not_in_documents" | "conflict", "value": ..., "unit": ...}, "evidence": [{"doc_id": ...,
"revision": ..., "amendment": ..., "clause": ..., "table": ..., "row": ..., "role": ...}], "note": ...}]}"""


def documents_text(chunks) -> str:
    """The documents as the drafter sees them: chunk texts in order, with a header per document. Rows of the long
    table are left out (its page chunks hold the same rows)."""
    out, current = [], None
    for c in chunks:
        if c.chunk_type == "table_row":
            continue
        key = (c.doc_id, c.revision, c.amendment)
        if key != current:
            current = key
            amd = f", amendment {c.amendment}" if c.amendment else ""
            out.append(f"\n===== {c.doc_id}, revision {c.revision}{amd} — {c.doc_title} — in force from "
                       f"{c.effective_from} to {c.effective_to or 'now'} — scope: stations {c.scope_stations}, "
                       f"equipment {c.scope_equipment_types} =====")
        out.append(re.sub(r"<[^>]+>", " ", c.text).strip())
    return "\n".join(out)


def draft(n: int, profile: str, more: bool = False) -> None:
    """Draft n questions; with `more`, add a batch to the drafts and show the drafter what it already wrote."""
    _, chunks = load_store(STORE_DIR)
    roles = "\n".join(f"- {k}: {v}" for k, v in ROLES.items())
    before = list(read_json_records(DRAFTS)) if more and DRAFTS.exists() else []
    done = ("\n\nAlready written, do not repeat them or ask the same with other numbers:\n"
            + "\n".join(f"- {q['question_ru']}" for q in before)) if before else ""
    user = (f"People:\n{roles}\n\nWrite {n} questions.{done}\n\nDocuments:\n{documents_text(chunks)}")
    client = LLMClient(get_profile(profile), logger=CallLogger(ROOT / "runs" / "lecturer" / "drafter.jsonl"))
    resp = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       response_format={"type": "json_object"})
    if not resp.ok:
        raise SystemExit(resp.error)
    data = json.loads(re.sub(r"^\s*```(?:json)?|```\s*$", "", resp.content.strip()))
    with open(DRAFTS, "a" if more else "w", encoding="utf-8") as f:
        for q in data["questions"]:
            f.write(json.dumps({**q, "drafted_by": resp.model}, ensure_ascii=False) + "\n")
    print(f"{len(data['questions'])} drafts by {resp.model}, {resp.prompt_tokens} prompt tokens")


HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})


def locators(e: dict) -> list[str]:
    """Locators for an evidence entry of the drafter, the most specific first: a row of a table that has no row
    chunks (Table 3: "Насосы II") falls back to the whole table."""
    doc = str(e["doc_id"]).translate(HYPHENS).strip()
    rev = f" r{e['revision']}" if e.get("revision") not in (None, "", "null") else ""
    amd = f"a{e['amendment']}" if e.get("amendment") not in (None, "", 0, "null") else ""
    if e.get("clause"):
        return [f"{doc}{rev}{amd} cl {str(e['clause']).strip()}"]
    table = re.sub(r"Приложение|Таблица|Table", "", str(e.get("table") or "")).strip()
    table = re.sub(r"^A", "А", table)  # captions use the Cyrillic А
    row = str(e.get("row") or "").translate(HYPHENS).strip()
    if table and row:
        return [f"{doc}{rev}{amd} t {table} row {row}", f"{doc}{rev}{amd} t {table}"]
    if table:
        return [f"{doc}{rev}{amd} t {table}"]
    return [doc]


def resolve_any(e: dict, chunks) -> tuple[str, list[str]]:
    for loc in locators(e):
        try:
            return loc, resolve(loc, chunks)
        except ValueError:
            continue
    raise ValueError(locators(e)[0])


def stems(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 2}


def build() -> None:
    _, chunks = load_store(STORE_DIR)
    by_id = {c.chunk_id: c for c in chunks}
    known = [stems(c.question_ru) for c in load_cases()]
    kept, review = [], []
    for i, q in enumerate(read_json_records(DRAFTS), start=1):
        reasons = []
        exp = q.get("expected") or {}
        etype = exp.get("type")
        evidence = []
        q["question_ru"], q["question_en"] = q["question_ru"].translate(HYPHENS), q["question_en"].translate(HYPHENS)
        for e in q.get("evidence") or []:
            try:
                loc, ids = resolve_any(e, chunks)
                evidence.append({"role": e.get("role") or "answer", "any_of": [loc], "chunk_ids": sorted(ids)})
            except (ValueError, KeyError) as exc:
                reasons.append(f"evidence not found: {exc}")
        if etype == "value":
            expected = {"value": str(exp.get("value")), "unit": exp.get("unit")}
            text = " ".join(by_id[c].text for ev in evidence for c in ev["chunk_ids"])
            if not evidence:
                reasons.append("no evidence")
            elif not has_value(text, expected["value"]):
                reasons.append(f"value {expected['value']} not in the evidence")
        elif etype in ("not_in_documents", "conflict"):
            expected = {"status": etype}
            if etype == "conflict" and len({by_id[ev["chunk_ids"][0]].doc_id for ev in evidence}) < 2:
                reasons.append("conflict needs two documents")
            if etype == "not_in_documents":
                evidence = []
        else:
            expected, reasons = {}, reasons + [f"unknown expected type {etype}"]
        qs = stems(q["question_ru"])
        answer_text = " ".join(by_id[c].text for ev in evidence if ev["role"] == "answer" for c in ev["chunk_ids"])
        leak = len(qs & stems(answer_text)) / len(qs) if qs and answer_text else 0.0
        if leak > LEAK_MAX:
            reasons.append(f"lexical leak {leak:.2f}")
        near = max((len(qs & k) / len(qs | k) for k in known), default=0.0)
        if near > NEAR_COPY:
            reasons.append(f"near-copy of the retrieval set ({near:.2f})")
        qa = set(tokenize(q["question_ru"]))  # numbers and dates count here: "станции 2" is not "станции 3"
        twin = max((len(qa & set(tokenize(k.question_ru))) / len(qa | set(tokenize(k.question_ru))) for k in kept),
                   default=0.0)
        if twin > NEAR_COPY:
            reasons.append(f"repeats a kept holdout question ({twin:.2f})")
        case_id = f"h{i:02d}"
        if case_id in REVIEWED_OUT and not reasons:
            reasons.append(f"rejected at review: {REVIEWED_OUT[case_id]}")
        review.append((case_id, q, expected, evidence, leak, reasons))
        if not reasons:
            as_of = q.get("as_of") or TODAY
            kept.append(TestCase(case_id, "holdout", "ru", q["question_ru"], q["question_en"], TODAY, as_of, expected,
                                 evidence, f"holdout ({q.get('drafted_by', 'drafter')})",
                                 f"role: {q.get('role')}; {q.get('note') or ''}".strip("; ")))
    write_cases(kept, HOLDOUT)
    lines = ["# Holdout questions: review list", "",
             f"Drafted by role by `{next(iter(read_json_records(DRAFTS)), {}).get('drafted_by')}` from the documents and "
             "four role descriptions only (never the ladder or the slices). Kept: questions whose evidence resolves, "
             f"whose value is in the evidence, with a lexical leak <= {LEAK_MAX} and no near-copy of the retrieval set.",
             "The lecturer checks every kept question (gold answer and evidence) before the numbers go to the slides.", "",
             f"Kept {len(kept)} of {len(review)}.", "",
             "| id | role | question (RU) | translation | expected | evidence | leak | kept / why not |",
             "|---|---|---|---|---|---|---|---|"]
    for case_id, q, expected, evidence, leak, reasons in review:
        exp = expected.get("value") or expected.get("status") or "?"
        unit = expected.get("unit") or ""
        ev = "; ".join(f"{e['role']}: {e['any_of'][0]}" for e in evidence) or "—"
        lines.append(f"| {case_id} | {q.get('role')} | {q['question_ru']} | {q['question_en']} | {exp} {unit} | {ev} | "
                     f"{leak:.2f} | {'kept' if not reasons else '; '.join(reasons)} |")
    (REFERENCE_DIR / "holdout_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    roles = {}
    for c in kept:
        r = c.note.split(";")[0].replace("role: ", "")
        roles[r] = roles.get(r, 0) + 1
    print(f"kept {len(kept)} of {len(review)}; by role {roles}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--profile", default="drafter")
    ap.add_argument("--more", action="store_true", help="add a batch of drafts to the existing ones")
    args = ap.parse_args()
    if args.draft or args.more or not DRAFTS.exists():
        draft(args.n, args.profile, args.more)
    build()


if __name__ == "__main__":
    main()
