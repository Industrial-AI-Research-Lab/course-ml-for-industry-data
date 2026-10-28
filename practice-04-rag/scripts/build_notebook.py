"""Build notebooks/practice_04_rag.ipynb from the cells below.

The notebook is generated so that its source stays readable in code review.
Edit the cells here, then run:  uv run python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "practice_04_rag.ipynb"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------------------------
md(r"""
# Practice 4. Finding the right clause: diagnose and fix

**ML in Industry 2026 · Topic 4: Retrieval-augmented generation**

## Goal

An engineer asks: *"After a bearing change, pump P-101 vibrates at 4.8 mm/s. Is this normal?"* The answer is in
regulation R-12, but the plant archive now has 16 documents, with old revisions, near-copies and a manual in English.
You measure where answers get lost, fix one cause at a time, and compare each fix with the step before it on the
same questions.
""")
md(r"""
**Glossary: measuring**

- **chunk**: a piece of a document (clause, table, table row) that search returns.
- **retrieval set**: 143 questions with known answers and known source chunks.
- **slice**: a group of questions of one kind, e.g. "the question names a past date".
- **gold evidence**: the chunks that hold the answer, found from the document structure.
- **recall@5**: the chunk with the answer is among the first five search results.
- **evidence recall**: every needed chunk (value, table, action) reached the model.
- **end-to-end**: the final answer is right: the value with its unit, or the right status.
- **correct + verified**: right, and every citation passes the three checks of step 7.
- **false refusal**: "not in the documents" or "needs a person" although the documents answer.
- **invented value**: a value given where the documents have none.
- **token**: a piece of text a model reads or writes; it measures size and cost.
- **ablation table**: one change per row; each row is compared with the row above.
- **paired comparison**: both rows answer the same questions; we count questions fixed and broken.
- **fixed / broke**: questions the new row fixed, and questions it broke.
- **verdict**: "B is better": new row B beats row A, and not by chance.
- **[a%–b%]**: the true share lies in this range with 95% confidence.
""")
md(r"""
**Glossary: the rows of the table**

- **tag**: the code of a piece of equipment, e.g. pump P-101.
- **S0**: plain BM25 keyword search: chunks ranked by the words they share with the question.
- **S1–S6, the ladder**: six plant steps: tags, table rows, applicability, date filter, evidence pack, contract.
- **S2b**: table rows searched apart from the clauses, at most two rows per question.
- **applicability**: the equipment registry tells which documents apply to this piece of equipment.
- **evidence pack**: the found chunks plus the clauses they refer to and their sections.
- **answer contract**: a fixed answer form with a status, a value and its sources.
- **statuses**: answered; not in the documents; conflict (documents disagree); needs a person.
- **vectors**: numbers an embedding model computes for the meaning of a text.
- **M1**: dense search, by meaning, not by words.
- **M2**: hybrid search, keywords and dense search together.
- **M3, reranker**: the course language model puts the top 20 chunks in order.
- **M4, cross-encoder**: a small reranking model that scores each chunk against the question.
- **M1t**: an English question is translated into Russian before search.
- **@S0 / @S6**: the mechanism instead of the plant steps / on top of them.
- **replay**: saved model answers; the same request gets the same answer offline.
- **holdout**: 27 extra questions the fixes were never tuned on.
""")
md(r"""
## Steps

| Step | You should see |
|------|----------------|
| 0 Final table | all rows at once |
| 1 Setup | the model answers, or replay |
| 2 Archive | 16 documents, 440 chunks |
| 3 Questions | 143 in 12 slices |
| 4 S0 | half right, losses in search |
| 5 S1–S5 | each step fixes its slice |
| 6 Mechanisms | wording fixed, dates not |
| 7 Contract | checked citations, conflicts |
| 8 Audit, holdout | a record, a gate, new questions |
| 9 Check | your numbers equal the reference |

**Live or replay.** Search runs on your laptop; model answers come from replay. `Mode: REPLAY` with a course
token: check `.env` (README, "Common problems").
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 0. Where we are going

The final table of this practice. Each row adds one change to the row in "compared with" and is compared with it
on the same questions.
""")
code(r"""
import warnings

import pandas as pd
from IPython.display import Markdown, display

from plant_assistant import REFERENCE_DIR
from plant_assistant.display import show

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 250)
reference = pd.read_csv(REFERENCE_DIR / "ablation.csv").fillna("")
show(reference, ["config", "vs", "recall@5", "evidence recall", "end-to-end", "e2e verdict"])
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 1. Setup

Model settings are in `models.toml`, the URL and the key in `.env`. The answers use the course model Qwen 3.8 27B
with thinking off: no hidden reasoning before the answer. Dense search and the cross-encoder use saved vectors and
scores; with access to those models you can rebuild them (see README).
""")
code(r"""
import os

from plant_assistant import REPLAY_DIR, RUNS_DIR, STORE_DIR
from plant_assistant.config import get_profile, list_profiles, load_env
from plant_assistant.llm import CallLogger, LLMClient, ReplayClient

load_env()
PROFILE = os.environ.get("PRACTICE_PROFILE", "course")
pd.DataFrame(list_profiles().items(), columns=["profile", "description"])
""")
code(r"""
def model_answers(profile_name: str) -> bool:
    # One call without network retries: if the model is not there, we want to know now.
    try:
        client = LLMClient(get_profile(profile_name), max_network_retries=0)
    except ValueError as exc:  # profile not configured (no URL or key)
        print(f"{profile_name}: {exc}")
        return False
    r = client.chat([{"role": "user", "content": "Reply with one word: OK"}])
    print(f"{profile_name}: {r.content.strip()[:20]!r} ({r.latency_s:.1f}s)" if r.ok else r.error[:200])
    return r.ok


LIVE = os.environ.get("PRACTICE_MODE") != "replay" and model_answers(PROFILE)
replay = {task: ReplayClient(REPLAY_DIR / f"{task}_course.jsonl")
          for task in ("answers", "answers_thinking", "rerank", "translate", "crossencoder")}
live = LLMClient(get_profile(PROFILE), logger=CallLogger(RUNS_DIR / f"my-{PROFILE}" / "calls.jsonl")) if LIVE \
    else replay["answers"]
print("Mode:", f"LIVE single questions on {PROFILE}, the full set from replay" if LIVE else
      "REPLAY of the lecturer's saved answers (course model)")
from plant_assistant.search import VectorStore

vectors = sorted((REPLAY_DIR / "vectors").glob("*.npz"))
print("Embedding vectors:", [v.stem for v in vectors] or "none: dense rows (M1, M2) are not measured")
VECTORS = VectorStore(vectors[0]) if vectors else None
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 2. The plant archive, version 2

The four documents of practice 3 plus 12 new synthetic ones. All are born-digital, so reading the PDFs does not
get in the way. The scale is small (tens of pages, 440 chunks); the process is the same at any scale. The
document registry (`data/documents.csv`) gives each document its dates of validity and the stations it covers.
""")
code(r"""
from plant_assistant.corpus import build_corpus, load_registry
from plant_assistant.store import load_store

summary = build_corpus(STORE_DIR)
print(summary)
_, chunks = load_store(STORE_DIR)
by_id = {c.chunk_id: c for c in chunks}
docs = pd.DataFrame(load_registry())
docs[["doc_id", "revision", "amendment", "kind", "language", "effective_from", "effective_to", "scope_stations",
      "scope_equipment_types", "scope_tags", "pages", "source"]]
""")
md(r"""
**What is new.** An **amendment**, an official change to one clause, moves a check interval of R-12 from 720 h
to 500 h on 01.09.2026. R-14 copies R-12 for pump station 3, with stricter limits. The English manual of pump
P-205 gives milder limits than R-12. The valve list VL-2 has a table of 110 rows over four pages.
""")
code(r"""
from plant_assistant.display import where

for cid in ("R-12-r3-table1", "R-14-r1-table1"):
    c = by_id[cid]
    print(f"{where(c)}  scope_stations={c.scope_stations}  valid {c.effective_from} → {c.effective_to or 'now'}")
    print("   ", c.text.splitlines()[0], "|", [line for line in c.text.splitlines() if "|II|" in line][0])
""")
md(r"""
Each chunk now also knows when it is valid, which stations and equipment it covers, and which clauses it refers
to and is referred to by.
""")
code(r"""
fields = ["chunk_id", "clause", "table_no", "row_key", "effective_from", "effective_to", "amended_by", "references",
          "referenced_by"]
pd.DataFrame([{f: getattr(by_id[cid], f) for f in fields} for cid in
              ("R-12-r3-4.4.1", "R-12-r3-a1-4.4.1", "R-12-r3-4.3.3", "R-12-r3-6.2", "R-20-r2-3.3",
               "VL-2-r4-tА.2-V-12")])
""")
md(r"""
The **index version** is a fingerprint of the chunks, the PDF reader and the embedding model. Every answer record
keeps it; when it changes, the questions run again.
""")
code(r"""
print("index_version:", summary["index_version"])
pd.crosstab(pd.Series([c.doc_id for c in chunks], name="document"), pd.Series([c.chunk_type for c in chunks],
                                                                              name="chunk type"), margins=True)
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 3. The retrieval set

143 questions in 12 slices. The 50 questions of practice 3 form the base slice; 20 of them also go to search in
English. Questions are asked in Russian, as at a Russian plant; the English translation is for reading. Gold
evidence is found by code from the document structure. Slice results are counts: 3 of 4, not 75%.
""")
code(r"""
from plant_assistant.cases import counts_by_slice, load_cases

cases = load_cases()
case_by_id = {c.case_id: c for c in cases}
show(pd.DataFrame([counts_by_slice(cases)], index=["questions"]))
""")
code(r"""
from plant_assistant.display import gold_frame

P101 = case_by_id["m01"]
print("Question (asked in Russian):", P101.question_ru)
print("Translation:                ", P101.question_en)
print("Expected:", P101.expected, "| asked on", P101.asked_on)
gold_frame(P101)
""")
md(r"""
One answer needs four chunks: the clause with the deadline, Table 3 with the limit, and two clauses with the
actions. recall@5 looks only for the chunk with the value; evidence recall needs all four.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 4. Baseline S0: keyword search, top 5, no filters

Search runs live; the model answers for all 143 questions come from replay.
""")
code(r"""
from plant_assistant import reval
from plant_assistant.display import answer_frame, context_frame, hits_frame
from plant_assistant.pipeline import CONFIGS, LADDER, System

system = System(chunks, answer_client=replay["answers"], rerank_client=replay["rerank"],
                translate_client=replay["translate"], cross_client=replay["crossencoder"], vectors=VECTORS,
                think_client=replay["answers_thinking"])
records, scores = {}, {}


def run(name: str) -> pd.DataFrame:
    records[name] = system.run(cases, CONFIGS[name], progress=False)
    scores[name] = reval.score(cases, records[name], by_id)
    return scores[name]


run("S0")
show(reval.ablation({"S0": scores["S0"]}, {}), ["config", "recall@5", "evidence recall", "end-to-end",
                                                 "correct + verified", "abstain ok", "conflict ok"])
""")
code(r"""
display(Markdown("**Wrong answers of S0 by link**"))
pd.DataFrame([reval.losses(scores["S0"])], index=["S0"])
""")
md(r"""
A wrong answer is charged to the first link that lost it: reading the PDF, search, building the context, or the
model. Most answers are lost **in search**: the chunk with the answer never reached the model. Nothing is lost in
reading, because the documents are born-digital. Below: the top five search results for the P-101 question.
""")
code(r"""
rec = next(r for r in records["S0"] if r["case_id"] == "m01")
hits_frame(rec, P101, by_id)
""")
md(r"""
None of the five holds the answer. They share "после замены подшипника" (after a bearing change) with the
question: one clause in three regulations, and two maintenance clauses. Table 3 and the inspection deadline are
missing. **Task 1 of the lecture** used five failures of this run:
""")
code(r"""
display(Markdown((REFERENCE_DIR / "s0_failures.md").read_text(encoding="utf-8")))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 5. The ladder: one plant step per row

Each step is added to the one before. We look at one question before and after, then at the table.

**S1: one spelling of tags.** "Р-101" with a Cyrillic Р, "P101" and "P 102" become P-101 and P-102, in the
chunks and in the question.
""")
code(r"""
for name in ("S0", "S1"):
    rec = system.run_case(case_by_id["g02"], CONFIGS[name], with_answer=False)
    display(Markdown(f"**{name}** · {case_by_id['g02'].question_ru}"), hits_frame(rec, case_by_id["g02"], by_id))
""")
md(r"""
**S2: table rows as chunks.** Before, a page of the valve table (33 rows) is one chunk. Now each row is a chunk
that repeats the table title, the column names and the units:
""")
code(r"""
print(by_id["VL-2-r4-tА.2-V-12"].text)
for name in ("S1", "S2"):
    rec = system.run_case(case_by_id["r01"], CONFIGS[name], with_answer=False)
    display(Markdown(f"**{name}** · {case_by_id['r01'].question_ru}"), hits_frame(rec, case_by_id["r01"], by_id, 3))
""")
md(r"""
**S2b: rows searched apart.** Mixed with the clauses, 110 short rows push the datasheets out of the results for
any question that names a pump. So rows get their own search and at most two places. The questions that mixed
rows broke:
""")
code(r"""
from plant_assistant.pipeline import PREVIOUS

previous = dict(PREVIOUS)
broken = ["d01", "d02", "d04", "z01", "z02", "z04", "z05", "i02", "g01"]
for name in ("S1", "S2", "S2b"):
    run(name)
pd.DataFrame({name: scores[name].set_index("case_id").loc[broken, "correct"] for name in ("S1", "S2", "S2b")}).T
""")
md(r"""
**S3: applicability.** The equipment registry gives each pump its station, class and documents. Search then sees
only the documents for that station. Pump P-310 is at station 3, so R-14 applies, not R-12. The manual of P-205
never names the pump; only the registry links them.
""")
code(r"""
from plant_assistant.applicability import load_registry

print(load_registry()["P-205"].describe())
rec = system.run_case(case_by_id["f01"], CONFIGS["S3"], with_answer=False)
print("linked by the registry:", [where(by_id[c]) for c in rec["linked"]])
""")
md(r"""
Two pumps of station 3, before and after S3: P-310 at 4.0 mm/s (`p01`) and P-311 at 3.0 mm/s (`p07`). Without
the registry, the second one is judged by R-12.
""")
code(r"""
from plant_assistant.applicability import equipment_context

# single questions: live if the course model answers (the reranker too), replay otherwise
system_live = System(chunks, answer_client=live, rerank_client=live if LIVE else replay["rerank"],
                     translate_client=live if LIVE else replay["translate"], cross_client=replay["crossencoder"],
                     vectors=VECTORS, think_client=replay["answers_thinking"])
for cid in ("p01", "p07"):  # P-310 at 4.0 mm/s; P-311 (class I, station 3) at 3.0 mm/s
    case = case_by_id[cid]
    for name in ("S2", "S3"):
        rec = system_live.run_case(case, CONFIGS[name])
        a = rec["answer"] or {}
        display(Markdown(f"**{cid} · {name}** · {case.question_ru} · scope: {rec['scope'] or 'none'}"),
                hits_frame(rec, case, by_id, 3))
        print(f"value {a.get('value')} (expected {case.expected['values'][0]}), right {reval.grade(case, a)[0]}:",
              a.get("answer"))
""")
md(r"""
**S4: a date filter.** Search keeps only chunks valid on the date of the question. The same question about the
check interval on three dates: 1000 h, 720 h, then 500 h after the amendment. At S4 the amendment still misses
the top five; the evidence pack of S5 brings it.
""")
code(r"""
for cid in ("v01", "v02", "v03"):
    for name in ("S3", "S4", "S5"):
        rec = system_live.run_case(case_by_id[cid], CONFIGS[name])
        a = rec["answer"] or {}
        print(f"{cid} {name}: as_of {rec['as_of']}  expected {case_by_id[cid].expected['value']} h  "
              f"answer {a.get('value')} {a.get('unit')}  top-1 {where(by_id[rec['retrieved'][0]])}")
""")
md(r"""
**Two wrong ways to filter by date.** Filtering after search can leave nothing. A flag "in force now" answers a
question about an incident in February 2025 with today's revision.
""")
code(r"""
from plant_assistant.filters import make_filter, post_filter, status_in_force
from plant_assistant.query import extract_rules

q = case_by_id["i01"]
fields = extract_rules(q.query, q.asked_on)
idx = system.bm25(CONFIGS["S4"])
by_date = make_filter(fields, None, use_scope=False, use_date=True)
pre = [where(by_id[h.chunk_id]) for h in idx.search(q.query, 5, by_date)]
post = [where(by_id[h.chunk_id]) for h in post_filter(idx.search(q.query, 5), by_id, by_date)]
flag = [where(by_id[h.chunk_id]) for h in idx.search(q.query, 5, status_in_force)]
print("question:", q.question_en)
pd.DataFrame({"pre-filter by date": pd.Series(pre), "post-filter by date": pd.Series(post),
              "status = in force": pd.Series(flag)})
""")
md(r"""
**S5: the evidence pack.** The found chunks, plus what they refer to, what refers to them, and their sections, up
to 2500 tokens. The question: after a bearing change, within how many hours to measure, and against which limit
(`m06`). At S4 the deadline clause is first but Table 3 is missing; the pack adds it:
""")
code(r"""
m06 = case_by_id["m06"]
for name in ("S4", "S5"):
    rec = system.run_case(m06, CONFIGS[name], with_answer=False)
    ctx = context_frame(rec, m06, by_id)
    print(f"{name}: top-1 {where(by_id[rec['retrieved'][0]])}; {len(ctx)} chunks, {ctx.tokens.sum()} tokens; "
          f"gold in the context: {sorted(set(ctx.gold) - {''})}; "
          f"evidence recall = {reval.retrieval_metrics(m06, rec)['evidence_recall']}")
ctx[ctx.gold != ""]
""")
md(r"""
The ladder in the table, each row against the row above:
""")
code(r"""
for name in ("S3", "S4", "S5"):
    run(name)
cols = ["config", "vs", "recall@5", "evidence recall", "end-to-end", "e2e: fixed / broke", "e2e verdict"]
show(reval.ablation(scores, previous), cols)
""")
code(r"""
display(Markdown("**End-to-end by slice** (counts)"))
show(reval.by_slice(scores, "correct"))
""")
md(r"""
- S1 fixes tag spellings; S2 fixes table rows but breaks base questions; S2b keeps the fix without the damage.
- S3 fixes applicability, S4 questions about a past date, S5 questions that need several chunks.
- Conflicts stay at zero until S6: a simple answer cannot say "the documents disagree".
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 6. Mechanisms: rows of the same table

Search methods from the Advanced NLP course, added in two places: instead of the plant steps (@S0) or on top of
them (@S6). M1 searches by meaning, M2 mixes keywords and meaning, M3 and M4 put the top 20 chunks in order, M1t
translates an English question first. We do not explain how they work; we look at where they help.
""")
code(r"""
MECH = ["S0", "M1@S0", "M2@S0", "M3@S0", "M4@S0", "S6", "M1@S6", "M2@S6", "M3@S6", "M4@S6"]
for name in ["M1t"] + MECH:
    if name not in scores and (VECTORS is not None or CONFIGS[name].retriever == "bm25"):
        run(name)
table = reval.ablation(scores, previous)
show(table[table.config.isin(MECH + ["M1t"])], cols)
""")
code(r"""
focus = ["language", "point_in_time", "applicability", "conflict", "table_row", "tag_variant", "multi_evidence"]
show(reval.by_slice({k: scores[k] for k in MECH if k in scores}, "correct")[focus])
""")
md(r"""
**What each row costs.** Seconds and model calls per question, and tokens of the chat model. The language-model
reranker (M3) adds a chat call of about 1800 tokens; the cross-encoder (M4) adds one short call and no chat tokens.
""")
code(r"""
show(reval.cost_table(records).reset_index(), ["config", "median s / question", "model calls / question",
                                                "chat calls / question", "prompt tokens / question",
                                                "completion tokens / question", "cross-encoder tokens / question"])
""")
code(r"""
lang = scores["M1t"].case_id
display(Markdown("**English questions**: as they are (S0), translated into Russian (M1t), "
                 "dense search on the English wording (M1@S0)"))
pd.DataFrame({name: [f"{int(scores[name].set_index('case_id').loc[lang, m].astype(bool).sum())}/{len(lang)}"
                     for m in ("recall@k", "correct")] for name in ("S0", "M1t", "M1@S0") if name in scores},
             index=["recall@5", "end-to-end"])
""")
md(r"""
- Mechanisms fix wording and language: dense search answers 17 of 20 English questions.
- Only the plant steps fix dates, applicability and conflicts: dense search alone answers 3 of 10 past-date questions.
- The cross-encoder gives most of the reranker's gain in a quarter of the time: 0.5 s extra against 2 s.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 7. The answer contract and the citation checks

S6 adds a fixed answer form: a status, the value with its unit, and up to five short claims with their sources. The
model may cite only chunks it was given; code then adds the document, revision, clause and page.
""")
md(r"""
**A place to think.** With thinking off, the model writes the answer at once. So the form starts with a short
internal field, `check`: do the sources agree, is the number in the cited text? It is logged for the audit, never
shown to the engineer and never graded.
""")
md(r"""
**Three checks of every citation**: it **exists** (the chunk was given to the model), it is **in force** on the
date of the question, and it **supports** the answer (the number and the unit are in its text). One citation must
support the value; the others back other claims and are checked in part 2 by a judge (a grading model).
""")
md(r"""
**The P-101 question.** At S6, keyword search puts Table 3 sixth, so the evidence pack has nothing to start from.
The reranker on top (M3@S6) brings Table 3 back:
""")
code(r"""
for name in ("S6", "M3@S6"):
    rec = system_live.run_case(P101, CONFIGS[name])
    print(f"{name}: top-5 {[where(by_id[c]) for c in rec['top']]}")
    print(f"     status {rec['answer']['status']}, correct {reval.grade(P101, rec['answer'])[0]}, "
          f"verified {rec['checks']['verified']}")
answer, checks = answer_frame(rec)
display(answer)
checks.insert(1, "where", checks.chunk_id.map(lambda c: where(by_id[c]) if c else ""))  # R-12-r3-table3 is Table А.1
checks
""")
md(r"""
**Task 3 of the lecture**: "Inspect P-101 within 24 hours (R-12, rev. 3, clause 4.3.4)". The clause exists and is
in force, but it is about the stop limit; "24 hours" is in the next clause, 4.3.3. A citation of a given chunk can
still be the wrong chunk: the checks catch it.
""")
code(r"""
from plant_assistant.answer import Answer, Claim
from plant_assistant.assemble import render
from plant_assistant.verify import check_answer

context = ["R-12-r3-4.3.4", "R-12-r3-4.3.3", "R-12-r2-4.3.3"]  # rev. 2 is there, as for a "what changed" question
_, sources = render(context, by_id, show_dates=True, ids_as_labels=True)
rows = []
for cid in context + ["R-12-r3-6.2"]:  # the last one was not in the context
    claim = Answer(status="answered", answer="Inspect P-101 within 24 hours.", value="24", unit="h",
                   claims=[Claim(text="Inspect within 24 hours.", sources=[cid])])
    r = check_answer(claim, sources, chunks, by_id, P101.asked_on)
    rows.append({"cited id": cid, "exists (in the context)": r["exists"], "in force": r["in_force"],
                 "supports 24 h": r["supports"], "verified": r["verified"]})
pd.DataFrame(rows)
""")
md(r"""
**A conflict.** For pump P-205, R-12 says inspect above 4.5 mm/s; the English manual says alarm at 7.1 mm/s. For
lubrication of P-101, R-12 says every 2000 h, the maintenance regulation R-15 every 3000 h. The system must show
both sides, not choose:
""")
code(r"""
for cid in ("f01", "f05"):
    rec = system_live.run_case(case_by_id[cid], CONFIGS["S6"])
    cited = sorted({by_id[c["chunk_id"]].doc_id for c in rec["checks"]["citations"] if c["chunk_id"]})
    print(f"{cid}: {rec['answer']['status']} | cited {cited} | {rec['answer']['answer'][:220]}")
""")
md(r"""
**Not in the documents.** Valve V-912 is in no document; valve V-112 is only in the registry. Both "not in the
documents" and "needs a person" are right here: no value is given.
""")
code(r"""
for cid in ("x01", "x03"):
    rec = system_live.run_case(case_by_id[cid], CONFIGS["S6"])
    print(cid, rec["answer"]["status"], "|", rec["answer"]["answer"][:160])
""")
md(r"""
**The contract in three versions.** An early version had no length limit: long answers were cut off, and the
model sometimes reasoned inside the answer. The final version adds limits and the `check` field. Thinking mode was
worse and three times slower. The table counts invalid answers, reasoning found in the answer, and answers handed
to a person.
""")
code(r"""
for name in ("S6:v1", "S6:v2", "S6:think"):
    if name not in scores:
        run(name)
pairs = {"S6:v2": "S6:v1", "S6": "S6:v2", "S6:think": "S6:v2"}
ba = reval.ablation({k: scores[k] for k in ("S6:v1", "S6:v2", "S6", "S6:think")}, pairs)
display(show(ba, ["config", "vs", "end-to-end", "e2e: fixed / broke", "e2e verdict", "false refusals",
                  "wrong or invented", "invalid", "leaks", "fallbacks"]))
show(pd.read_csv(REFERENCE_DIR / "contract_before_after.csv", index_col="new vs old"),
     ["invalid", "end-to-end old", "end-to-end new", "new vs old: fixed / broke", "s / q", "tokens out / q"])
""")
md(r"""
**What `check` shows.** Pump P-205 at 6.0 mm/s, where R-12 and the manual disagree (`f03`). At S6 the model sees
both and still picks R-12, by a rule nobody gave it. `check` does not fix this, but the log shows why, so a person
can fix the rule or the prompt. With dense search or the reranker on top, the answer is right.
""")
code(r"""
f03 = case_by_id["f03"]
for name in ("S6:v2", "S6"):
    rec = system_live.run_case(f03, CONFIGS[name])
    print(f"{name}: {rec['answer']['status']} | answer: {rec['answer']['answer'][:200]}")
    if rec.get("check"):
        print(f"      check (internal): {rec['check']}")
""")

md(r"""
The whole ladder with S6:
""")
code(r"""
full = reval.ablation(scores, previous)
show(full[full.config.isin([c.name for c in LADDER])], cols + ["correct + verified", "abstain ok", "conflict ok"])
""")
md(r"""
**How a row fails matters.** A refusal is safe but costs the engineer time; a wrong value goes into a decision.
S0 fails both ways, S6 almost only by refusing, and M3@S0 gives wrong and invented values.
""")
code(r"""
modes = ["false refusals", "wrong or invented", "right abstentions", "invalid", "fallbacks"]
show(full[full.config.isin(["S0", "S5", "S6", "M3@S0", "M3@S6"])], ["config", "end-to-end"] + modes)
""")
md(r"""
**Needs a person.** Six extra questions where the rule is in the documents but a fact is missing: a pump with no
class in the registry, a tag that matches two pumps, a measurement in one direction of three. The right answer
names what is missing. Giving a value here is the dangerous failure.
""")
code(r"""
from plant_assistant import DATA_DIR

needs_human = load_cases(DATA_DIR / "needs_human.jsonl")
rows = {}
for name in ["S6", "M3@S6", "M4@S6"] + (["M1@S6", "M2@S6"] if VECTORS is not None else []):
    recs = system.run(needs_human, CONFIGS[name], workers=4, progress=False)
    rows[name] = {r["case_id"]: (r.get("answer") or {}).get("status", "invalid") for r in recs}
pd.DataFrame(rows).assign(question=[c.question_en for c in needs_human])
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 8. The audit log and the regression gate

Half a year later, someone asks why the system said 5,0. The **audit record** keeps the question, its date, the
index version, the chunks with their revisions, the answer and the checks.
""")
code(r"""
import json

from plant_assistant.audit import audit_record

rec = next(r for r in records["M3@S6"] if r["case_id"] == "m01")
print(json.dumps(audit_record(rec, system.index_version, by_id), ensure_ascii=False, indent=1)[:1800])
""")
md(r"""
The **regression gate** is a pass/fail test on the question set: the new row must not be clearly worse and may
break at most two questions. Topic 6 runs it after every new revision or index.
""")
code(r"""
g = reval.gate(scores["S5"], scores["S6"])
print(f"S6 vs S5: {g['A right']} -> {g['B right']} of {g['n']}, fixed {g['B fixed']}, broke {g['B broke']} "
      f"{g['broken_cases']}, p = {g['p_value']}  ->  GATE {'PASS' if g['passed'] else 'FAIL'} {g['reasons']}")
""")

md(r"""
**Holdout.** 27 questions written by a model from the documents and four job roles only: operator, mechanic,
reliability engineer, auditor. The fixes were never tuned on them, so they show whether the gains carry over.
""")
code(r"""
holdout = load_cases(REPLAY_DIR.parent / "holdout.jsonl")
h_scores = {}
for name in [c.name for c in LADDER] + ["M3@S0", "M3@S6"]:
    h_scores[name] = reval.score(holdout, system.run(holdout, CONFIGS[name], progress=False), by_id)
show(reval.ablation(h_scores, previous), ["config", "vs", "recall@5", "evidence recall", "end-to-end",
                                           "e2e: fixed / broke", "e2e verdict"])
""")
md(r"""
- The plant steps still help on the holdout; applicability (S3) is the big step.
- Between the best rows the differences are within noise: 27 questions are too few.
- Everyday questions rarely ask about dates or conflicts, so a reranker alone does about as well there.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 9. Check yourself

Your search columns must equal the reference exactly; end-to-end from replay too. A live run may differ by a few
questions.
""")
code(r"""
mine = reval.ablation(scores, previous).set_index("config")
ref = reference.set_index("config")
common = [c for c in mine.index if c in ref.index]
pd.DataFrame({"recall@5 (yours)": mine.loc[common, "recall@5"], "recall@5 (reference)": ref.loc[common, "recall@5"],
              "end-to-end (yours)": mine.loc[common, "end-to-end"],
              "end-to-end (reference)": ref.loc[common, "end-to-end"]})
""")
md(r"""
**Success criteria**

- Your recall@5 and evidence recall equal `reference/ablation.csv`.
- The P-101 question at M3@S6: "not normal, inspect within 24 hours", all three checks passed.
- Valve V-912 gets no value; P-205 and the lubrication of P-101 get "conflict" with both documents cited.
""")
md(r"""
**What you built:** a question set, losses by link, five plant steps, an answer contract with checks, an audit
record and a regression gate.
""")
md(r"""
**Typical mistakes**

- Filtering by date after search: nothing is left.
- "In force now" instead of dates: a past incident gets today's rule.
- A tag normalized in the chunks but not in the question.
- "Dense is better" from two averages, without a paired comparison.
- A citation check that looks only at the clause number.
""")
md(r"""
**Optional tasks** (details in the README)

1. Read the scanned PDF with Docling, the reader of practice 3, and run S6.
2. Another embedding model through Ollama (local models).
3. Query fields extracted by the model instead of by rules.
4. A smaller evidence pack: 500, 1000, 2000 tokens.
5. Near-copies of a regulation without the station fields.
6. A document hierarchy in code: the stricter rule wins.
7. The index in a vector database (pgvector or Qdrant).
""")


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t) for kind, t in cells]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
