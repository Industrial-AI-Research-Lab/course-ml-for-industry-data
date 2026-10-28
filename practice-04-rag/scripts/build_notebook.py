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

The engineer asks again: *"After a bearing change, pump P-101 vibrates at 4.8 mm/s. Is this normal?"*
The answer is in regulation R-12, but the plant archive now has 16 documents: two revisions of R-12 and an
amendment, a near-copy of R-12 for another pump station, a maintenance regulation, a CMMS procedure, datasheets,
an English manual of an imported pump, a valve list with 110 rows and some unrelated instructions.

Search mechanisms (embeddings, hybrid search, rerankers) are taught in the Advanced NLP course. Here search is a
ready module, and we do the engineering around it:

1. a **retrieval set** built from the structure of the documents, with known evidence for every question;
2. measure the baseline and **attribute every lost answer** to a link: reading / search / context assembly / model;
3. **fix one link at a time** with plant knowledge, each fix is a row of an **ablation table** with a paired
   comparison (practice 2);
4. mechanisms from Advanced NLP are added as **rows of the same table**;
5. the answer follows a **contract**; code checks the citations; "not in the documents", "conflict" and "needs a person" are answers.

```
question ─► fields (tags, date) ─► registry: station, class ─► filter: scope, date ─► BM25 ─► evidence pack
                                                                                               │
                         audit log ◄── 3 citation checks ◄── answer contract (course Qwen) ◄───┘
```

## Steps and what you should see

| Step | What we do | Expected result |
|------|------------|-----------------|
| 0 | The final table | the ablation table S0-S6 and the mechanism rows |
| 1 | Setup | the course model answers; otherwise replay |
| 2 | Archive v2 | 16 documents, 440 chunks, new fields, `index_version` |
| 3 | Retrieval set | 143 questions in 12 slices, gold chunks computed by code; a holdout of 27 written by role |
| 4 | Baseline S0 | about half of the answers right; most losses in search |
| 5 | Ladder S1-S5 | tags, table rows (mixed or separate), applicability, dates, evidence pack: one row each |
| 6 | Mechanisms | a reranker and query translation as rows of the same table, with their cost |
| 7 | Answer and checks | P-101 answered with checked citations; a neighbour clause caught; conflict; abstention |
| 8 | Audit, regression, holdout | an audit record; the regression gate S6 vs S5; the ladder on questions written by role |
| 9 | Reproduce | the mandatory path, success criteria, optional tasks |

## Live or replay

Search, filters and evidence packs run **live** (milliseconds, no model). Model answers for all 143 questions
in every configuration come from the **replay** of the lecturer's run: identical requests get identical saved
answers. Single questions in steps 5 and 7 go to the course model live if it answers. Dense vectors
(e5-mistral) and the answers of both rerankers (LLM and cross-encoder) are replayed too.

## Common problems

| Problem | What to do |
|---------|-----------|
| `Mode: REPLAY` but you have the course token | check `COURSE_LLM_BASE_URL` and `COURSE_LLM_API_KEY` in `.env`; run `uv run python -m plant_assistant.smoke_test course` |
| `ReplayMiss` | you changed a prompt, the chunking, a filter or the questions: replay works only for unchanged requests. Run live, or undo the change |
| your table differs from `reference/ablation.csv` in retrieval columns | the store differs: compare `index_version` with `reference/store/manifest.json` |
| the answer is empty, `finish_reason = length` | thinking mode is on: keep `course` (thinking off) for the answer contract |
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 0. Where we are going

The final table of this practice, measured by the lecturer. At the end you will know where every number comes from.
Every row adds one change to the row in `vs` and is compared with it **in pairs, on the same questions**.
""")
code(r"""
import warnings

import pandas as pd
from IPython.display import Markdown, display

from plant_assistant import REFERENCE_DIR

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 250)
reference = pd.read_csv(REFERENCE_DIR / "ablation.csv").fillna("")
reference[["config", "vs", "recall@5", "evidence recall", "end-to-end", "correct + verified", "e2e verdict"]]
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 1. Setup

Profiles are in `models.toml`, secrets in `.env` (as in practices 1-3). The answers use the course model
Qwen 3.8 27B with thinking off. The embedding (`embed-course`, e5-mistral-7b-instruct) and cross-encoder
(`rerank-course`, bge-reranker-v2-m3) profiles are optional: on 28.10.2026 they answer for the lecturer's key, and
it is not known whether a student token sees them. The notebook uses the saved vectors (`data/replay/vectors/`) and
the saved cross-encoder answers.
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
          for task in ("answers", "rerank", "translate", "crossencoder")}
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

The store of practice 3 (R-12 rev. 2 and rev. 3, two datasheets) plus 12 new **synthetic** documents. They are
born-digital: reading was topic 3 and must not get in the way. This is a teaching scale (tens of pages, hundreds
of chunks), not "thousands of pages": the process is the same, the numbers are illustrative.

`build_corpus` reads the new PDFs, takes dates and scope from the document registry (`data/documents.csv`, as from
a document management system), adds the new fields and writes `store/`.
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
**What is new, and why.** An **amendment** to R-12 rev. 3 changes one clause (4.4.1: 720 h → 500 h from
01.09.2026). **R-14** is a near-copy of R-12 for pump station No. 3, with the same title and clause numbers and a
stricter Table 3. The **manual of P-205** is in English and gives milder limits than R-12: a conflict.
The **valve list** has a table of 110 rows over four pages. R-20 defines what "priority high" in clause 6.2 of
R-12 means.

> **Chat:** "R-12 and R-14 have the same title. Name one field that tells them apart."
""")
code(r"""
from plant_assistant.display import where

for cid in ("R-12-r3-table1", "R-14-r1-table1"):
    c = by_id[cid]
    print(f"{where(c)}  scope_stations={c.scope_stations}  valid {c.effective_from} → {c.effective_to or 'now'}")
    print("   ", c.text.splitlines()[0], "|", [line for line in c.text.splitlines() if "|II|" in line][0])
""")
md(r"""
New fields of a chunk (the practice 3 schema is kept, the fields are added):
`effective_from` / `effective_to` (validity), `amended_by`, `scope_stations` / `scope_equipment_types` /
`scope_tags`, `references` / `referenced_by` (links in both directions), `table_no`, `row_key`, `doc_language`.
""")
code(r"""
fields = ["chunk_id", "clause", "table_no", "row_key", "effective_from", "effective_to", "amended_by", "references",
          "referenced_by"]
pd.DataFrame([{f: getattr(by_id[cid], f) for f in fields} for cid in
              ("R-12-r3-4.4.1", "R-12-r3-a1-4.4.1", "R-12-r3-4.3.3", "R-12-r3-6.2", "R-20-r2-3.3",
               "VL-2-r4-tА.2-V-12")])
""")
md(r"""
The index has a version: a fingerprint of the chunks, the parser and the embedding model. It goes into every
answer log (step 8). If any of the three changes, the index is rebuilt and the retrieval set runs again (topic 6).
""")
code(r"""
print("index_version:", summary["index_version"])
pd.crosstab(pd.Series([c.doc_id for c in chunks], name="document"), pd.Series([c.chunk_type for c in chunks],
                                                                              name="chunk type"), margins=True)
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 3. The retrieval set

143 questions in the format of the practice 2 harness. The 50 questions of practice 3 (known page) are the
`base` slice; 20 of them also go to search **in English as they are** (the `language` slice); the rest are new.
Every question has a Russian wording (it goes to search and to the prompt: the engineer of a Russian plant asks in
Russian) and an English translation for us. Evidence is written as **locators**; the gold chunk ids are computed
by code. Slices are reported as **counts**: 3 of 4 is not "75%".
""")
code(r"""
from plant_assistant.cases import counts_by_slice, load_cases

cases = load_cases()
case_by_id = {c.case_id: c for c in cases}
pd.DataFrame([counts_by_slice(cases)], index=["questions"])
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
Four evidence chunks for one answer: the clause with the deadline (4.3.3), the table with the limit (Table 3),
and the actions (6.1, 6.2). recall@k counts only the answer; **evidence recall** asks for all four in the context.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 4. Baseline S0: BM25 over raw chunks, top-5, no filters

> **Predict, then run.** Where will most answers be lost: reading, search, context assembly or the model?

Search runs live; the answers of the course model for 143 questions come from replay.
""")
code(r"""
from plant_assistant import reval
from plant_assistant.display import answer_frame, context_frame, hits_frame
from plant_assistant.pipeline import CONFIGS, LADDER, System

system = System(chunks, answer_client=replay["answers"], rerank_client=replay["rerank"],
                translate_client=replay["translate"], cross_client=replay["crossencoder"], vectors=VECTORS)
records, scores = {}, {}


def run(name: str) -> pd.DataFrame:
    records[name] = system.run(cases, CONFIGS[name], progress=False)
    scores[name] = reval.score(cases, records[name], by_id)
    return scores[name]


run("S0")
reval.ablation({"S0": scores["S0"]}, {})[["config", "recall@5", "evidence recall", "MRR", "end-to-end",
                                          "correct + verified", "abstain ok", "conflict ok"]]
""")
code(r"""
display(Markdown("**Wrong answers of S0 by link**"))
pd.DataFrame([reval.losses(scores["S0"])], index=["S0"])
""")
md(r"""
Most answers are lost **in search**: the chunk with the answer never reached the model. Nothing is lost in
reading (the documents are born-digital and the practice 3 store was read well). In practice 3 the model answered
98% of these questions when it was given the page: everything below that is lost by search and assembly here.

The same view for the P-101 question: the top-5 of BM25.
""")
code(r"""
rec = next(r for r in records["S0"] if r["case_id"] == "m01")
hits_frame(rec, P101, by_id)
""")
md(r"""
No word of the question is in clause 4.3.3 ("при превышении границы осмотра ... осматривают в течение 24 часов"):
lexical search cannot find it from "вибрирует 4,8 мм/с, это норма?". Only Table 3 comes up, by "мм/с".

**Task 1 of the lecture** used five failures of exactly this run:
""")
code(r"""
display(Markdown((REFERENCE_DIR / "s0_failures.md").read_text(encoding="utf-8")))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 5. The ladder: one fix per row

Every row adds one change to the previous one and is compared with it in pairs. We look at one question before and
after, then at the row of the table.

**S1: tag normalization.** "Р-101" with a Cyrillic Р, "P101", "P 102": one form in the chunks **and** in the query.
""")
code(r"""
for name in ("S0", "S1"):
    rec = system.run_case(case_by_id["g02"], CONFIGS[name], with_answer=False)
    display(Markdown(f"**{name}** · {case_by_id['g02'].question_ru}"), hits_frame(rec, case_by_id["g02"], by_id))
""")
md(r"""
**S2: rows of long tables are chunks.** The valve list has 110 rows on 4 pages. A page of 33 rows is one chunk in
S0-S1; in S2 every row is a chunk with the caption, the column names and the units:
""")
code(r"""
print(by_id["VL-2-r4-tА.2-V-12"].text)
for name in ("S1", "S2"):
    rec = system.run_case(case_by_id["r01"], CONFIGS[name], with_answer=False)
    display(Markdown(f"**{name}** · {case_by_id['r01'].question_ru}"), hits_frame(rec, case_by_id["r01"], by_id, 3))
""")
md(r"""
**S2b: rows in a separate collection.** 110 short row chunks mixed with the clauses win BM25 for every question
that names a pump (the rows say "нагнетание P-101"), and push datasheets out of the top-5. Instead: rows are
searched in their own index and get a small quota (top-5 chunks + at most 2 rows that share a term with the
question). The questions that mixed rows broke in the lecturer's first run:
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
**S3: applicability through the registry.** The tag goes to the equipment registry: station, type, class and the
**documents of the unit** (datasheet, manufacturer's manual). The filter keeps only documents whose scope fits,
**before** search; the class goes into the prompt; the tables of the linked documents go into the context. P-310
is at station No. 3, so R-14 applies, not R-12. The manual of P-205 does not name the plant tag (the manufacturer
does not know it): only the registry links it.
""")
code(r"""
from plant_assistant.applicability import load_registry

print(load_registry()["P-205"].describe())
rec = system.run_case(case_by_id["f01"], CONFIGS["S3"], with_answer=False)
print("linked by the registry:", [where(by_id[c]) for c in rec["linked"]])
""")
code(r"""
from plant_assistant.applicability import equipment_context

# single questions: live if the course model answers (the reranker too), replay otherwise
system_live = System(chunks, answer_client=live, rerank_client=live if LIVE else replay["rerank"],
                     translate_client=live if LIVE else replay["translate"], cross_client=replay["crossencoder"],
                     vectors=VECTORS)
for name in ("S2", "S3"):
    rec = system_live.run_case(case_by_id["p01"], CONFIGS[name])
    display(Markdown(f"**{name}** · {case_by_id['p01'].question_ru} · scope: {rec['scope'] or 'none'}"),
            hits_frame(rec, case_by_id["p01"], by_id, 3))
    print("answer:", (rec["answer"] or {}).get("answer"))
""")
md(r"""
**S4: a revision filter by the date of the question.** `effective_from <= date < effective_to`, not
`status = in force`. The same question on three dates: 1000 h (rev. 2), 720 h (rev. 3), 500 h (amendment No. 1).
At S4 the old wordings are gone, but the amendment is not in the top-5 for "now" (its text differs from the
question): the model says "not in the documents". The evidence pack of S5 brings it as a part of section 4.4.
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
**Typical mistakes, measured.** A filter *after* search empties the top-k; the flag `status = in force` answers a
question about an incident in February 2025 with rev. 3.
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
**S5: the evidence pack.** For every hit: the chunks it refers to and the chunks that refer to it (6.2 refers to
4.3.3, not the other way round), one more step back, then the whole sections, within 2500 tokens.
""")
code(r"""
for name in ("S4", "S5"):
    rec = system.run_case(P101, CONFIGS[name], with_answer=False)
    ctx = context_frame(rec, P101, by_id)
    print(f"{name}: {len(ctx)} chunks, {ctx.tokens.sum()} tokens; gold in the context: "
          f"{sorted(set(ctx.gold) - {''})}, evidence recall = {reval.retrieval_metrics(P101, rec)['evidence_recall']}")
ctx[ctx.gold != ""]
""")
md(r"""
> **Poll:** "Which fix will give the biggest gain on the whole set? Tag normalization / Table rows /
> Applicability / Revision filter / Evidence pack"

The rows of the table, each against the previous one:
""")
code(r"""
for name in ("S3", "S4", "S5"):
    run(name)
cols = ["config", "vs", "recall@5", "evidence recall", "end-to-end", "e2e: fixed / broke", "e2e verdict",
        "evidence verdict"]
reval.ablation(scores, previous)[cols].fillna("")
""")
code(r"""
display(Markdown("**End-to-end by slice** (counts)"))
reval.by_slice(scores, "correct")
""")
md(r"""
Read the table by slices, not only by the total: S2 fixes table rows and **breaks** base questions, S2b keeps
the fix without the damage; S3 moves `applicability` and `base`, S4 `point_in_time` and `incident`, S5
`multi_evidence` and `cross_doc`. The `conflict` slice stays at 0 until S6: without the contract the model has no
way to say "the documents disagree".
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 6. Mechanisms from Advanced NLP: rows of the same table

Two points of attachment: **instead of** plant knowledge (on S0) and **on top of** it (on S6). We do not explain
how they work; we look at where they help.

- **M1** dense search (e5-mistral-7b-instruct on the course server, saved vectors) instead of BM25;
- **M2** hybrid search: BM25 and dense top-20 fused by reciprocal rank (RRF);
- **M3** an LLM reranker (listwise, course Qwen) over the top-20 of the current search;
- **M4** a cross-encoder (bge-reranker-v2-m3) over the same top-20: one `/rerank` call, no chat tokens;
- **M1t** translation of an English question into Russian before search (the `language` slice only).

> **Chat:** "Will a reranker fix the point-in-time slice? yes / no"
""")
code(r"""
MECH = ["S0", "M1@S0", "M2@S0", "M3@S0", "M4@S0", "S6", "M1@S6", "M2@S6", "M3@S6", "M4@S6"]
for name in ["M1t"] + MECH:
    if name not in scores and (VECTORS is not None or CONFIGS[name].retriever == "bm25"):
        run(name)
table = reval.ablation(scores, previous)
table[table.config.isin(MECH + ["M1t"])][cols].fillna("")
""")
code(r"""
focus = ["language", "point_in_time", "applicability", "conflict", "table_row", "tag_variant", "multi_evidence"]
reval.by_slice({k: scores[k] for k in MECH if k in scores}, "correct")[focus]
""")
md(r"""
**What each row costs.** Median seconds per question (search here + model latency saved with the answers), model
calls (all: chat, the question embedding, the cross-encoder), chat calls and tokens per question. A row that is
better must also be affordable: the LLM reranker adds a chat call with ~1800 prompt tokens; the cross-encoder adds
one short call and no chat tokens.
""")
code(r"""
reval.cost_table(records)
""")
code(r"""
lang = scores["M1t"].case_id
display(Markdown("**The `language` slice**: English questions as they are (S0), translated to Russian (M1t), "
                 "dense search on the English wording (M1@S0)"))
pd.DataFrame({name: [f"{int(scores[name].set_index('case_id').loc[lang, m].astype(bool).sum())}/{len(lang)}"
                     for m in ("recall@k", "correct")] for name in ("S0", "M1t", "M1@S0") if name in scores},
             index=["recall@5", "end-to-end"])
""")
md(r"""
**What the mechanism rows say** (the lecturer's run; compare with your table):

- **Instead of plant knowledge** every mechanism beats S0, and none comes close to S6 (best: the LLM reranker,
  71% against 91%). They fix what is about words: dense search answers 17/20 English questions (translation:
  13/20) and all tag variants. They do not fix what is about the plant: dense search is 3/10 on `point_in_time`,
  3/12 on `applicability` and 0/9 on `conflict`. Versions, applicability and conflicts need the filters, the
  registry and the contract.
- **On top of plant knowledge** no mechanism changes end-to-end detectably (M1@S6 95%, +13 / −5). Dense search
  answers all English questions and finds the last conflict; the rerankers raise evidence recall, not end-to-end.
- **The cross-encoder** gives most of the reranker gain for a quarter of the time: +0.5 s and no chat tokens per
  question, against +2 s and ~2000 chat tokens for the LLM reranker.
- **New failures come with new context:** with the first answer contract, 5 answers of M1@S6 did not fit into the
  token limit (long lists of claims and citations); contract v2 (step 7) removes them. Dense search alone invents 5
  values where the answer is "not in the documents" (S0: 1). A nearest neighbour always exists.

> **Discussion:** "Your project has 300 documents in two languages and versions. Which row do you build first, and
> what do you measure to decide whether a second one is worth its cost?"
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 7. The answer contract and the citation checks

S6 = S5 + the answer contract: a status (`answered`, `not_in_documents`, `conflict`, `needs_human`), the value with
its unit, and at most 5 claims, each with at most 2 sources. A source is the **id of a chunk in the context**
(`R-12-r3-4.3.3`), and the JSON schema of the question makes it an `enum` of exactly the ids that were put into the
prompt: the model cannot cite a chunk it did not get (guided decoding on the course server enforces it; code checks
it anyway). Code fills the document, revision, clause, table and page from the index. An answer cut by the token
limit is asked once more with a short instruction; if it fails again, the answer is "needs a person" (a fallback,
counted). Three checks of every citation:

1. **exists**: the cited chunk was in the context (and so in the index);
2. **in force**: it is valid on the date of the question;
3. **supports**: the number and the unit of the answer are in its text (or in a table it refers to).

The P-101 question. With the archive v2, BM25 puts Table 3 at rank 6 for this wording (a lubrication clause of
R-15 shares "подшипник" and "насос"), so S6 has nothing to build the evidence pack from. The reranker over the
top-20 brings Table 3 back, and the pack does the rest:
""")
code(r"""
for name in ("S6", "M3@S6"):
    rec = system_live.run_case(P101, CONFIGS[name])
    print(f"{name}: top-5 {[where(by_id[c]) for c in rec['top']]}")
    print(f"     status {rec['answer']['status']}, correct {reval.grade(P101, rec['answer'])[0]}, "
          f"verified {rec['checks']['verified']}")
answer, checks = answer_frame(rec)
display(answer)
checks
""")
md(r"""
> **Chat:** "The answer cites R-12 rev. 2. Which check fails?"

Task 3 of the lecture: *"Inspect P-101 within 24 hours (R-12, rev. 3, clause 4.3.4, page 3)"*. The clause exists
and is in force; its text is about the **stop** limit, "24 hours" is in the neighbouring clause 4.3.3. The enum
makes sure the cited chunk was in the context; it does not make it the right one: the checks still matter.
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
**Conflict.** P-205 is a class II pump at station No. 2: R-12 says inspect above 4.5 mm/s; the manufacturer's
manual says "alarm 7.1 mm/s". BM25 never finds the English manual from a Russian question, but the registry links
it (S3), so both sides are in the context. In the same language: R-15 (maintenance) says lubricate every 3000 h,
R-12 says 2000 h, both in force for station No. 2. The system must show both, not choose:
""")
code(r"""
for cid in ("f01", "f05"):
    rec = system_live.run_case(case_by_id[cid], CONFIGS["S6"])
    cited = sorted({by_id[c["chunk_id"]].doc_id for c in rec["checks"]["citations"] if c["chunk_id"]})
    print(f"{cid}: {rec['answer']['status']} | cited {cited} | {rec['answer']['answer'][:220]}")
""")
md(r"""
**Not in the documents.** V-912 is in no document (V-112 is in the registry but in no document either). Both
`not_in_documents` and `needs_human` are right here: no value is given.
""")
code(r"""
for cid in ("x01", "x03"):
    rec = system_live.run_case(case_by_id[cid], CONFIGS["S6"])
    print(cid, rec["answer"]["status"], "|", rec["answer"]["answer"][:160])
""")
md(r"""
**The contract before and after.** The first version (rows `S6:v1`, `M*@S6:v1`, replayed from the first reference
run) let the model copy citation objects without limits: with a long context, the claims and citations did not fit
into 3000 tokens and the answer was invalid. Same search, same context, only the contract changes. v2 removes every
invalid answer (no retry was even needed: the longest answer is 666 tokens) and halves the output tokens. It also
costs something: with a short answer the model no longer "thinks aloud" in the answer field, and two questions break
in most rows (f03: a conflict answered with one side; n04: "not in the documents" answered with a neighbouring
tolerance).

> **Discussion:** "The old contract was right on f03 because the model reasoned inside the answer. How would you
> give it room to check for a conflict without letting the answer grow without limit?"
""")
code(r"""
for name in ("S6:v1", "M1@S6:v1", "M1@S6") if VECTORS is not None else ("S6:v1",):
    if name not in scores:
        run(name)
pairs = {"S6": "S6:v1", "M1@S6": "M1@S6:v1"} if VECTORS is not None else {"S6": "S6:v1"}
ba = reval.ablation({k: scores[k] for p in pairs.items() for k in p[::-1]}, pairs)
display(ba[["config", "vs", "end-to-end", "e2e: fixed / broke", "e2e verdict", "false refusals", "wrong or invented",
            "invalid", "fallbacks"]].fillna(""))
pd.read_csv(REFERENCE_DIR / "contract_before_after.csv", index_col="row")
""")
md(r"""
The whole ladder with S6:
""")
code(r"""
full = reval.ablation(scores, previous)
full[full.config.isin([c.name for c in LADDER])][cols + ["correct + verified", "abstain ok", "conflict ok"]].fillna("")
""")
md(r"""
**How a row fails matters, not only how often.** Every question is one of: right, a **false refusal** ("not in
the documents" or "needs a person" when the answer is in the documents), a **wrong or invented** value, or an
invalid answer. A refusal is safe but not free: the engineer searches the archive by hand or waits for a person,
and a system that refuses a third of the time is not used. A wrong value is worse: it goes into a decision. Compare
S0 (fails both ways), S6 (fails almost only by refusing) and M3@S0 (refuses less, but gives wrong values and invents
values where the answer is "not in the documents"):
""")
code(r"""
modes = ["false refusals", "wrong or invented", "right abstentions", "invalid", "fallbacks"]
full[full.config.isin(["S0", "S5", "S6", "M3@S0", "M3@S6"])][["config", "end-to-end"] + modes]
""")
md(r"""
**Needs a person.** The fourth status has its own small set (`data/needs_human.jsonl`, not in the 143): the norm
is in the documents, but the answer depends on data that nobody has given the system: a pump with no power and no
class in its card, a tag that matches two pumps (P-220A and P-220B), no tag at all, a spare pump without a station
(R-12 or R-14?), a grease named only in a missing datasheet, a measurement in one direction of three. The right
answer names what is missing and who has it. A value here is the dangerous failure: the system chose the class or
the station itself. On the retrieval set, `needs_human` is accepted on a "not in the documents" question (x01:
no value is given) and counted as a false refusal on a question with an answer (k02 at S6: R-12 Table 3 did not
reach the context).
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

Half a year later, "why did the system say 5,0?" must have an answer. The record keeps the question, its date,
the index version, the chunks with their revisions, the answer and the checks.
""")
code(r"""
import json

from plant_assistant.audit import audit_record

rec = next(r for r in records["M3@S6"] if r["case_id"] == "m01")
print(json.dumps(audit_record(rec, system.index_version, by_id), ensure_ascii=False, indent=1)[:1800])
""")
md(r"""
The retrieval set is a regression test: the gate of practice 2 (not significantly worse, at most 2 questions
broken) on S6 against S5. Topic 6 puts this into CI and runs it after every new revision or index version.
""")
code(r"""
g = reval.gate(scores["S5"], scores["S6"])
print(f"S6 vs S5: {g['A right']} -> {g['B right']} of {g['n']}, fixed {g['B fixed']}, broke {g['B broke']} "
      f"{g['broken_cases']}, p = {g['p_value']}  ->  GATE {'PASS' if g['passed'] else 'FAIL'} {g['reasons']}")
""")

md(r"""
**Holdout: are the fixes fitted to our own slices?** The retrieval set was written together with the fixes. The
holdout has 27 questions written by a model (gpt-oss-120b on the course server) from the documents and four role
descriptions only: an operator, a mechanic, a reliability engineer, an auditor. It never saw the ladder or the
slices. Code kept the gradable ones (the review list: `reference/holdout_review.md`).
""")
code(r"""
holdout = load_cases(REPLAY_DIR.parent / "holdout.jsonl")
h_scores = {}
for name in [c.name for c in LADDER] + ["M3@S0", "M3@S6"]:
    h_scores[name] = reval.score(holdout, system.run(holdout, CONFIGS[name], progress=False), by_id)
reval.ablation(h_scores, previous)[["config", "vs", "recall@5", "evidence recall", "end-to-end", "e2e: fixed / broke",
                                    "e2e verdict"]].fillna("")
""")
md(r"""
On the holdout the plant-side ladder still helps (S3 is the big step), but the reranker alone on raw BM25 (M3@S0)
is at least as good as S6, and 27 questions cannot tell them apart. The honest reading: plant data are what fix versions,
applicability and conflicts, and those questions are rare in a sample of everyday questions; a reranker fixes
wording. Measure both on your own questions, and look at the cost columns before you choose.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 9. Check yourself and reproduce

**Your table against the reference.** Retrieval columns must match exactly (BM25 and filters are deterministic);
end-to-end from replay matches too, a live run may differ by a few questions (look at the intervals).
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

- the retrieval columns of your ablation table equal `reference/ablation.csv`;
- the P-101 question at M3@S6: "not normal, inspect within 24 hours", citations R-12 rev. 3 cl. 4.3.3, Table 3,
  cl. 6.1-6.2, all three checks pass (at S6 with BM25 alone it is lost in search: see step 7);
- V-912: no value (`not_in_documents` or `needs_human`); P-205 and the lubrication interval of P-101: `conflict` with both documents cited.

**What you built:** a retrieval set with evidence from the structure of the documents, loss attribution by link,
tag normalization, table rows, applicability through the registry, a date filter that understands amendments,
an evidence pack with references in both directions, an answer contract with three citation checks, an audit
record with the index version, and the regression gate. In the demo project this becomes the search tool of the
agent (topic 5), the regression test of the index (topic 6) and the audit log (topics 6 and 7).

**Typical mistakes** (check before you trust a number):
- the revision filter after search: the top-k is full of the old revision, after the filter nothing is left;
- `status = in force` instead of the dates: a wrong answer to a question about an incident;
- amendment No. 1 not applied: clause 4.4.1 of rev. 3 looks valid;
- the tag normalized in the chunks but not in the question (or the other way round);
- the date filter on for a "what changed" question: rev. 2 disappears;
- "dense is better than BM25" from two averages without a paired comparison; a percentage on a slice of 4;
- a citation check that only looks for the clause: the neighbouring clause 4.3.4 passes;
- `question_en` sent to search instead of `question_ru`: BM25 finds only tags and numbers;
- a new embedding profile with the old vectors: the manifest (`index_version`) must catch it;
- the course model in thinking mode spent the token limit before the JSON of the contract.

**Optional tasks** (see the README for details):
1. Reading against search: S6 on a store read from the scan by Docling (practice 3). How many losses move to "reading"?
2. Your own embedder: `bge-m3` against `qwen3-embedding:0.6b` via Ollama, paired comparison on the `language` slice.
3. LLM extraction of the query fields (`query.extract_llm`) against the rules: where do the rules miss a date or a tag?
4. Context budget: the evidence pack at 500 / 1000 / 2000 tokens: evidence recall and end-to-end.
5. Near-duplicates without metadata: remove `scope` and count the P-310 questions answered from R-12.
6. Document hierarchy: a company standard above R-12 and the rule "the stricter wins" in code; the `conflict` slice.
7. A vector database: the index in pgvector or Qdrant (installed from PyPI or a mirror); time and result.
8. Your documents: the process on your project documents: 20 questions from the structure, S0 and one ladder row.
""")
md(r"""
**Use it in your project**

The same process works on the documents of your project. Do it step by step:

1. Write 20–50 questions from the roles of your users. For each question, note the answer type (number, short
   fact, text, not in the documents). Note the gold evidence too (document, revision, clause or page). Take it from
   the structure of the document, not from search results.
2. Add slices. Always add "not in the documents" (at least 5 questions). Then add the slices that matter for your
   data: versions, near-duplicate documents, tables, languages.
3. Build the baseline: BM25 over your chunks, top 5, the answer contract with citations. Measure recall@5,
   evidence recall, end-to-end, false refusals and invented values.
4. Count the losses by link with `reval.py`: reading, search, assembly, model.
5. Fix the biggest link with one step. Compare the new row in pairs with the row above (practice 2 statistics).
6. Keep the ablation table. It is your evidence for the search criterion at the defense.
7. Run the set again after every change of documents, parser or model.

Files to copy (from `src/plant_assistant/`):
- `cases.py`: the question format, `TestCase` with `expected` and `evidence` locators, turned into chunk ids by
  `resolve`;
- `reval.py`, with `units.py`, `evals/stats.py` and `Chunk` from `store.py`: grading, metrics, losses by link,
  failure modes, the ablation table;
- `answer.py`: the contract (`Answer`, `contract_v2_format` with the enum of context ids, `parse`,
  `fallback_answer`), and `System.answer` in `pipeline.py` for the retry and the fallback;
- `verify.py`: the three citation checks (`check_answer`);
- `applicability.py`, if your answers depend on the equipment: it needs a registry in the format of
  `data/equipment_registry.csv`, with the `documents` column.

Text answers are not graded by code here: part 2 grades them with a judge.
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
