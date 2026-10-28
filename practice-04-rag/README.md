# Practice 4. Finding the right clause: diagnose and fix

Part of the course **ML in Industry 2026** (topic 4: retrieval-augmented generation). The project is the
**Plant Engineer Assistant**. After this practice it searches a plant archive for the clause that is not only
similar to the question but **applicable, in force, complete and provable**, measures where answers are lost, and
answers with citations that code checks.

The running example comes from lectures 1–4: *"After a bearing change, pump P-101 vibrates at 4.8 mm/s. Is this
normal?"* Search mechanisms (embeddings, hybrid search, rerankers) are taught in the Advanced NLP course; here they
are ready tools and rows of an ablation table. The practice is about the engineering around them:

1. a **retrieval set** built from the structure of the documents, with known evidence for every question;
2. the baseline is measured and **every lost answer is attributed to a link**: reading / search / context assembly / model;
3. links are **fixed one at a time** with plant knowledge; every fix is a row of the ablation table with a
   **paired comparison** (practice 2);
4. mechanisms are added as **rows of the same table**;
5. the answer follows a **contract**; code checks the citations; "not in the documents" and "conflict" are answers.

Everything you need is in this folder: synthetic documents, the store of practice 3, the retrieval set, the
lecturer's saved model answers (replay files) and reference results.

A second part of practice 4 will follow as a separate notebook: dense and hybrid search in depth (here they are only
rows of the ablation table), text answers graded by a judge, and a document-context step.

## What is inside

| Path | What it is |
|------|------------|
| `inputs/practice-03/store/` | the reference chunk store of practice 3 (R-12 rev. 2 and rev. 3, PS-101, DS-P-205) and its 50 questions |
| `inputs/practice-01/equipment_registry.csv` | the equipment registry of practice 1, unchanged |
| `data/docs/` | 12 new born-digital PDFs of the plant archive v2 (see below) |
| `data/documents.csv` | registry of the whole archive: dates of validity, scope, language (as from a document management system) |
| `data/equipment_registry.csv` | the registry of practice 4: practice 1 + station, motor power, machine class, datasheet, documents of the unit (P-205: DS-P-205 and OM-P-205) |
| `data/questions.jsonl` | the retrieval set: 143 questions with evidence and gold chunk ids |
| `data/holdout.jsonl`, `data/holdout_drafts.jsonl` | 27 holdout questions written by role (and all 75 drafts); `scripts/draft_holdout.py` |
| `data/ground_truth/` | page texts and the clause and table pages of the new documents |
| `data/replay/` | the lecturer's model calls: answers, LLM reranker rankings, cross-encoder scores, translations (`vectors/`: e5-mistral vectors, see "Dense search") |
| `reference/` | `results.md`, `ablation.csv`, `cost.csv`, slices, losses, per-question scores, `s0_failures.md`, the holdout (`holdout_ablation.csv`, `holdout_review.md`), the audit record of the P-101 answer, the store `reference/store/`, the executed notebook |
| `scripts/` | `generate_corpus.py`, `build_questions.py`, `build_reference.py`, `build_vectors.py`, `build_notebook.py`; content in `corpus_content.py`, `questions_content.py` |
| `src/plant_assistant/` | `corpus.py`, `normalize.py`, `search.py`, `query.py`, `applicability.py`, `filters.py`, `assemble.py`, `answer.py`, `verify.py`, `reval.py`, `audit.py`, `pipeline.py`; the client, the store and the statistics from practices 1–3 |

## The plant archive, version 2

All documents are **synthetic** and marked as teaching material: the plant, the regulations, the pumps, the valves
and the numbers are invented. Table 3 of R-12 and R-14 is in the style of ISO 20816 but its values are not from
the standard. 16 documents, 47 pages, 440 chunks (110 of them rows of one long table): a teaching scale, not
"thousands of pages".

| Document | What it is | Why it is here |
|----------|------------|----------------|
| R-12 rev. 2, rev. 3 | from practice 3 | the core; revisions (inspection limit of class II: 5.0 → 4.5 mm/s) |
| R-12 rev. 3, Amendment No. 1 | changes clause 4.4.1 (720 → 500 h) from 01.09.2026 | an amendment, not a revision: a question "on a date" has three answers |
| R-14 | the same title and clauses as R-12, for pump station No. 3, stricter Table 3 (class II: 3.5 / 5.6) | a near-duplicate: tells apart only by scope |
| R-15 | maintenance of centrifugal pumps, stations 2 and 3 | a distractor on the same topic; disagrees with R-12 / R-14 on lubrication (3000 h vs 2000 h) and alignment (0.08 vs 0.05 mm): a conflict in one language |
| R-20 | work requests in the CMMS: priorities and deadlines | clause 6.2 of R-12 says "priority high", R-20 says what it means |
| PS-101, DS-P-205 | datasheets from practice 3 | applicability: class, power |
| PS-102, PS-310, PS-311 | datasheets of pumps of stations 2 and 3 | P-310 falls under R-14, not R-12 |
| OM-P-205 (English) | manufacturer's manual of the imported pump P-205 | a conflict with R-12 (alarm 7.1 / trip 11.2 mm/s) and a second language |
| VL-2 | valve list of station 2: a table of 110 rows on 4 pages | rows as chunks; "allowed pressure for valve V-12" (lecture 1) |
| FS-07, LAB-03, WH-04 | fire safety, laboratory access, lubricant storage | realistic background |

New fields of a chunk (the schema of practice 3 is kept, fields are added): `effective_from` / `effective_to`,
`amendment`, `amended_by`, `scope_stations` / `scope_equipment_types` / `scope_tags`, `references` /
`referenced_by`, `table_no`, `row_key`, `doc_language`; `index_version` for the whole store.

**The equipment registry is aligned with R-12.** In the registry of practice 1, pump P-101 was at "Pump station 1",
while R-12 (scope: pump station No. 2) and its Table A.1 put it at station No. 2. Applicability depends on it, so
`data/equipment_registry.csv` of this practice moves P-101 to pump station 2. The rest of the practice 1 registry
is unchanged; new rows describe the equipment of R-12 and R-14.

**One answer of practice 3 changed.** Question c06 ("how often must periodic measurements with a portable vibration
meter be done") was 720 h in practice 3. The archive now has Amendment No. 1, in force from 01.09.2026: the answer
on the day the questions are asked (27.10.2026) is 500 h.

## The retrieval set

143 questions in the format of the practice 2 harness. Every question has a Russian wording (it goes to search and
to the prompt: the engineer of a Russian plant asks in Russian) and an English translation for display. Evidence is
written as locators (`"R-12 r3 cl 4.3.3"`, `"R-12 r3 t 3"`, `"VL-2 r4 t А.2 row V-12"`); the gold chunk ids are
computed by code (`cases.resolve`).

| Slice | N | Example |
|-------|---|---------|
| base | 50 | the questions of practice 3, translated into Russian |
| language | 20 | 20 of them in English, sent to search as they are |
| applicability | 12 | "P-310 vibrates at 4.0 mm/s: inspect?" (R-14: yes, 3.5) |
| point_in_time | 10 | the check interval of P-101 on 01.05.2024 / 01.06.2025 / now: 1000 / 720 / 500 h |
| incident | 3 | "10.02.2025, P-101, 4.8 mm/s, no inspection: a violation?" (no, rev. 2: 5.0) |
| what_changed | 4 | "how did the limit change in rev. 3?" |
| multi_evidence | 8 | "P-101 at 4.8 mm/s: what to do and by when?" (4.3.3 + Table 3 + 6.1 + 6.2) |
| cross_doc | 3 | "a request under clause 6.2 of R-12: taken into work within?" (R-20) |
| conflict | 9 | "the vibration limit of P-205?" (R-12 against the English manual); "how often to lubricate the bearings of P-101?" (R-12 against R-15, both in Russian) |
| table_row | 8 | "allowed pressure for valve V-12?" |
| tag_variant | 6 | "Р-101" (Cyrillic Р), "P101", "P 102" |
| not_in_documents | 10 | "valve V-912", "noise level of P-101" |

**The status "needs a person" has its own small set** (`data/needs_human.jsonl`, 6 questions, slice
`needs_human`, not in the 143): the norm is in the documents, but the answer depends on data that nobody gave the
system. P-215 has no power and no class in its registry card; "P-220" matches P-220A (class I) and P-220B
(class II); a question names no pump; spare pump P-216 has no station (R-12 or R-14?); the grease is named only in a
datasheet that is not in the archive; P-101 was measured in one direction of three. Only `needs_human` is right.

## The ladder and the mechanisms

| Row | Configuration |
|-----|---------------|
| S0 | BM25 over raw chunks, top-5, no filters, context = top-5 |
| S1 | + tag normalization (chunks and query) |
| S2 | + rows of long tables as chunks, mixed with the clauses |
| S2b | + rows of long tables in a separate collection: top-5 chunks + at most 2 rows (S3 and later build on S2b) |
| S3 | + applicability: registry → station, type, class, documents of the unit; scope filter before search; the tables of the linked documents and the class in the prompt |
| S4 | + revision filter by the date of the question (`effective_from <= date < effective_to`) |
| S5 | + evidence pack: references both ways, one more step back, whole sections, 2500 tokens |
| S6 | + answer contract and three citation checks |
| M3@S0, M3@S6 | an LLM reranker (course Qwen, listwise over the top-20) on S0 and on S6 |
| M1t | translation of an English question into Russian before search (the `language` slice) |
| M1, M2 @S0, @S6 | dense and hybrid search: **TODO**, see "Dense search" |

## Reproduce the practice (30–60 minutes)

### 1. Install

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone https://github.com/Industrial-AI-Research-Lab/course-ml-for-industry-data.git
cd course-ml-for-industry-data/practice-04-rag
uv sync --extra notebook --extra dev
uv run pytest        # no network: builds the store, checks the reference from replay
```

### 2. Choose a model

| Profile | Model | What you need |
|---------|-------|---------------|
| `course` (default) | Qwen 3.8 27B, course server, thinking off | URL and token from the lecturers |
| none | replay of the lecturer's run | nothing: the notebook switches to replay automatically |

```bash
cp .env.example .env          # fill in COURSE_LLM_*
uv run python -m plant_assistant.smoke_test course
```

The answers for all 143 questions in every configuration come from replay even with a live model (about a thousand
calls); single questions in steps 5 and 7 go to the live model. `PRACTICE_MODE=replay` forces replay.

### 3. Run the notebook

```bash
uv run jupyter lab notebooks/practice_04_rag.ipynb
```

Run all cells from top to bottom. Search, filters and evidence packs run live on your machine in milliseconds.

### 4. Rebuild everything (the lecturer, or you after a change)

```bash
uv run --extra generate python scripts/generate_corpus.py   # the PDFs and the ground truth (deterministic)
uv run python -m plant_assistant.corpus                      # store/
uv run python scripts/build_questions.py                     # data/questions.jsonl, data/needs_human.jsonl
uv run python scripts/build_reference.py --replay            # reference/ from replay; without --replay: live
uv run python scripts/build_notebook.py
```

### Dense search and the cross-encoder (M1, M2, M4)

On 28.10.2026 the course server has an embedding model and a cross-encoder for the lecturer's key; whether a student
token sees them is not known, so the notebook replays them:

- **M1** dense search: `iairlab/e5-mistral-7b-instruct`, 4096-dim, cosine in numpy (`search.DenseIndex`). The model is
  asymmetric: a question is embedded with the instruction of the profile (`query_prefix` in `models.toml`:
  `Instruct: Given a question about plant equipment, retrieve the regulation or manual passages that answer
  it\nQuery: `), a passage without it. The vectors of all 440 chunks and of every question wording that goes to
  search (the 143 questions, the holdout, the needs_human set, the M1t translations, each also with normalized tags)
  are in `data/replay/vectors/e5-mistral.npz` (float16, 4.9 MB) with a manifest (`.json`) that refuses vectors of
  another corpus and keeps the latency of each query embedding call.
- **M2** hybrid: reciprocal rank fusion of the BM25 and the dense top-20 (`search.rrf`, c = 60).
- **M4** cross-encoder: `iairlab/bge-reranker-v2-m3` via `POST <server root>/rerank` (not under `/v1`), one call per
  question over the top-20 of the row, the same passages as the LLM reranker (M3). It uses no chat tokens. Its
  answers are in `data/replay/crossencoder_course.jsonl`, replayed like the LLM calls.

To rebuild with your own key (or with Ollama: `--profile embed-bge-m3 --name bge-m3`, then change `VECTORS` in
`scripts/build_reference.py`):

```bash
uv run python -m plant_assistant.smoke_test embed-course
uv run python -m plant_assistant.smoke_test rerank-course
uv run python scripts/build_vectors.py --profile embed-course --name e5-mistral
uv run python scripts/build_reference.py        # live: saved requests are replayed, new ones go to the server
```

## Expected results

See `reference/results.md` (the full table with Wilson intervals, paired verdicts, slices and losses by link).

In short (lecturer's run, course Qwen 3.8 27B, thinking off, 143 questions; seconds = median per question;
calls = all model calls per question: chat, the question embedding of dense rows, the cross-encoder; tokens in =
chat prompt tokens only: the cross-encoder reads ~1500–1750 tokens per question of its own, no chat model):

| Row | recall@5 | evidence recall | end-to-end | paired verdict vs `vs` | s / q | calls / q | tokens in / q |
|-----|----------|-----------------|------------|------------------------|-------|-----------|---------------|
| S0 BM25, raw chunks | 50% | 44% | 51% | — | 1.5 | 1 | 843 |
| S1 + tag normalization | 53% | 47% | 55% | vs S0: +5 / −0, not detectable (p = 0.06); tag slice 0/6 → 5/6 | 1.5 | 1 | 864 |
| S2 + table rows, mixed | 56% | 48% | 54% | vs S1: +6 / −7: table rows 6/8 → 8/8, but short row chunks push datasheets out | 1.4 | 1 | 727 |
| S2b + table rows, separate collection | 54% | 49% | 56% | vs S1: +3 / −1: table rows 8/8, and all 7 broken questions that S1 answered come back | 1.5 | 1 | 839 |
| S3 + applicability (on S2b) | 67% | 68% | 71% | better (+24 / −2) | 1.7 | 1 | 1083 |
| S4 + date filter | 76% | 76% | 76% | better (+7 / −0) | 1.8 | 1 | 1135 |
| S5 + evidence pack | 76% | 88% | 86% | better (+14 / −0) | 3.1 | 1 | 2811 |
| S6 + contract and checks | 76% | 88% | 91% | better (+8 / −1): conflicts 0/9 → 8/9 | 4.8 | 1 | 3036 |
| M1@S0 dense (e5-mistral) instead of BM25 | 68% | 62% | 63% | better than S0 (+32 / −15, p = 0.02) | 2.0 | 2 | 1298 |
| M2@S0 hybrid BM25 + dense (RRF) | 63% | 53% | 60% | better than S0 (+23 / −10, p = 0.04) | 1.9 | 2 | 1073 |
| M3@S0 LLM reranker on BM25 | 76% | 73% | 71% | better than S0 (+32 / −4) | 3.5 | 2 | 2895 |
| M4@S0 cross-encoder on BM25 | 73% | 65% | 66% | better than S0 (+27 / −6) | 2.0 | 2 | 1341 |
| M1@S6 S6 with dense search | 81% | 93% | 93% | not different from S6 (+11 / −8) | 5.5 | 2 | 3386 |
| M2@S6 S6 with hybrid search | 80% | 92% | 91% | not different from S6 (+6 / −6) | 5.4 | 2 | 3384 |
| M3@S6 LLM reranker on S6 | 85% | 93% | 91% | not different from S6 (+5 / −5); evidence better (+7 / −0) | 6.4 | 2 | 4534 |
| M4@S6 cross-encoder on S6 | 84% | 93% | 90% | not different from S6 (+5 / −6); evidence better (+6 / −0) | 5.2 | 2 | 3295 |
| M1t translation, `language` slice | 9/18 | 9/18 | 13/20 | better than S0 on the slice (5/20) | 1.9 | 2 | 769 |

- At S0, 60 of 70 wrong answers are lost **in search**, 2 in assembly, 8 in the model, none in reading.
  At S6, 12 of 13 are still lost in search: mostly the English questions of the `language` slice.
- Point in time: 4/10 at S0, 7/10 with the reranker alone, 9/10 with the date filter (S4), 10/10 at S5.
- `conflict` (9 questions: R-12 against the English manual of P-205, and R-15 against R-12 / R-14 in Russian):
  0/9 up to S5 (the simple answer has no way to say "the documents disagree"), 8/9 at S6. The manual of P-205 is
  never found by BM25 from a Russian question; it reaches the context only because the registry links it to P-205
  (S3). f09 fails: the registry brings the tables of the manual, and the answer is in its clause 6.4.
- **The P-101 question is lost in search at S6:** with the archive v2, BM25 puts Table 3 at rank 6 for this
  wording, and the evidence pack has nothing to start from. With the reranker (M3@S6) it is answered with all
  four citations and all checks pass.
- **Holdout** (27 questions written by role by gpt-oss-120b, never shown the ladder; `reference/holdout_review.md`):
  S0 56%, S3 74%, S6 78%, M3@S0 85%, M3@S6 85% (Wilson intervals about ±15 points). On everyday questions the
  reranker alone is at least as good as the plant-side ladder; the ladder is what fixes versions, applicability
  and conflicts, and those are rare in the holdout. 27 questions cannot separate S6 from M3@S0.
- **Mechanisms against plant knowledge** (end-to-end by slice in `reference/results.md`). Instead of plant
  knowledge, every mechanism beats S0, and none comes close to S6 (best: M3@S0, 71% against 91%). They fix what is
  about words: dense search answers 17/20 English questions (translation M1t: 13/20) and 6/6 tag variants. They do
  not fix what is about the plant: dense search is 3/10 on `point_in_time`, 3/12 on `applicability` and 0/9 on
  `conflict` (S6: 10/10, 12/12, 8/9). On top of S6, no mechanism changes end-to-end detectably. Dense search on S6
  answers all 20 English questions and the last conflict (f09: it finds clause 6.4 of the manual), and loses a few
  base, table and applicability questions; the rerankers raise evidence recall (+7 / −0, +6 / −0), not end-to-end.
- **The cross-encoder against the LLM reranker:** most of the gain for a quarter of the time: +0.5 s and no chat
  tokens per question against +2 s and ~2000 chat tokens (on S0: 66% against 71%; holdout 81% against 85%).
- **The token limit of the contract:** with more sources in the context, the claims and citations of a long answer
  do not fit into 3000 tokens: 5 invalid answers at M1@S6 and M2@S6, 4 at M4@S6, 3 at M3@S6, 0 at S6 (mostly the
  P-205 datasheet questions; m01 at M1@S6). A limit on the number of claims and citations in the schema is the
  fix; it changes every S6 prompt, so it is not in this reference.
- Dense search alone invents more: 5 values on "not in the documents" questions (S0: 1). A nearest neighbour
  always exists.
- Holdout: M1@S0 59%, M2@S0 67%, M4@S0 81%, M1@S6 81%, M2@S6 78%, M4@S6 81% (S6 78%, M3@S0 85%): the rerankers alone
  are as good as the whole ladder on everyday questions; dense search alone is not.

**How the rows fail matters, not only how often.** A refusal ("not in the documents" or "needs a person" when the
answer is in the documents) is safe but not free: the engineer searches the archive by hand or waits for a person,
and a system that refuses a third of the time is not used. A wrong value is worse: it goes into a decision.
False refusals are out of the 123 questions with an answer (or a conflict) in the documents; wrong or invented out
of all 143 ("invented": a value where the answer is not in the documents); right abstentions out of the 20
"not in the documents" questions (`needs_human` is accepted there: no value is given). Invalid: no answer of the
schema. Every question is exactly one of: right, false refusal, wrong or invented, invalid.

| Row | end-to-end | false refusals | wrong or invented | right abstentions | invalid |
|-----|------------|----------------|-------------------|-------------------|---------|
| S0 | 73 | 42 (34%) | 28 (1 invented) | 19/20 | 0 |
| S2b | 80 | 35 (28%) | 27 (1 invented) | 19/20 | 1 |
| S3 | 102 | 19 (15%) | 22 (1 invented) | 19/20 | 0 |
| S4 | 109 | 18 (15%) | 16 (1 invented) | 19/20 | 0 |
| S5 | 123 | 11 (9%) | 9 | 20/20 | 0 |
| S6 | 130 | 12 (10%), 1 of them `needs_human` (k02) | 1 | 20/20 (x01: `needs_human`) | 0 |
| M3@S0 | 101 | 15 (12%) | 27 (3 invented: n03, n04, n07) | 17/20 | 0 |
| M3@S6 | 130 | 7 (6%) | 3 (1 invented) | 19/20 (x01: `needs_human`) | 3 |
| M1@S0 | 90 | 26 (21%) | 27 (5 invented) | 15/20 | 0 |
| M4@S0 | 94 | 22 (18%) | 27 (3 invented) | 17/20 | 0 |
| M1@S6 | 133 | 4 (3%) | 1 | 20/20 | 5 |
| M4@S6 | 129 | 8 (7%) | 2 | 20/20 | 4 |
| Holdout S6 (27) | 21 | 3/23 | 3 (1 invented) | 3/4 | 0 |
| Holdout M3@S0 | 23 | 2/23 | 2 (1 invented) | 3/4 | 0 |
| Holdout M3@S6 | 23 | 2/23 | 2 (1 invented) | 3/4 | 0 |

S6 and M3@S6 have the same end-to-end, but M3@S6 refuses less and fails more often by a wrong value or an invalid
answer. The reranker alone (M3@S0) refuses three times less than S0 and gives as many wrong values.

**Needs a person** (6 questions; cells: status (value)):

| Question | S5 (no status field) | S6 | M1@S6 | M2@S6 | M3@S6 | M4@S6 |
|----------|----------------------|----|-------|-------|-------|-------|
| nh01 P-215, no power, no class | answered (2,8) | invalid: needs_human, then the token limit | needs_human | needs_human | needs_human | needs_human |
| nh02 "P-220": P-220A or P-220B | not_in_documents | needs_human | needs_human | needs_human | needs_human | needs_human |
| nh03 no tag, station 2 | not_in_documents | not_in_documents | needs_human | needs_human | needs_human | needs_human |
| nh04 spare P-216, no station | answered (4,5) | answered (4,5) | needs_human | conflict | conflict | conflict |
| nh05 grease from a missing datasheet | not_in_documents | needs_human | needs_human | needs_human | needs_human | needs_human |
| nh06 one direction of three measured | answered (4,5) | needs_human | needs_human | needs_human | needs_human | needs_human |
| right | 0/6 | 3/6 | 6/6 | 5/6 | 5/6 | 5/6 |

The contract gives the system a way to stop; S5 picks a class or a station itself in half of the cases. The station
case (nh04) is the hardest: the system either applies R-12 or calls the two regulations a conflict, though only one
of them applies.

## Success criteria

- `store/manifest.json` has the same `index_version` as `reference/store/manifest.json`;
- the retrieval columns of your ablation table (recall@5, evidence recall, MRR) equal `reference/ablation.csv`;
  end-to-end from replay is equal too, a live run is within the interval;
- the P-101 question at M3@S6: "not normal, inspect within 24 hours", citations R-12 rev. 3 cl. 4.3.3, Table 3,
  cl. 6.1–6.2, all three citation checks pass (at S6 it is lost in search: the reason is the point of step 7);
- V-912: no value (`not_in_documents` or `needs_human`, both accepted); P-205 and the lubrication interval of P-101 at S6: `conflict` with both documents cited.

## Use it in your project

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
- `answer.py`: the contract (`Answer`, `CONTRACT_SCHEMA`, `CONTRACT_FORMAT`, `parse`);
- `verify.py`: the three citation checks (`check_answer`);
- `applicability.py`, if your answers depend on the equipment: it needs a registry in the format of
  `data/equipment_registry.csv`, with the `documents` column.

Text answers are not graded by code here: part 2 grades them with a judge.

## Optional tasks (for better understanding)

Each is a section you add to the notebook or a script; a team can take any of them into its project.

1. **Reading against search.** Build the R-12 part of the store from the Docling output of the scan (practice 3,
   `data/parsed/docling.jsonl`) and run S6 on it. How many losses move from "search" to "reading"?
2. **Your own embedder.** `bge-m3` against `qwen3-embedding:0.6b` via Ollama (`scripts/build_vectors.py`);
   paired comparison on the `language` slice; M1 against M1t (translation of the question).
3. **LLM extraction of the query fields against the rules.** `query.extract_llm` on the course model: where do the
   rules miss a date or a tag, and where does the model invent one?
4. **Context budget.** The evidence pack at 500 / 1000 / 2000 tokens (`Config.budget`): evidence recall and end-to-end.
5. **Near-duplicates without metadata.** Remove `scope_*` and count the P-310 questions answered from R-12.
6. **Document hierarchy.** Add a company standard above R-12 and the rule "the stricter requirement wins" in code;
   the `conflict` slice.
7. **A vector database.** Move the index to pgvector or Qdrant (installed from PyPI or a mirror: Docker Hub is not
   available from Russia), compare time and results with numpy.
8. **Your documents.** The process on your project documents: 20 questions from the structure, S0 and one ladder row.
