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

## What is inside

| Path | What it is |
|------|------------|
| `inputs/practice-03/store/` | the reference chunk store of practice 3 (R-12 rev. 2 and rev. 3, PS-101, DS-P-205) and its 50 questions |
| `inputs/practice-01/equipment_registry.csv` | the equipment registry of practice 1, unchanged |
| `data/docs/` | 12 new born-digital PDFs of the plant archive v2 (see below) |
| `data/documents.csv` | registry of the whole archive: dates of validity, scope, language (as from a document management system) |
| `data/equipment_registry.csv` | the registry of practice 4: practice 1 + station, motor power, machine class, datasheet |
| `data/questions.jsonl` | the retrieval set: 138 questions with evidence and gold chunk ids |
| `data/ground_truth/` | page texts and the clause and table pages of the new documents |
| `data/replay/` | the lecturer's model calls: answers, reranker rankings, translations (`vectors/`: embedding vectors, see "Dense search") |
| `reference/` | `results.md`, `ablation.csv`, slices, losses, per-question scores, `s0_failures.md`, the audit record of the P-101 answer, the store `reference/store/`, the executed notebook |
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
| R-15 | maintenance of centrifugal pumps, stations 2 and 3 | a distractor on the same topic, refers to R-12 and R-14 |
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

138 questions in the format of the practice 2 harness. Every question has a Russian wording (it goes to search and
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
| conflict | 4 | "the vibration limit of P-205?" (R-12 against the manual) |
| table_row | 8 | "allowed pressure for valve V-12?" |
| tag_variant | 6 | "Р-101" (Cyrillic Р), "P101", "P 102" |
| not_in_documents | 10 | "valve V-912", "noise level of P-101" |

## The ladder and the mechanisms

| Row | Configuration |
|-----|---------------|
| S0 | BM25 over raw chunks, top-5, no filters, context = top-5 |
| S1 | + tag normalization (chunks and query) |
| S2 | + rows of long tables as chunks |
| S3 | + applicability: registry → station, type, class; scope filter before search; class in the prompt |
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

The answers for all 138 questions in every configuration come from replay even with a live model (about a thousand
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
uv run python scripts/build_questions.py                     # data/questions.jsonl
uv run python scripts/build_reference.py --replay            # reference/ from replay; without --replay: live
uv run python scripts/build_notebook.py
```

### Dense search (M1, M2): not measured yet

On 27.10.2026 the course LiteLLM has no embedding and no rerank model, and the machine used for the reference had no
Ollama, so **the rows M1 and M2 of the reference table are TODO**. The code is ready: `search.DenseIndex` (cosine in
numpy over saved vectors), `search.HybridIndex` (RRF), `search.VectorStore` (vectors with a manifest that refuses
vectors of another corpus or model). To fill the rows:

```bash
ollama pull bge-m3                                          # ~1.2 GB, from ollama.com, no Hugging Face
uv run python -m plant_assistant.smoke_test embed-bge-m3
uv run python scripts/build_vectors.py --profile embed-bge-m3 --name bge-m3
uv run python scripts/build_reference.py --configs S0 S1 S2 S3 S4 S5 S6 M1@S0 M2@S0 M1@S6 M2@S6
```

The reranker (M3) works today: it is an LLM reranker on the course model (one listwise call per question over the
top-20); its answers are in `data/replay/rerank_course.jsonl`.

## Expected results

See `reference/results.md` (the full table with Wilson intervals, paired verdicts, slices and losses by link).

In short (lecturer's run, course Qwen 3.8 27B, thinking off, 138 questions):

| Row | recall@5 | evidence recall | end-to-end | paired verdict vs the previous row |
|-----|----------|-----------------|------------|-------------------------------------|
| S0 BM25, raw chunks | 51% | 45% | 54% | — |
| S1 + tag normalization | 54% | 48% | 58% | +5 / −0, not detectable on 138 (p = 0.06); tag slice 0/6 → 5/6 |
| S2 + table rows | 55% | 47% | 56% | +6 / −9: table rows 6/8 → 8/8, but 110 short row chunks with pump tags push datasheets out of the top-5 |
| S3 + applicability | 66% | 58% | 70% | better (+21 / −1) |
| S4 + date filter | 73% | 67% | 75% | better (+7 / −0) |
| S5 + evidence pack | 73% | 79% | 85% | better (+13 / −0) |
| S6 + contract and checks | 73% | 79% | 83% | not different (+0 / −2); correct and verified 83% (S5: 85%) |
| M3@S0 LLM reranker on BM25 | 76% | 74% | 73% | better than S0 (+30 / −4) |
| M3@S6 LLM reranker on S6 | 85% | 91% | 90% | better than S6 (+11 / −2) |
| M1t translation, `language` slice | 8/20 | 8/20 | 12/20 | better than S0 on the slice (5/20) |

- At S0, 56 of 63 wrong answers are lost **in search**, 2 in assembly, 5 in the model, none in reading.
  At S6, 20 of 23 are still lost in search: the English questions and the English manual of P-205.
- Versions and applicability are fixed by plant data (S3, S4), not by the reranker: point-in-time 4/10 at S0,
  6/10 with the reranker, 10/10 at S4-S5.
- The `conflict` slice is 0/4 in every row: Russian questions do not find the English manual with BM25, so the
  conflict is never shown. With both sources in the context, the contract returns `conflict` (step 7).
- Dense and hybrid rows: TODO (see "Dense search").

## Success criteria

- `store/manifest.json` has the same `index_version` as `reference/store/manifest.json`;
- the retrieval columns of your ablation table (recall@5, evidence recall, MRR) equal `reference/ablation.csv`;
  end-to-end from replay is equal too, a live run is within the interval;
- the P-101 question at S6: "not normal, inspect within 24 hours", citations R-12 rev. 3 cl. 4.3.3, Table 3,
  cl. 6.1–6.2, all three citation checks pass;
- V-912: `not_in_documents`; P-205 with both sources in the context: `conflict`.

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
