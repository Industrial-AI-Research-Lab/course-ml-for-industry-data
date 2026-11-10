# Practice 4. Finding the right clause: diagnose and fix

**ML in Industry 2026 · Retrieval-augmented generation · Plant Engineer Assistant**

An engineer asks: *“After a bearing change, pump P-101 vibrates at 4.8 mm/s. Is this normal?”*
Find the applicable, current evidence, locate where answers are lost, and measure one fix at a time.
Search mechanisms are covered in Advanced NLP; here they are tools compared on the same questions.

The practice includes synthetic documents, saved model responses, embedding vectors and reference results.
The replay path requires no API key, model download, Docker or GPU.

## Run the practice

Use Python 3.12 (the version in `.python-version`) and [uv](https://docs.astral.sh/uv/).
Run these commands from `practice-04-rag/` in a checkout containing this directory:

```bash
uv sync --locked --extra notebook --extra dev
uv run --offline --locked pytest -q
PRACTICE_MODE=replay uv run --offline --locked jupyter lab notebooks/practice_04_rag.ipynb
```

Install dependencies once with network access. The tests and replay notebook then run offline.
Run all notebook cells in order. BM25, filters and context assembly run locally; model responses,
reranker scores and embedding vectors come from the saved reference run.

To execute without opening Jupyter:

```bash
PRACTICE_MODE=replay uv run --offline --locked jupyter nbconvert \
  --to notebook --execute notebooks/practice_04_rag.ipynb \
  --output practice_04_checked.ipynb --output-dir runs \
  --ExecutePreprocessor.timeout=1200
```

Verified on macOS with Python 3.12.12 on 2026-11-10: fresh environment, empty package cache,
40 passing tests and 40 executed notebook code cells. Replay reference tables reproduced exactly;
runtime checks used no credentials and blocked external Python connections.

### Optional live answers

```bash
cp .env.example .env
# Fill in COURSE_LLM_BASE_URL and COURSE_LLM_API_KEY.
uv run --locked python -m plant_assistant.smoke_test course
uv run --locked jupyter lab notebooks/practice_04_rag.ipynb
```

The notebook calls the course Qwen for individual examples when available; full evaluations stay on replay.
`PRACTICE_MODE=replay` forces replay. Live results may differ from the saved run.

## Student work and scope

Allow 60–75 minutes for the guided notebook. Write three short checkpoint answers: diagnose failures, compare
one fix, and identify invalid citations. Keep them with the final metric comparison and a three-sentence
conclusion. Each checkpoint provides an expected format and a reference result. Optional experiments are extensions.

The notebook applies the lecture's process: diagnose a loss, change one link, compare paired results.
It reuses practice 3's reading output and prepares search as a tool (topic 5) and regression checks (topic 6).
Permissions/ACL are deferred to topic 7. Text-answer evaluation with a calibrated judge, discussed in the lecture,
is outside the implemented notebook; numbers/statuses and limited citation checks are evaluated here.

## Files

| Path | Contents |
|------|----------|
| `notebooks/practice_04_rag.ipynb` | the practice, sections 0–9 |
| `inputs/practice-03/` | the copied store and 50 questions from practice 3 |
| `inputs/practice-01/` | the original equipment registry |
| `data/docs/`, `data/ground_truth/` | 12 new PDFs, their texts and structure |
| `data/documents.csv`, `data/equipment_registry.csv` | document dates and scope; equipment classes and linked documents |
| `data/questions.jsonl` | 143 questions with expected values or statuses and gold chunk ids |
| `data/holdout.jsonl`, `data/needs_human.jsonl` | 27 role-based questions; 6 cases requiring missing equipment data |
| `data/replay/` | saved chat answers, translations, both rerankers and e5-mistral vectors |
| `reference/` | ablation, slice, cost and contract-comparison tables; audit, store and executed notebook |
| `scripts/` | corpus, questions, vectors, reference and notebook builders |
| `src/plant_assistant/` | retrieval, filters, assembly, answers, checks, evaluation and audit modules |

## Corpus and questions

All documents and numbers are **synthetic teaching material**. Table 3 resembles ISO 20816 tables but does
not contain values from that standard. The corpus has 16 documents, 47 pages and 440 chunks, including
110 valve-table rows.

- **R-12 revisions 2 and 3:** class II inspection limits 5.0 and 4.5 mm/s; stop limit 7.1 mm/s.
- **Amendment No. 1:** clause 4.4.1 changes from 720 to 500 h on 2026-09-01.
- **R-14:** near-copy for station 3; class II limits 3.5 / 5.6 mm/s.
- **R-15 and R-20:** maintenance conflicts and CMMS priorities/deadlines.
- **Five datasheets and the English P-205 manual:** equipment classes, technical values and conflicting limits.
- **VL-2:** 110 valve rows on pages 4–7; three unrelated instructions supply background.

The copied practice 1 registry places P-101 at station 1. This practice's registry corrects it to station 2,
matching R-12. Question c06 changes from 720 h in practice 3 to 500 h on the fixed question date, 2026-10-27.

Each question has Russian search wording and an English translation. The `language` slice sends English
wording directly. Evidence locators identify clauses, tables or rows; code resolves them to chunk ids.

| Slice | Questions | Tests |
|-------|----------:|-------|
| base | 50 | translated practice 3 questions |
| language | 20 | English queries against multilingual documents |
| applicability | 12 | station, equipment type and machine class |
| point_in_time | 10 | rules valid on a past date |
| incident | 3 | a past event evaluated under its original rules |
| what_changed | 4 | evidence from multiple revisions |
| multi_evidence | 8 | values, rules and actions in separate chunks |
| cross_doc | 3 | evidence across documents |
| conflict | 9 | different limits in applicable documents |
| table_row | 8 | a specific valve-table row |
| tag_variant | 6 | Cyrillic letters, spaces and missing hyphens |
| not_in_documents | 10 | absent facts; another 10 occur in base/language |

The holdout was drafted from documents and four job roles, without the ladder: 27 of 75 drafts were kept
after code checks and manual filtering. The separate `needs_human` set covers missing class, station,
unambiguous tag, datasheet or measurements. It is outside the 143-question set.

## Configurations

| Row | Change |
|-----|--------|
| S0 | BM25, top 5, no filters; context is the search result |
| S1 | normalize tags in chunks and questions |
| S2 | mix individual table rows with clauses |
| S2b | search rows separately: top 5 other chunks plus at most 2 rows |
| S3 | build on S2b: registry scope filter, equipment class and linked document tables |
| S4 | filter by date, including clause amendments |
| S5 | add references in both directions and parent sections within 2500 estimated tokens |
| S6 | structured answer, bounded internal `check`, source ids restricted to the context, citation checks |
| M1 | dense search with e5-mistral instead of BM25 |
| M2 | BM25 + dense, fused by reciprocal rank |
| M3 | course Qwen reranks the top 20 |
| M4 | bge-reranker-v2-m3 cross-encoder reranks the top 20 |
| M1t | translate the English question into Russian; language slice only |

`@S0` and `@S6` apply a mechanism to the baseline and the full pipeline. `:v1` and `:v2` retain earlier answer
contracts; the main S6 rows use v3. The schema allows five claims, two sources per claim, a 300-character internal
check and a 600-character answer. A failed or reasoning-contaminated answer gets one retry, then `needs_human`.

## Reference results

These are **code-grader scores on saved answers**, not a full assessment of engineering advice.
See the limitations below and `reference/results.md` for intervals, paired comparisons and slices.

| Row | Correct / 143 | Holdout / 27 | Median seconds/question |
|-----|--------------:|-------------:|------------------------:|
| S0 | 73 (51%) | 15 (56%) | 1.46 |
| S3 | 102 (71%) | 20 (74%) | 1.68 |
| S4 | 109 (76%) | 19 (70%) | 1.76 |
| S5 | 123 (86%) | 20 (74%) | 3.05 |
| S6 | 129 (90%) | 21 (78%) | 5.79 |
| M1@S6 | 138 (97%) | 25 (93%) | 6.01 |
| M2@S6 | 133 (93%) | 23 (85%) | 6.87 |
| M3@S6 | 133 (93%) | 24 (89%) | 6.73 |
| M4@S6 | 132 (92%) | 22 (81%) | 5.87 |

- **Losses:** S0 loses 60 answers in search, 2 in assembly and 8 in the model. S6 loses 12 in search and 2 in
  the model: 14 errors total.
- **Dates:** the past-date slice rises from 4/10 at S0 to 10/10 at S6.
- **Conflicts:** S6 returns the expected status on 7/9 cases; M1@S6 on 9/9. S6 chooses one document on the
  P-205 6.0 mm/s case (`f03`) and misses a manual clause on `f09`.
- **Missing data:** S6 gets 3/6 `needs_human` cases right; M1@S6 gets 5/6. Every main S6 row incorrectly gives
  a value when only one of three vibration directions was measured (`nh06`).
- **P-101:** S6 misses Table 3 in its top 5 and refuses. M3@S6 retrieves it and answers “inspect within 24 hours”
  with checked citations. The automatic grade does not certify every action in the prose.
- **Paired tests:** S6 versus S5 is inconclusive (+7 fixed / 1 broken, p = 0.070). M1@S6 versus S6 passes the
  unadjusted 0.05 threshold (+13 / −4, p = 0.049); treat this as exploratory after many comparisons.

Contract v2 removed the long invalid outputs of v1. Version 3 adds an internal check, with mixed quality gains
and roughly 1–1.7 extra seconds per question. Thinking mode scores 86% at 17.78 seconds/question.
Detailed before/after results are in `reference/contract_before_after.csv`.

## What the evaluation does and does not prove

`reval.grade` checks:

- **One value:** value and unit, allowing decimal commas and unit synonyms.
- **Several values:** expected numbers occur in the answer text; their units, roles and the recommended action
  are not checked. An answer can mention the right limits and still reach the wrong decision.
- **Conflict:** the status is `conflict`; the grader does not assess the explanation or both conflicting values.
- **Not in documents:** an explicit refusal, or a non-conflict response without a value, is accepted. This also
  accepts `needs_human`; it does not establish that the refusal explanation is correct.
- **Needs a person:** only the status `needs_human`, not whether the right clarification was requested.

`verify.check_answer` checks source existence, validity dates and whether at least one cited chunk contains the
key number and unit. A linked table may supply that number. For conflicts, it requires citations from at least
two documents; it does not prove those documents contradict each other. It does not check each claim, the correct
table row, numerical comparisons or action deadlines. A refusal may pass citation checks while being wrong.
`correct + verified` combines these two limited checks; it is not proof of a fully correct answer.

Retrieval scores concern the annotated evidence, which may omit alternative valid sources. From S2b onwards,
“recall@5” includes up to two extra row results; registry-linked documents count in evidence recall only.
Loss attribution is a deterministic diagnostic based on these gold chunks, not a causal experiment.

All cases are synthetic, some English questions translate base questions, and the holdout has only 27 cases.
Wilson intervals and paired tests describe this sample; repeated tuning and multiple comparisons limit wider
claims. Replay proves reproducibility for saved requests, not current live-model quality. Times describe the
saved server run plus local search, not a throughput benchmark.

Free-text correctness and claim-by-claim support need additional manual or judge-based evaluation; neither is
implemented here. Do not use these scores to approve real equipment decisions.

## Rebuild data and reference

Normal use builds `store/` in the notebook. To regenerate source data, install the generator dependency too:

```bash
uv sync --locked --extra notebook --extra dev --extra generate
uv run --locked python scripts/generate_corpus.py
uv run --offline --locked python -m plant_assistant.corpus
uv run --offline --locked python scripts/build_questions.py
uv run --offline --locked python scripts/build_reference.py --replay
uv run --offline --locked python scripts/build_notebook.py
```

The reference builder overwrites `reference/`; use a separate checkout for experiments.
Changing questions, prompts or chunks can invalidate replay requests.

To recompute vectors or cross-encoder scores with an authorized course profile:

```bash
uv run --locked python -m plant_assistant.smoke_test embed-course
uv run --locked python -m plant_assistant.smoke_test rerank-course
uv run --locked python scripts/build_vectors.py --profile embed-course --name e5-mistral
uv run --locked python scripts/build_reference.py --live-mechanisms
```

For e5-mistral, question embeddings use `query_prefix` from `models.toml`; passages do not.
The vector manifest records model, dimension, corpus fingerprint and query latency. Cross-encoder replay is in
`data/replay/crossencoder_course.jsonl`. Local embedding profiles are optional; changing the model also requires
rebuilding vectors and selecting their path in `build_reference.py`.

## Check your run

- The built store's `index_version` matches `reference/store/manifest.json`.
- All search and code-grader columns match `reference/ablation.csv` in replay mode.
- M3@S6 answers the P-101 question and passes citation checks; S6 alone refuses.
- Valve V-912 gets no value. The demonstrated P-205 limit and P-101 lubrication cases return `conflict`.

## Common problems

| Symptom | Check |
|---------|-------|
| Replay despite a configured key | unset `PRACTICE_MODE=replay` to try live examples; check `.env` and the smoke test |
| `ReplayMiss` | a question, prompt, context or model differs from the saved request |
| Search columns differ | compare store fingerprints and rebuild with the locked dependencies |
| `finish_reason = length` | the output exhausted its token budget; inspect the contract version, thinking setting and retry record |

## Optional experiments

1. Replace the R-12 chunks with a scan reader's output from practice 3; recount reading losses.
2. Compare embedding profiles on English queries with paired tests.
3. Compare model-extracted query fields with the rules in `query.py`.
4. Reduce the evidence-pack budget to 500, 1000 or 2000 tokens.
5. Remove station metadata and count answers drawn from the wrong regulation.
6. Add an explicit document hierarchy; evaluate conflicting rules.
7. Move the vector index to pgvector or Qdrant; compare results and latency.
