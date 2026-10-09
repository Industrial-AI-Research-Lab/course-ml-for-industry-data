# Practice 1. Structured extraction from maintenance logs

Part of the course **ML in Industry 2026** (topic 1: LLM-based systems in industry). The project is the
**Plant Engineer Assistant**: after this practice it extracts structured, validated records from free-text
maintenance logs and compares models by quality, latency and cost.

Everything you need is in this folder: code, notebook, synthetic data, the lecturer's saved model answers
(replay files) and reference results. Nothing has to be downloaded separately.

## Reproduce the practice (30–60 minutes)

### 1. Install

You need Python 3.11–3.13 (3.14 is not supported yet: `pydantic` has no wheels for it) and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone https://github.com/Industrial-AI-Research-Lab/course-ml-for-industry-data.git
cd course-ml-for-industry-data/practice-01-structured-extraction
uv sync --all-extras
```

Without uv: `python -m venv .venv`, activate it, then `pip install -e ".[notebook,dev]"`.

### 2. Check the data

The data is already in the repository, there is no archive to download:

| Path | What it is |
|------|------------|
| `data/maintenance_log.jsonl` | 300 synthetic log entries with ground truth |
| `data/splits.json`, `data/equipment_registry.csv`, `data/dataset_card.json` | splits `dev_20` / `eval_50`, equipment registry, how the data was made |
| `data/replay/*.jsonl` | the lecturer's saved model answers: the notebook uses them when your model is not available |
| `data/generation_calls.jsonl` | call log of the dataset generator (for the curious) |
| `runs/<setup>/` | the lecturer's reference runs: results and summary of every setup in the comparison table |
| `reference/` | reference table, per-field accuracy, chart, executed notebook |

```bash
uv run pytest        # 26 tests, no network; also checks that the data loads
```

### 3. Choose a model

| Profile | Model | What you need |
|---------|-------|---------------|
| `local` (default) | Qwen3.5 4B, Q4, via Ollama | [Ollama](https://ollama.com/download), then `ollama pull qwen3.5:4b-q4_K_M` (3.3 GB), 8 GB RAM |
| `local-small` | Qwen3.5 2B, Q4, via Ollama | `ollama pull qwen3.5:2b-q4_K_M` (1.9 GB), for weak laptops: see the next section |
| `course` | Qwen 3.8 27B, course server | URL and token from the lecturers (from practice 2) |
| none of these | replay of the lecturer's runs | nothing: the notebook switches to replay automatically |

```bash
cp .env.example .env          # fill in COURSE_LLM_* when you get the course token
uv run python -m plant_assistant.smoke_test                # checks the "local" profile
uv run python -m plant_assistant.smoke_test local-small    # weak laptop: checks the small model
uv run python -m plant_assistant.smoke_test course         # checks the course model
```

The smoke test ends with `OK`. The first call to a local model can take a minute: the model is loading.

### Weak laptop or a small model: how to run every step

A laptop without a GPU and with less than 8 GB of free RAM does not run the 4B model comfortably.
Use the `local-small` profile (Qwen3.5 2B) and fewer records. The same two settings are read by every
stage of the practice: the smoke test, the notebook steps and the final comparison.

| Setting | Value for a weak laptop | What it changes |
|---------|-------------------------|-----------------|
| `PRACTICE_PROFILE` | `local-small` | which model the notebook calls live (default `local`) |
| `PRACTICE_EVAL_N` | `20` | how many of the 50 `eval_50` records step 7 uses (default `50`) |
| `timeout_s` of `local-small` in `models.toml` | `300` (already set) | one answer on a CPU can take over a minute; raise it if you see timeouts |
| `LOCAL_SMALL_LLM_MODEL` in `.env` | any other Ollama model name | try a different small model with the same profile |

Start the notebook with these settings:

```bash
# Linux, macOS, Git Bash
PRACTICE_PROFILE=local-small PRACTICE_EVAL_N=20 uv run jupyter lab notebooks/practice_01_structured_extraction.ipynb
```

```powershell
# Windows PowerShell
$env:PRACTICE_PROFILE = "local-small"; $env:PRACTICE_EVAL_N = "20"; uv run jupyter lab notebooks/practice_01_structured_extraction.ipynb
```

Or set `PROFILE` and `EVAL_N` directly in the first code cell of section 1 of the notebook: the cell has a comment where.

What to expect with the small model, step by step (reference numbers from [reference/results.md](reference/results.md)):

- **Step 1, smoke test.** `uv run python -m plant_assistant.smoke_test local-small` must end with `OK`.
  The notebook then prints `Mode: LIVE calls to local-small`.
- **Step 3, baseline.** Even fewer valid answers than the 4B model (20%). This is the point of the step.
- **Step 4, schema + retry.** The 2B model reaches about **60% valid after retries and 51% field accuracy**
  on `eval_50`, not the ≥ 95% success criterion. The criterion is for the 4B and the course model.
  Your check with the small model: validity goes up a lot after adding the schema, and the retry fixes some records.
- **Step 5, edge cases.** Expect more invented values. The replay for comparison (`local-structured-edge`) is the 4B model.
- **Step 7, comparison.** Live run of `local-small-structured` on `EVAL_N` records; all other models are replays,
  so the table is still complete. On a CPU the live run takes about 1–3 minutes for 20 records.
- **Too slow even so?** Do not start Ollama: the notebook detects that the model does not answer and uses
  replay for everything. Replay needs no model and runs in under a minute.

### 4. Run the notebook

```bash
uv run jupyter lab notebooks/practice_01_structured_extraction.ipynb
```

Run all cells from top to bottom. To use another profile, start Jupyter with
`PRACTICE_PROFILE=course` (or `local-small`, see above). On a slow laptop, `PRACTICE_EVAL_N=20` uses fewer records.

### 5. Check yourself

- Valid answers after retries ≥ 95% on `dev_20` (step 4 of the notebook) with the `local` or `course` profile.
  With `local-small`, compare with the `local-small-structured` row of the reference table instead.
- Your comparison table is close to [reference/results.md](reference/results.md).

## Project structure

```
practice-01-structured-extraction/
├── models.toml                 # model profiles: URL, model, parameters, price (no secrets)
├── .env.example                # copy to .env: keys and URLs
├── src/plant_assistant/
│   ├── config.py               # profiles + .env
│   ├── llm.py                  # OpenAI-compatible client, call log, replay client
│   ├── jsonl.py                # tolerant JSON Lines reader (survives "reformat on save")
│   ├── schema.py               # MaintenanceRecord: the contract; strict JSON Schema
│   ├── prompts.py              # all prompts in one place
│   ├── validation.py           # business rules: registry, dates, durations, quotes
│   ├── extraction.py           # baseline and structured extraction, retry, fallback
│   ├── rules_baseline.py       # regex + dictionaries + registry lookup, no model
│   ├── metrics.py              # field accuracy, invented values, latency, cost
│   ├── runner.py               # run a setup over records, save results
│   ├── report.py               # comparison table and chart
│   └── smoke_test.py
├── scripts/
│   ├── generate_dataset.py     # synthetic maintenance log with ground truth
│   ├── build_reference.py      # lecturer: reference runs, replay files, reference tables
│   └── build_notebook.py       # builds the practice notebook from readable source
├── notebooks/practice_01_structured_extraction.ipynb
├── fixtures/edge_cases.jsonl   # hand-written edge cases
├── data/                       # dataset, splits, registry and replay files (in git)
├── runs/                       # lecturer's reference runs (in git); your runs go to runs/my-* (not in git)
├── reference/                  # reference results (table, per-field accuracy, chart, executed notebook)
└── tests/                      # pytest, no network
```

## How the pieces fit

```
log entry ──► prompt ──► model (schema in the API call) ──► JSON ──► Pydantic schema ──► business rules
                ▲                                                         │ fail               │ fail
                └──────────── retry with the error text (up to 3) ◄───────┴────────────────────┘
                                         │ still failing
                                         ▼
                              fallback model ──► human review queue
every call ──► runs/<setup>/calls.jsonl  (prompt, answer, tokens, latency, cost)  = replay file
```

## Notes

- **Data is synthetic.** The generator samples the facts first (the ground truth), then an LLM writes the entry
  and a checker verifies that the text states exactly these facts. Numbers measured on it are illustrative.
- **Proxy cache.** The course LiteLLM server caches identical requests. Profiles set a unique `user` field per
  run (`bypass_proxy_cache`), so latency and randomness are measured honestly.
- **Thinking mode.** Qwen models think by default. For extraction we turn it off
  (`chat_template_kwargs.enable_thinking=false` on the course server, `reasoning_effort="none"` in Ollama);
  otherwise the reasoning can use the whole token budget and the answer is empty.
- **Secrets** stay in `.env`. Call logs and replay files contain prompts and answers, never keys.
- **Frontier models** (GPT, Claude, DeepSeek) appear only as replay files: they are not available to students
  in Russia, and this is a realistic constraint for a Russian plant too.
