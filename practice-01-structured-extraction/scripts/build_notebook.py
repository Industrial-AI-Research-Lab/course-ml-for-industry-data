"""Build notebooks/practice_01_structured_extraction.ipynb from the cells below.

The notebook is generated so that its source stays readable in code review.
Edit the cells here, then run:  uv run python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "practice_01_structured_extraction.ipynb"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------------------------
md(r"""
# Practice 1. Structured extraction from maintenance logs

**ML in Industry 2026 · Topic 1: LLM-based systems in industry**

## Goal

Technicians write maintenance logs as free text: abbreviations, typos, Russian and English mixed.
We turn each entry into a validated record that a database, a dashboard or a reliability model can use:

```
"Р-101 замена подшипника, гудел, 12.02.2026, 4 ч, простой"
        ↓
{"equipment_id": "P-101", "equipment_type": "pump", "work_type": "corrective_repair",
 "failure_mode": "bearing_wear", "replaced_parts": ["bearing"], "duration_hours": 4,
 "work_date": "2026-02-12", "downtime": true, ...}
```

## Steps and what you should see

| Step | What we do | Expected result |
|------|------------|-----------------|
| 1 | Setup and smoke test | the model answers |
| 2 | Look at the data | 300 synthetic log entries with ground truth |
| 3 | Baseline: "return JSON" in the prompt | many answers are not valid JSON |
| 4 | Schema + constrained output + validation + retry | almost all answers are valid |
| 5 | Edge cases | valid JSON is not correct data |
| 6 | Rules baseline | rules are strong on some fields |
| 7 | Compare models on `eval_50` | table and chart: quality, latency, cost |
| 8 | Call logs | every call is saved: the base for topics 2 and 6 |

## Success criteria

- After step 4: **valid answers after retries ≥ 95%** for your model on `dev_20`.
- After step 7: your numbers are close to the **reference table** (`reference/results.md`).

## Live or replay

Every model call goes through one OpenAI-compatible client. If your model answers, the notebook calls it
**live**. If not (no Ollama, no token yet), it uses **replay**: answers that the lecturer saved for exactly
the same requests. The validation, retry and metrics code is the same in both cases.

Frontier models (GPT, Claude, DeepSeek) are always replayed: they are not available to students in Russia,
and this is a realistic constraint for a Russian plant too.

## Common problems

| Problem | What to do |
|---------|-----------|
| `Connection error` to `localhost:11434` | Ollama is not running: start it from the Start menu (or `ollama serve`); the smoke test prints a hint |
| model not found | `ollama pull qwen3.5:4b-q4_K_M` |
| VPN or proxy | requests to `localhost` always bypass the proxy; for the course server the proxy settings of your terminal are used |
| empty answers, `finish_reason = length` | the thinking mode is on and used all tokens: keep `reasoning_effort = "none"` for local models |
| weak laptop (no GPU, < 8 GB RAM) | `ollama pull qwen3.5:2b-q4_K_M`, then start Jupyter with `PRACTICE_PROFILE=local-small PRACTICE_EVAL_N=20`; the small model reaches ~60% valid, not 95%: see "Weak laptop" in the README |
| everything is very slow on CPU | set `PRACTICE_EVAL_N=20` before starting Jupyter, or use replay (do not start Ollama: the notebook switches to replay by itself) |
| `FileNotFoundError: data/...` | the data is in the repository: your clone is incomplete, run `git status` and `git pull` |
| `ReplayMiss` | you changed a prompt or the schema: replay works only for unchanged requests, use a live model |
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 0. Where we are going

This is the final table of this practice, measured by the lecturer. At the end you will understand
where every number comes from.
""")
code(r"""
from pathlib import Path
from IPython.display import Markdown, display

from plant_assistant import REFERENCE_DIR

display(Markdown((REFERENCE_DIR / "results.md").read_text(encoding="utf-8").split("## Field accuracy")[0]))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 1. Setup and smoke test

A **profile** is everything needed to call one model: URL, key, model name, parameters, price
(`models.toml`). Secrets are in `.env`. To switch the model you change the profile name, not the code.
""")
code(r"""
import json
import os

import pandas as pd

from plant_assistant.config import get_profile, list_profiles, load_env
from plant_assistant.data import load_edge_cases, load_registry, load_split
from plant_assistant.llm import LLMClient, ReplayClient
from plant_assistant.runner import new_logger, replay_path, run_extraction, run_rules

pd.set_option("display.max_colwidth", 120)
load_env()

# Which model to call live and how many eval_50 records to use in step 7.
# Defaults: the 4B model via Ollama and all 50 records.
# Weak laptop (no GPU, < 8 GB RAM)? Either start Jupyter with PRACTICE_PROFILE=local-small PRACTICE_EVAL_N=20,
# or replace the two lines below with:  PROFILE = "local-small"  and  EVAL_N = 20
PROFILE = os.environ.get("PRACTICE_PROFILE", "local")  # local, local-small or course
EVAL_N = int(os.environ.get("PRACTICE_EVAL_N", "50"))  # 20 is enough on a slow machine
print(f"PROFILE = {PROFILE}, EVAL_N = {EVAL_N}")

pd.DataFrame([{"profile": k, "description": v} for k, v in list_profiles().items()])
""")
code(r"""
def model_answers(profile_name: str) -> bool:
    try:
        # no network retries: if the model is not there, we want to know at once
        client = LLMClient(get_profile(profile_name), max_network_retries=0)
        reply = client.chat([{"role": "user", "content": "Reply with one word: OK"}])
    except ValueError as exc:  # profile not configured (no URL or key)
        print(f"{profile_name}: {exc}")
        return False
    print(f"{profile_name}: {'OK' if reply.ok else reply.error} ({reply.latency_s:.1f}s)")
    return reply.ok


LIVE = model_answers(PROFILE)
print("Mode:", "LIVE calls to " + PROFILE if LIVE else "REPLAY of the lecturer's saved answers")
""")
md(r"""
**Small model (`local-small`, Qwen3.5 2B)?** Everything below runs the same way, only the numbers differ:
the reference run of the 2B model gets about 60% valid answers after retries and 51% field accuracy on `eval_50`
(see `local-small-structured` in `reference/results.md`). The ≥ 95% criterion of step 4 is for the 4B and the
course model. With the small model, watch *how much* the schema and the retry improve validity, not the absolute number.
""")
code(r"""
# Saved answers exist for the "local" (4B) and "course" setups. With local-small the steps on dev_20
# run live on the 2B model; if it does not answer, the replay of the 4B model is used instead.
SETUP_PREFIX = "course" if PROFILE == "course" else "local"


def client_for(setup: str):
    # Live model if it answers, otherwise the lecturer's saved answers for the same setup.
    if LIVE:
        return LLMClient(get_profile(PROFILE), logger=new_logger(f"my-{setup}"))
    return ReplayClient(replay_path(setup))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 2. The data

A synthetic maintenance log: 300 entries. The generator first sampled the **facts** (this is the ground truth),
then an LLM wrote each entry in a technician's style, and a checker made sure the text states exactly these
facts. A quarter of the entries got typos afterwards. The data is synthetic: numbers measured on it are
illustrative, on a real log you measure again.

Splits: `dev_20` for the steps below, `eval_50` for the final comparison.
""")
code(r"""
registry = load_registry()
dev = load_split("dev_20")
eval_records = load_split("eval_50")[:EVAL_N]
print(f"registry: {len(registry)} equipment items, dev_20: {len(dev)}, eval: {len(eval_records)}")
pd.DataFrame([{"record_id": r.record_id, "language": r.language, "text": r.text} for r in dev[:8]])
""")
code(r"""
example = dev[0]
print(example.text)
example.ground_truth
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 3. Baseline: "return JSON" in the prompt

The model sees a field list in the prompt and the instruction "Return only JSON". No schema in the API call.
We parse the answer with `json.loads` and validate it with the schema.

> **Predict, then run.** What share of answers will be valid? (a) more than 90% (b) 60–90% (c) less than 60%
""")
code(r"""
# 20 records, 1 call each: seconds on a GPU, a few minutes on a CPU-only laptop (local-small has a 300 s timeout per call).
# Small model: expect fewer valid answers than the 20% of the 4B reference run. That is the point of this step.
baseline = run_extraction(f"my-{SETUP_PREFIX}-baseline-dev20", client_for(f"{SETUP_PREFIX}-baseline-dev20"),
                          dev, mode="baseline", registry=registry)
print(f"valid answers: {baseline.summary['valid_final_pct']}%")
pd.Series([r.first_error_type or "valid" for r in baseline.results]).value_counts()
""")
code(r"""
broken = next((r for r in baseline.results if not r.valid), None)
if broken:
    print("error type:", broken.first_error_type, "|", broken.attempts[0].errors[0])
    print("-" * 80)
    print(broken.attempts[0].content[:800])
""")
md(r"""
Typical problems: JSON inside a markdown code block, text around the JSON, a date like `12.02.2026`
instead of ISO, a value that is not in the list of allowed values. The model "understood" the task; the
format is still not usable by a program.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 4. Schema, constrained output, validation, retry

**One Pydantic model is the single source of truth.** We generate the JSON Schema for the API from it,
and we validate the answer with it.
""")
code(r"""
from plant_assistant.schema import MaintenanceRecord, strict_json_schema

pd.DataFrame([{"field": name, "description": f.description} for name, f in MaintenanceRecord.model_fields.items()])
""")
code(r"""
schema = strict_json_schema()
print(json.dumps(schema["properties"]["failure_mode"], indent=2))
print("required:", schema["required"])
""")
md(r"""
The schema goes into the API call as `response_format`. For a local model Ollama turns it into a grammar:
the model **cannot** generate a token that breaks the schema (constrained decoding).

Important: self-hosted servers (Ollama, vLLM) use the schema **only to constrain tokens**. The model never
reads the field descriptions. So we also put the field definitions into the prompt
(`prompts.FIELD_GUIDE`). Without them the course model writes `null` instead of `"none"` for
"no problem found" in almost half of the records (see `course-structured-schema-only` in the comparison).
**Constrained decoding guarantees the format, not the meaning.**

After the schema check, **business rules** check what a schema cannot express (`validation.py`):
the tag exists in the registry and has the right type, the date is not in the future, the duration is
plausible, every evidence quote is really in the text. If a check fails, we **retry** with the error text in the
prompt. After 3 attempts the record goes to **human review** (or to a fallback model).

> **Predict, then run.** The format is now guaranteed. Will the data now always be correct? Type yes or no.
""")
code(r"""
# 20 records, up to 3 calls each (retries): about a minute on a GPU, several minutes on a CPU-only laptop.
# Success criterion: valid after retries >= 95% for "local" (4B) and "course".
# Small model (local-small): the reference is ~60% valid after retries; compare the jump from step 3 instead.
structured = run_extraction(f"my-{SETUP_PREFIX}-structured-dev20", client_for(f"{SETUP_PREFIX}-structured-dev20"),
                            dev, mode="structured", registry=registry)
s = structured.summary
print(f"valid on the 1st attempt: {s['valid_first_pct']}%   valid after retries: {s['valid_final_pct']}%")
print(f"field accuracy: {s['field_accuracy_pct']}%   invented values: {s['invented_pct']}%")
pd.Series([r.status for r in structured.results]).value_counts()
""")
code(r"""
# Which checks failed on the first attempt?
first_errors = [e for r in structured.results for a in r.attempts[:1] if a.error_type for e in a.errors]
pd.Series([e.split("'")[0].split(":")[0][:60] for e in first_errors]).value_counts().head(10)
""")
code(r"""
# One record that needed a retry: the error text goes back to the model.
retried = next((r for r in structured.results if len(r.attempts) > 1), None)
if retried:
    for a in retried.attempts:
        print(f"attempt {a.number}: {a.error_type or 'passed'}  {a.errors[:2]}")
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 5. Edge cases: valid JSON is not correct data

Eight hand-written entries (`fixtures/edge_cases.jsonl`). For each one we know what a good system should do.

> **Predict, then run.** Which of these entries will make the model invent a value?
""")
code(r"""
# 8 records, fast on any model. Small model: expect more invented values and more "needs_review";
# the replay you fall back to (local-structured-edge) is the 4B model.
edge = load_edge_cases()
edge_run = run_extraction(f"my-{SETUP_PREFIX}-structured-edge", client_for(f"{SETUP_PREFIX}-structured-edge"),
                          edge, mode="structured", registry=registry, progress=False)
rows = []
for rec, res in zip(edge, edge_run.results):
    out = res.record or {}
    rows.append({
        "case": rec.meta["case"], "text": rec.text, "status": res.status,
        "equipment_id": out.get("equipment_id"), "work_type": out.get("work_type"),
        "failure_mode": out.get("failure_mode"), "work_date": out.get("work_date"),
        "duration_h": out.get("duration_hours"), "expected": rec.meta["expected"],
    })
pd.DataFrame(rows)
""")
md(r"""
What to look for:
- **no date**: `work_date` must be `null`. A model that invents a date is worse than no answer.
- **two equipment items**: the schema has room for one tag. This is a schema design problem, not a model problem.
- **minutes**: 40 minutes is 0.67 hours, not 40.
- **Cyrillic look-alike**: `Р-101` with a Cyrillic `Р` must become `P-101`, or the registry lookup fails.
- **future date, unknown tag**: business rules catch them. The right result is `needs_review`, not a "fixed" value.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 6. Rules baseline: start simple

Before we compare models, we compare with **no model at all**: regular expressions, keyword dictionaries
and a registry lookup (`rules_baseline.py`, about 150 lines). Zero cost, microseconds per record, fully
deterministic.
""")
code(r"""
from plant_assistant.metrics import coverage
from plant_assistant.runner import result_rows

rules = run_rules("my-rules", eval_records, registry)
per_field = pd.DataFrame({
    "rules accuracy, %": rules.summary["per_field_pct"],
    "rules coverage, %": coverage(result_rows(eval_records, rules.results)),
})
per_field
""")
md(r"""
Rules are very good where the format is regular (tag, date, duration) and where a lookup gives the answer
(equipment type from the registry). They are weaker where people describe things in their own words
(failure mode, work type), and they break on typos. The right question is not "rules or LLM" but
"which fields do the rules cover, and what do we do with the rest".
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 7. Compare models on `eval_50`

Your model runs live (or replay). The other models are replays of the lecturer's runs on the same records,
with the same prompts and the same code.

> **Poll.** Which model would you deploy at a plant without internet access? Why? Write one reason in the chat.
""")
code(r"""
# Only your profile runs live, on EVAL_N records (50 by default, 20 with PRACTICE_EVAL_N=20); every other
# setup is a replay and takes seconds. The live run takes 1-2 min on a GPU and can take 10+ min on a CPU-only
# laptop with 50 records: use EVAL_N = 20 there. Replays use the same EVAL_N records, so the table stays comparable.
COMPARE = [  # setup name -> how to get its answers
    ("local-structured-schema-only", "replay"),
    ("local-structured", "live" if (LIVE and PROFILE == "local") else "replay"),
    ("local-small-structured", "live" if (LIVE and PROFILE == "local-small") else "replay"),
    ("course-structured-schema-only", "replay"),
    ("course-structured", "live" if (LIVE and PROFILE == "course") else "replay"),
    ("course-thinking-structured", "replay"),
    ("gpt-oss-120b-structured", "replay"),
    ("deepseek-v3.2-structured", "replay"),
    ("gpt-6-luna-structured", "replay"),
    ("gpt-6.1-sol-structured", "replay"),
    ("claude-sonnet-5-structured", "replay"),
]

summaries = [rules.summary]
for setup, how in COMPARE:
    profile = setup.split("-structured")[0]
    mode = "structured-schema-only" if setup.endswith("schema-only") else "structured"
    if how == "live":
        client = LLMClient(get_profile(profile), logger=new_logger(f"my-{setup}"))
    else:
        try:
            client = ReplayClient(replay_path(setup))
        except FileNotFoundError:
            print(f"skip {setup}: no replay file")
            continue
    out = run_extraction(f"my-{setup}", client, eval_records, mode=mode, registry=registry)
    out.summary["name"] = setup + (" (live)" if how == "live" else "")
    summaries.append(out.summary)
""")
code(r"""
from plant_assistant.report import comparison_table, per_field_table, plot_quality_latency_cost

comparison_table(summaries)
""")
code(r"""
plot_quality_latency_cost(summaries, title=f"Quality vs latency vs cost, {len(eval_records)} records");
""")
code(r"""
per_field_table(summaries)
""")
md(r"""
How to read it:
- **Validity** is close to 100% for every model with a schema and retries. Validity is not the problem anymore.
- **Field accuracy** separates the models. Look at the per-field table: which fields are hard for everybody?
- **Latency** depends on hardware: a 4B model on a laptop CPU and on a GPU differ by an order of magnitude.
  Replayed latency is the lecturer's measurement, not yours.
- **Cost**: self-hosted models have no per-token price, you pay with hardware and operations.
  Frontier APIs are not available in Russia at all.
- **Rules** are in the table too. For which fields is an LLM worth it?
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 8. Call logs

Every live call is written to `runs/my-<setup>/calls.jsonl`: prompt, answer, model, tokens, latency, cost.
The lecturer's runs are next to yours in `runs/<setup>/` (results and summaries) and `data/replay/` (call logs).
In topic 2 these logs become test cases, in topic 6 they feed monitoring. A call log is also a replay file:
this is how the lecturer's runs got into your notebook.
""")
code(r"""
from plant_assistant import PROJECT_ROOT, RUNS_DIR

log_files = sorted(RUNS_DIR.glob("my-*/calls.jsonl"), key=lambda p: p.stat().st_mtime)
source = log_files[-1] if log_files else replay_path("local-structured-dev20")
entry = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
print("file:", source.relative_to(PROJECT_ROOT))
print(json.dumps({k: entry[k] for k in ("ts", "profile", "model", "meta", "response_format")}, indent=2))
print(json.dumps({k: entry["response"][k] for k in ("prompt_tokens", "completion_tokens", "latency_s", "cost_usd")}, indent=2))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 9. Results and next steps

**Check yourself:** valid after retries ≥ 95% on `dev_20` with the `local` or `course` profile; your table close
to `reference/results.md`. With `local-small`, compare your row with `local-small-structured` in the reference table.

**What you built:** a typed contract (`MaintenanceRecord`), constrained output, business rules,
retry with the error text, fallback to human review, a rules baseline, a comparison harness and call logs.

**Homework (optional):**
1. When you get the course token: set `PRACTICE_PROFILE=course`, run the notebook live, compare thinking on and off.
2. Normalize in code, not in the model: convert Cyrillic look-alike letters and add the missing hyphen in
   `equipment_id` before the business rules. How many retries disappear? (Look at `FV305` in the logs:
   the small model could not fix it and returned `null` just to pass the check.)
3. Hybrid: run the rules first and call the LLM only for the fields the rules left empty. Compare accuracy,
   number of LLM calls and cost with "LLM for everything".
4. Two-step extraction: free-text reasoning first, then formatting. Does accuracy change?
5. Split the schema into two smaller schemas. Does accuracy change?
6. Connect a Russian API (Yandex AI Studio or GigaChat) with your own account as a new profile.
   Check that it really follows the JSON Schema.

**Next practice (topic 2):** we turn these runs into an evaluation harness with a golden set.
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
