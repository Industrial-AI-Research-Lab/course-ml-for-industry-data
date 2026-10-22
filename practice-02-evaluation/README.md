# Practice 2. Did the new prompt make the system better?

Part of the course **ML in Industry 2026**, topic 2. Evaluating LLM systems.

In practice 1 the system learned to turn a maintenance log entry into a record with eight typed fields and one free-text field, `defect_summary`. In this practice we change the extraction prompt for `defect_summary` and measure the effect on the typed fields and on the summaries. Then we check the LLM judge that grades the summaries.

## Plan

| Part | Block | Run | Measure |
|------|-------|-----|---------|
| A. The system | A1 The change | prompt v1 and v2 | |
| | A2 Typed fields | v1 and v2, 3 runs each, 100 entries | field accuracy, all fields right, eval_50 and holdout |
| | A3 Free text | the same runs, graded by an LLM judge | faithful summaries, interval, paired test |
| | A4 Gate | v2 against v1 | pass or fail |
| B. The judge | B1 Human labels | 213 summaries from practice 1 | PASS or FAIL with a reason |
| | B2 Judges | 2 judge prompts, thinking off and on | agreement, kappa, caught failures |
| | B3 Disagreements | judge and person | causes of disagreement |
| | B4 Few labels | judge on all, 40 human labels | PPI estimate with an interval |

## Run

You need Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
cd course-ml-for-industry-data/practice-02-evaluation
uv sync --all-extras
uv run pytest
```

Copy `.env.example` to `.env` and add the URL and token of the course model from the course channel. The first run of the notebook makes about 1 600 calls to the course model and takes 10–20 minutes. Results are stored in `runs/` and reused on the next run.

```bash
uv run python -m plant_assistant.smoke_test course
uv run jupyter lab notebooks/practice_02_evaluation.ipynb
```

Run all cells from top to bottom and compare your numbers with `reference/results.md`. Live runs may differ by a few points.

The gate also runs from the command line and returns exit code 0 or 1.

```bash
uv run python -m plant_assistant.evals.gate course-v1-r1 course-v2-r1
```

## Files

| Path | Content |
|------|---------|
| `notebooks/practice_02_evaluation.ipynb` | the practice |
| `reference/` | results and the executed notebook |
| `src/plant_assistant/evals/` | test cases, harness, error analysis, code checks, judge, statistics, gate |
| `src/plant_assistant/prompts.py` | prompt v1 and the rule of prompt v2 (`SUMMARY_RULE`) |
| `evals/` | golden set, holdout and the labelling guideline |
| `data/` | log entries, registry, splits, summary pool and human labels |
| `inputs/practice-01/runs/` | results of practice 1 |
| `runs/` | your runs, created by the notebook |
| `scripts/` | scripts that rebuild the golden set, the summary pool, the reference results and the notebook |
