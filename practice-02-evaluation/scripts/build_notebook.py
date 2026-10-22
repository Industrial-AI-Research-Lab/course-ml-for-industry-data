"""Build notebooks/practice_02_evaluation.ipynb from the cells below.

Edit the cells here, then run  uv run python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "practice_02_evaluation.ipynb"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# =============================================================================================
md(r"""
# Practice 2. Did the new prompt make the system better?

**ML in Industry 2026 · Topic 2. Evaluating LLM systems**

## After practice 1

The system reads one maintenance log entry and returns one record that follows a JSON Schema.

```
21.07.25, агрегат остановлен, M-310: устранение неисправности, обнаружено низкое сопротивление изоляции, 6 hrs, Гусев К.
  ↓
{"equipment_id": "M-310", "equipment_type": "electric_motor", "work_type": "corrective_repair",
 "failure_mode": "electrical_fault", "replaced_parts": [], "duration_hours": 6, "work_date": "2025-07-21",
 "downtime": true, "defect_summary": "Low insulation resistance detected.", "evidence": [...]}
```

- The log has 300 synthetic entries in Russian, English or both, with typos. Every entry is labelled, because the generator chose the facts first and wrote the text second.
- Eight typed fields are compared with the labels. The course model gets 99.8% field accuracy on `eval_50`.
- The free-text field `defect_summary` has no label and was not evaluated in practice 1.

## Plan

Change the extraction prompt for `defect_summary` and measure the effect.

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

| Split | Size | Role |
|-------|------|------|
| `eval_50` | 50 | golden set, used to build practice 1 |
| `holdout_50` | 50 | entries nobody looked at, used to confirm results |

The notebook needs the course model token in `.env`. The first run makes about 1 600 calls and takes 10–20 minutes. Results are stored in `runs/` and reused on the next run.
""")

code(r"""
import os

import pandas as pd

from plant_assistant import PROJECT_ROOT, RUNS_DIR, prompts
from plant_assistant.config import get_profile, load_env
from plant_assistant.evals.cases import fingerprint, load_cases
from plant_assistant.llm import LLMClient

pd.set_option("display.max_colwidth", 110)
load_env()
course = LLMClient(get_profile("course"))
print("course model:", "OK" if course.chat([{"role": "user", "content": "Reply with one word: OK"}]).ok else "NOT AVAILABLE, check .env")
JUDGE = "course-thinking"  # binary judge with thinking on, it has the highest kappa in part B


golden = [c for c in load_cases(PROJECT_ROOT / "evals" / "cases_v1.jsonl") if c.source == "log"]
holdout = load_cases(PROJECT_ROOT / "evals" / "holdout_v1.jsonl")
cases = golden + holdout
print(f"eval_50: {len(golden)} entries ({fingerprint(golden)}), holdout_50: {len(holdout)} entries ({fingerprint(holdout)})")
""")

# =============================================================================================
md(r"""
# Part A. The system, prompt v1 and prompt v2

## A1. The change

In practice 1 the model often added a cause or a location that the entry does not state. Examples labelled by a person.
""")
code(r"""
from plant_assistant.evals.analysis import labelled_pool

pool = labelled_pool()
pool[pool["rule"] == "cause"][["text", "summary", "note"]].head(4)
""")
md(r"""
Prompt v1 is the prompt of practice 1. Prompt v2 adds one rule at the end of the system prompt.
""")
code(r"""
v1 = prompts.extraction_messages("<entry>")[0]["content"]
v2 = prompts.extraction_messages("<entry>", summary_rule=True)[0]["content"]
print("v1 ends with\n  ..." + v1[-160:])
print("\nv2 adds\n  " + v2[len(v1):].strip())
""")
md(r"""
Prompt v2 on five entries, next to prompt v1.
""")
code(r"""
from plant_assistant.evals.harness import load_results, run_cases

five = golden[:5]
v1_five = run_cases("five-v1", course, five, mode="structured", progress=False)
v2_five = run_cases("five-v2", course, five, mode="structured-v2", progress=False)
pd.DataFrame([{"entry": c.text, "v1 summary": (v1_five[c.case_id]["record"] or {}).get("defect_summary"),
               "v2 summary": (v2_five[c.case_id]["record"] or {}).get("defect_summary")} for c in five])
""")
md(r"""
The blocks below use both prompts, 3 runs each, on 100 entries (`eval_50` and `holdout_50`), course model.

## A2. Typed fields

> **Predict.** Will the typed fields stay the same with prompt v2?
""")
code(r"""
from plant_assistant.evals.analysis import experiment_table
from plant_assistant.evals.runs import extraction_runs, run_judges

RUNS = extraction_runs(cases)  # 6 runs of 100 entries, about 5 minutes on the first run
run_judges(RUNS, profile=JUDGE)  # the judge grades every summary, about 3 minutes on the first run
exp = experiment_table(cases, JUDGE)
exp[["prompt", "setup", "valid, %", "field accuracy, %", "all fields right, %"]]
""")
md(r"""
**Result.** Both prompts give about 96.6% field accuracy and 89–90% of entries with all fields right. Runs of one prompt differ a little even at temperature 0.

The same runs on `eval_50` and on the holdout.
""")
code(r"""
from plant_assistant.evals.harness import score_cases, summary_row

rows = []
for run in ["course-v1-r1", "course-v2-r1"]:
    for part, cs in (("eval_50", golden), ("holdout_50", holdout)):
        rows.append({**summary_row(score_cases(cs, load_results(run)), run), "part": part})
pd.DataFrame(rows)[["setup", "part", "field accuracy, %", "all fields right, %"]]
""")
md(r"""
**Result.** All fields are right in 96–98% of `eval_50` entries and in 80–84% of holdout entries. `eval_50` is saturated, because practice 1 was tuned on it.

Error categories over the 100 entries. Invented means the text does not state the value. Missed means the text states it.
""")
code(r"""
from plant_assistant.evals.errors import error_counts, error_table

pd.concat({run: error_counts(error_table(cases, load_results(run))) for run in ["course-v1-r1", "course-v2-r1"]})
""")
md(r"""
**Result.** The most frequent typed error is a wrong `work_type` for both prompts.

## A3. Free text

Code checks first. English, at most 15 words, no string "null", null when no defect was found.
""")
code(r"""
from collections import Counter

from plant_assistant.evals.checks import check_summary

for run in ["course-v1-r1", "course-v2-r1"]:
    res = load_results(run)
    problems = Counter(p for c in cases for p in check_summary((res[c.case_id]["record"] or {}).get("defect_summary"),
                                                               c.expected.get("failure_mode")))
    print(run, dict(problems) or "no code problems")
""")
md(r"""
Both prompts pass the code checks. An LLM judge decides whether each summary says only what the entry says. Part B tests this judge against a person (kappa 0.82).

> **Predict.** Will prompt v2 give more faithful summaries? By how many points?
""")
code(r"""
exp[["prompt", "setup", "summary faithful, %", "faithful 95% interval"]]
""")
md(r"""
**Result.** 91–93% of summaries are faithful with v1 and 97–98% with v2, in every run. The intervals overlap a little, so the prompts are compared on the same entries with the McNemar test.
""")
code(r"""
from plant_assistant.evals.analysis import experiment_compare, experiment_compare_pooled

pd.DataFrame([experiment_compare(cases, JUDGE, i) for i in range(3)] + [experiment_compare_pooled(cases, JUDGE)])
""")
md(r"""
**Result.** v2 fixed 5–7 summaries and broke none. Runs 1 and 3 give p < 0.05, run 2 gives p = 0.063. With few changed entries one entry moves the p-value a lot. All three runs together give p = 0.031.

Entries needed for a reliable decision.
""")
code(r"""
from plant_assistant.evals.stats import n_independent, n_paired

r = experiment_compare(cases, JUDGE, 0)
p1, p2, changed = r["A right"] / r["n"], r["B right"] / r["n"], (r["B fixed"] + r["B broke"]) / r["n"]
print(f"v1 {p1:.0%} -> v2 {p2:.0%}, changed cases {changed:.0%}")
print(f"separate samples {n_independent(p1, p2)} entries per version, paired {n_paired(abs(p2 - p1), changed)} entries")
""")

md(r"""
## A4. Gate

The gate checks that the candidate is not worse than the baseline on the same entries. It fails when the candidate is significantly worse or breaks more than 2 entries.
""")
code(r"""
from plant_assistant.evals.gate import main as gate

exit_codes = [gate(["course-v1-r1", "course-v2-r1", "--metric", "all_fields_right"])]
print()
exit_codes.append(gate(["local-structured", "local-small-structured"]))  # 2B model instead of 4B, from practice 1
print("exit codes", exit_codes)
""")
md(r"""
**Result.** v2 passes. The 2B model fails with 22 broken entries. Wrong entries become draft cases for the next golden set.
""")
code(r"""
from plant_assistant.data import load_records
from plant_assistant.evals.cases import cases_from_calls, save_cases

eval_records = {r.record_id: r for r in load_records() if r.record_id in {c.case_id for c in golden}}
drafts = cases_from_calls(RUNS_DIR / "course-v2-r1" / "results.jsonl", eval_records)
save_cases(drafts, RUNS_DIR / "cases_v2_draft.jsonl")
pd.DataFrame([{"case": d.case_id, "note": d.note, "status": d.status} for d in drafts])
""")

# =============================================================================================
md(r"""
# Part B. The judge

## B1. Human labels

A person labelled the 213 unique summaries written by the 14 setups of practice 1. FAIL means the summary says something the entry does not support. The rules are in `evals/guideline_summary.md`.
""")
code(r"""
print(pool["human"].value_counts().to_dict())
pool[pool["human"] == "FAIL"]["rule"].value_counts().to_frame("FAIL labels")
""")
md(r"""
**Result.** 32 of 213 summaries fail, mostly because of an invented cause.

## B2. Judges

- **Scale.** "You are an expert. Rate the helpfulness of this answer from 1 to 10." The judge sees only the summary. A score of 7 or more is PASS.
- **Binary.** One failure mode, the entry and the registry line in the prompt, the guideline rules, a reason and then PASS or FAIL.

Both prompts run on all 213 summaries with thinking off and on. The first 20 verdicts of the binary judge are below.
""")
code(r"""
from plant_assistant.evals.analysis import load_judgements
from plant_assistant.evals.runs import pool_judges

pool_judges()  # 4 judges on 213 summaries, about 8 minutes on the first run
first20 = load_judgements("judge-binary-course")
pd.DataFrame([{"summary": row["summary"], "person": row["human"], "judge": first20[row["item_id"]]["verdict"],
               "reason": first20[row["item_id"]]["reason"]} for _, row in pool.head(20).iterrows()])
""")
md(r"""
> **Predict.** Which judge agrees with the person best?
""")
code(r"""
from plant_assistant.evals.analysis import judge_vs_human

JUDGES = ["judge-scale-course", "judge-scale-course-thinking", "judge-binary-course", "judge-binary-course-thinking"]
judge_vs_human(JUDGES)[["judge", "agreement, %", "kappa", "catches failures, %", "passes good answers, %"]]
""")
code(r"""
# Scores of the scale judge
pd.Series([d["score"] for d in load_judgements("judge-scale-course").values()]).value_counts().sort_index().to_frame("summaries")
""")
md(r"""
**Result.** The binary judge with thinking reaches kappa 0.82 and catches 88% of failures. Without thinking it reaches 0.72 and 69%. The scale judge gives 1 or 2 to almost every short summary, so it catches all failures and has kappa 0.

## B3. Disagreements
""")
code(r"""
best = load_judgements(f"judge-binary-{JUDGE}")
pool["judge"] = pool["item_id"].map(lambda i: (best.get(i) or {}).get("verdict"))
pool["judge reason"] = pool["item_id"].map(lambda i: (best.get(i) or {}).get("reason"))
pool[pool["judge"] != pool["human"]][["text", "summary", "human", "judge", "judge reason"]]
""")
md(r"""
**Result.** 11 disagreements. Some are judge errors, some come from unclear rules, and one revealed two swapped human labels.

## B4. Few labels

An expert labels 40 summaries, the judge labels all of them. Prediction-powered inference (PPI) corrects the judge estimate with the 40 labels. The full labels give the true value for comparison.
""")
code(r"""
from plant_assistant.evals.analysis import ppi_demo

ppi_demo(f"judge-binary-{JUDGE}", n_labelled=40)
""")
md(r"""
**Result.** The expert alone gives 71–94%. PPI gives 74–89% and contains the true 84.9%.

# Conclusions

| Observation | Meaning |
|-------------|---------|
| v2 raised faithful summaries from 91–93% to 97–98% and broke nothing | measure the targeted field and check the rest |
| run 2 gave p = 0.063 | one run is one sample |
| 96–98% on `eval_50`, 80–84% on the holdout | a set used for tuning saturates |
| the scale judge catches 100% of failures and has kappa 0 | check both error rates of a judge |
| thinking raised the judge's catch rate from 69% to 88% | judge settings are measurable |
| a disagreement found a labelling error | read every disagreement |

# About the data

The data is synthetic and was built from the labels, so every entry is labelled automatically. Real data needs manual labelling. Synthetic test data has the traps from the lecture (block 3), such as easy cases, copied wording, one model family and polite simulated users.

1. Start from real logs, and add synthetic cases only to fill gaps.
2. Generate by dimensions with a different model family.
3. Remove duplicates, copied wording and easy cases, and let an expert check a sample.
4. Label with a written guideline and two people, and measure kappa.
5. Keep a holdout and add harder cases when the set saturates.

# Homework

1. Label 30 summaries with `evals/guideline_summary.md` and compute Cohen's kappa against `data/summary_labels.jsonl`.
2. Write a judge for the criterion "the summary describes the defect, not only the work". Label 30 cases and report kappa.
3. Write prompt v3 for the `work_type` error from A2 and compare it with v2 on the holdout with the paired test.
""")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    nb["cells"] = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t) for kind, t in cells]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"{len(cells)} cells -> {OUT}")


if __name__ == "__main__":
    main()
