"""Retrieval eval: metrics, end-to-end grading by code, loss attribution, the ablation table.

Metrics of one question (questions with the status "not in documents" have no evidence: no retrieval metrics):
- recall@k: every "answer" evidence item has a chunk in the top-k of search (with separate rows: top-5 + 2 rows);
  documents linked by the registry are not search, they count only in evidence recall;
- evidence recall: every evidence item (answer, support, action) has a chunk in the context that the model saw;
- MRR: 1 / rank of the first chunk of the first answer item in the search list (0 if not in the top-`depth`);
- end-to-end: code checks the value and the unit, all values of a "what changed" answer, or the status;
- verified: the answer is right AND its citations pass the three checks (verify.py).

How a configuration fails matters as much as how often (`outcome`, one per question):
- false refusal: a value (or a conflict) was expected, the answer gives none: "not in the documents" or
  "needs a person". Safe, but it has a price: the engineer searches the archive by hand or waits for a person;
- wrong: a value that is not the expected one, or one side of a conflict presented as the answer;
- invented: a value where the right answer is "not in the documents";
- invalid: no answer of the schema (the run cannot use it at all).
Statuses on "not in the documents" questions: "not_in_documents" and "needs_human" are both accepted (no value is
given, the engineer is sent to the documents or to a person); a value or "conflict" is not.
The needs_human set (data/needs_human.jsonl, not in the retrieval set): only "needs_human" is right. "Not in the
documents" is wrong there (the norm is in the documents, the equipment data are not) but safe; a value is the
dangerous failure: the system chose a class, a station or a measurement that nobody gave it.

A wrong answer is attributed to one link of the chain, the first that lost it:
reading (the value is in no chunk) -> search (an answer chunk is not in the context) -> assembly (the answer
chunk is there, another required chunk is not) -> model (everything was in the context).
Rows of the ablation table are compared in pairs on the same questions (McNemar, practice 2).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .cases import SLICES, TestCase
from .evals.stats import compare, wilson
from .store import Chunk
from .units import canonical_unit, has_value, same_value

LINKS = ["reading", "search", "assembly", "model"]


# ---------------------------------------------------------------------------------------------
# Grading by code
# ---------------------------------------------------------------------------------------------
def grade(case: TestCase, ans: dict | None) -> tuple[bool, str]:
    """(correct, reason)."""
    if ans is None:
        return False, "invalid answer"
    exp = case.expected
    status = ans.get("status", "answered")
    if exp.get("status") == "conflict":
        return (True, "ok") if status == "conflict" else (False, f"status {status}, expected conflict")
    if exp.get("status") == "not_in_documents":
        if status == "not_in_documents" or (status != "conflict" and ans.get("value") in (None, "")):
            return True, "ok"
        return False, "invented a value"
    if exp.get("status") == "needs_human":
        if status == "needs_human":
            return True, "ok"
        if status == "answered" and ans.get("value") not in (None, ""):
            return False, "answered without the missing data"
        return False, f"status {status}, expected needs_human"
    if status != "answered":
        return False, f"status {status}"
    if "values" in exp:
        text = f"{ans.get('answer', '')} {ans.get('value') or ''}"
        missing = [v for v in exp["values"] if not has_value(text, v)]
        return (True, "ok") if not missing else (False, f"missing {missing}")
    if not same_value(ans.get("value"), exp["value"]):
        return False, "wrong value" if ans.get("value") not in (None, "") else "said not in documents"
    if exp.get("unit") and canonical_unit(ans.get("unit")) != canonical_unit(exp["unit"]):
        return False, "wrong unit"
    return True, "ok"


# ---------------------------------------------------------------------------------------------
# Per question
# ---------------------------------------------------------------------------------------------
REFUSALS = ("not_in_documents", "needs_human")
OUTCOMES = ["correct", "false refusal", "wrong", "invented", "invalid"]


def outcome(case: TestCase, ans: dict | None, correct: bool) -> str:
    """How the answer failed (see the module docstring)."""
    if correct:
        return "correct"
    if ans is None:
        return "invalid"
    no_value = ans.get("status") in REFUSALS or (ans.get("status", "answered") == "answered"
                                                 and ans.get("value") in (None, ""))
    if case.status == "not_in_documents":
        return "invented"
    return "false refusal" if no_value else "wrong"


def retrieval_metrics(case: TestCase, rec: dict, k: int = 5) -> dict:
    if not case.evidence:
        return {"recall@k": None, "evidence_recall": None, "mrr": None}
    top = set(rec.get("top") or rec["retrieved"][:k])  # S2b+: top-5 chunks and up to 2 rows of the row collection
    context = set(rec["context"])
    answers = case.gold("answer")
    recall = all(g & top for g in answers)
    evidence = all(g & context for g in case.gold())
    first = answers[0]
    rank = next((i for i, cid in enumerate(rec["retrieved"], start=1) if cid in first), None)
    return {"recall@k": recall, "evidence_recall": evidence, "mrr": 1 / rank if rank else 0.0}


def reading_lost(case: TestCase, chunks_by_id: dict[str, Chunk]) -> bool:
    """The value is in no evidence chunk of the store: lost when the documents were read (topic 3)."""
    values = case.expected.get("values") or ([case.expected["value"]] if "value" in case.expected else [])
    if not values:
        return False
    text = " ".join(chunks_by_id[c].text for g in case.gold() for c in g if c in chunks_by_id)
    return not all(has_value(text, v) for v in values)


def attribute(case: TestCase, rec: dict, correct: bool, chunks_by_id: dict[str, Chunk]) -> str | None:
    if correct:
        return None
    if not case.evidence:  # "not in documents": nothing to find, the model invented a value
        return "model"
    if reading_lost(case, chunks_by_id):
        return "reading"
    context = set(rec["context"])
    if not all(g & context for g in case.gold("answer")):
        return "search"
    if not all(g & context for g in case.gold()):
        return "assembly"
    return "model"


def score(cases: list[TestCase], records: list[dict], chunks_by_id: dict[str, Chunk], k: int = 5) -> pd.DataFrame:
    """One row per question of one configuration."""
    by_case = {c.case_id: c for c in cases}
    rows = []
    for rec in records:
        case = by_case[rec["case_id"]]
        ok, reason = grade(case, rec.get("answer")) if "answer" in rec else (None, "")
        checks = rec.get("checks") or {}
        rows.append({"case_id": case.case_id, "slice": case.slice, "config": rec["config"],
                     **retrieval_metrics(case, rec, k), "correct": ok, "reason": reason,
                     "verified": bool(ok and checks.get("verified")),
                     "link": attribute(case, rec, bool(ok), chunks_by_id) if ok is not None else None,
                     "status": (rec.get("answer") or {}).get("status"),
                     "value": (rec.get("answer") or {}).get("value"),
                     "expected": case.status,
                     "outcome": outcome(case, rec.get("answer"), bool(ok)) if ok is not None else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# The ablation table
# ---------------------------------------------------------------------------------------------
def _share(series: pd.Series) -> tuple[int, int]:
    s = series.dropna()
    return int(s.astype(bool).sum()), len(s)


def _fmt(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def ablation(scores: dict[str, pd.DataFrame], previous: dict[str, str | None]) -> pd.DataFrame:
    """Rows: configurations. Columns: metrics with Wilson intervals, and the paired verdict against the row
    named in `previous` (on the same questions)."""
    out = []
    for name, df in scores.items():
        row = {"config": name, "n": len(df)}
        for col, label in (("recall@k", "recall@5"), ("evidence_recall", "evidence recall"), ("correct", "end-to-end"),
                           ("verified", "correct + verified")):
            k, n = _share(df[col])
            row[label] = _fmt(k, n)
            row[f"_{label}"] = k / n if n else None
        row["MRR"] = round(df["mrr"].dropna().mean(), 3) if df["mrr"].notna().any() else None
        nid = df[df.slice == "not_in_documents"]
        row["abstain ok"] = f"{int(nid.correct.sum())}/{len(nid)}" if len(nid) else "—"
        conf = df[df.slice == "conflict"]
        row["conflict ok"] = f"{int(conf.correct.sum())}/{len(conf)}" if len(conf) else "—"
        row.update(failure_modes(df))
        prev = previous.get(name)
        if prev and prev in scores:
            a = scores[prev].set_index("case_id")
            b = df.set_index("case_id")
            common = [c for c in b.index if c in a.index]
            r = compare(list(a.loc[common, "correct"].astype(bool)), list(b.loc[common, "correct"].astype(bool)))
            row["vs"] = prev
            row["e2e: fixed / broke"] = f"+{r['B fixed']} / -{r['B broke']}"
            row["e2e verdict"] = f"{r['decision']} (p = {r['p_value']:.3f})"
            ev_a = a.loc[common, "evidence_recall"]
            ev_b = b.loc[common, "evidence_recall"]
            mask = ev_a.notna() & ev_b.notna()
            r2 = compare(list(ev_a[mask].astype(bool)), list(ev_b[mask].astype(bool)))
            row["evidence verdict"] = f"{r2['decision']} (+{r2['B fixed']}/-{r2['B broke']}, p = {r2['p_value']:.3f})"
        out.append(row)
    return pd.DataFrame(out)


def failure_modes(df: pd.DataFrame) -> dict[str, str]:
    """Columns of the ablation table: false refusals of the questions with an answer in the documents, wrong and
    invented values of all questions, right abstentions of the "not in the documents" questions."""
    if "outcome" not in df or df["outcome"].isna().all():
        return {}
    answerable = df[df.expected.isin(["answered", "conflict"])]
    nid = df[df.expected == "not_in_documents"]
    o = df.outcome.value_counts()
    fr = int((answerable.outcome == "false refusal").sum())
    escalated = int(((answerable.outcome == "false refusal") & (answerable.status == "needs_human")).sum())
    ok_nid = nid[nid.correct.astype(bool)]
    nh = int((ok_nid.status == "needs_human").sum())
    pct = lambda k, n: f"{k}/{n} = {k / n:.0%}" if n else "—"  # noqa: E731
    return {"false refusals": pct(fr, len(answerable)) + (f" ({escalated} needs a person)" if escalated else ""),
            "wrong or invented": f"{int(o.get('wrong', 0)) + int(o.get('invented', 0))}"
                                 + (f" ({int(o.get('invented', 0))} invented)" if o.get("invented", 0) else ""),
            "right abstentions": pct(len(ok_nid), len(nid)) + (f" ({nh} needs a person)" if nh else ""),
            "invalid": str(int(o.get("invalid", 0)))}


def by_slice(scores: dict[str, pd.DataFrame], metric: str = "correct") -> pd.DataFrame:
    """Counts per slice (k/n), not percentages: a slice of 4 questions has no meaningful percentage."""
    rows = []
    for name, df in scores.items():
        row = {"config": name}
        for s in SLICES:
            part = df[df.slice == s][metric].dropna()
            row[s] = f"{int(part.astype(bool).sum())}/{len(part)}" if len(part) else "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("config")


def losses(df: pd.DataFrame) -> dict[str, int]:
    c = Counter(df["link"].dropna())
    return {link: c.get(link, 0) for link in LINKS}


def gate(baseline: pd.DataFrame, candidate: pd.DataFrame, metric: str = "correct", max_broken: int = 2) -> dict:
    """The regression gate of practice 2 on the retrieval set: the candidate fails if it is significantly worse
    or breaks more than `max_broken` questions that the baseline answered right. Topic 6 runs it in CI."""
    a, b = baseline.set_index("case_id"), candidate.set_index("case_id")
    common = [c for c in b.index if c in a.index]
    xa, xb = list(a.loc[common, metric].astype(bool)), list(b.loc[common, metric].astype(bool))
    r = compare(xa, xb)
    reasons = []
    if r["decision"] == "B is worse":
        reasons.append("significantly worse than the baseline")
    if r["B broke"] > max_broken:
        reasons.append(f"breaks {r['B broke']} questions that the baseline answered (limit {max_broken})")
    broken = [c for c, x, y in zip(common, xa, xb) if x and not y]
    return {**r, "metric": metric, "passed": not reasons, "reasons": reasons, "broken_cases": broken}


def cost(records: list[dict]) -> dict:
    """Latency and cost per question: median seconds (search measured here + model latency as measured when the
    answer was made), model calls (all: chat, embedding of the question, cross-encoder) and chat calls, chat tokens,
    cross-encoder tokens. A replayed answer keeps the latency of the live call."""
    c = [r["cost"] for r in records if r.get("cost")]
    if not c:
        return {}
    secs = pd.Series([x["search_s"] + x["model_s"] for x in c])
    return {"median s / question": round(float(secs.median()), 2), "p90 s": round(float(secs.quantile(0.9)), 2),
            "search ms (median)": round(1000 * float(pd.Series([x["search_s"] for x in c]).median()), 1),
            "model calls / question": round(sum(x["calls"] for x in c) / len(c), 2),
            "chat calls / question": round(sum(x.get("chat_calls", x["calls"]) for x in c) / len(c), 2),
            "prompt tokens / question": round(sum(x["prompt_tokens"] for x in c) / len(c)),
            "completion tokens / question": round(sum(x["completion_tokens"] for x in c) / len(c)),
            # chat tokens above; the cross-encoder reads its own tokens (no chat model, no generation)
            "cross-encoder tokens / question": round(sum(x.get("reranker_tokens", 0) for x in c) / len(c))}


def cost_table(records: dict[str, list[dict]]) -> pd.DataFrame:
    return pd.DataFrame({name: cost(recs) for name, recs in records.items()}).T.rename_axis("config")
