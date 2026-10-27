"""Build the reference results of practice 4 and the replay files.

Live (the lecturer, with the course token):   uv run python scripts/build_reference.py
From replay (anyone, no network):              uv run python scripts/build_reference.py --replay

Steps:
1. the store (store/ and its committed copy reference/store/) and the retrieval set (data/questions.jsonl);
2. every configuration of the ladder (S0-S6) and of the mechanism branch on all questions;
   answers, reranker calls and translations go to the course model (live) or come from data/replay/;
3. reference/ablation.csv, reference/slices_*.csv, reference/losses.csv, reference/per_question.csv,
   reference/s0_failures.md (five failed answers for task 1 of the lecture) and reference/results.md.

Live calls are logged; identical requests are sent once (configurations that give the same context ask the
same question). The logs become the replay files: one entry per request, without the prompt text (the replay
key is a hash of the request), so the files stay small.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from plant_assistant import REFERENCE_DIR, REPLAY_DIR, RUNS_DIR, STORE_DIR  # noqa: E402
from plant_assistant.audit import audit_record  # noqa: E402
from plant_assistant.cases import SLICES, counts_by_slice, load_cases  # noqa: E402
from plant_assistant import DATA_DIR  # noqa: E402
from plant_assistant.config import get_profile  # noqa: E402
from plant_assistant.corpus import build_corpus  # noqa: E402
from plant_assistant.jsonl import read_json_records  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient, ReplayClient  # noqa: E402
from plant_assistant.pipeline import CONFIGS, CONFLICT_DEMO, LADDER, PREVIOUS, System  # noqa: E402
from plant_assistant import reval  # noqa: E402
from plant_assistant.store import load_store  # noqa: E402

TASKS = ("answers", "rerank", "translate")
RUN_CONFIGS = [c.name for c in LADDER] + ["M1t", "M3@S0", "M3@S6"]
TODO_CONFIGS = {"M1@S0": "S0", "M2@S0": "S0", "M1@S6": "S6", "M2@S6": "S6"}
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
HOLDOUT_CONFIGS = [c.name for c in LADDER] + ["M3@S0", "M3@S6"]
# the questions that mixed table rows (S2) broke in the first reference run
BROKEN_BY_MIXED_ROWS = ["d01", "d02", "d04", "z01", "z02", "z04", "z05", "i02", "g01"]


class MemoClient:
    """Sends each distinct request once: the same prompt gets the same saved answer in every configuration."""

    def __init__(self, client):
        self.client, self.profile_name, self.model = client, client.profile_name, client.model
        self._memo: dict[str, object] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def chat(self, messages, response_format=None, meta=None):
        from plant_assistant.llm import request_hash

        key = request_hash(self.model, messages, response_format)
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._memo:
                resp = self.client.chat(messages, response_format=response_format, meta=meta)
                if not resp.ok:  # one more try for a network error
                    resp = self.client.chat(messages, response_format=response_format, meta=meta)
                self._memo[key] = resp
            return self._memo[key]


def compact_replay(log: Path, out: Path, merge: bool = False) -> int:
    """Call log -> replay file: one entry per request hash, successful answers only, no prompt text.
    `merge` keeps the entries already in the replay file (adding demo calls to a finished run)."""
    seen: dict[str, dict] = {}
    if merge and out.exists():
        for e in read_json_records(out):
            seen[e["request_hash"]] = e
    for e in read_json_records(log):
        if e["response"].get("error"):
            continue
        e = {k: v for k, v in e.items() if k not in ("messages", "ts", "run_id")}
        e["response"] = {k: v for k, v in e["response"].items() if k not in ("reasoning",)}
        seen.setdefault(e["request_hash"], e)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in seen.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(seen)


class ReplayFirst:
    """Live building that keeps the saved answers: a request that is already in the replay file gets its saved
    answer (a re-run does not change rows whose prompts did not change); a new request goes to the model."""

    def __init__(self, replay: ReplayClient, live):
        self.replay, self.live = replay, live
        self.profile_name, self.model = live.profile_name, live.model

    def chat(self, messages, response_format=None, meta=None):
        r = self.replay.chat(messages, response_format=response_format, meta=meta)
        return self.live.chat(messages, response_format=response_format, meta=meta) if r.error == "ReplayMiss" else r


def clients(replay: bool, fresh: bool = False) -> dict:
    if replay:
        return {t: ReplayClient(REPLAY_DIR / f"{t}_course.jsonl") for t in TASKS}
    profile = get_profile("course")
    out = {}
    for t in TASKS:
        log = RUNS_DIR / "lecturer" / f"{t}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.unlink(missing_ok=True)
        logger = CallLogger(log)
        live = MemoClient(LLMClient(profile, logger=logger))
        saved = REPLAY_DIR / f"{t}_course.jsonl"
        out[t] = live if fresh or not saved.exists() else \
            ReplayFirst(ReplayClient(saved, logger=logger, strict=False), live)
    return out


# ---------------------------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_markdown(index=index)


def describe_chunk(c) -> str:
    where = (f"cl. {c.clause}" if c.chunk_type == "clause" and c.clause else
             f"Table {c.table_no}" + (f", row {c.row_key}" if c.row_key else "") if c.table_no else c.chunk_type)
    rev = f" rev. {c.revision}" if c.revision is not None else ""
    amd = f" Amd. {c.amendment}" if c.amendment else ""
    return f"{c.doc_id}{rev}{amd} {where}"


# Chosen for task 1 of the lecture (all about P-101): the lecture question itself, a Cyrillic tag, an old wording
# found instead of the amendment (search); the clause without its table (assembly); rev. 2 in the context but
# rev. 3 in the answer (model). The store has no reading losses: born-digital documents.
PREFERRED_FAILURES = ["m01", "g01", "v03", "m06", "i01"]


def pick_failures(cases, df: pd.DataFrame, n: int = 5) -> list[str]:
    """Five failed S0 answers for task 1 of the lecture: the preferred ones if they still fail, otherwise
    one per link where possible, P-101 first."""
    by_case = {c.case_id: c for c in cases}
    wrong = df[df.correct == False]  # noqa: E712
    if set(PREFERRED_FAILURES) <= set(wrong.case_id):
        return PREFERRED_FAILURES
    about_p101 = {cid for cid in wrong.case_id if "P-101" in by_case[cid].question_en or "Р-101" in
                  by_case[cid].question_ru or "P101" in by_case[cid].question_ru}
    picked: list[str] = []
    for link in reval.LINKS:
        part = wrong[wrong.link == link]
        if part.empty:
            continue
        pool = [c for c in part.case_id if c in about_p101] or list(part.case_id)
        picked.append(pool[0])
    rest = [c for c in wrong.case_id if c not in picked]
    rest.sort(key=lambda c: (c not in about_p101, c))
    for c in rest:
        if len(picked) >= n:
            break
        if df.set_index("case_id").loc[c, "link"] not in [df.set_index("case_id").loc[p, "link"] for p in picked] \
                or len(picked) >= len(set(wrong.link)):
            picked.append(c)
    return picked[:n]


def failures_md(cases, records: dict, df: pd.DataFrame, by_id: dict, picked: list[str]) -> str:
    by_case = {c.case_id: c for c in cases}
    recs = {r["case_id"]: r for r in records}
    out = ["# Five failed answers of S0 (task 1 of lecture 4)\n",
           "Configuration S0: BM25 over raw chunks, top-5, no filters, the course model answers from the top-5.",
           "Task: for each answer write where it was lost: R = reading, S = search, C = context assembly, M = model.\n"]
    for i, cid in enumerate(picked, start=1):
        c, r = by_case[cid], recs[cid]
        row = df.set_index("case_id").loc[cid]
        ans = r.get("answer") or {}
        exp = c.expected.get("value") or c.expected.get("values") or c.expected.get("status")
        out += [f"## {i}. {cid} ({c.slice}) — lost in: **{row.link}**\n",
                (f"- Question (asked in English, sent to search as is): {c.question_en}" if c.language == "en" else
                 f"- Question (asked in Russian): {c.question_ru}\n- Translation: {c.question_en}"),
                f"- Expected: {exp} {c.expected.get('unit') or ''}".rstrip(),
                f"- System answer: {ans.get('answer', '(invalid)')} (value: {ans.get('value')})",
                f"- Gold evidence: {'; '.join(e['any_of'][0] for e in c.evidence) or 'none: not in the documents'}",
                "- Top-5 of search: " + "; ".join(describe_chunk(by_id[x]) for x in r["retrieved"][:5]),
                f"- Why this link: {reason_text(row.link)}\n"]
    return "\n".join(out)


def reason_text(link: str) -> str:
    return {"reading": "the value is in no chunk of the store",
            "search": "no chunk with the answer reached the context",
            "assembly": "the chunk with the answer is in the context, a required support or action chunk is not",
            "model": "every needed chunk was in the context, the answer is still wrong"}[link]


def demo(system: System, cases) -> dict:
    """The demo calls of the notebook that are not part of a configuration run."""
    cid, ctx = CONFLICT_DEMO
    case = next(c for c in cases if c.case_id == cid)
    return system.answer_from_context(case, ctx, CONFIGS["S6"])


# ---------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="answers from data/replay/ instead of the course model")
    ap.add_argument("--configs", nargs="*", default=RUN_CONFIGS)
    ap.add_argument("--only-demos", action="store_true", help="live: only the demo calls, merged into the replay")
    ap.add_argument("--fresh", action="store_true", help="live: ask the model again even for saved requests")
    args = ap.parse_args()
    if args.only_demos:
        _, chunks = load_store(STORE_DIR)
        cl = clients(False)
        system = System(chunks, answer_client=cl["answers"])
        demo(system, load_cases())
        print("replay answers:", compact_replay(RUNS_DIR / "lecturer" / "answers.jsonl",
                                                REPLAY_DIR / "answers_course.jsonl", merge=True))
        return

    summary = build_corpus(STORE_DIR)
    shutil.copytree(STORE_DIR, REFERENCE_DIR / "store", dirs_exist_ok=True)
    print("store:", summary)
    _, chunks = load_store(STORE_DIR)
    cases = load_cases()
    cl = clients(args.replay, args.fresh)
    system = System(chunks, answer_client=cl["answers"], rerank_client=cl["rerank"], translate_client=cl["translate"])
    holdout = load_cases(HOLDOUT_FILE)

    records, scores, h_records, h_scores = {}, {}, {}, {}
    for name in args.configs:
        cfg = CONFIGS[name]
        records[name] = system.run(cases, cfg, workers=8)
        scores[name] = reval.score(cases, records[name], system.by_id)
        df = scores[name]
        print(f"{name}: recall@5 {df['recall@k'].dropna().mean():.3f}, evidence {df['evidence_recall'].dropna().mean():.3f}, "
              f"e2e {df.correct.mean():.3f}, verified {df.verified.mean():.3f}")

    for name in HOLDOUT_CONFIGS:
        if name in args.configs:
            h_records[name] = system.run(holdout, CONFIGS[name], workers=8, progress=False)
            h_scores[name] = reval.score(holdout, h_records[name], system.by_id)
            print(f"holdout {name}: e2e {h_scores[name].correct.mean():.3f}")
    conflict_demo = demo(system, cases)
    if not args.replay:
        for t in TASKS:
            n = compact_replay(RUNS_DIR / "lecturer" / f"{t}.jsonl", REPLAY_DIR / f"{t}_course.jsonl")
            print(f"replay {t}: {n} requests")

    REFERENCE_DIR.mkdir(exist_ok=True)
    table = reval.ablation(scores, PREVIOUS)
    for name, base in TODO_CONFIGS.items():
        table = pd.concat([table, pd.DataFrame([{"config": name, "n": "TODO",
                                                 "recall@5": "TODO: no embedding vectors yet (see README)",
                                                 "vs": base}])], ignore_index=True)
    public = table[[c for c in table.columns if not c.startswith("_")]]
    public.to_csv(REFERENCE_DIR / "ablation.csv", index=False)
    slices_e2e = reval.by_slice(scores, "correct")
    slices_recall = reval.by_slice(scores, "recall@k")
    slices_evidence = reval.by_slice(scores, "evidence_recall")
    slices_e2e.to_csv(REFERENCE_DIR / "slices_end_to_end.csv")
    slices_recall.to_csv(REFERENCE_DIR / "slices_recall.csv")
    slices_evidence.to_csv(REFERENCE_DIR / "slices_evidence.csv")
    losses = pd.DataFrame({name: reval.losses(scores[name]) for name in ("S0", "S6") if name in scores}).T
    losses.index.name = "config"
    losses.to_csv(REFERENCE_DIR / "losses.csv")
    pd.concat(scores.values()).to_csv(REFERENCE_DIR / "per_question.csv", index=False)
    costs = reval.cost_table(records)
    costs.to_csv(REFERENCE_DIR / "cost.csv")
    h_table = reval.ablation(h_scores, PREVIOUS)
    h_public = h_table[[c for c in h_table.columns if not c.startswith("_") and c not in ("abstain ok", "conflict ok")]]
    h_public.to_csv(REFERENCE_DIR / "holdout_ablation.csv", index=False)
    pd.concat(h_scores.values()).to_csv(REFERENCE_DIR / "holdout_per_question.csv", index=False)
    broken = pd.DataFrame({name: scores[name].set_index("case_id").loc[BROKEN_BY_MIXED_ROWS, "correct"]
                           for name in ("S1", "S2", "S2b", "S3") if name in scores})

    # five S0 failures for task 1 of the lecture
    picked = pick_failures(cases, scores["S0"])
    (REFERENCE_DIR / "s0_failures.md").write_text(failures_md(cases, records["S0"], scores["S0"], system.by_id, picked),
                                                  encoding="utf-8")

    # the P-101 question: S6 (BM25) and M3@S6 (with the reranker), the audit record of the answer that holds
    p101_s6 = next(r for r in records["S6"] if r["case_id"] == "m01")
    p101 = next(r for r in records["M3@S6"] if r["case_id"] == "m01")
    (REFERENCE_DIR / "audit_p101_S6.json").unlink(missing_ok=True)
    audit = audit_record(p101, system.index_version, system.by_id)
    (REFERENCE_DIR / "audit_p101.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    gate = reval.gate(scores["S5"], scores["S6"])

    lines = [
        "# Practice 4 — reference results",
        "",
        "Lecturer's run on the course model `iairlab/qwen3.8-27b` (thinking off), "
        f"{'replayed' if args.replay else 'live'}. Search and filters are deterministic; model answers may differ "
        "by a question or two in a live run (compare with the intervals).",
        "",
        f"- Store: {summary['documents']} documents, {summary['pages']} pages, {summary['chunks']} chunks "
        f"({summary['table_rows']} of them rows of the long valve table); index version `{summary['index_version']}`.",
        f"- Retrieval set: {len(cases)} questions; by slice: {counts_by_slice(cases)}.",
        "- Dense search (M1, M2) is **not measured yet**: the course server has no embedding model and this machine "
        "had no Ollama. The code path and the replay format are ready (README, 'Dense search').",
        "",
        "## Ablation table",
        "",
        "Every row adds one change to the row named in `vs` and is compared with it in pairs on the same questions "
        "(exact McNemar test, practice 2). Intervals: Wilson, 95%. recall@5 and evidence recall exclude the "
        "questions whose right answer is \"not in the documents\" (no evidence). From S2b on, the search result is the "
        "top-5 chunks plus up to 2 rows of the row collection; documents that the registry links to the equipment "
        "(S3+) are not search: they count in evidence recall, not in recall@5.",
        "",
        md_table(public.fillna("")),
        "",
        "## Latency and cost per question",
        "",
        "Median seconds per question = search (measured on the lecturer's laptop) + model latency of every call "
        "(as measured when the answer was made; replay keeps it). Calls and tokens are per question, all calls "
        "counted (an uncached system).",
        "",
        md_table(costs, index=True),
        "",
        "## Mixed rows (S2) against a separate row collection (S2b): the questions S2 broke",
        "",
        "The first reference run (rows mixed with the clauses) broke these questions; True = right.",
        "",
        md_table(broken.astype(str), index=True),
        "",
        "## Holdout: questions written by role, not by slice",
        "",
        f"{len(holdout)} questions drafted by `{next(iter(c.source for c in holdout), '')}` from the documents and four "
        "role descriptions (operator, mechanic, reliability engineer, auditor), filtered by code (evidence resolves, "
        "value in the evidence, lexical leak <= 0.6, no near-copy of the retrieval set) and reviewed; the review list "
        "is `holdout_review.md`. The ladder was never shown to the drafter.",
        "",
        md_table(h_public.fillna("")),
        "",
        "## End-to-end by slice (counts, not percentages)",
        "",
        md_table(slices_e2e, index=True),
        "",
        "## recall@5 by slice",
        "",
        md_table(slices_recall, index=True),
        "",
        "## Evidence recall by slice",
        "",
        md_table(slices_evidence, index=True),
        "",
        "## Where answers are lost (wrong answers by link)",
        "",
        md_table(losses, index=True),
        "",
        "Reading: the value is in no chunk; search: no chunk with the answer in the context; assembly: the answer "
        "chunk is there, a required support or action chunk is not; model: everything was in the context. "
        "Practice 3 ceiling with the known page: 98%.",
        "",
        "## The P-101 question",
        "",
        f"S6 (BM25): status `{(p101_s6.get('answer') or {}).get('status')}`; top-5 "
        + ", ".join(describe_chunk(system.by_id[c]) for c in p101_s6["top"])
        + ". Table 3 is at rank 6: the evidence pack has nothing to start from. M3@S6 (reranker over the top-20):",
        "",
        f"Question (asked in Russian): {next(c for c in cases if c.case_id == 'm01').question_ru}",
        "",
        "```json",
        json.dumps(p101.get("answer"), ensure_ascii=False, indent=1),
        "```",
        "",
        f"Checks: verified = {p101['checks']['verified']}, exists = {p101['checks']['exists']}, "
        f"in force = {p101['checks']['in_force']}, supports = {p101['checks']['supports']}. "
        "Citations: " + "; ".join(f"{x['chunk_id']} (exists {x['exists']}, in force {x['in_force']}, supports "
                                   f"{x['supports']})" for x in p101["checks"]["citations"]),
        "",
        "The audit record is in `audit_p101.json`.",
        "",
        "## The contract on a conflict (demo: both sides of the P-205 conflict put into the context by hand)",
        "",
        f"Status: `{(conflict_demo['answer'] or {}).get('status')}`; answer: {(conflict_demo['answer'] or {}).get('answer')}",
        "",
        "## Regression gate S6 against S5 (practice 2 rule)",
        "",
        f"{gate['A right']} -> {gate['B right']} right of {gate['n']}, fixed {gate['B fixed']}, broke {gate['B broke']}"
        f" ({', '.join(gate['broken_cases']) or 'none'}), p = {gate['p_value']}: **{'PASS' if gate['passed'] else 'FAIL'}**"
        + (f" ({'; '.join(gate['reasons'])})" if gate["reasons"] else ""),
        "",
        "## Five S0 failures for task 1 of the lecture",
        "",
        "See `s0_failures.md`: " + ", ".join(f"{cid} ({scores['S0'].set_index('case_id').loc[cid, 'link']})"
                                             for cid in picked) + ".",
    ]
    (REFERENCE_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(public.to_string())


if __name__ == "__main__":
    main()
