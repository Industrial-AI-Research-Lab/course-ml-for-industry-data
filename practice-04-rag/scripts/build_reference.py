"""Build the reference results of practice 4 and the replay files.

Live (the lecturer, with the course token):   uv run python scripts/build_reference.py
From replay (anyone, no network):              uv run python scripts/build_reference.py --replay
Only the needs_human set, live, merged into the replay:  uv run python scripts/build_reference.py --only-needs-human

Steps:
1. the store (store/ and its committed copy reference/store/) and the retrieval set (data/questions.jsonl);
2. every configuration of the ladder (S0-S6) and of the mechanism branch on all questions;
   answers, reranker calls (LLM and cross-encoder) and translations go to the course server (live) or come from
   data/replay/; dense rows use the saved vectors data/replay/vectors/e5-mistral.* (scripts/build_vectors.py);
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
from plant_assistant import answer as A  # noqa: E402
from plant_assistant.audit import audit_record  # noqa: E402
from plant_assistant.cases import SLICES, counts_by_slice, load_cases  # noqa: E402
from plant_assistant import DATA_DIR  # noqa: E402
from plant_assistant.config import get_profile  # noqa: E402
from plant_assistant.corpus import build_corpus  # noqa: E402
from plant_assistant.jsonl import read_json_records  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient, ReplayClient  # noqa: E402
from plant_assistant.crossencoder import CrossEncoderClient  # noqa: E402
from plant_assistant.pipeline import CONFIGS, CONFLICT_DEMO, CONTRACT_VARIANTS, LADDER, PREVIOUS, System  # noqa: E402
from plant_assistant.search import VectorStore  # noqa: E402
from plant_assistant import reval  # noqa: E402
from plant_assistant.store import load_store  # noqa: E402

TASKS = ("answers", "answers_thinking", "rerank", "translate", "crossencoder")
TASK_PROFILES = {"crossencoder": "rerank-course", "answers_thinking": "course-thinking"}  # others: the course profile
# measured rows that a live build must not call again: replay only (a request that is not saved fails loudly)
# unless --live-mechanisms; the embedding vectors are always the saved ones
REPLAY_ONLY_TASKS = ("rerank", "translate", "crossencoder")
VECTORS = REPLAY_DIR / "vectors" / "e5-mistral"
MECHANISM_ROWS = ["M1@S0", "M2@S0", "M3@S0", "M4@S0", "M1@S6", "M2@S6", "M3@S6", "M4@S6"]
# the rows of the first answer contract (citation objects, no limits), kept for "before and after"
VARIANT_ROWS = [c.name for c in CONTRACT_VARIANTS]  # "S6:v1", "M*@S6:v1", "S6:v2", "M*@S6:v2", "S6:think"
# (new, old): every S6-based row with contract v1 -> v2, then v2 -> v3; thinking mode against v2
S6_ROWS = ("S6", "M1@S6", "M2@S6", "M3@S6", "M4@S6")
BEFORE_AFTER = [(f"{c}:v2", f"{c}:v1") for c in S6_ROWS] + [(c, f"{c}:v2") for c in S6_ROWS] + [("S6:think", "S6:v2")]
RUN_CONFIGS = [c.name for c in LADDER] + ["M1t"] + MECHANISM_ROWS + VARIANT_ROWS
HOLDOUT_FILE = DATA_DIR / "holdout.jsonl"
HOLDOUT_CONFIGS = [c.name for c in LADDER] + MECHANISM_ROWS + VARIANT_ROWS
NEEDS_HUMAN_FILE = DATA_DIR / "needs_human.jsonl"
# S5 has no status field (the simple answer gives a value or none): the row shows what the contract adds
NEEDS_HUMAN_CONFIGS = ["S5", "S6", "M1@S6", "M2@S6", "M3@S6", "M4@S6"] + VARIANT_ROWS
CONTRACT_SLICES = ["multi_evidence", "what_changed", "conflict", "language", "point_in_time"]
# measured times of the rows already in the reference: a rebuild keeps them (search time on a laptop varies by a
# millisecond between runs, and the rows did not change)
TIMING_COLUMNS = ["median s / question", "p90 s", "search ms (median)"]
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


def clients(replay: bool, fresh: bool = False, live_mechanisms: bool = False) -> dict:
    if replay:
        return {t: ReplayClient(REPLAY_DIR / f"{t}_course.jsonl") for t in TASKS}
    out = {}
    for t in TASKS:
        if t in REPLAY_ONLY_TASKS and not live_mechanisms:
            out[t] = ReplayClient(REPLAY_DIR / f"{t}_course.jsonl")
            continue
        log = RUNS_DIR / "lecturer" / f"{t}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.unlink(missing_ok=True)
        logger = CallLogger(log)
        profile = get_profile(TASK_PROFILES.get(t, "course"))
        live = MemoClient((CrossEncoderClient if t == "crossencoder" else LLMClient)(profile, logger=logger))
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


MECHANISM_VIEW = ["S0", "M1@S0", "M2@S0", "M3@S0", "M4@S0", "S6", "M1@S6", "M2@S6", "M3@S6", "M4@S6"]
FOCUS_SLICES = ["language", "point_in_time", "applicability", "conflict", "table_row", "tag_variant", "multi_evidence"]


def mechanisms_table(public: pd.DataFrame, h_public: pd.DataFrame, e2e: pd.DataFrame, recall: pd.DataFrame) -> pd.DataFrame:
    rows = [r for r in MECHANISM_VIEW if r in set(public.config)]
    t = public.set_index("config").loc[rows, ["vs", "recall@5", "end-to-end", "e2e: fixed / broke", "e2e verdict"]]
    t["recall@5"] = t["recall@5"].str.extract(r"= (\d+%)")[0]
    t["end-to-end"] = t["end-to-end"].str.extract(r"= (\d+%)")[0]
    t["language recall@5"] = recall.loc[rows, "language"]
    for s in FOCUS_SLICES:
        t[s] = e2e.loc[rows, s]
    h = h_public.set_index("config")
    t["holdout"] = [h.loc[r, "end-to-end"].split(" [")[0] if r in h.index else "" for r in rows]
    return t.fillna("")


def before_after(scores: dict[str, pd.DataFrame], costs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One contract against another on the same questions: paired verdict, failure modes, cost."""
    out = []
    for new, old in BEFORE_AFTER:
        if new not in scores or old not in scores:
            continue
        t = reval.ablation({old: scores[old], new: scores[new]}, {new: old}).set_index("config")
        a, b = t.loc[old], t.loc[new]
        row = {"new vs old": f"{new} vs {old}", "invalid": f"{a['invalid']} → {b['invalid']}",
               "leaks": f"{a['leaks']} → {b['leaks']}", "fallbacks": f"{a['fallbacks']} → {b['fallbacks']}",
               "end-to-end old": a["end-to-end"].split(" [")[0], "end-to-end new": b["end-to-end"].split(" [")[0],
               "new vs old: fixed / broke": b["e2e: fixed / broke"], "verdict": b["e2e verdict"],
               "false refusals": f"{a['false refusals'].split(' = ')[0]} → {b['false refusals'].split(' = ')[0]}",
               "wrong or invented": f"{a['wrong or invented']} → {b['wrong or invented']}",
               "verified": f"{a['correct + verified'].split(' [')[0]} → {b['correct + verified'].split(' [')[0]}",
               "evidence recall": b["evidence recall"].split(" [")[0]}
        if costs is not None and old in costs.index and new in costs.index:
            row["s / q"] = f"{costs.loc[old, 'median s / question']} → {costs.loc[new, 'median s / question']}"
            row["chat calls / q"] = f"{costs.loc[old, 'chat calls / question']} → {costs.loc[new, 'chat calls / question']}"
            row["tokens in / q"] = f"{costs.loc[old, 'prompt tokens / question']:.0f} → {costs.loc[new, 'prompt tokens / question']:.0f}"
            row["tokens out / q"] = f"{costs.loc[old, 'completion tokens / question']:.0f} → {costs.loc[new, 'completion tokens / question']:.0f}"
        out.append(row)
    return pd.DataFrame(out).set_index("new vs old") if out else pd.DataFrame()


def before_after_slices(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = list(dict.fromkeys(r for pair in BEFORE_AFTER for r in pair[::-1] if r in scores))
    return reval.by_slice({r: scores[r] for r in rows}, "correct")[CONTRACT_SLICES]


def p101_table(records: dict[str, list[dict]], scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The P-101 questions (m01: the lecture question, i01: the incident) in every contract row."""
    rows = []
    order = [f"{r}{v}" for r in S6_ROWS for v in (":v1", ":v2", "")] + ["S6:think"]
    for name in [r for r in order if r in records]:
        sc = scores[name].set_index("case_id")
        for cid in ("m01", "i01", "f03", "n04"):
            rec = next(r for r in records[name] if r["case_id"] == cid)
            a = rec.get("answer") or {}
            rows.append({"row": name, "question": cid, "status": a.get("status", "invalid"),
                         "value": a.get("value"), "right": bool(sc.loc[cid, "correct"]),
                         "verified": bool(sc.loc[cid, "verified"]), "retried": bool(rec.get("retried")),
                         "fallback": bool(rec.get("fallback")), "leaks": len(rec.get("leaks") or []),
                         "check (internal)": (rec.get("check") or "")[:160]})
    return pd.DataFrame(rows)


def needs_human_table(scores: dict[str, pd.DataFrame], records: dict[str, list[dict]]) -> pd.DataFrame:
    """Status per question and configuration on the needs_human set, with the summary row. An invalid answer
    shows why the model stopped (finish reason "length": the token limit)."""
    def cell(rec: dict) -> str:
        ans = rec.get("answer")
        if ans is None:
            return f"invalid ({rec.get('finish_reason') or 'not JSON'})"
        return ans.get("status", "") + (f" ({ans['value']})" if ans.get("value") else "")

    t = pd.DataFrame({name: {r["case_id"]: cell(r) for r in recs} for name, recs in records.items()})
    t.loc["right (needs_human)"] = [f"{int(df.correct.sum())}/{len(df)}" for df in scores.values()]
    t.loc["gave a value"] = [str(int((df.value.notna() & (df.value != "")).sum())) for df in scores.values()]
    t.index.name = "question"
    return t


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
    ap.add_argument("--live-mechanisms", action="store_true",
                    help="live: also call the LLM reranker, the translation and the cross-encoder for new requests")
    ap.add_argument("--only-needs-human", action="store_true",
                    help="live: only the needs_human set, merged into the replay (the other rows are not re-run)")
    args = ap.parse_args()
    if args.only_needs_human:
        _, chunks = load_store(STORE_DIR)
        cl = clients(False)
        system = System(chunks, answer_client=cl["answers"], rerank_client=cl["rerank"],
                        translate_client=cl["translate"], cross_client=cl["crossencoder"],
                        vectors=VectorStore(VECTORS) if VECTORS.with_suffix(".npz").exists() else None)
        nh = load_cases(NEEDS_HUMAN_FILE)
        for name in NEEDS_HUMAN_CONFIGS:
            df = reval.score(nh, system.run(nh, CONFIGS[name], workers=6, progress=False), system.by_id)
            print(name, df[["case_id", "status", "value", "correct"]].to_string(index=False))
        for t in ("answers", "rerank", "crossencoder"):
            print(f"replay {t}:", compact_replay(RUNS_DIR / "lecturer" / f"{t}.jsonl", REPLAY_DIR / f"{t}_course.jsonl",
                                                 merge=True))
        return
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
    cl = clients(args.replay, args.fresh, args.live_mechanisms)
    vectors = VectorStore(VECTORS) if VECTORS.with_suffix(".npz").exists() else None
    if vectors:
        vectors.check(summary["index_version"])
    system = System(chunks, answer_client=cl["answers"], rerank_client=cl["rerank"], translate_client=cl["translate"],
                    cross_client=cl["crossencoder"], vectors=vectors, think_client=cl["answers_thinking"])
    holdout = load_cases(HOLDOUT_FILE)
    if vectors is None:  # no vectors: the dense rows are skipped
        args.configs = [c for c in args.configs if CONFIGS[c].retriever == "bm25"]

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
    needs_human = load_cases(NEEDS_HUMAN_FILE)
    nh_records = {name: system.run(needs_human, CONFIGS[name], workers=6, progress=False)
                  for name in NEEDS_HUMAN_CONFIGS if name in args.configs}
    nh_scores = {name: reval.score(needs_human, recs, system.by_id) for name, recs in nh_records.items()}
    conflict_demo = demo(system, cases)
    if not args.replay:
        for t in [t for t in TASKS if args.live_mechanisms or t not in REPLAY_ONLY_TASKS]:
            n = compact_replay(RUNS_DIR / "lecturer" / f"{t}.jsonl", REPLAY_DIR / f"{t}_course.jsonl")
            print(f"replay {t}: {n} requests")

    REFERENCE_DIR.mkdir(exist_ok=True)
    main = {k: v for k, v in scores.items() if k not in VARIANT_ROWS}
    table = reval.ablation(scores, PREVIOUS)
    table = table[~table.config.isin(VARIANT_ROWS)]
    public = table[[c for c in table.columns if not c.startswith("_")]]
    public.to_csv(REFERENCE_DIR / "ablation.csv", index=False)
    slices_e2e = reval.by_slice(main, "correct")
    slices_recall = reval.by_slice(main, "recall@k")
    slices_evidence = reval.by_slice(main, "evidence_recall")
    slices_e2e.to_csv(REFERENCE_DIR / "slices_end_to_end.csv")
    slices_recall.to_csv(REFERENCE_DIR / "slices_recall.csv")
    slices_evidence.to_csv(REFERENCE_DIR / "slices_evidence.csv")
    losses = pd.DataFrame({name: reval.losses(scores[name]) for name in ("S0", "S6") if name in scores}).T
    losses.index.name = "config"
    losses.to_csv(REFERENCE_DIR / "losses.csv")
    pd.concat(scores.values()).to_csv(REFERENCE_DIR / "per_question.csv", index=False)
    costs = reval.cost_table(records)
    if (REFERENCE_DIR / "cost.csv").exists():
        # keep the measured time of a row whose calls and tokens did not change (a ":v1" row may take it from the
        # row of the same name measured before the contract changed)
        old = pd.read_csv(REFERENCE_DIR / "cost.csv", index_col="config")
        same = ["model calls / question", "prompt tokens / question", "completion tokens / question"]
        for c in costs.index:
            src = c if c in old.index else c.split(":")[0]
            if src in old.index and all(float(old.loc[src, x]) == float(costs.loc[c, x]) for x in same):
                costs.loc[c, TIMING_COLUMNS] = old.loc[src, TIMING_COLUMNS].values
    costs.to_csv(REFERENCE_DIR / "cost.csv")
    contract_table = before_after(scores, costs)
    contract_table.to_csv(REFERENCE_DIR / "contract_before_after.csv")
    contract_slices = before_after_slices(scores)
    p101_rows = p101_table(records, scores)
    h_contract = before_after(h_scores)
    violations = sum(len(r.get("schema_violations") or []) for name, recs in list(records.items())
                     + list(h_records.items()) + list(nh_records.items())
                     if CONFIGS[name].contract and CONFIGS[name].contract_version >= 2 for r in recs)
    h_table = reval.ablation(h_scores, PREVIOUS)
    h_table = h_table[~h_table.config.isin(VARIANT_ROWS)]
    h_public = h_table[[c for c in h_table.columns if not c.startswith("_") and c not in ("abstain ok", "conflict ok")]]
    h_public.to_csv(REFERENCE_DIR / "holdout_ablation.csv", index=False)
    pd.concat(h_scores.values()).to_csv(REFERENCE_DIR / "holdout_per_question.csv", index=False)
    nh_table = needs_human_table(nh_scores, nh_records)
    nh_table.to_csv(REFERENCE_DIR / "needs_human.csv")
    pd.concat(nh_scores.values()).to_csv(REFERENCE_DIR / "needs_human_per_question.csv", index=False)
    on_nid = {name: df[(df.expected == "not_in_documents") & (df.status == "needs_human")].case_id.tolist()
              for name, df in scores.items()}
    escalated = {name: df[(df.expected != "not_in_documents") & (df.status == "needs_human")].case_id.tolist()
                 for name, df in scores.items()}
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
        "- Mechanisms (the course server): dense search with `iairlab/e5-mistral-7b-instruct` "
        "(4096-dim; questions with the instruction, documents without; vectors in `data/replay/vectors/e5-mistral.*`), "
        "hybrid = RRF of the BM25 and dense top-20, the LLM reranker (M3, course Qwen, listwise) and the cross-encoder "
        "`iairlab/bge-reranker-v2-m3` (M4, `/rerank`), both over the top-20 of the row they are added to.",
        "",
        "## Evaluation limits",
        "",
        "End-to-end is the code-grader score: one value and its unit, the presence of required numbers in a "
        "multi-value answer, or an expected status. It does not grade the full explanation or recommended action. "
        "Citation checks test sources, dates and numeric presence; they do not establish support for every claim. "
        "The small synthetic holdout and unadjusted paired tests are exploratory. See README, 'What the evaluation "
        "does and does not prove'.",
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
        "## Mechanisms against plant knowledge",
        "",
        "Each mechanism at two points: instead of plant knowledge (on S0) and on top of it (on S6), compared in pairs "
        "with S0 or S6. End-to-end by slice in counts; the holdout column is end-to-end on the 27 role questions.",
        "",
        md_table(mechanisms_table(public, h_public, slices_e2e, slices_recall), index=True),
        "",
        "## The answer contract: v1, v2, v3 (and v2 in thinking mode)",
        "",
        "Search and context are the same in every version: evidence recall does not change; only the answer step does.",
        "",
        "- **v1** (`S6:v1`, `M*@S6:v1`, the first reference run): the model copies citation objects (document, "
        "revision, clause, table, row, page) from the source headers; no limits. Long answers ran out of the 3000 tokens.",
        f"- **v2** (`S6:v2`, `M*@S6:v2`): a claim cites the **ids of the chunks** in the context, and "
        f"the JSON schema makes the id an `enum` of exactly those ids (built per question); at most {A.MAX_CLAIMS} "
        f"claims, at most {A.MAX_SOURCES_PER_CLAIM} sources per claim; code fills the citation fields from the index. An "
        "answer cut by the token limit (3000, unchanged) or not of the schema is asked once more with a short "
        "instruction; if it fails again, the answer is `needs_human` with no value (a fallback).",
        f"- **v3** (`S6` and every row built on it): v2 plus an internal field `check` first in the schema (at most {A.CHECK_MAX} characters: do "
        "the sources agree, is the number in the cited text, is the answer in the documents at all), logged and never "
        f"shown or graded, and `maxLength` {A.ANSWER_MAX} on `answer`. A leak detector in code (markers such as "
        "\"Wait\", \"Let me\", \"Actually\", or an answer cut at the limit) asks once more, then falls back to "
        "`needs_human`. The internal field records a brief consistency check. It is not evidence of the model's "
        "reasoning or of answer correctness; the version comparison measures observed outputs only.",
        "- **v2 in thinking mode** (`S6:think`, profile `course-thinking`): the reasoning goes to its own channel "
        "(not logged in the replay); the server applies the JSON schema to the content after it (checked).",
        "",
        "Every S6-based row (S6, M1@S6..M4@S6) is measured with v3; its v1 and v2 answers stay in the replay as "
        "`X:v1` and `X:v2` rows.",
        "",
        f"Guided decoding (vLLM behind LiteLLM) enforces the enum, `maxItems` and `maxLength`: asked to cite an id that "
        f"is not in the context, or to write 9 claims with 4 sources each, or a long essay, the model returned only "
        f"context ids, 5 claims, 2 sources per claim, strings cut at the limit. Code checks it anyway: {violations} "
        "schema violations in all v2 and v3 answers.",
        "",
        "Main set (143 questions), new against old on the same questions; s / q and tokens are medians and means "
        "per question, all chat calls (a retry counts; tokens out of the thinking row include the reasoning):",
        "",
        md_table(contract_table, index=True),
        "",
        "End-to-end by slice:",
        "",
        md_table(contract_slices, index=True),
        "",
        "Questions to watch (m01: the lecture question; i01: the incident; f03: a conflict answered correctly "
        "by v1; n04: \"not in the documents\"); `check` is the internal field of v3: logged for the audit, "
        "never shown to the engineer:",
        "",
        md_table(p101_rows.fillna("")),
        "",
        "Holdout (27):",
        "",
        md_table(h_contract[[c for c in ("invalid", "leaks", "fallbacks", "end-to-end old", "end-to-end new",
                                         "new vs old: fixed / broke", "false refusals", "wrong or invented")
                             if c in h_contract.columns]], index=True) if len(h_contract) else "",
        "",
        "Needs a person (6): see the table in the next section (the main rows are v3; `:v1`, `:v2`, `S6:think` are the "
        "other versions).",
        "",
        "## How the rows fail: refusing against inventing",
        "",
        "The columns `false refusals` .. `invalid` split the wrong answers by kind (every question is exactly one of: "
        "right, false refusal, wrong or invented, invalid). A false refusal is \"not in the documents\" or \"needs a "
        "person\" on a question whose answer (or conflict) is in the documents: out of the "
        f"{int((scores['S0'].expected != 'not_in_documents').sum())} such questions. Wrong or invented: a value "
        "that is not the expected one, one side of a conflict given as the answer, or a value where the right answer "
        f"is \"not in the documents\". Right abstentions: out of the {int((scores['S0'].expected == 'not_in_documents').sum())} "
        "questions whose answer is not in the documents (\"needs a person\" is accepted there: no value is given). "
        "Invalid: no answer of the schema.",
        "",
        "False refusals add manual work. Wrong or invented values can mislead decisions. "
        "S0 fails both ways; S6 almost only by refusing; the reranker alone (M3@S0) refuses less than S0 but keeps "
        "giving wrong values and invents values on \"not in the documents\" questions.",
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
        "value in the evidence, lexical leak <= 0.6, no near-copy of the retrieval set) and by hand: 27 of 75 drafts "
        "kept. The ladder was never shown to the drafter.",
        "",
        md_table(h_public.fillna("")),
        "",
        "## The status \"needs a person\": its own small set",
        "",
        f"{len(needs_human)} questions (`data/needs_human.jsonl`, slice `needs_human`, not in the retrieval set): the "
        "norm is in the documents, but the answer depends on equipment data that no document and no registry card "
        "has (a pump without power and class, a tag that matches two pumps, no tag, a spare pump without a station, "
        "a grease named only in a missing datasheet, a measurement in one direction of three). Right: only "
        "`needs_human`. `not_in_documents` does not meet the expected status; a value means the system chose a class, a station or "
        "a measurement that nobody gave it. S5 has no status field: it shows what the contract adds. Cells: status "
        "(value).",
        "",
        md_table(nh_table, index=True),
        "",
        "Notes: " + "; ".join(f"{c.case_id}: {c.note}" for c in needs_human) + ".",
        "",
        "`needs_human` on the retrieval set. On a \"not in the documents\" question it is accepted (grader rule, "
        "unchanged): " + "; ".join(f"{k}: {', '.join(v)}" for k, v in on_nid.items() if v) + ". On a question with "
        "an answer it is a false refusal (counted in `false refusals`): "
        + "; ".join(f"{k}: {', '.join(v)}" for k, v in escalated.items() if v) + ".",
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
