"""The retrieval pipeline as configurations: every row of the ablation table is one `Config`.

question -> fields (query.py) -> scope (applicability.py) -> filter (filters.py) -> search (search.py)
-> [rerank] -> context (assemble.py) -> answer (answer.py) -> checks (verify.py)

The ladder S0-S6 adds one plant-side change per row (S2 and S2b are two ways to index the rows of a long table:
mixed with the clauses, or in a separate collection with a small quota; S3 and later build on S2b); the mechanism branch (M*) adds dense search, hybrid
search, an LLM reranker (M3) and a cross-encoder (M4) at two points: instead of plant knowledge (on S0) and on top
of it (on S6).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

from tqdm.auto import tqdm

from . import answer as A
from .applicability import Scope, equipment_context, linked_chunks, load_registry, scope_for
from .assemble import evidence_pack, render, top_context
from .cases import TestCase
from .corpus import index_version
from .filters import make_filter
from .query import extract_rules
from .search import BM25Index, DenseIndex, Hit, HybridIndex, VectorStore, cross_rerank, rerank, view
from .store import Chunk
from .verify import check_answer


@dataclass(frozen=True)
class Config:
    name: str
    description: str = ""
    normalize: bool = False  # S1: tags in one form in chunks and query
    rows: str = "pages"  # long tables: "pages" | "mixed" (S2) | "separate" (S2b: own collection, `row_quota` rows)
    row_quota: int = 2
    applicability: bool = False  # S3: registry: scope filter, linked documents in the context, class in the prompt
    date_filter: bool = False  # S4: chunks valid on the date of the question, dates in the source headers
    evidence_pack: bool = False  # S5: references both ways and parent sections within a budget
    contract: bool = False  # S6: answer contract and citation checks
    contract_version: int = 3  # 3: v2 + internal check field, leak detector; 2: chunk ids as an enum, limits, one
    # retry, fallback (":v2" rows); 1: citation objects (":v1" rows)
    thinking: bool = False  # the answer by the course model in thinking mode (reasoning in its own channel)
    retriever: str = "bm25"  # bm25 | dense | hybrid
    rerank: bool = False  # reranker over the top `depth`
    reranker: str = "llm"  # "llm": listwise, course Qwen (M3); "cross": cross-encoder bge-reranker-v2-m3 (M4)
    translate: bool = False  # translate an English question into Russian before search
    k: int = 5
    depth: int = 20
    budget: int = 2500
    only_slices: tuple[str, ...] = ()  # run this row only on these slices (M1t: the language slice)


S0 = Config("S0", "BM25 over raw chunks, top-5, no filters, context = top-5")
S1 = replace(S0, name="S1", description="+ tag normalization", normalize=True)
S2 = replace(S1, name="S2", description="+ rows of long tables as chunks, mixed with the clauses", rows="mixed")
S2B = replace(S1, name="S2b", description="+ rows of long tables in a separate collection: top-5 chunks + top-2 rows",
              rows="separate")
S3 = replace(S2B, name="S3", description="+ applicability: registry scope filter, linked documents, class in the prompt",
             applicability=True)
S4 = replace(S3, name="S4", description="+ revision filter by the date of the question", date_filter=True)
S5 = replace(S4, name="S5", description="+ evidence pack: references both ways, parents", evidence_pack=True)
S6 = replace(S5, name="S6", description="+ answer contract and citation checks", contract=True)
LADDER = [S0, S1, S2, S2B, S3, S4, S5, S6]
PREVIOUS = {"S1": "S0", "S2": "S1", "S2b": "S1", "S3": "S2b", "S4": "S3", "S5": "S4", "S6": "S5", "M1t": "S0",
            "M3@S0": "S0", "M3@S6": "S6", "M1@S0": "S0", "M2@S0": "S0", "M1@S6": "S6", "M2@S6": "S6",
            "M4@S0": "S0", "M4@S6": "S6"}

M3_S0 = replace(S0, name="M3@S0", description="S0 + LLM reranker (course Qwen) over BM25 top-20", rerank=True)
M3_S6 = replace(S6, name="M3@S6", description="S6 + LLM reranker (course Qwen) over the filtered top-20",
                rerank=True)
M1T = replace(S0, name="M1t", description="S0 + translation of the English question into Russian (course Qwen)",
              translate=True, only_slices=("language",))
M1_S0 = replace(S0, name="M1@S0", description="dense search (e5-mistral) instead of BM25", retriever="dense")
M2_S0 = replace(S0, name="M2@S0", description="hybrid BM25 + dense (RRF over the two top-20)", retriever="hybrid")
M1_S6 = replace(S6, name="M1@S6", description="S6 with dense search", retriever="dense")
M2_S6 = replace(S6, name="M2@S6", description="S6 with hybrid search", retriever="hybrid")
M4_S0 = replace(S0, name="M4@S0", description="S0 + cross-encoder (bge-reranker-v2-m3) over BM25 top-20", rerank=True,
                reranker="cross")
M4_S6 = replace(S6, name="M4@S6", description="S6 + cross-encoder over the filtered top-20", rerank=True,
                reranker="cross")
MECHANISMS = [M1_S0, M1T, M2_S0, M3_S0, M4_S0, M1_S6, M2_S6, M3_S6, M4_S6]
# The same rows with the first answer contract (citation objects, no limits): "before" for the comparison
CONTRACT_V1 = [replace(c, name=f"{c.name}:v1", description=f"{c.name} with the answer contract v1", contract_version=1)
               for c in (S6, M1_S6, M2_S6, M3_S6, M4_S6)]
# S6 with contract v2 (before the check field), and v2 with the course model in thinking mode
CONTRACT_V2 = [replace(c, name=f"{c.name}:v2", description=f"{c.name} with the answer contract v2", contract_version=2)
               for c in (S6, M1_S6, M2_S6, M3_S6, M4_S6)]
S6_THINK = replace(S6, name="S6:think", description="S6, contract v2, the answer in thinking mode",
                   contract_version=2, thinking=True)
CONTRACT_VARIANTS = CONTRACT_V1 + CONTRACT_V2 + [S6_THINK]
CONFIGS = {c.name: c for c in LADDER + MECHANISMS + CONTRACT_VARIANTS}

# Demo of the contract: the two sides of the P-205 conflict put into the context by hand
CONFLICT_DEMO = ("f01", ["R-12-r3-table1", "OM-P-205-r2-table2"])

TRANSLATE_SYSTEM = """Translate the engineer's question into Russian. Keep equipment tags (P-101, V-12), document
numbers (R-12) and numbers exactly as they are. Return only the translation."""


class System:
    """Everything a configuration needs: the store, the registry, the indexes and the model clients."""

    def __init__(self, chunks: list[Chunk], answer_client=None, rerank_client=None, translate_client=None,
                 vectors: VectorStore | None = None, embedder_name: str | None = None, cross_client=None,
                 embed_query=None, think_client=None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.registry = load_registry()
        self.answer_client = answer_client
        self.rerank_client = rerank_client
        self.translate_client = translate_client
        self.vectors = vectors
        self.cross_client = cross_client
        self.think_client = think_client  # the course model in thinking mode (row "S6:think")
        self.embed_query = embed_query  # live embedding of a question without a saved vector
        self.index_version = index_version(chunks, embedder_name)
        self._bm25: dict[tuple, BM25Index] = {}
        self._dense: dict[tuple, DenseIndex] = {}
        self._lock = threading.Lock()

    def bm25(self, cfg: Config, rows: str | None = None) -> BM25Index:
        key = (cfg.normalize, rows or ("main" if cfg.rows == "separate" else cfg.rows))
        with self._lock:
            if key not in self._bm25:
                self._bm25[key] = BM25Index(self.chunks, cfg.normalize, key[1])
        return self._bm25[key]

    def dense(self, cfg: Config, rows: str | None = None) -> DenseIndex:
        if self.vectors is None:
            raise RuntimeError("no embedding vectors: dense rows need data/replay/vectors/<model>.npz or a live "
                               "embedding profile (see README, 'Dense search')")
        key = (cfg.normalize, rows or ("main" if cfg.rows == "separate" else cfg.rows))
        with self._lock:
            if key not in self._dense:
                self._dense[key] = DenseIndex(self.chunks, self.vectors, key[1], self.embed_query, cfg.normalize)
        return self._dense[key]

    def index(self, cfg: Config, rows: str | None = None):
        """The index of a configuration; `rows` picks a collection ("main" or "rows") in the separate mode."""
        if cfg.retriever == "bm25":
            return self.bm25(cfg, rows)
        if cfg.retriever == "dense":
            return self.dense(cfg, rows)
        return HybridIndex(self.bm25(cfg, rows), self.dense(cfg, rows), cfg.depth)

    def view_ids(self, cfg: Config) -> set[str]:
        if cfg.rows == "separate":
            return {c.chunk_id for c in view(self.chunks, "main")} | {c.chunk_id for c in view(self.chunks, "rows")}
        return {c.chunk_id for c in view(self.chunks, cfg.rows)}

    # ---------------------------------------------------------------------------------------------
    def translate(self, question: str, meta: dict) -> tuple[str, object]:
        resp = self.translate_client.chat([{"role": "system", "content": TRANSLATE_SYSTEM},
                                           {"role": "user", "content": question}], meta=meta)
        return (resp.content.strip() if resp.ok and resp.content.strip() else question), resp

    def retrieve(self, case: TestCase, cfg: Config) -> dict:
        t0 = time.perf_counter()
        # model calls made during retrieval: (latency, prompt tokens, completion tokens, wall seconds, kind);
        # kind "chat" uses chat tokens; "embed" (the question vector) and "cross" (the cross-encoder) do not
        calls = []
        query = case.query
        if cfg.translate and case.language == "en":
            t = time.perf_counter()
            query, resp = self.translate(query, {"case_id": case.case_id, "task": "translate"})
            calls.append((resp.latency_s, resp.prompt_tokens, resp.completion_tokens, time.perf_counter() - t,
                          "chat"))
        if cfg.retriever in ("dense", "hybrid"):  # one embedding call for the question (latency as saved)
            t = time.perf_counter()
            latency = self.dense(cfg).query_vector(query)[1]
            calls.append((latency, 0, 0, time.perf_counter() - t, "embed"))
        fields = extract_rules(query, case.asked_on)
        scope = scope_for(fields, query, self.registry) if cfg.applicability else Scope()
        allowed = make_filter(fields, scope, cfg.applicability, cfg.date_filter)
        separate = cfg.rows == "separate"
        hits = self.index(cfg, "main" if separate else None).search(query, cfg.depth, allowed)
        rerank_info = None
        if cfg.rerank:
            t = time.perf_counter()
            if cfg.reranker == "cross":
                hits, rerank_info = cross_rerank(self.cross_client, query, hits, self.by_id, cfg.depth,
                                                 meta={"case_id": case.case_id, "config": cfg.name, "task": "cross"})
            else:
                hits, rerank_info = rerank(self.rerank_client, query, hits, self.by_id, cfg.depth,
                                           meta={"case_id": case.case_id, "config": cfg.name, "task": "rerank"})
            calls.append((rerank_info["latency_s"], rerank_info["prompt_tokens"], rerank_info["completion_tokens"],
                          time.perf_counter() - t, "cross" if cfg.reranker == "cross" else "chat"))
        top = hits[:cfg.k]
        if separate:  # the row collection gives at most `row_quota` rows, and only rows that share a term
            rows = [h for h in self.index(cfg, "rows").search(query, cfg.row_quota, allowed) if h.score > 0]
            top = top + rows
            hits = top + hits[cfg.k:]
        # documents the registry links to the equipment (S3+): outside search, inside the scope and the date
        expand_filter = make_filter(fields, scope, cfg.applicability, cfg.date_filter, use_doc_ids=False)
        top_ids = {h.chunk_id for h in top}
        linked = [c for c in linked_chunks(scope, self.chunks, expand_filter) if c not in top_ids] \
            if cfg.applicability else []
        seeds = top + [Hit(c, 0.0, len(top) + i + 1, "registry") for i, c in enumerate(linked)]
        if cfg.evidence_pack:
            # references are followed inside the scope and the date, but not inside the "named document" filter:
            # clause 6.2 of R-12 leads to R-20 even when the question names only R-12
            context = evidence_pack(seeds, self.by_id, self.view_ids(cfg), expand_filter, len(seeds),
                                    budget=cfg.budget)
        else:
            context = top_context(seeds, len(seeds))
        model_wall = sum(c[3] for c in calls)
        cost = {"search_s": round(time.perf_counter() - t0 - model_wall, 4), "model_s": sum(c[0] for c in calls),
                "calls": len(calls), "chat_calls": sum(c[4] == "chat" for c in calls),
                "prompt_tokens": sum(c[1] for c in calls), "completion_tokens": sum(c[2] for c in calls),
                "reranker_tokens": (rerank_info or {}).get("reranker_tokens", 0)}
        return {"query": query, "fields": fields, "scope": scope, "hits": hits, "top": [h.chunk_id for h in top],
                "linked": linked, "context": context, "rerank": rerank_info, "cost": cost}

    def run_case(self, case: TestCase, cfg: Config, with_answer: bool = True) -> dict:
        r = self.retrieve(case, cfg)
        rec = {"case_id": case.case_id, "config": cfg.name, "query": r["query"], "asked_on": case.asked_on,
               "as_of": r["fields"].as_of, "scope": r["scope"].describe() if cfg.applicability else "",
               "retrieved": [h.chunk_id for h in r["hits"]], "top": r["top"], "linked": r["linked"],
               "context": r["context"], "rerank": r["rerank"], "cost": r["cost"]}
        if not with_answer:
            return rec
        equipment = equipment_context(r["scope"]) if cfg.applicability else ""
        out = self.answer(case, r["query"], r["context"], r["fields"].as_of, equipment, cfg, "answer")
        cost = rec["cost"]
        for resp in out["responses"]:
            cost.update({"model_s": round(cost["model_s"] + resp.latency_s, 3), "calls": cost["calls"] + 1,
                         "chat_calls": cost["chat_calls"] + 1,
                         "prompt_tokens": cost["prompt_tokens"] + resp.prompt_tokens,
                         "completion_tokens": cost["completion_tokens"] + resp.completion_tokens})
        resp, ans = out["responses"][-1], out["answer"]
        rec["check"] = out["check"]  # internal (contract v3): logged, never shown, never graded
        rec.update({"model": resp.model, "request_hash": out["responses"][0].request_hash, "error": resp.error,
                    "from_replay": resp.from_replay, "latency_s": resp.latency_s,
                    "finish_reason": out["responses"][0].finish_reason,
                    "answer": ans.model_dump() if ans else None, "raw": None if ans else resp.content[:500],
                    "retried": len(out["responses"]) > 1, "fallback": out["fallback"], "leaks": out["leaks"],
                    "schema_violations": out["violations"], "checks": out["checks"]})
        return rec

    def answer(self, case: TestCase, query: str, context: list[str], as_of: str | None, equipment: str, cfg: Config,
               task: str) -> dict:
        """The answer step. Simple answer (S0-S5) and contract v1: one call, an answer that is not of the schema is
        invalid. Contract v2: the schema allows only the chunk ids of this context; an answer cut by the token
        limit or not of the schema is asked once more with a short instruction; if it fails again, the answer is
        "needs a person" with no value (`fallback`)."""
        v2 = cfg.contract and cfg.contract_version >= 2
        v3 = cfg.contract and cfg.contract_version >= 3
        context_text, sources = render(context, self.by_id, show_dates=cfg.date_filter, ids_as_labels=v2)
        fmt = A.contract_v3_format(context) if v3 else A.contract_v2_format(context) if v2 else \
            A.CONTRACT_FORMAT if cfg.contract else A.SIMPLE_FORMAT
        client = self.think_client if cfg.thinking else self.answer_client
        responses, ans, fallback, check, leaks = [], None, False, "", []
        for retry in ((False, True) if v2 else (False,)):
            msgs = A.messages(query, context_text, case.asked_on, as_of, equipment, cfg.contract,
                              cfg.contract_version, retry)
            resp = client.chat(msgs, response_format=fmt, meta={"case_id": case.case_id, "config": cfg.name,
                                                                "task": task + ("-retry" if retry else "")})
            responses.append(resp)
            if not resp.ok:  # a network error is not a length problem: no retry
                break
            ans, check = A.parse_v3(resp.content) if v3 else (A.parse(resp.content, cfg.contract), "")
            leaked = A.leak(ans.answer) if v3 and ans is not None else None
            if leaked:
                leaks.append(leaked)
            if not v2 or (ans is not None and resp.finish_reason != "length" and not leaked):
                break
            ans = None
        violations = []
        if v2 and ans is None and responses[-1].ok:
            ans, fallback = A.fallback_answer(), True
        if v2 and ans is not None:
            violations = A.schema_violations(ans, set(context))
            ans.citations = [A.Citation(source=cid, doc_id=c.doc_id, revision=c.revision, amendment=c.amendment,
                                        clause=c.clause if c.chunk_type == "clause" else None, table_no=c.table_no,
                                        row_key=c.row_key, page=c.page)
                             for cid in A.cited_ids(ans) if cid in set(context) for c in [self.by_id[cid]]]
        checks = check_answer(ans, sources, self.chunks, self.by_id, as_of) if ans is not None else None
        return {"answer": ans, "responses": responses, "fallback": fallback, "violations": violations,
                "checks": checks, "check": check, "leaks": leaks}

    def answer_from_context(self, case: TestCase, chunk_ids: list[str], cfg: Config) -> dict:
        """Answer from a given context (a demo: what the contract does when both sides of a conflict are there)."""
        fields = extract_rules(case.query, case.asked_on)
        scope = scope_for(fields, case.query, self.registry) if cfg.applicability else Scope()
        out = self.answer(case, case.query, chunk_ids, fields.as_of,
                          equipment_context(scope) if cfg.applicability else "", cfg, "demo")
        ans, resp = out["answer"], out["responses"][-1]
        return {"case_id": case.case_id, "context": chunk_ids, "answer": ans.model_dump() if ans else None,
                "checks": out["checks"], "from_replay": resp.from_replay, "latency_s": resp.latency_s,
                "fallback": out["fallback"]}

    def run(self, cases: list[TestCase], cfg: Config, with_answer: bool = True, workers: int = 8,
            progress: bool = True) -> list[dict]:
        todo = [c for c in cases if not cfg.only_slices or c.slice in cfg.only_slices]
        if not with_answer and not cfg.rerank and not cfg.translate:
            return [self.run_case(c, cfg, False) for c in todo]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            it = ex.map(lambda c: self.run_case(c, cfg, with_answer), todo)
            return list(tqdm(it, total=len(todo), desc=cfg.name, disable=not progress))


def save_records(records: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def config_table() -> list[dict]:
    return [{"config": c.name, "description": c.description,
             **{k: v for k, v in asdict(c).items() if k not in ("name", "description", "only_slices")}}
            for c in LADDER + MECHANISMS]
