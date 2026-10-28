"""Search with one interface: `search(query, k, allowed) -> list[Hit]`.

The mechanisms (BM25, dense vectors, hybrid with RRF, an LLM reranker) are taught in the Advanced NLP course;
here they are ready tools that become rows of the ablation table. Two plant-side switches change what is
indexed, not how it is searched:
- `normalize`: equipment tags in one form in the chunks and in the query (normalize.py);
- `rows`: how a long table is indexed: "pages" (a page of the table is a chunk), "mixed" (every row is a chunk,
  in the same index as the clauses), or two collections, "main" (clauses and short tables) and "rows" (rows
  only), searched separately with a small quota for rows.

`allowed` is a pre-filter: search ranks only the allowed chunks, so the top-k is never filled with chunks
that a filter would throw away later (post-filtering, a typical mistake).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from .crossencoder import parse_rerank, rerank_request
from .normalize import normalize_tags
from .store import Chunk
from .units import plain


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int  # 1-based
    source: str = "bm25"


# ---------------------------------------------------------------------------------------------
# Tokens: technical text keeps "P-101", "4,5", "4.3.3", "мм/с" as one token; words are stemmed
# ---------------------------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-.,/][0-9A-Za-zА-Яа-яЁё]+)*")
STOP = set("""и в во на по с со к ко о об от до из за для при не ни но а или же ли то это как что какой какая
какое какие каких какую каком который которые ее её его их он она оно они мы вы я ты у же бы был была было быть
the a an of in on at to for by with from is are was were be been what which how does do did must can this that
it its as or and if than then there these those""".split())
_STEMMERS = threading.local()  # a snowball stemmer keeps state: one per thread


def _stem(word: str) -> str:
    if not hasattr(_STEMMERS, "ru"):
        _STEMMERS.ru = snowballstemmer.stemmer("russian")
        _STEMMERS.en = snowballstemmer.stemmer("english")
    return _STEMMERS.ru.stemWord(word) if re.match(r"[а-я]", word) else _STEMMERS.en.stemWord(word)


def tokenize(text: str) -> list[str]:
    out = []
    for t in TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if t in STOP:
            continue
        out.append(_stem(t) if t.isalpha() else t)
    return out


# ---------------------------------------------------------------------------------------------
# Index views
# ---------------------------------------------------------------------------------------------
ROW_MODES = ("pages", "mixed", "main", "rows")


def view(chunks: list[Chunk], rows: str | bool) -> list[Chunk]:
    """Which chunks are in an index: see `rows` in the module docstring (True / False mean mixed / pages)."""
    rows = {True: "mixed", False: "pages"}.get(rows, rows)
    long_page = [c.chunk_type == "table" and c.split_into_rows for c in chunks]
    if rows == "pages":
        return [c for c in chunks if c.chunk_type != "table_row"]
    if rows == "mixed":
        return [c for c, lp in zip(chunks, long_page) if not lp]
    if rows == "main":
        return [c for c, lp in zip(chunks, long_page) if not lp and c.chunk_type != "table_row"]
    if rows == "rows":
        return [c for c in chunks if c.chunk_type == "table_row"]
    raise ValueError(f"unknown rows mode {rows!r}, use one of {ROW_MODES}")


def index_text(c: Chunk, normalize: bool) -> str:
    """The text that is indexed: without HTML and Markdown markup (a datasheet read by a VLM is HTML)."""
    text = plain(c.text)
    return normalize_tags(text) if normalize else text


def prepare_query(query: str, normalize: bool) -> str:
    return normalize_tags(query) if normalize else query


class BM25Index:
    """BM25 over a view of the store. Scores of all chunks are computed, then the filter, then the top-k."""

    def __init__(self, chunks: list[Chunk], normalize: bool = False, rows: str | bool = "pages"):
        self.normalize, self.rows = normalize, rows
        self.chunks = view(chunks, rows)
        self.ids = [c.chunk_id for c in self.chunks]
        self._bm25 = BM25Okapi([tokenize(index_text(c, normalize)) for c in self.chunks])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(prepare_query(query, self.normalize))), dtype=float)

    def search(self, query: str, k: int = 5, allowed: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        return top_k(self.ids, self.chunks, self.scores(query), k, allowed, "bm25")


def top_k(ids: list[str], chunks: list[Chunk], scores: np.ndarray, k: int,
          allowed: Callable[[Chunk], bool] | None, source: str) -> list[Hit]:
    """The k best allowed chunks. Ties keep the store order (stable sort): an honest, visible rule."""
    mask = np.array([allowed(c) for c in chunks], dtype=bool) if allowed else np.ones(len(chunks), dtype=bool)
    s = np.where(mask, scores, -np.inf)
    order = np.argsort(-s, kind="stable")
    hits = []
    for i in order[:k]:
        if not np.isfinite(s[i]):
            break
        hits.append(Hit(ids[i], float(s[i]), len(hits) + 1, source))
    return hits


# ---------------------------------------------------------------------------------------------
# Dense vectors: live from an OpenAI-compatible /embeddings endpoint or replayed from data/replay/vectors/
# ---------------------------------------------------------------------------------------------
class VectorStore:
    """Vectors of the chunks and of the questions of one embedding model.

    Files: <dir>/<name>.npz with arrays `chunk_vectors`, `query_vectors` and <dir>/<name>.json with the
    manifest: model, dimension, index_version of the corpus, chunk ids and query texts in the same order, the
    instruction the questions were embedded with (`query_prefix`) and the latency of each query embedding call.
    Vectors from another corpus or another model are refused: that is what the manifest is for.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        self.manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        arrays = np.load(path.with_suffix(".npz"))
        self.chunk_ids: list[str] = self.manifest["chunk_ids"]
        self.chunk_vectors = _unit(arrays["chunk_vectors"])
        self.queries: dict[str, int] = {q: i for i, q in enumerate(self.manifest["queries"])}
        self.query_vectors = _unit(arrays["query_vectors"])
        self.query_prefix: str = self.manifest.get("query_prefix", "")
        self.latency: list[float] = self.manifest.get("query_latency_s") or [0.0] * len(self.queries)

    def check(self, index_version_corpus: str) -> None:
        if not index_version_corpus.startswith(self.manifest["corpus"]):
            raise ValueError(f"vectors were built for {self.manifest['corpus']}, the store is {index_version_corpus}: "
                             "rebuild the vectors (scripts/build_vectors.py)")

    def query_vector(self, query: str) -> np.ndarray:
        if query not in self.queries:
            raise KeyError(f"no saved vector for this query: {query[:60]!r}. Use a live embedding profile.")
        return self.query_vectors[self.queries[query]]

    def query_latency(self, query: str) -> float:
        """Seconds of the embedding call for this question when the vectors were built (replay keeps it)."""
        return float(self.latency[self.queries[query]]) if query in self.queries else 0.0


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class DenseIndex:
    """Cosine similarity in numpy: at a thousand chunks no vector database is needed."""

    def __init__(self, chunks: list[Chunk], vectors: VectorStore, rows: str | bool = "pages",
                 embed_query: Callable[[str], np.ndarray] | None = None, normalize: bool = False):
        """`embed_query` embeds a question that has no saved vector (live; it adds the instruction itself).
        `normalize`: tags of the question in one form (the chunks were embedded with normalized tags)."""
        self.chunks = view(chunks, rows)
        self.normalize = normalize
        self._live: dict[str, tuple[np.ndarray, float]] = {}
        self.ids = [c.chunk_id for c in self.chunks]
        pos = {cid: i for i, cid in enumerate(vectors.chunk_ids)}
        missing = [cid for cid in self.ids if cid not in pos]
        if missing:
            raise ValueError(f"{len(missing)} chunks have no vector, e.g. {missing[:3]}")
        self.matrix = vectors.chunk_vectors[[pos[cid] for cid in self.ids]]
        self.vectors = vectors
        self.embed_query = embed_query

    def query_vector(self, query: str) -> tuple[np.ndarray, float]:
        """The vector of the question and the seconds its embedding call took (saved, or measured live)."""
        q = normalize_tags(query) if self.normalize else query
        if q in self.vectors.queries or self.embed_query is None:
            return self.vectors.query_vector(q), self.vectors.query_latency(q)
        if q not in self._live:
            t = time.perf_counter()
            v = _unit(np.asarray(self.embed_query(q), dtype=np.float32))
            self._live[q] = (v, time.perf_counter() - t)
        return self._live[q]

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.query_vector(query)[0]

    def search(self, query: str, k: int = 5, allowed: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        return top_k(self.ids, self.chunks, self.scores(query), k, allowed, "dense")


def rrf(rankings: Iterable[list[Hit]], k: int = 5, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum of 1 / (c + rank) over the rankings."""
    score: dict[str, float] = {}
    for hits in rankings:
        for h in hits:
            score[h.chunk_id] = score.get(h.chunk_id, 0.0) + 1.0 / (c + h.rank)
    order = sorted(score, key=lambda cid: -score[cid])
    return [Hit(cid, score[cid], i + 1, "hybrid") for i, cid in enumerate(order[:k])]


class HybridIndex:
    def __init__(self, bm25: BM25Index, dense: DenseIndex, depth: int = 20):
        self.bm25, self.dense, self.depth = bm25, dense, depth
        self.chunks = bm25.chunks

    def search(self, query: str, k: int = 5, allowed: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        return rrf([self.bm25.search(query, self.depth, allowed), self.dense.search(query, self.depth, allowed)], k)


# ---------------------------------------------------------------------------------------------
# LLM reranker (listwise) on the course model: one call per question, answers are replayed
# ---------------------------------------------------------------------------------------------
RERANK_SYSTEM = """You rank passages of plant documents by how useful they are to answer an engineer's question.
A useful passage states the value, the rule or the action that the question asks about, for the equipment,
the document and the date in the question. Return JSON: {"ranking": ["P3", "P1", ...]} with the labels of the
most useful passages first. Include at most 10 labels. Use only labels from the list."""
RERANK_SCHEMA = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
                 "required": ["ranking"], "additionalProperties": False}
RERANK_FORMAT = {"type": "json_schema", "json_schema": {"name": "rerank", "strict": True, "schema": RERANK_SCHEMA}}


def rerank_messages(query: str, candidates: list[Chunk], max_chars: int = 500) -> list[dict]:
    lines = []
    for i, c in enumerate(candidates, start=1):
        text = re.sub(r"\s+", " ", c.text)[:max_chars]
        lines.append(f"[P{i}] {c.doc_id}: {text}")
    user = f"Question: {query}\n\nPassages:\n" + "\n".join(lines)
    return [{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": user}]


def passage(c: Chunk, max_chars: int = 500) -> str:
    return re.sub(r"\s+", " ", c.text)[:max_chars]


def rerank(client, query: str, hits: list[Hit], chunks_by_id: dict[str, Chunk], k: int = 5,
           meta: dict | None = None) -> tuple[list[Hit], dict]:
    """Reorder the candidates by the model. Labels the model did not return keep their order after the ranked
    ones; an invalid answer keeps the original order (and says so)."""
    candidates = [chunks_by_id[h.chunk_id] for h in hits]
    resp = client.chat(rerank_messages(query, candidates), response_format=RERANK_FORMAT, meta=meta)
    order: list[int] = []
    try:
        for label in json.loads(resp.content)["ranking"]:
            m = re.fullmatch(r"P?(\d+)", str(label).strip())
            if m and 1 <= int(m.group(1)) <= len(hits) and int(m.group(1)) - 1 not in order:
                order.append(int(m.group(1)) - 1)
        ok = True
    except (json.JSONDecodeError, KeyError, TypeError):
        ok = False
    order += [i for i in range(len(hits)) if i not in order]
    out = [Hit(hits[i].chunk_id, hits[i].score, r + 1, "rerank") for r, i in enumerate(order[:k])]
    return out, {"ok": ok and resp.ok, "error": resp.error, "latency_s": resp.latency_s,
                 "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens}


# ---------------------------------------------------------------------------------------------
# Cross-encoder (bge-reranker-v2-m3 on the course server): one /rerank call per question, no chat tokens
# ---------------------------------------------------------------------------------------------
def cross_rerank(client, query: str, hits: list[Hit], chunks_by_id: dict[str, Chunk], k: int = 5,
                 meta: dict | None = None) -> tuple[list[Hit], dict]:
    """Reorder the candidates by the cross-encoder score. The passages are the same as for the LLM reranker
    (document id + the first 500 characters). An unusable response keeps the original order (and says so)."""
    candidates = [chunks_by_id[h.chunk_id] for h in hits]
    resp = client.chat(rerank_request(query, [f"{c.doc_id}: {passage(c)}" for c in candidates]), meta=meta)
    order = parse_rerank(resp.content, len(hits)) if resp.ok else None
    ok = order is not None
    order = order or list(range(len(hits)))
    out = [Hit(hits[i].chunk_id, hits[i].score, r + 1, "cross") for r, i in enumerate(order[:k])]
    return out, {"ok": ok, "error": resp.error, "latency_s": resp.latency_s, "prompt_tokens": 0,
                 "completion_tokens": 0, "reranker_tokens": resp.prompt_tokens}

