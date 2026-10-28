"""Embedding vectors of the chunks and of the questions, for dense search (M1) and hybrid search (M2).

Needs an embedding profile that answers (models.toml): e5-mistral-7b-instruct on the course server (the
lecturer's reference, `embed-course`) or Ollama on your laptop (`ollama pull bge-m3`). Writes
data/replay/vectors/<name>.npz (chunk_vectors, query_vectors, float16) and <name>.json (model, dimension, corpus
version, chunk ids and query texts in the same order, the query instruction, the latency of each query call).

Every chunk of the store gets a vector (both views: page tables and table rows), embedded as it is. Every wording
that can go to search gets one too: the question of every case of the retrieval set, the holdout and the
needs_human set (Russian; English for the `language` slice), the translations of the M1t row, and each of them
with normalized tags (the rows built on S1+). An asymmetric model embeds a question with the instruction of the
profile (`query_prefix`): the vectors are saved under the question text without it. Questions are embedded one per
call, so the manifest keeps the latency of a real query embedding.

Run:  uv run python scripts/build_vectors.py --profile embed-course --name e5-mistral
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_assistant import DATA_DIR, REPLAY_DIR, STORE_DIR  # noqa: E402
from plant_assistant.cases import load_cases  # noqa: E402
from plant_assistant.config import get_profile  # noqa: E402
from plant_assistant.corpus import index_version  # noqa: E402
from plant_assistant.normalize import normalize_tags  # noqa: E402
from plant_assistant.smoke_test import embed  # noqa: E402
from plant_assistant.store import load_store  # noqa: E402
from plant_assistant.units import plain  # noqa: E402


def batched(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="embed-course")
    ap.add_argument("--name", default="e5-mistral")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    prefix = get_profile(args.profile).query_prefix
    _, chunks = load_store(STORE_DIR)
    texts = [normalize_tags(plain(c.text)) for c in chunks]
    cases = load_cases() + load_cases(DATA_DIR / "holdout.jsonl") + load_cases(DATA_DIR / "needs_human.jsonl")
    queries = [c.query for c in cases]
    translations = REPLAY_DIR / "translate_course.jsonl"
    if translations.exists():
        for line in translations.read_text(encoding="utf-8").splitlines():
            queries.append(json.loads(line)["response"]["content"].strip())
    queries = list(dict.fromkeys(q for x in queries for q in (x, normalize_tags(x))))
    chunk_vectors = [v for b in tqdm(list(batched(texts, args.batch)), desc="chunks") for v in embed(args.profile, b)]
    query_vectors, latency = [], []
    for q in tqdm(queries, desc="queries"):
        t = time.perf_counter()
        query_vectors.append(embed(args.profile, [q], query=True)[0])
        latency.append(round(time.perf_counter() - t, 3))
    out = REPLAY_DIR / "vectors" / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), chunk_vectors=np.asarray(chunk_vectors, dtype=np.float16),
                        query_vectors=np.asarray(query_vectors, dtype=np.float16))
    version = index_version(chunks, args.name)
    manifest = {"model": args.name, "profile": args.profile, "dimension": len(chunk_vectors[0]),
                "corpus": version.split(".")[0], "index_version": version,
                "query_prefix": prefix, "chunk_ids": [c.chunk_id for c in chunks], "queries": queries,
                "query_latency_s": latency,
                "note": "chunk text: tags normalized, markup removed, no instruction; a query: the instruction "
                        "(query_prefix) + the question, saved under the question; vectors stored as float16"}
    out.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(chunk_vectors)} chunk vectors, {len(query_vectors)} query vectors -> {out}.npz")


if __name__ == "__main__":
    main()
