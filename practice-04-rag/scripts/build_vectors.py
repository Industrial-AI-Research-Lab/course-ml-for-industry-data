"""Embedding vectors of the chunks and of the questions, for dense search (M1) and hybrid search (M2).

Needs an embedding profile that answers (models.toml): Ollama on your laptop (`ollama pull bge-m3`) or an
embedding model on the course server. Writes data/replay/vectors/<name>.npz (chunk_vectors, query_vectors)
and <name>.json (model, dimension, corpus version, chunk ids and query texts in the same order).
Every chunk of the store gets a vector (both views: page tables and table rows); every wording of every
question that can go to search (Russian, English, and the translations of the M1t row) gets one too.

Run:  uv run python scripts/build_vectors.py --profile embed-bge-m3 --name bge-m3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_assistant import REPLAY_DIR, STORE_DIR  # noqa: E402
from plant_assistant.cases import load_cases  # noqa: E402
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
    ap.add_argument("--profile", default="embed-bge-m3")
    ap.add_argument("--name", default="bge-m3")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    _, chunks = load_store(STORE_DIR)
    texts = [normalize_tags(plain(c.text)) for c in chunks]
    queries = sorted({q for c in load_cases() for q in (c.question_ru, c.question_en)})
    translations = REPLAY_DIR / "translate_course.jsonl"
    if translations.exists():
        for line in translations.read_text(encoding="utf-8").splitlines():
            queries.append(json.loads(line)["response"]["content"].strip())
    queries = list(dict.fromkeys(queries))
    chunk_vectors = [v for b in tqdm(list(batched(texts, args.batch)), desc="chunks") for v in embed(args.profile, b)]
    query_vectors = [v for b in tqdm(list(batched(queries, args.batch)), desc="queries") for v in embed(args.profile, b)]
    out = REPLAY_DIR / "vectors" / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), chunk_vectors=np.asarray(chunk_vectors, dtype=np.float16),
                        query_vectors=np.asarray(query_vectors, dtype=np.float16))
    version = index_version(chunks, args.name)
    manifest = {"model": args.name, "profile": args.profile, "dimension": len(chunk_vectors[0]),
                "corpus": version.split(".")[0], "index_version": version,
                "chunk_ids": [c.chunk_id for c in chunks], "queries": queries,
                "note": "chunk text: tags normalized, markup removed; vectors stored as float16"}
    out.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(chunk_vectors)} chunk vectors, {len(query_vectors)} query vectors -> {out}.npz")


if __name__ == "__main__":
    main()
