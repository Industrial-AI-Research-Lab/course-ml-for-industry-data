"""Embedding calls shared by vector generation, live search and connection checks."""

from __future__ import annotations

from .config import get_profile


def embed(profile_name: str, texts: list[str], query: bool = False) -> list[list[float]]:
    """Vectors from an OpenAI-compatible /embeddings endpoint (Ollama, vLLM behind LiteLLM). `query=True`: the
    texts are questions and get the instruction of the profile (`query_prefix`; e5-mistral needs it)."""
    from openai import OpenAI

    from .crossencoder import query_text

    p = get_profile(profile_name)
    client = OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=p.timeout_s)
    if query:
        texts = [query_text(t, p.query_prefix) for t in texts]
    out = client.embeddings.create(model=p.model, input=texts)
    return [d.embedding for d in out.data]

