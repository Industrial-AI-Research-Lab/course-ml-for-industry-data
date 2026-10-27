"""Check that a model profile answers. Usage:

    uv run python -m plant_assistant.smoke_test course          # course Qwen: a chat call
    uv run python -m plant_assistant.smoke_test embed-bge-m3    # an embedding profile: one vector
"""

from __future__ import annotations

import sys

from .config import get_profile, load_env
from .llm import LLMClient


def embed(profile_name: str, texts: list[str]) -> list[list[float]]:
    """Vectors from an OpenAI-compatible /embeddings endpoint (Ollama, vLLM behind LiteLLM)."""
    from openai import OpenAI

    p = get_profile(profile_name)
    client = OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=p.timeout_s)
    out = client.embeddings.create(model=p.model, input=texts)
    return [d.embedding for d in out.data]


def check(profile_name: str) -> bool:
    env_path = load_env()
    profile = get_profile(profile_name)
    print(f"profile : {profile.name} ({profile.description})")
    print(f"model   : {profile.model}")
    print(f"url     : {profile.base_url or '<missing>'}")
    print(f".env    : {env_path or '<not found>'}")
    missing = profile.missing_settings()
    if missing:
        print("FAIL    : " + "; ".join(missing))
        return False
    if profile_name.startswith("embed"):
        try:
            v = embed(profile_name, ["Граница осмотра для насоса P-101"])[0]
        except Exception as exc:  # no server, no model
            print(f"FAIL    : {type(exc).__name__}: {str(exc)[:200]}")
            print(f"hint    : for Ollama: `ollama pull {profile.model}`")
            return False
        print(f"vector  : dimension {len(v)}")
        print("OK")
        return True
    client = LLMClient(profile, max_network_retries=0)
    reply = client.chat([{"role": "user", "content": "Reply with one word: OK"}])
    if not reply.ok:
        print(f"FAIL    : {reply.error}")
        return False
    print(f"text    : {reply.content.strip()[:40]!r} in {reply.latency_s:.1f}s")
    print("OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else "course") else 1)
