"""Check that a model profile answers. Usage:

    uv run python -m plant_assistant.smoke_test course          # course Qwen: a chat call
    uv run python -m plant_assistant.smoke_test embed-bge-m3    # an embedding profile: one vector
"""

from __future__ import annotations

import sys

from .config import get_profile, load_env
from .llm import LLMClient
from .embeddings import embed


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
    if profile_name.startswith("rerank"):
        from .crossencoder import CrossEncoderClient, parse_rerank, rerank_request

        r = CrossEncoderClient(profile, retries=0).chat(rerank_request(
            "Граница осмотра насоса класса II", ["Таблица 3: класс II, осмотр 4,5 мм/с", "Смазка каждые 2000 ч"]))
        order = parse_rerank(r.content, 2) if r.ok else None
        if order is None:
            print(f"FAIL    : {r.error or 'unusable response'}")
            return False
        print(f"ranking : {order} in {r.latency_s:.2f}s")
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
