"""Check that a model profile answers. Usage:

    uv run python -m plant_assistant.smoke_test            # profile "local"
    uv run python -m plant_assistant.smoke_test course     # course Qwen
"""

from __future__ import annotations

import os
import re
import sys
from urllib.parse import urlsplit

from .config import get_profile, load_env
from .llm import LLMClient, is_local_url
from .schema import response_format
from . import prompts

SAMPLE = "Р-101 замена подшипника, гудел, 12.02.2026, 4 ч, простой"
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def _mask(value: str) -> str:
    return re.sub(r"//[^@/]+@", "//***@", value)  # hide user:password in proxy URLs


def diagnose_local(base_url: str, model: str) -> None:
    """Local server down, or up but the model call failed? (Requests to localhost bypass proxies.)"""
    import httpx

    found = {k: v for k, v in os.environ.items() if k.upper() in PROXY_VARS}
    print("proxy   : " + (", ".join(f"{k}={_mask(v)}" for k, v in sorted(found.items())) or "no proxy variables")
          + " (not used for localhost)")
    parts = urlsplit(base_url)
    probe = f"{parts.scheme}://{parts.netloc}/api/version"
    try:
        answer = f"HTTP {httpx.get(probe, timeout=5, trust_env=False).status_code}"
    except Exception as exc:
        answer = f"{type(exc).__name__}: {str(exc)[:120]}"
    print(f"probe   : {probe} -> {answer}")
    if answer.startswith("HTTP"):
        print(f"hint    : Ollama is running, but the call failed. Is the model downloaded? `ollama pull {model}`")
    else:
        print("hint    : Ollama does not answer. Start it from the Start menu (or run `ollama serve`), then retry.")


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
    client = LLMClient(profile, max_network_retries=0)
    reply = client.chat([{"role": "user", "content": "Reply with one word: OK"}])
    if not reply.ok:
        print(f"FAIL    : {reply.error}")
        if is_local_url(profile.base_url):
            diagnose_local(profile.base_url, profile.model)
        return False
    print(f"reply   : {reply.content.strip()[:40]!r} in {reply.latency_s:.1f}s (the first call can be slow: model loading)")
    structured = client.chat(prompts.extraction_messages(SAMPLE), response_format=response_format())
    if not structured.ok or not structured.content.strip().startswith("{"):
        print(f"FAIL    : structured output does not work: {structured.error or structured.content[:120]!r}")
        if structured.finish_reason == "length":
            print("hint    : the answer hit the token limit; is the thinking mode off for this profile?")
        return False
    print(f"schema  : OK, {structured.completion_tokens} tokens in {structured.latency_s:.1f}s")
    print("OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else "local") else 1)
