"""One client for every model: OpenAI-compatible API, call logging, replay.

- LLMClient calls a live model (Ollama, LiteLLM, any OpenAI-compatible server).
- ReplayClient answers from a saved call log. It has the same interface, so the rest
  of the pipeline (validation, retries, metrics) runs unchanged on saved answers.
- CallLogger writes every call to JSONL: prompt, answer, model, tokens, latency, cost.
  A call log of one run is also a replay file for the next run.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .config import Profile
from .jsonl import read_json_records


@dataclass
class ChatResponse:
    content: str
    finish_reason: str | None
    model: str
    profile: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning: str = ""
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    request_hash: str = ""
    from_replay: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class ChatClient(Protocol):
    profile_name: str
    model: str

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ChatResponse: ...


def request_hash(model: str, messages: list[dict[str, str]], response_format: dict[str, Any] | None) -> str:
    """Stable key of a request. Replay finds saved answers by this key."""
    payload = json.dumps(
        {"model": model, "messages": messages, "response_format": response_format},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class CallLogger:
    """Append-only JSONL log of model calls. Safe to use from several threads."""

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._lock = threading.Lock()

    def log(self, response: ChatResponse, messages, response_format, params, meta) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "request_hash": response.request_hash,
            "profile": response.profile,
            "model": response.model,
            "meta": meta or {},
            "params": params,
            "response_format": (response_format or {}).get("json_schema", {}).get("name")
            if response_format
            else None,
            "messages": messages,
            "response": {k: v for k, v in asdict(response).items() if k not in ("request_hash",)},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local_url(url: str) -> bool:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "") in LOCAL_HOSTS


def _describe_error(exc: Exception) -> str:
    """The SDK says only "Connection error."; the real reason is in the cause chain."""
    text = f"{type(exc).__name__}: {str(exc)[:300]}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        text += f" (cause: {type(cause).__name__}: {str(cause)[:200]})"
    return text


class LLMClient:
    """Live model behind an OpenAI-compatible API."""

    def __init__(self, profile: Profile, logger: CallLogger | None = None, max_network_retries: int = 2):
        from openai import DefaultHttpxClient, OpenAI  # imported here so replay works without network setup

        missing = profile.missing_settings()
        if missing:
            raise ValueError(f"Profile '{profile.name}' is not configured: {', '.join(missing)}")
        self.profile = profile
        self.profile_name = profile.name
        self.model = profile.model
        self.logger = logger
        # A local server (Ollama) never needs a proxy. A VPN that sets HTTP_PROXY without NO_PROXY
        # would otherwise send requests for localhost to the proxy, and they fail.
        http_client = DefaultHttpxClient(trust_env=False) if is_local_url(profile.base_url) else None
        # Network-level retries (timeouts, 429, 5xx) with exponential backoff are done by the SDK.
        # Validation retries (bad output) are done by our extraction code. These are different things.
        self._client = OpenAI(
            base_url=profile.base_url,
            api_key=profile.api_key,
            timeout=profile.timeout_s,
            max_retries=max_network_retries,
            http_client=http_client,
        )
        self._run_tag = uuid.uuid4().hex[:8]

    def chat(self, messages, response_format=None, meta=None) -> ChatResponse:
        p = self.profile
        params: dict[str, Any] = {"max_tokens": p.max_tokens, "temperature": p.temperature}
        extra_body = dict(p.extra_body)
        if p.bypass_proxy_cache:
            extra_body["user"] = f"run-{self._run_tag}"
        rhash = request_hash(p.model, messages, response_format)
        kwargs: dict[str, Any] = dict(model=p.model, messages=messages, **params)
        if response_format:
            kwargs["response_format"] = response_format
        if extra_body:
            kwargs["extra_body"] = extra_body

        start = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - start
            choice = completion.choices[0]
            message = choice.message
            usage = completion.usage
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            response = ChatResponse(
                content=message.content or "",
                finish_reason=choice.finish_reason,
                model=p.model,
                profile=p.name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
                reasoning=reasoning,
                latency_s=round(latency, 3),
                cost_usd=p.cost_usd(prompt_tokens, completion_tokens),
                request_hash=rhash,
            )
        except Exception as exc:  # network error, timeout, HTTP error after SDK retries
            response = ChatResponse(
                content="",
                finish_reason=None,
                model=p.model,
                profile=p.name,
                latency_s=round(time.perf_counter() - start, 3),
                error=_describe_error(exc),
                request_hash=rhash,
            )
        if self.logger:
            self.logger.log(response, messages, response_format, {**params, "extra_body": p.extra_body}, meta)
        return response


class ReplayMiss(KeyError):
    pass


class ReplayClient:
    """Answers from a saved call log instead of calling the model.

    The key is the request itself (model + messages + schema). If you change a prompt,
    the saved answer no longer matches: ReplayMiss tells you that you need a live model.
    """

    def __init__(self, path: str | Path, logger: CallLogger | None = None, strict: bool = True,
                 model: str | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Replay file not found: {self.path}. Replay files ship with the repository in data/replay/: "
                f"check that your clone is complete (git status, git pull)."
            )
        self.logger = logger
        self.strict = strict
        self._by_hash: dict[str, list[dict[str, Any]]] = {}
        self._used: dict[str, int] = {}
        self._lock = threading.Lock()
        first = None
        for entry in read_json_records(self.path):
            first = first or entry
            self._by_hash.setdefault(entry["request_hash"], []).append(entry)
        if first is None:
            raise ValueError(f"Replay file is empty: {self.path}")
        # One file can hold several models (main + fallback): pick one with `model`.
        chosen = next((e for v in self._by_hash.values() for e in v if e["model"] == model), first) if model else first
        self.profile_name = chosen["profile"]
        self.model = chosen["model"]

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_hash.values())

    def chat(self, messages, response_format=None, meta=None) -> ChatResponse:
        rhash = request_hash(self.model, messages, response_format)
        with self._lock:
            entries = self._by_hash.get(rhash)
            if not entries:
                if self.strict:
                    raise ReplayMiss(
                        f"No saved answer for this request in {self.path.name}. "
                        "Did you change the prompt or the schema? Replay only works for unchanged requests."
                    )
                return ChatResponse(
                    content="", finish_reason=None, model=self.model, profile=self.profile_name,
                    error="ReplayMiss", request_hash=rhash, from_replay=True,
                )
            # Identical requests can appear more than once (e.g. repeated runs): serve them in order.
            i = self._used.get(rhash, 0)
            entry = entries[min(i, len(entries) - 1)]
            self._used[rhash] = i + 1
        saved = entry["response"]
        response = ChatResponse(**{**saved, "request_hash": rhash, "from_replay": True})
        if self.logger:
            self.logger.log(response, messages, response_format, entry.get("params", {}), meta)
        return response


def make_client(profile_name: str, logger: CallLogger | None = None) -> LLMClient:
    from .config import get_profile

    return LLMClient(get_profile(profile_name), logger=logger)
