"""Cross-encoder reranker (bge-reranker-v2-m3 on the course server) and query embeddings with an instruction.

A cross-encoder reads the question and one passage together and returns a relevance score: one HTTP call per
question for all candidates, no chat tokens, no prompt to write. The endpoint is `POST <server root>/rerank`
(Cohere / Jina format, served by vLLM): {"model", "query", "documents"} -> {"results": [{"index",
"relevance_score"}]}.

The request is stored like a chat request (the query and the documents as "messages"), so the call log, the
replay files and `ReplayClient` are the same as for the LLM reranker: data/replay/crossencoder_course.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .config import Profile
from .llm import CallLogger, ChatResponse, _describe_error, request_hash


def rerank_request(query: str, documents: list[str]) -> list[dict[str, str]]:
    """The request as "messages": the replay key is computed from them like for a chat call."""
    return [{"role": "query", "content": query}] + [{"role": "document", "content": d} for d in documents]


def server_root(base_url: str) -> str:
    """/rerank lives at the server root, not under /v1."""
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def parse_rerank(content: str, n: int) -> list[int] | None:
    """Indexes of the documents from the most to the least relevant; None if the response is not usable.
    An index the server did not return keeps its original place after the scored ones."""
    try:
        results = json.loads(content)["results"]
        scored = sorted(((float(r["relevance_score"]), int(r["index"])) for r in results), key=lambda x: -x[0])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    order = []
    for _, i in scored:
        if 0 <= i < n and i not in order:
            order.append(i)
    return order + [i for i in range(n) if i not in order]


class CrossEncoderClient:
    """Live cross-encoder behind /rerank. `chat` takes the messages of `rerank_request`, so MemoClient,
    ReplayFirst and CallLogger work unchanged; the content of the response is {"results": [...]}."""

    def __init__(self, profile: Profile, logger: CallLogger | None = None, retries: int = 2):
        import httpx

        missing = profile.missing_settings()
        if missing:
            raise ValueError(f"Profile '{profile.name}' is not configured: {', '.join(missing)}")
        self.profile, self.profile_name, self.model, self.logger, self.retries = \
            profile, profile.name, profile.model, logger, retries
        self.url = server_root(profile.base_url) + "/rerank"
        self._http = httpx.Client(timeout=profile.timeout_s, headers={"Authorization": f"Bearer {profile.api_key}"})

    def chat(self, messages: list[dict[str, Any]], response_format=None, meta=None) -> ChatResponse:
        query = next(m["content"] for m in messages if m["role"] == "query")
        documents = [m["content"] for m in messages if m["role"] == "document"]
        rhash = request_hash(self.model, messages, response_format)
        start = time.perf_counter()
        error, content, tokens = None, "", 0
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                r = self._http.post(self.url, json={"model": self.model, "query": query, "documents": documents})
                r.raise_for_status()
                data = r.json()
                content = json.dumps({"results": [{"index": x["index"], "relevance_score": x["relevance_score"]}
                                                  for x in data["results"]]})
                meta_tokens = (data.get("meta") or {}).get("tokens") or {}
                tokens = int(meta_tokens.get("input_tokens") or (data.get("usage") or {}).get("total_tokens") or 0)
                error = None
                break
            except Exception as exc:  # network error, HTTP error, unexpected body
                error = _describe_error(exc)
        response = ChatResponse(content=content, finish_reason="stop" if error is None else None, model=self.model,
                                profile=self.profile_name, prompt_tokens=tokens,
                                latency_s=round(time.perf_counter() - start, 3), error=error, request_hash=rhash)
        if self.logger:
            self.logger.log(response, messages, response_format, {"endpoint": "/rerank"}, meta)
        return response


def query_text(question: str, prefix: str = "") -> str:
    """What an asymmetric embedding model embeds for a question: the instruction, then the question.
    Documents are embedded as they are."""
    return f"{prefix}{question}" if prefix else question
