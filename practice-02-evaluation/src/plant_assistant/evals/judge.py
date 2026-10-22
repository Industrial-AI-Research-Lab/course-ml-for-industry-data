"""LLM-as-judge for the free-text field defect_summary.

Two judge prompts, both from the lecture:
- "scale": the common bad prompt. "You are an expert. Rate the helpfulness from 1 to 10." No source,
  no definition, a scale. We keep it to measure how bad it is.
- "binary": one failure mode (a claim that the entry does not support), the source in the prompt,
  the rules of the labelling guideline, a reason first, then PASS or FAIL.

The judge answers with structured output (a Pydantic schema), so its verdicts are fields that code
can count. A judge is a model call like any other: it goes through the same client, it is logged,
and it can be replayed.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm.auto import tqdm

from ..llm import ChatClient
from ..schema import response_format


class ScaleVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int = Field(description="Helpfulness of the answer from 1 (useless) to 10 (perfect).")


class BinaryVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(description="One sentence: which claim of the summary is or is not supported by the entry.")
    verdict: Literal["PASS", "FAIL"] = Field(description="FAIL if the summary contains at least one unsupported claim.")


SCALE_SYSTEM = "You are an expert. Rate the helpfulness of this answer from 1 to 10."
SCALE_USER = "Answer: {summary}"

BINARY_SYSTEM = """You check one thing: does a short defect summary contain a claim that the maintenance log entry does not support?

The entry was written by a technician at an industrial plant, in Russian, English or a mix. The summary was written by an extraction system and should restate the defect in English.

FAIL if the summary has at least one of these:
- a cause or a causal link that the entry does not state ("causing", "due to", "indicating"), even if it sounds plausible;
- a fact that the entry does not state: a part, a location of the defect, a type of work, a symptom;
- a statement that contradicts the entry, for example "no defect" when the entry reports a defect;
- a wrong technical term that changes the meaning (a wrong translation of a part name).

PASS if every claim is stated in the entry. Paraphrases and translations are fine. A summary may be shorter than the entry and may leave things out. The equipment type given in the registry line is a known fact.

First write one sentence with your reason. Then give the verdict."""

BINARY_USER = """Log entry:
<<<
{text}
>>>
{context}
Summary to check:
<<<
{summary}
>>>"""

PROMPTS = {"scale": (SCALE_SYSTEM, SCALE_USER, ScaleVerdict), "binary": (BINARY_SYSTEM, BINARY_USER, BinaryVerdict)}
SCALE_PASS_FROM = 7  # the scale judge "passes" a summary with a score of 7 or more


def judge_messages(prompt: str, text: str, summary: str, context: str = "") -> list[dict[str, str]]:
    system, user, _ = PROMPTS[prompt]
    ctx = f"{context}\n" if context else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(text=text, summary=summary, context=ctx)}]


@dataclass
class JudgeResult:
    item_id: str
    prompt: str
    verdict: str | None  # PASS / FAIL, None if the judge answer was unusable
    score: int | None
    reason: str
    error: str | None
    latency_s: float


def judge_one(client: ChatClient, prompt: str, item: dict[str, Any]) -> JudgeResult:
    _, _, schema = PROMPTS[prompt]
    messages = judge_messages(prompt, item["text"], item["summary"], item.get("context", ""))
    resp = client.chat(messages, response_format=response_format(schema),
                       meta={"item_id": item["item_id"], "judge_prompt": prompt})
    if not resp.ok:
        return JudgeResult(item["item_id"], prompt, None, None, "", resp.error, resp.latency_s)
    try:
        parsed = schema.model_validate(json.loads(resp.content))
    except (json.JSONDecodeError, ValidationError) as exc:
        return JudgeResult(item["item_id"], prompt, None, None, "", f"unparsable: {str(exc)[:80]}", resp.latency_s)
    if isinstance(parsed, ScaleVerdict):
        verdict = "PASS" if parsed.score >= SCALE_PASS_FROM else "FAIL"
        return JudgeResult(item["item_id"], prompt, verdict, parsed.score, "", None, resp.latency_s)
    return JudgeResult(item["item_id"], prompt, parsed.verdict, None, parsed.reason, None, resp.latency_s)


def judge_items(client: ChatClient, items: list[dict[str, Any]], prompt: str = "binary",
                concurrency: int | None = None, progress: bool = True) -> list[JudgeResult]:
    """items: dicts with item_id, text, summary and an optional context line (e.g. the registry type)."""
    if concurrency is None:
        concurrency = getattr(getattr(client, "profile", None), "concurrency", 4)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        it = pool.map(lambda d: judge_one(client, prompt, d), items)
        return list(tqdm(it, total=len(items), desc=f"judge-{prompt}", disable=not progress))
