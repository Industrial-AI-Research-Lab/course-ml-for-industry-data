"""Extraction: from one free-text log entry to a validated MaintenanceRecord.

Two ways:
- extract_baseline: "return JSON" in the prompt, parse the answer. No schema, no retries.
- extract_structured: JSON Schema in the API call (schema-constrained output or constrained
  decoding), then validation -> business rules -> retry with the error text -> fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ValidationError

from . import prompts
from .data import Equipment
from .llm import ChatClient, ChatResponse
from .schema import MaintenanceRecord, response_format
from .validation import check_business_rules

# Status of a record after extraction.
VALID_FIRST = "valid_first"  # passed all checks on the first attempt
VALID_AFTER_RETRY = "valid_after_retry"  # passed after one or more retries
FALLBACK_VALID = "fallback_valid"  # the main model failed, the fallback model succeeded
NEEDS_REVIEW = "needs_review"  # nothing worked: goes to a human
INVALID = "invalid"  # baseline only: the answer is not valid (no retries in the baseline)

VALID_STATUSES = {VALID_FIRST, VALID_AFTER_RETRY, FALLBACK_VALID}


@dataclass
class Attempt:
    number: int
    profile: str
    content: str
    error_type: str | None  # None = passed
    errors: list[str]
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float


@dataclass
class ExtractionResult:
    record_id: str
    mode: str
    profile: str
    status: str
    record: dict[str, Any] | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status in VALID_STATUSES

    @property
    def first_error_type(self) -> str | None:
        return self.attempts[0].error_type if self.attempts else None

    @property
    def latency_s(self) -> float:
        return round(sum(a.latency_s for a in self.attempts), 3)

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def prompt_tokens(self) -> int:
        return sum(a.prompt_tokens for a in self.attempts)

    @property
    def completion_tokens(self) -> int:
        return sum(a.completion_tokens for a in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            valid=self.valid,
            latency_s=self.latency_s,
            cost_usd=self.cost_usd,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
        return d


@dataclass
class ParseOutcome:
    record: MaintenanceRecord | None
    error_type: str | None
    errors: list[str]


def _pydantic_errors(exc: ValidationError) -> list[str]:
    out = []
    for e in exc.errors()[:8]:
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        got = e.get("input")
        got_text = f" (got {json.dumps(got, ensure_ascii=False)[:60]})" if not isinstance(got, dict) else ""
        out.append(f"{loc}: {e['msg']}{got_text}")
    return out


def classify_parse_failure(response: ChatResponse) -> tuple[str, str]:
    """Why could the answer not be parsed as JSON? Returns (error_type, message)."""
    text = response.content or ""
    if response.error:
        return "api_error", response.error
    if not text.strip():
        if response.reasoning_tokens or response.reasoning:
            return "empty", "The answer is empty: the reasoning used the whole token budget."
        return "empty", "The answer is empty."
    if response.finish_reason == "length":
        return "truncated", "The answer was cut off by the token limit."
    if "```" in text:
        return "markdown_fence", "The answer is wrapped in a markdown code block. Return plain JSON."
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return "extra_text", "The answer contains text around the JSON. Return only JSON."
        except json.JSONDecodeError:
            pass
    return "invalid_json", "The answer is not valid JSON."


def parse_answer(
    response: ChatResponse,
    source_text: str,
    registry: dict[str, Equipment] | None,
) -> ParseOutcome:
    """JSON -> schema -> business rules. Stops at the first stage that fails."""
    try:
        data = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        error_type, message = classify_parse_failure(response)
        return ParseOutcome(None, error_type, [message])
    try:
        record = MaintenanceRecord.model_validate(data)
    except ValidationError as exc:
        return ParseOutcome(None, "schema_error", _pydantic_errors(exc))
    if registry is not None:
        problems = check_business_rules(record, source_text, registry)
        if problems:
            return ParseOutcome(record, "business_rule", problems)
    return ParseOutcome(record, None, [])


def _attempt(number: int, response: ChatResponse, outcome: ParseOutcome) -> Attempt:
    return Attempt(
        number=number,
        profile=response.profile,
        content=response.content,
        error_type=outcome.error_type,
        errors=outcome.errors,
        latency_s=response.latency_s,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
        cost_usd=response.cost_usd,
    )


def extract_baseline(client: ChatClient, record_id: str, text: str) -> ExtractionResult:
    """Prompt-only JSON. The answer is valid if it parses and matches the schema."""
    messages = prompts.baseline_messages(text)
    response = client.chat(messages, response_format=None, meta={"record_id": record_id, "mode": "baseline", "attempt": 1})
    outcome = parse_answer(response, text, registry=None)
    status = VALID_FIRST if outcome.error_type is None else INVALID
    return ExtractionResult(
        record_id=record_id,
        mode="baseline",
        profile=client.profile_name,
        status=status,
        record=outcome.record.model_dump(mode="json") if outcome.record and status == VALID_FIRST else None,
        attempts=[_attempt(1, response, outcome)],
    )


def _structured_loop(
    client: ChatClient,
    record_id: str,
    text: str,
    registry: dict[str, Equipment] | None,
    max_attempts: int,
    attempts: list[Attempt],
    stage: str,
    field_guide: bool = True,
) -> MaintenanceRecord | None:
    messages = prompts.extraction_messages(text, field_guide=field_guide)
    rf = response_format()
    for i in range(1, max_attempts + 1):
        response = client.chat(
            messages,
            response_format=rf,
            meta={"record_id": record_id, "mode": "structured", "attempt": i, "stage": stage},
        )
        outcome = parse_answer(response, text, registry)
        attempts.append(_attempt(len(attempts) + 1, response, outcome))
        if outcome.error_type is None:
            return outcome.record
        if outcome.error_type == "api_error":
            continue  # the SDK already retried the network; try once more, without changing the prompt
        messages = prompts.retry_messages(messages, response.content, outcome.errors)
    return None


def extract_structured(
    client: ChatClient,
    record_id: str,
    text: str,
    registry: dict[str, Equipment] | None,
    max_attempts: int = 3,
    fallback: ChatClient | None = None,
    field_guide: bool = True,
) -> ExtractionResult:
    """Schema-constrained call -> validate -> retry with errors -> fallback model -> human review.

    field_guide=False sends the schema only, without field definitions in the prompt (for comparison).
    """
    attempts: list[Attempt] = []
    record = _structured_loop(client, record_id, text, registry, max_attempts, attempts, "main", field_guide)
    if record is not None:
        status = VALID_FIRST if len(attempts) == 1 else VALID_AFTER_RETRY
    elif fallback is not None:
        record = _structured_loop(fallback, record_id, text, registry, max_attempts, attempts, "fallback", field_guide)
        status = FALLBACK_VALID if record is not None else NEEDS_REVIEW
    else:
        status = NEEDS_REVIEW
    return ExtractionResult(
        record_id=record_id,
        mode="structured" if field_guide else "structured-schema-only",
        profile=client.profile_name,
        status=status,
        record=record.model_dump(mode="json") if record is not None else None,
        attempts=attempts,
    )
