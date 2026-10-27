"""The answer: a simple JSON answer (configurations S0-S5) or the answer contract (S6).

The contract makes the answer a checkable artifact: a status (answered / not_in_documents / conflict /
needs_human), the value with its unit, claims with the sources they rest on, and citations as fields
(doc_id, revision, amendment, clause, table, row, page) that code can check (verify.py).
"Not in the documents" and "the documents contradict each other" are normal answers, not failures.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

STATUSES = ["answered", "not_in_documents", "conflict", "needs_human"]


class Citation(BaseModel):
    source: str = Field(description="label of the source, e.g. S2")
    doc_id: str
    revision: int | None = None
    amendment: int | None = None
    clause: str | None = None
    table_no: str | None = None
    row_key: str | None = None
    page: int | None = None


class Claim(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    status: Literal["answered", "not_in_documents", "conflict", "needs_human"] = "answered"
    answer: str = ""
    value: str | None = None
    unit: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # simple answers: labels only


# ---------------------------------------------------------------------------------------------
# Simple answer (S0-S5): value, unit, source labels
# ---------------------------------------------------------------------------------------------
SIMPLE_SYSTEM = """You help engineers of a plant. Answer the question using only the numbered sources.
The question may be in Russian; answer in English.
Return JSON with these fields:
- answer: one to three sentences: the decision, the number with its unit, what to do and by when.
- value: the key number of the answer exactly as written in the source (keep a decimal comma), without the unit;
  for a machine class the roman numeral; null if the sources do not contain the answer.
- unit: the unit of the value, or null.
- sources: labels of the sources you used, like ["S1", "S3"].
Do not guess a value that is not in the sources."""

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "value": {"type": ["string", "null"]},
                   "unit": {"type": ["string", "null"]}, "sources": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "value", "unit", "sources"], "additionalProperties": False,
}
SIMPLE_FORMAT = {"type": "json_schema", "json_schema": {"name": "simple_answer", "strict": True, "schema": SIMPLE_SCHEMA}}

# ---------------------------------------------------------------------------------------------
# Answer contract (S6)
# ---------------------------------------------------------------------------------------------
CONTRACT_SYSTEM = """You help engineers of a plant. Answer the question using only the numbered sources.
The question may be in Russian; answer in English. Every source has a header with its citation fields.
Return JSON by the schema:
- status:
  "answered" - the sources answer the question;
  "not_in_documents" - no source states the answer (do not guess, do not use general knowledge);
  "conflict" - two sources that both apply to this equipment give different values for the same thing
     (for example a plant regulation and a manufacturer's manual); show both, do not choose;
  "needs_human" - the answer depends on data about the equipment that are not in the sources.
- answer: a short answer for the engineer: the decision, the number with its unit, and every action the sources
  require with its deadline.
- value: the key number exactly as written in the source (keep a decimal comma), without the unit; for a machine
  class the roman numeral; null if status is not "answered".
- unit: the unit of the value, or null.
- claims: the statements of the answer, each with the labels of the sources it rests on.
- citations: one entry per source you rely on, with the fields copied from its header (doc_id, revision,
  amendment, clause, table_no, row_key, page; null if the header has no such field) and its label in "source".
Rules: cite the clause or the table row that states the value, not a neighbouring one. If the sources show dates
of validity, use only sources in force on the date the question is about. Decide first, then write: the "answer"
field holds the result for the engineer, never your reasoning."""

CITATION_SCHEMA = {
    "type": "object",
    "properties": {"source": {"type": "string"}, "doc_id": {"type": "string"},
                   "revision": {"type": ["integer", "null"]}, "amendment": {"type": ["integer", "null"]},
                   "clause": {"type": ["string", "null"]}, "table_no": {"type": ["string", "null"]},
                   "row_key": {"type": ["string", "null"]}, "page": {"type": ["integer", "null"]}},
    "required": ["source", "doc_id", "revision", "amendment", "clause", "table_no", "row_key", "page"],
    "additionalProperties": False,
}
CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": STATUSES},
        "answer": {"type": "string"},
        "value": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "claims": {"type": "array", "items": {
            "type": "object", "properties": {"text": {"type": "string"},
                                             "sources": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "sources"], "additionalProperties": False}},
        "citations": {"type": "array", "items": CITATION_SCHEMA},
    },
    "required": ["status", "answer", "value", "unit", "claims", "citations"], "additionalProperties": False,
}
CONTRACT_FORMAT = {"type": "json_schema", "json_schema": {"name": "answer_contract", "strict": True,
                                                          "schema": CONTRACT_SCHEMA}}


def messages(question: str, context: str, asked_on: str, as_of: str | None, equipment: str = "",
             contract: bool = False) -> list[dict]:
    when = f"The question is asked on {asked_on}."
    if as_of and as_of != asked_on:
        when += f" It is about the date {as_of}."
    parts = [when, f"Question: {question}"]
    if equipment:
        parts.append(equipment)
    parts.append(f"Sources:\n{context}" if context else "Sources: none")
    return [{"role": "system", "content": CONTRACT_SYSTEM if contract else SIMPLE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def parse(content: str, contract: bool) -> Answer | None:
    """The model output as an Answer, or None if it is not valid JSON of the schema."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content or "")
    try:
        data = json.loads(text)
        if not contract:
            data = {"status": "answered" if data.get("value") not in (None, "") else "not_in_documents", **data}
        return Answer(**data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None
