"""The answer: a simple JSON answer (configurations S0-S5) or the answer contract (S6).

The contract makes the answer a checkable artifact: a status (answered / not_in_documents / conflict /
needs_human), the value with its unit, and claims with the sources they rest on, which code checks (verify.py).
"Not in the documents" and "the documents contradict each other" are normal answers, not failures.

Three versions of the contract:
- v1 (the first reference, rows "S6:v1", "M*@S6:v1"): the model copies citation objects (doc_id, revision,
  amendment, clause, table, row, page) from the source headers. Long answers ran out of the 3000 tokens.
- v2 (rows "S6:v2", "M*@S6:v2"): a claim cites the **ids of the chunks** in the
  context, and the JSON schema makes the id an `enum` of exactly those ids (built per question), so the model
  cannot cite a chunk it did not get. At most 5 claims, at most 2 sources per claim. Code fills document,
  revision, clause, table and page from the index. An answer cut by the token limit or not of the schema is
  asked once more with a short instruction; if that fails too, the answer is "needs a person" with no value (a
  fallback, counted).
- v3 (S6 and every row built on it): v2 plus an internal field `check` (first in the schema, at most 300 characters): do the sources agree,
  is the number in the cited text, is the answer in the documents at all. With thinking off, the JSON is the only
  place the model can reason; in v1 it reasoned inside "answer" ("...Wait, let's r...") and that caught a
  conflict, v2 took the room away. `check` is logged, never shown to the engineer and never graded. "answer" gets
  a `maxLength`; a leak detector in code (markers like "Wait", "Let me", "Actually", or an answer cut at the
  limit) asks once more, then falls back to "needs a person".
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


# ---------------------------------------------------------------------------------------------
# Answer contract v2: citations are chunk ids from the context (an enum in the schema), limits on length
# ---------------------------------------------------------------------------------------------
MAX_CLAIMS, MAX_SOURCES_PER_CLAIM = 5, 2

CONTRACT_V2_SYSTEM = f"""You help engineers of a plant. Answer the question using only the sources.
The question may be in Russian; answer in English. Every source starts with its id in square brackets, then its
document, revision, clause or table, page.
Return JSON by the schema:
- status:
  "answered" - the sources answer the question;
  "not_in_documents" - no source states the answer (do not guess, do not use general knowledge);
  "conflict" - two sources that both apply to this equipment give different values for the same thing
     (for example a plant regulation and a manufacturer's manual); show both, do not choose;
  "needs_human" - the answer depends on data about the equipment that are not in the sources.
- answer: a short answer for the engineer: the decision, the number with its unit, and every action the sources
  require with its deadline. At most 3 sentences.
- value: the key number exactly as written in the source (keep a decimal comma), without the unit; for a machine
  class the roman numeral; null if status is not "answered".
- unit: the unit of the value, or null.
- claims: at most {MAX_CLAIMS} short statements of the answer, each with the ids of at most {MAX_SOURCES_PER_CLAIM}
  sources it rests on (copy the id from the square brackets).
Rules: cite the clause or the table row that states the value, not a neighbouring one. If the sources show dates
of validity, use only sources in force on the date the question is about. Decide first, then write: the "answer"
field holds the result for the engineer, never your reasoning."""

# The second try after an answer cut by the token limit or not of the schema
RETRY_INSTRUCTION = """Your previous answer was too long or not valid JSON. Answer again, briefly: "answer" at most
2 sentences; at most 3 claims of at most 20 words, each with 1 source id. Nothing else."""

FALLBACK_TEXT = "No valid answer within the token limit: a person must answer (the question is logged)."


def contract_v2_format(chunk_ids: list[str]) -> dict:
    """The schema of one question: a claim may cite only the chunks that were put into the prompt."""
    ids = list(dict.fromkeys(chunk_ids)) or ["none"]
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": STATUSES},
            "answer": {"type": "string"},
            "value": {"type": ["string", "null"]},
            "unit": {"type": ["string", "null"]},
            "claims": {"type": "array", "maxItems": MAX_CLAIMS, "items": {
                "type": "object",
                "properties": {"text": {"type": "string"},
                               "sources": {"type": "array", "maxItems": MAX_SOURCES_PER_CLAIM,
                                           "items": {"type": "string", "enum": ids}}},
                "required": ["text", "sources"], "additionalProperties": False}},
        },
        "required": ["status", "answer", "value", "unit", "claims"], "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "answer_contract_v2", "strict": True, "schema": schema}}


def cited_ids(ans: "Answer") -> list[str]:
    """The chunk ids an answer v2 rests on, in order, without repeats."""
    return list(dict.fromkeys(src for cl in ans.claims for src in cl.sources))


def schema_violations(ans: "Answer", allowed_ids: set[str]) -> list[str]:
    """What guided decoding should have prevented: checked in code anyway (an unknown id fails its citation)."""
    out = []
    if len(ans.claims) > MAX_CLAIMS:
        out.append(f"{len(ans.claims)} claims")
    out += [f"{len(cl.sources)} sources in a claim" for cl in ans.claims if len(cl.sources) > MAX_SOURCES_PER_CLAIM]
    out += [f"unknown id {i}" for i in cited_ids(ans) if i not in allowed_ids]
    return out


def fallback_answer() -> "Answer":
    return Answer(status="needs_human", answer=FALLBACK_TEXT)


# ---------------------------------------------------------------------------------------------
# Answer contract v3: v2 + an internal "check" field, a length limit on "answer", a leak detector
# ---------------------------------------------------------------------------------------------
CHECK_MAX, ANSWER_MAX = 300, 600

CONTRACT_V3_SYSTEM = CONTRACT_V2_SYSTEM.replace(
    "Return JSON by the schema:\n- status:",
    f"Return JSON by the schema:\n"
    f"- check: your internal check before you decide, at most {CHECK_MAX} characters; the engineer never sees it.\n"
    f"  Do the sources that apply to this equipment and date agree or give different values? Is the number in the\n"
    f"  text of the source you will cite? Is the answer in the sources at all, or only something close to it?\n"
    f"- status:",
).replace(
    "  require with its deadline. At most 3 sentences.",
    f"  require with its deadline. At most 3 sentences, at most {ANSWER_MAX} characters, only the result: your\n"
    f"  reasoning goes into \"check\", never into \"answer\".",
)

RETRY_V3_INSTRUCTION = """Your previous answer was too long, not valid JSON, or had your reasoning in the "answer"
field. Put the reasoning into "check". "answer" holds only the result for the engineer, at most 2 sentences; at
most 3 claims, each with 1 source id."""

# reasoning that leaked into the answer for the engineer
LEAK_MARKERS = re.compile(r"\b(wait|let me|let's|let us|actually|hmm+|hold on|on second thought|re-?check\w*|"
                          r"i think|i need to|re-?read\w*)\b", re.I)


def contract_v3_format(chunk_ids: list[str]) -> dict:
    """The v2 schema with `check` first (the model checks before it decides) and a length limit on `answer`."""
    v2 = contract_v2_format(chunk_ids)["json_schema"]["schema"]
    props = {"check": {"type": "string", "maxLength": CHECK_MAX}, **v2["properties"]}
    props["answer"] = {"type": "string", "maxLength": ANSWER_MAX}
    schema = {**v2, "properties": props, "required": ["check", *v2["required"]]}
    return {"type": "json_schema", "json_schema": {"name": "answer_contract_v3", "strict": True, "schema": schema}}


def leak(answer_text: str) -> str | None:
    """Why the answer for the engineer looks like reasoning, or None."""
    m = LEAK_MARKERS.search(answer_text or "")
    if m:
        return f"marker {m.group(0)!r}"
    if len(answer_text or "") >= ANSWER_MAX:
        return f"cut at {ANSWER_MAX} characters"
    return None


def parse_v3(content: str) -> tuple[Answer | None, str]:
    """The answer for the engineer and the internal check, kept apart: the check never enters `Answer`."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content or "")
    try:
        data = json.loads(text)
        check = str(data.pop("check", ""))
        return Answer(**data), check
    except (json.JSONDecodeError, ValidationError, TypeError, AttributeError):
        return None, ""


def messages(question: str, context: str, asked_on: str, as_of: str | None, equipment: str = "",
             contract: bool = False, version: int = 1, retry: bool = False) -> list[dict]:
    when = f"The question is asked on {asked_on}."
    if as_of and as_of != asked_on:
        when += f" It is about the date {as_of}."
    parts = [when, f"Question: {question}"]
    if equipment:
        parts.append(equipment)
    parts.append(f"Sources:\n{context}" if context else "Sources: none")
    system = SIMPLE_SYSTEM if not contract else {1: CONTRACT_SYSTEM, 2: CONTRACT_V2_SYSTEM}.get(version,
                                                                                            CONTRACT_V3_SYSTEM)
    if retry:
        system += "\n\n" + (RETRY_V3_INSTRUCTION if version >= 3 else RETRY_INSTRUCTION)
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


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
