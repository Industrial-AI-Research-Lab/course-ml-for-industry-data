"""Fields of a question: equipment tags, named documents, the station, the date, "what changed".

Rules first: they are cheap, deterministic and testable. A schema on the course model (the approach of
topic 1) is a fallback for what the rules did not take; it is an optional task of the practice.
The number in the question ("4,8 мм/с") is a value to compare, not a search key: we keep it as a field.
"""

from __future__ import annotations

import json
import re
from datetime import date

from pydantic import BaseModel, Field

from .normalize import find_tags

DOC_RE = re.compile(r"\b(R-\d+|PS-\d+|VL-\d+|OM-P-\d+|DS-P-\d+)\b")
STATION_RE = re.compile(r"(?:станци[яиюейм]+|station)\s*(?:№|No\.?)?\s*(\d+)", re.I)
DATE_DOTS = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
MONTHS_RU = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7, "август": 8,
             "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}
MONTHS_EN = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                         "september", "october", "november", "december"], start=1)}
DATE_RU = re.compile(r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|"
                     r"декабря)\s+(\d{4})", re.I)
DATE_EN = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS_EN) + r")\s+(\d{4})", re.I)
HISTORY_RE = re.compile(r"изменил|поменял|по сравнению|что изменилось|раньше было|\bchanged?\b|compared with|"
                        r"\bdifference\b", re.I)
VALUE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(мм/с|mm/s)", re.I)


class QueryFields(BaseModel):
    tags: list[str] = Field(default_factory=list, description="equipment tags, normalized: P-101")
    doc_ids: list[str] = Field(default_factory=list, description="documents named in the question: R-12")
    stations: list[str] = Field(default_factory=list, description="pump stations named in the question: '3'")
    as_of: str | None = Field(None, description="the date the question is about, YYYY-MM-DD")
    date_in_text: bool = False
    history: bool = Field(False, description="the question asks what changed between revisions")
    measured_value: str | None = Field(None, description="a measured vibration velocity in the question")


def _iso(d: int, m: int, y: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_date(text: str) -> str | None:
    m = DATE_DOTS.search(text)
    if m:
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = DATE_RU.search(text)
    if m:
        month = next(v for k, v in MONTHS_RU.items() if m.group(2).lower().startswith(k))
        return _iso(int(m.group(1)), month, int(m.group(3)))
    m = DATE_EN.search(text)
    if m:
        return _iso(int(m.group(1)), MONTHS_EN[m.group(2).lower()], int(m.group(3)))
    return None


def extract_rules(question: str, asked_on: str) -> QueryFields:
    """Fields by rules. Without a date in the text the question is about the day it is asked;
    a "what changed" question is about no single date (every revision counts)."""
    when = find_date(question)
    history = bool(HISTORY_RE.search(question))
    value = VALUE_RE.search(question)
    return QueryFields(
        tags=find_tags(question),
        doc_ids=list(dict.fromkeys(DOC_RE.findall(question))),
        stations=list(dict.fromkeys(STATION_RE.findall(question))),
        as_of=None if history else (when or asked_on),
        date_in_text=when is not None,
        history=history,
        measured_value=value.group(1) if value else None,
    )


# ---------------------------------------------------------------------------------------------
# Fallback: the same schema filled by the course model (optional task 3)
# ---------------------------------------------------------------------------------------------
LLM_SYSTEM = """Extract fields from an engineer's question about plant documents. Return JSON by the schema.
Tags are equipment tags like P-101, K-301, V-12: write them with Latin letters and a hyphen.
as_of is the date the question is about (YYYY-MM-DD) or null if the question names no date.
history is true only if the question asks what changed between revisions of a document."""


def extract_llm(client, question: str, asked_on: str) -> QueryFields:
    schema = QueryFields.model_json_schema()
    rf = {"type": "json_schema", "json_schema": {"name": "query_fields", "schema": schema}}
    resp = client.chat([{"role": "system", "content": LLM_SYSTEM},
                        {"role": "user", "content": f"Today is {asked_on}.\nQuestion: {question}"}], response_format=rf)
    data = json.loads(resp.content)
    fields = QueryFields(**data)
    if not fields.history and not fields.as_of:
        fields.as_of = asked_on
    return fields
