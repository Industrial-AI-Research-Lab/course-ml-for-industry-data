"""Mini-eval of reading end-to-end: questions about the scans, answers checked by code.

For every question we know the page with the answer. The reader's text of that page goes to the model
(search is topic 4: here the chunk is given), the model answers in JSON, and code checks the number, the
unit and "not stated". The same model answers for every reader, so a difference between readers is a
difference in what they read.

Statistics come from lecture 2 and are not explained again: accuracy with a 95% interval, readers
compared in pairs on the same questions, and for every wrong answer the step where it was lost
(`attribute`: the same question with the born-digital text shows whether the reader or the model failed).
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .metrics import number_value, numbers

SYSTEM = """You answer an engineer's question about one page of a plant document.
Use only the page text. The text comes from a document reader and can contain recognition errors.
Return JSON with these fields:
- value: the number exactly as written on the page (keep a decimal comma if it is there), without the unit.
- unit: the unit of that value as written on the page, or null if the value has no unit.
- clause: where the value is: a clause number like "4.3.3", a table like "Table 3", or a line name; null if unclear.
- not_stated: true if the page does not contain the answer. Then value, unit and clause are null.
Do not guess a value that is not on the page."""

SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": ["string", "null"]},
        "unit": {"type": ["string", "null"]},
        "clause": {"type": ["string", "null"]},
        "not_stated": {"type": "boolean"},
    },
    "required": ["value", "unit", "clause", "not_stated"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "page_answer", "strict": True, "schema": SCHEMA}}


def messages(question: str, page_text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Page text:\n<<<\n{page_text.strip()}\n>>>\n\nQuestion: {question}"}]


# ---------------------------------------------------------------------------------------------
# Grading by code
# ---------------------------------------------------------------------------------------------
UNIT_SYNONYMS = {
    "mm/s": ["mm/s", "мм/с", "mm/sec"],
    "%": ["%", "percent", "процента", "процентов"],
    "months": ["months", "month", "мес", "месяц", "месяца", "месяцев"],
    "min": ["min", "minutes", "minute", "мин", "минут", "минуты"],
    "h": ["h", "hours", "hour", "hrs", "ч", "час", "часа", "часов", "operating hours", "часов наработки",
          "ч наработки", "моточасов"],
    "kW": ["kw", "квт"],
    "°C": ["°c", "c", "degc", "°с", "градусов", "℃"],
    "mm": ["mm", "мм"],
    "years": ["years", "year", "лет", "года", "год"],
    "rpm": ["rpm", "об/мин", "r/min", "rev/min", "1/min", "min-1", "мин-1"],
    "Hz": ["hz", "гц"],
    "m³/h": ["m³/h", "m3/h", "м³/ч", "м3/ч", "m^3/h", "m3/hr", "m³/hr"],
    "m": ["m", "м"],
    "kg": ["kg", "кг"],
}
_UNIT = {s.replace(" ", ""): canon for canon, syn in UNIT_SYNONYMS.items() for s in syn}


def canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = unit.strip().lower().replace(" ", "").rstrip(".")
    return _UNIT.get(u, u)


def parse_answer(content: str) -> dict | None:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content or "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def grade(question: dict, answer: dict | None) -> tuple[bool, str]:
    """(correct, reason). The answer is right when the number and the unit match, or both say "not stated"."""
    if answer is None:
        return False, "invalid answer"
    said_none = bool(answer.get("not_stated")) or answer.get("value") in (None, "")
    if question["value"] is None:
        return (True, "ok") if said_none else (False, "invented a value")
    if said_none:
        return False, "said not stated"
    got = number_value(str(answer["value"]).strip())
    if got is None or abs(got - number_value(question["value"])) > 1e-9:
        return False, "wrong value"
    if question["unit"] is not None and canonical_unit(answer.get("unit")) != question["unit"]:
        return False, "wrong unit"
    return True, "ok"


def answer_on_page(question: dict, page_text: str) -> bool | None:
    """Did the reader keep the right number on the page? None for "not stated" questions."""
    if question["value"] is None:
        return None
    want = number_value(question["value"])
    return any(number_value(n) == want for n in numbers(page_text))


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------
def run_qa(reader: str, pages: dict[tuple[str, int], str], questions: list[dict], client,
           concurrency: int = 4) -> list[dict]:
    """Ask every question with the reader's text of the page that holds the answer."""

    def one(q: dict) -> dict:
        text = pages.get((q["file"], q["page"]), "")
        r = client.chat(messages(q["question"], text), response_format=RESPONSE_FORMAT,
                        meta={"task": "qa", "reader": reader, "question": q["id"]})
        answer = parse_answer(r.content) if r.ok else None
        correct, reason = grade(q, answer)
        kept = answer_on_page(q, text)
        return {"reader": reader, "id": q["id"], "slice": q["slice"], "question": q["question"],
                "expected": q["value"], "expected_unit": q["unit"], "correct": correct, "reason": reason,
                "number_on_page": kept, "value": (answer or {}).get("value"), "unit": (answer or {}).get("unit"),
                "clause": (answer or {}).get("clause"), "not_stated": (answer or {}).get("not_stated"),
                "latency_s": r.latency_s, "error": r.error}

    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        return list(pool.map(one, questions))


def attribute(results: list[dict], ceiling: str = "digital") -> list[dict]:
    """Where was a wrong answer lost? Replace the reader's text by the ground truth and ask again (lecture 2).

    The ceiling run is that replacement: the same model with the text layer of the born-digital original.
    If the model answered right there and wrong on the reader's text, the reader lost the answer ("reading").
    If it was wrong there too, the model is the failing link ("model").
    """
    ok = {r["id"]: r["correct"] for r in results if r["reader"] == ceiling}
    for r in results:
        r["lost_at"] = "" if r["correct"] else ("reading" if ok.get(r["id"]) else "model")
    return results


# ---------------------------------------------------------------------------------------------
# Statistics (lecture 2): interval, paired comparison
# ---------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of a share k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def paired(a: list[bool], b: list[bool], n_boot: int = 5000, seed: int = 0) -> dict:
    """A versus B on the same questions: difference of accuracy, 95% bootstrap interval, verdict."""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    verdict = "better" if lo > 0 else "worse" if hi < 0 else "not different"
    return {"diff": float(d.mean()), "low": float(lo), "high": float(hi), "verdict": verdict,
            "a_only": int((d > 0).sum()), "b_only": int((d < 0).sum())}
