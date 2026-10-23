"""A VLM on a drawing and on a photo: tags from a P&ID fragment, fields from a nameplate.

What is realistic in 2026 without your own labelled data: the tags and the title block of a drawing,
and the fields of a nameplate. What is not: the topology (what is connected to what).
A dense sheet in small print, and worse its scanned paper copy, is where a VLM misses and invents tags:
tiles help to read small print and cost a call per tile.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from PIL import Image
from pydantic import BaseModel, Field

from .images import vision_message
from .metrics import normalize_tag, tag_scores
from .qa import parse_answer

# ---------------------------------------------------------------------------------------------
# P&ID tags
# ---------------------------------------------------------------------------------------------
PID_PROMPT = """This is a fragment of a P&ID (piping and instrumentation diagram).
List every equipment tag and instrument tag written on it: vessels, pumps, valves, instruments.
An instrument bubble shows letters on top and a number below: write it as LETTERS-NUMBER, for example PT-101.
Do not list line numbers (like 100-PL-001) and do not add tags that are not on the drawing.
Also read the title block: drawing number, revision and date."""

PID_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "drawing_number": {"type": ["string", "null"]},
        "revision": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
    },
    "required": ["tags", "drawing_number", "revision", "date"],
    "additionalProperties": False,
}


def pid_request(image) -> tuple[list[dict], dict]:
    return [vision_message(PID_PROMPT, image)], {
        "type": "json_schema", "json_schema": {"name": "pid_tags", "strict": True, "schema": PID_SCHEMA}}


def tiles(image, rows: int = 2, cols: int = 2, overlap: float = 0.15) -> list[Image.Image]:
    """Cut a drawing into rows x cols parts that overlap by `overlap` of a part, so no tag is cut in two."""
    image = Image.open(image) if isinstance(image, (str, Path)) else image
    w, h = image.width / cols, image.height / rows
    dx, dy = w * overlap, h * overlap
    return [image.crop((int(max(c * w - dx, 0)), int(max(r * h - dy, 0)),
                        int(min((c + 1) * w + dx, image.width)), int(min((r + 1) * h + dy, image.height))))
            for r in range(rows) for c in range(cols)]


def merge_tile_answers(answers: list[dict | None]) -> dict:
    """Tags: the union over the tiles (the overlap gives duplicates, normalization removes them).
    Title block: a drawing keeps it in the bottom right corner, so it is read from the last tile only.
    The other tiles also "find" a drawing number: the off-sheet connectors name other drawings."""
    tags, seen = [], set()
    for a in answers:
        for t in (a or {}).get("tags") or []:
            if normalize_tag(t) not in seen:
                seen.add(normalize_tag(t))
                tags.append(t)
    corner = answers[-1] or {}
    return {"tags": tags, **{k: corner.get(k) for k in ("drawing_number", "revision", "date")}}


def score_pid(answer: dict | None, truth: dict) -> dict:
    answer = answer or {"tags": []}
    s = tag_scores(truth["tags"], answer.get("tags") or [])
    s["title_block"] = sum([
        normalize_tag(str(answer.get("drawing_number") or "")) == normalize_tag(truth["drawing"]),
        re.sub(r"\D", "", str(answer.get("revision") or "")) == truth["revision"],  # "Ред. 3" -> "3"
        str(answer.get("date") or "").strip() == truth["date"],
    ])
    return s


# ---------------------------------------------------------------------------------------------
# Nameplate: the Pydantic schema from practice 1, applied to a photo
# ---------------------------------------------------------------------------------------------
class Nameplate(BaseModel):
    """Fields of a pump nameplate. null when the field is not readable on the photo."""

    model: Optional[str] = Field(description="pump model, as written")
    serial_number: Optional[str] = Field(description="serial number (Зав. №), as written")
    year: Optional[int] = Field(description="year of manufacture")
    flow_m3h: Optional[float] = Field(description="flow Q in m³/h")
    head_m: Optional[float] = Field(description="head H in m")
    speed_rpm: Optional[float] = Field(description="speed n in rpm (об/мин)")
    power_kw: Optional[float] = Field(description="motor power N in kW")
    mass_kg: Optional[float] = Field(description="mass in kg")


NAMEPLATE_PROMPT = """This is a photo of a pump nameplate. Read the fields and return JSON.
Field definitions:
""" + "\n".join(f"- {name}: {f.description}" for name, f in Nameplate.model_fields.items()) + """
Write numbers as numbers. If a field is not readable, return null: do not guess."""


def nameplate_request(image) -> tuple[list[dict], dict]:
    schema = Nameplate.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(Nameplate.model_fields)
    return [vision_message(NAMEPLATE_PROMPT, image)], {
        "type": "json_schema", "json_schema": {"name": "nameplate", "strict": True, "schema": schema}}


def score_nameplate(answer: dict | None, truth: dict) -> dict:
    """Field by field: exact match after trimming; numbers compared as numbers."""
    answer = answer or {}
    fields = {}
    for name in Nameplate.model_fields:
        want, got = truth[name], answer.get(name)
        if isinstance(want, (int, float)):
            try:
                ok = got is not None and abs(float(got) - float(want)) < 1e-9
            except (TypeError, ValueError):
                ok = False
        else:
            ok = got is not None and str(got).strip().upper().translate(_LAT) == str(want).upper().translate(_LAT)
        fields[name] = {"expected": want, "got": got, "ok": ok}
    return {"accuracy": sum(f["ok"] for f in fields.values()) / len(fields), "fields": fields}


# The model number НК 65/35-125 is Cyrillic on the plate; a model may answer in Latin look-alikes.
_LAT = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")


def run_image_task(client, task: str, image, truth: dict) -> dict:
    msgs, fmt = pid_request(image) if task == "pid" else nameplate_request(image)
    r = client.chat(msgs, response_format=fmt, meta={"task": task})
    answer = parse_answer(r.content) if r.ok else None
    score = score_pid(answer, truth) if task == "pid" else score_nameplate(answer, truth)
    return {"profile": client.profile_name, "task": task, "answer": answer, "score": score,
            "latency_s": r.latency_s, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "cost_usd": r.cost_usd, "error": r.error}


def run_pid_tiles(client, image, truth: dict, rows: int = 2, cols: int = 2, overlap: float = 0.15) -> dict:
    """The same task on tiles: one call per tile, then merge. Every tile is a separate image, so it costs
    about rows x cols times the tokens of one call."""
    parts = tiles(image, rows, cols, overlap)
    responses = []
    for k, part in enumerate(parts):
        msgs, fmt = pid_request(part)
        responses.append(client.chat(msgs, response_format=fmt, meta={"task": "pid_tiles", "tile": k}))
    answer = merge_tile_answers([parse_answer(r.content) if r.ok else None for r in responses])
    errors = [r.error for r in responses if r.error]
    return {"profile": client.profile_name, "task": "pid_tiles", "tiles": len(parts), "answer": answer,
            "score": score_pid(answer, truth),
            "latency_s": round(sum(r.latency_s or 0 for r in responses), 2),
            "prompt_tokens": sum(r.prompt_tokens or 0 for r in responses),
            "completion_tokens": sum(r.completion_tokens or 0 for r in responses),
            "cost_usd": sum(r.cost_usd or 0 for r in responses), "error": "; ".join(errors) or None}


def answer_json(result: dict) -> str:
    return json.dumps(result["answer"], ensure_ascii=False, indent=1)
