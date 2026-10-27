"""The retrieval set: test cases in the style of the practice 2 harness, extended with evidence.

A test case knows where its answer is: `evidence` lists locators ("R-12 r3 cl 4.3.3", "R-12 r3 t 3",
"VL-2 r4 t А.2 row V-12", "PS-101") with a role (answer / support / action). The gold chunk ids are computed
from the locators by code (`resolve`), never marked by hand: change the chunking, rebuild the set.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import DATA_DIR
from .jsonl import read_json_records
from .store import Chunk

QUESTIONS_FILE = DATA_DIR / "questions.jsonl"

# slices in the order of the lecture
SLICES = ["base", "language", "applicability", "point_in_time", "incident", "what_changed", "multi_evidence",
          "cross_doc", "conflict", "table_row", "tag_variant", "not_in_documents"]


@dataclass
class TestCase:
    case_id: str
    slice: str
    language: str  # the language of the wording that goes to search: "ru" (default) or "en" (slice "language")
    question_ru: str
    question_en: str
    asked_on: str  # the day the question is asked
    as_of: str | None  # the date the question is about (None: "what changed", every revision counts)
    expected: dict[str, Any]  # {"value", "unit"} | {"values", "unit"} | {"status"}
    evidence: list[dict[str, Any]] = field(default_factory=list)  # {"role", "any_of", "chunk_ids"}
    source: str = "practice 4"
    note: str = ""

    __test__ = False  # not a pytest test class

    @property
    def query(self) -> str:
        """What goes to search and to the prompt."""
        return self.question_en if self.language == "en" else self.question_ru

    @property
    def status(self) -> str:
        return self.expected.get("status", "answered")

    def gold(self, role: str | None = None) -> list[set[str]]:
        """Evidence items as sets of acceptable chunk ids (one set per item), optionally of one role."""
        return [set(e["chunk_ids"]) for e in self.evidence if role is None or e["role"] == role]


def load_cases(path: str | Path = QUESTIONS_FILE) -> list[TestCase]:
    return [TestCase(**r) for r in read_json_records(path)]


def write_cases(cases: list[TestCase], path: str | Path = QUESTIONS_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------
# Locators -> chunk ids
# ---------------------------------------------------------------------------------------------
LOCATOR_RE = re.compile(r"^(?P<doc>\S+)(?:\s+r(?P<rev>\d+)(?:a(?P<amd>\d+))?)?"
                        r"(?:\s+cl\s+(?P<clause>\S+)|\s+t\s+(?P<table>\S+)(?:\s+row\s+(?P<row>\S+))?)?$")


def parse_locator(loc: str) -> dict[str, Any]:
    m = LOCATOR_RE.match(loc.strip())
    if not m:
        raise ValueError(f"bad locator: {loc!r}")
    d = m.groupdict()
    return {"doc_id": d["doc"], "revision": int(d["rev"]) if d["rev"] else None,
            "amendment": int(d["amd"]) if d["amd"] else None, "clause": d["clause"], "table_no": d["table"],
            "row_key": d["row"]}


def resolve(loc: str, chunks: list[Chunk]) -> list[str]:
    """Chunk ids that hold the place the locator points to. A table row is found both as its row chunk and
    as the page chunk of the table that contains it (whichever the index has)."""
    p = parse_locator(loc)
    out = []
    for c in chunks:
        if c.doc_id != p["doc_id"] or (p["revision"] is not None and c.revision != p["revision"]):
            continue
        if c.amendment != p["amendment"]:
            continue
        if p["clause"]:
            ok = c.clause == p["clause"] and c.chunk_type == "clause"
        elif p["row_key"]:
            ok = c.table_no == p["table_no"] and (
                (c.chunk_type == "table_row" and c.row_key == p["row_key"])
                or (c.chunk_type == "table" and f"|{p['row_key']}|" in c.text))
        elif p["table_no"]:
            ok = c.chunk_type == "table" and c.table_no == p["table_no"]
        else:  # a whole document (a datasheet: one chunk)
            ok = True
        if ok:
            out.append(c.chunk_id)
    if not out:
        raise ValueError(f"locator {loc!r} matches no chunk")
    return out


def counts_by_slice(cases: list[TestCase]) -> dict[str, int]:
    return {s: sum(c.slice == s for c in cases) for s in SLICES}
