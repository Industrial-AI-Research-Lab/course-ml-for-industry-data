"""Three checks of a citation, by code: the cheap grader of lecture 2.

1. exists:    a chunk with this document, revision, amendment, clause / table / row is in the index;
2. in_force:  that chunk is valid on the date the question is about;
3. supports:  the number and the unit of the answer are in the text of the cited chunk (or of a table it
              refers to).

A citation can look right and still not support the answer: "Inspect within 24 hours (R-12, rev. 3,
clause 4.3.4)" passes checks 1 and 2 and fails check 3, because "24 hours" is in the neighbouring clause 4.3.3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .answer import Answer, Citation
from .assemble import Source
from .filters import in_force
from .store import Chunk
from .units import canonical_unit, has_value, unit_in_text


@dataclass
class CitationCheck:
    source: str
    chunk_id: str | None
    exists: bool
    in_force: bool | None  # None: not checked (no date, a "what changed" question)
    supports: bool | None  # None: the answer has no value


def find_chunk(cit: Citation, chunks: list[Chunk]) -> Chunk | None:
    """The chunk the citation fields point to (not the label: the model may copy a wrong field)."""
    for c in chunks:
        if c.doc_id != cit.doc_id or (cit.revision is not None and c.revision != cit.revision):
            continue
        if (c.amendment or None) != (cit.amendment or None):
            continue
        if c.chunk_type == "datasheet":  # one chunk per datasheet: a parameter name in row_key changes nothing
            return c
        if cit.row_key:
            if c.chunk_type == "table_row" and c.row_key == cit.row_key and (not cit.table_no or c.table_no == cit.table_no):
                return c
            if c.chunk_type == "table" and c.table_no == cit.table_no and f"|{cit.row_key}|" in c.text:
                return c
        elif cit.clause:
            if c.chunk_type == "clause" and c.clause == cit.clause:
                return c
        elif cit.table_no:
            if c.chunk_type == "table" and c.table_no == cit.table_no:
                return c
    return None


def support_text(c: Chunk, chunks_by_id: dict[str, Chunk]) -> str:
    """The cited text and the tables it refers to ("по таблице 3")."""
    extra = [chunks_by_id[r].text for r in c.references if r in chunks_by_id and chunks_by_id[r].chunk_type == "table"]
    return "\n".join([c.text, *extra])


def check_answer(ans: Answer, sources: list[Source], chunks: list[Chunk], chunks_by_id: dict[str, Chunk],
                 as_of: str | None) -> dict:
    """Checks of every citation and the verdict for the answer.

    A contract v1 answer is checked by its citation fields; a contract v2 answer by the chunk ids of its claims
    (an id that was not in the context fails "exists"); a simple answer (S0-S5) only names source labels, so its
    citations are the chunks behind the labels. The answer is verified when every citation exists and
    is in force, and at least one cited chunk holds the value with its unit (for an answer with a value)."""
    label_to_chunk = {s.label: s.chunk_id for s in sources}
    cits: list[tuple[str, Chunk | None]] = []
    if sources and all(s.label == s.chunk_id for s in sources):  # contract v2: labels are the chunk ids
        in_context = {s.chunk_id for s in sources}
        for cid in dict.fromkeys(src for cl in ans.claims for src in cl.sources):
            cits.append((cid, chunks_by_id.get(cid) if cid in in_context else None))
    elif ans.citations:
        for cit in ans.citations:
            cits.append((cit.source, find_chunk(cit, chunks)))
    else:
        for label in ans.sources:
            cid = label_to_chunk.get(label.strip().strip("[]"))
            cits.append((label, chunks_by_id.get(cid) if cid else None))
    unit = canonical_unit(ans.unit)
    checks = []
    for label, c in cits:
        if c is None:
            checks.append(CitationCheck(label, None, False, None, None))
            continue
        force = None if as_of is None else in_force(c, as_of)
        sup = None
        if ans.status == "answered" and ans.value not in (None, ""):
            text = support_text(c, chunks_by_id)
            sup = has_value(text, ans.value) and (unit is None or unit_in_text(unit, text))
        checks.append(CitationCheck(label, c.chunk_id, True, force, sup))
    all_exist = bool(checks) and all(ch.exists for ch in checks)
    all_in_force = all(ch.in_force is not False for ch in checks)
    if ans.status == "answered" and ans.value not in (None, ""):
        supported = any(ch.supports for ch in checks)
    elif ans.status == "conflict":
        supported = len({chunks_by_id[ch.chunk_id].doc_id for ch in checks if ch.chunk_id}) >= 2
    else:
        supported = True
    needs_citation = ans.status in ("answered", "conflict")
    verified = (all_exist and all_in_force and supported) if needs_citation else all_in_force and all(
        ch.exists for ch in checks)
    return {"verified": verified, "exists": all_exist, "in_force": all_in_force, "supports": supported,
            "citations": [asdict(ch) for ch in checks]}
