"""Filters before search: the date of the question and the scope of the documents.

A revision is a field with dates, not a flag: a chunk is valid on day d if effective_from <= d < effective_to.
"status = in force" answers only "what is valid today", and gives a wrong answer with a correct-looking
citation to a question about an incident of last year. An amendment ends the validity of the clause it
replaces (corpus.apply_amendments), so the base clause drops out by the same rule.
"""

from __future__ import annotations

from typing import Callable

from .applicability import Scope, in_scope
from .query import QueryFields
from .search import Hit
from .store import Chunk


def in_force(c: Chunk, as_of: str | None) -> bool:
    """Valid on the date. No date (a "what changed" question): every revision counts."""
    if as_of is None:
        return True
    start = c.effective_from or c.effective_date or "0000-00-00"
    return start <= as_of and (c.effective_to is None or as_of < c.effective_to)


def status_in_force(c: Chunk) -> bool:
    """The typical mistake: the flag of the document instead of the dates of the chunk."""
    return c.status == "in force"


def make_filter(fields: QueryFields, scope: Scope | None, use_scope: bool, use_date: bool,
                use_doc_ids: bool = True) -> Callable[[Chunk], bool] | None:
    if not use_scope and not use_date:
        return None

    def allowed(c: Chunk) -> bool:
        if use_date and not in_force(c, fields.as_of):
            return False
        if use_scope and scope is not None and not in_scope(c, scope, use_doc_ids):
            return False
        return True

    return allowed


def post_filter(hits: list[Hit], chunks_by_id: dict[str, Chunk], allowed: Callable[[Chunk], bool]) -> list[Hit]:
    """Filter AFTER search (the typical mistake): what is left of the top-k may be nothing."""
    return [h for h in hits if allowed(chunks_by_id[h.chunk_id])]
