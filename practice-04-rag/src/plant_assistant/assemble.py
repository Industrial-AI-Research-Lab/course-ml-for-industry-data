"""Context assembly: which chunks go into the prompt, in what order, within a budget.

Base: the top-k found chunks. Evidence pack: for the best hits, also the chunks they refer to and the chunks
that refer to them (references in both directions: clause 6.2 points to 4.3.3, not the other way round),
then the whole parent section of each of them if the budget allows. Evidence for the answer first, context
"for understanding" later. Long context is not free: the budget is part of the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .search import Hit
from .store import Chunk


def tokens(text: str) -> int:
    """A rough token count for Russian and English text with Qwen-like tokenizers."""
    return len(text) // 3 + 1


def top_context(hits: list[Hit], k: int = 5) -> list[str]:
    return [h.chunk_id for h in hits[:k]]


def evidence_pack(hits: list[Hit], chunks_by_id: dict[str, Chunk], in_view: set[str],
                  allowed: Callable[[Chunk], bool] | None = None, k: int = 5, budget: int = 2500) -> list[str]:
    """Chunk ids of the pack, in order of priority, cut at `budget` tokens (the first hit always stays):

    1. evidence: every hit of the top-k with the chunks it refers to and the chunks that refer to it; one more
       step back along incoming references ("Table 3" <- 4.3.3 <- 6.2: the action refers to the rule);
    2. the sections that references led to, whole (6.2 brings section 6: 6.1 to 6.4);
    3. the sections of the hits, whole (a hit that is a heading brings its own clauses).
    """
    children: dict[str, list[str]] = {}
    for c in chunks_by_id.values():
        if c.parent_id and c.chunk_id in in_view:
            children.setdefault(c.parent_id, []).append(c.chunk_id)

    def ok(cid: str) -> bool:
        return cid in chunks_by_id and cid in in_view and (allowed is None or allowed(chunks_by_id[cid]))

    def section(cid: str) -> list[str]:
        c = chunks_by_id[cid]
        head = cid if c.chunk_type == "section" else c.parent_id
        if not head or head not in chunks_by_id:
            return []
        return [x for x in [head] + children.get(head, []) if ok(x)]

    hit_ids = [h.chunk_id for h in hits[:k]]
    evidence: list[str] = []
    linked: list[str] = []
    for cid in hit_ids:
        c = chunks_by_id[cid]
        evidence.append(cid)
        for r in c.references + c.referenced_by:
            if ok(r):
                evidence.append(r)
                linked.append(r)
                linked += [x for x in chunks_by_id[r].referenced_by if ok(x)]
                evidence += [x for x in chunks_by_id[r].referenced_by if ok(x)]
    tier2 = [x for cid in linked for x in section(cid)]
    tier3 = [x for cid in hit_ids for x in section(cid)]
    order: list[str] = []
    used = 0
    for cid in evidence + tier2 + tier3:
        if cid in order:
            continue
        cost = tokens(chunks_by_id[cid].text)
        if order and used + cost > budget:
            continue
        order.append(cid)
        used += cost
    return order


# ---------------------------------------------------------------------------------------------
# Rendering the context for the prompt
# ---------------------------------------------------------------------------------------------
@dataclass
class Source:
    label: str  # S1, S2, ...
    chunk_id: str
    header: str


def header(c: Chunk, show_dates: bool) -> str:
    """The citation fields of a chunk, as the model must copy them."""
    parts = [f"doc_id={c.doc_id}", f"revision={c.revision if c.revision is not None else '-'}"]
    if c.amendment:
        parts.append(f"amendment={c.amendment}")
    if c.chunk_type == "clause" and c.clause:
        parts.append(f"clause={c.clause}")
    if c.table_no:
        parts.append(f"table={c.table_no}")
    if c.row_key:
        parts.append(f"row={c.row_key}")
    parts.append(f"page={c.page}")
    if show_dates:
        parts.append(f"in force from {c.effective_from} to {c.effective_to or 'now'}")
    return "; ".join(parts)


def render(chunk_ids: list[str], chunks_by_id: dict[str, Chunk], show_dates: bool,
           ids_as_labels: bool = False) -> tuple[str, list[Source]]:
    """The context for the prompt. Labels S1, S2, ... (simple answer, contract v1) or the chunk ids themselves
    (contract v2: the model cites an id, the schema allows only these ids)."""
    sources, blocks = [], []
    for i, cid in enumerate(chunk_ids, start=1):
        c = chunks_by_id[cid]
        s = Source(cid if ids_as_labels else f"S{i}", cid, header(c, show_dates))
        sources.append(s)
        blocks.append(f"[{s.label}] {s.header}\n{c.text.strip()}")
    return "\n\n".join(blocks), sources
