"""The audit log of an answer: what was asked, on which date, from which index, which chunks of which
revisions were found and shown, what the model answered and what the checks said.

Half a year later, "why did the system say 5,0?" must have an answer: this record. Topic 6 keeps these
records and replays them against a new index version; topic 7 adds who asked and what they may see.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def audit_record(rec: dict, index_version: str, chunks_by_id: dict) -> dict:
    """A record of one answer from a pipeline result (pipeline.run_case)."""

    def ref(cid: str) -> dict:
        c = chunks_by_id[cid]
        return {"chunk_id": cid, "doc_id": c.doc_id, "revision": c.revision, "amendment": c.amendment,
                "clause": c.clause, "table_no": c.table_no, "row_key": c.row_key, "page": c.page,
                "effective_from": c.effective_from, "effective_to": c.effective_to}

    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": rec["case_id"], "question": rec["query"], "asked_on": rec["asked_on"], "as_of": rec["as_of"],
        "config": rec["config"], "index_version": index_version,
        "filters": rec["scope"], "retrieved": [ref(c) for c in rec["retrieved"][:5]],
        "context": [ref(c) for c in rec["context"]], "model": rec.get("model"),
        "request_hash": rec.get("request_hash"), "answer": rec.get("answer"), "checks": rec.get("checks"),
    }


def write_audit(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
