"""Code checks for the free-text field defect_summary. The bottom of the grader pyramid.

The schema describes defect_summary as "Short English summary of the defect, at most 15 words.
null if there was no defect." Three of these four requirements need no model at all.
Only "does the summary say something that the entry does not?" needs a judge.
"""

from __future__ import annotations

import re
from typing import Any

MAX_WORDS = 15
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def check_summary(summary: str | None, failure_mode: str | None) -> list[str]:
    """Problems of one summary that code can find. Empty list: the code checks pass."""
    problems = []
    if summary is None:
        return problems
    text = summary.strip()
    if text.lower() in {"null", "none", "n/a", ""}:
        problems.append("placeholder text instead of null")
    if CYRILLIC.search(text):
        problems.append("not in English")
    if len(text.split()) > MAX_WORDS:
        problems.append(f"longer than {MAX_WORDS} words")
    if failure_mode == "none":
        problems.append("a summary although no defect was found (should be null)")
    return problems


def check_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """rows: dicts with summary and failure_mode (ground truth). Adds 'code_problems'."""
    return [{**r, "code_problems": check_summary(r["summary"], r.get("failure_mode"))} for r in rows]
