"""Business rules: checks that a schema cannot express.

Valid JSON is not correct data. The schema guarantees the shape; these rules catch values
that are well-formed but impossible: an unknown equipment tag, a date in the future,
a quote that is not in the source text.
"""

from __future__ import annotations

import difflib
import re
from datetime import date

from .data import Equipment
from .schema import MaintenanceRecord

EQUIPMENT_ID_PATTERN = re.compile(r"^[A-Z]{1,3}-\d{2,4}[A-Z]?$")
# The log covers 2024-01-01 .. 2026-09-30 (the dataset "today").
EARLIEST_DATE = date(2020, 1, 1)
LOG_TODAY = date(2026, 9, 30)
MAX_DURATION_HOURS = 48.0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def check_business_rules(
    record: MaintenanceRecord,
    source_text: str,
    registry: dict[str, Equipment],
    today: date = LOG_TODAY,
) -> list[str]:
    """Return a list of human-readable problems. Empty list means the record passed."""
    problems: list[str] = []

    eid = record.equipment_id
    if eid is not None:
        if not EQUIPMENT_ID_PATTERN.match(eid):
            problems.append(
                f"equipment_id '{eid}' has a wrong format. Use Latin capital letters, a hyphen and digits, "
                "e.g. 'P-101'."
            )
        elif eid not in registry:
            close = difflib.get_close_matches(eid, list(registry), n=3, cutoff=0.6)
            hint = f" Similar tags in the registry: {', '.join(close)}." if close else ""
            problems.append(f"equipment_id '{eid}' is not in the equipment registry.{hint}")
        else:
            expected_type = registry[eid].equipment_type
            if record.equipment_type.value != expected_type:
                problems.append(
                    f"equipment_type '{record.equipment_type.value}' does not match the registry: "
                    f"{eid} is a {expected_type}."
                )

    if record.work_date is not None:
        if record.work_date > today:
            problems.append(f"work_date {record.work_date.isoformat()} is in the future.")
        elif record.work_date < EARLIEST_DATE:
            problems.append(f"work_date {record.work_date.isoformat()} is too old for this log.")

    if record.duration_hours is not None and not (0 < record.duration_hours <= MAX_DURATION_HOURS):
        problems.append(f"duration_hours {record.duration_hours} is outside the range (0, {MAX_DURATION_HOURS}].")

    source = _normalize(source_text)
    for quote in record.evidence:
        if _normalize(quote) not in source:
            problems.append(f"evidence quote '{quote}' is not an exact quote from the entry.")

    return problems
