"""Small read-only tools. All data in this practice is synthetic."""
import json
from datetime import date
from pathlib import Path

import pandas as pd
from langchain.tools import tool

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'


def select_batches(line_id, start, end):
    if line_id not in ('L-1', 'L-2', 'L-3'):
        raise ValueError('Unknown line. Choose L-1, L-2 or L-3.')
    date.fromisoformat(start)
    date.fromisoformat(end)
    if start > end:
        raise ValueError('Start must not be after end.')
    rows = pd.read_csv(DATA / 'batches.csv')
    return rows[(rows.line_id == line_id) & rows.date.between(start, end)]


def totals(rows):
    produced = int(rows.produced.sum())
    defects = int(rows.defects.sum())
    return dict(produced=produced, defects=defects,
                defect_rate_pct=round(100 * defects / produced, 4) if produced else None)


@tool
def get_quality_summary(line_id: str, start: str, end: str) -> dict:
    """Get weighted defect rates by product for a line and inclusive ISO date range.

    Use counts, not an average of batch percentages. L-3 has no recorded batches.
    """
    try:
        rows = select_batches(line_id, start, end)
    except ValueError as exc:
        return dict(status='invalid_input', error=str(exc))
    products = [dict(product_id=p, **totals(g)) for p, g in rows.groupby('product_id')]
    return dict(status='ok' if len(rows) else 'no_data', line_id=line_id, start=start, end=end,
                **totals(rows), products=products, batch_ids=rows.batch_id.tolist())


@tool
def compare_periods(line_id: str, before_start: str, before_end: str,
                    after_start: str, after_end: str) -> dict:
    """Compare weighted defect rates for the same products in two non-overlapping periods.

    Return percentage point changes. A change is not evidence of its cause.
    """
    try:
        before = select_batches(line_id, before_start, before_end)
        after = select_batches(line_id, after_start, after_end)
        if before_end >= after_start:
            raise ValueError('The before period must end before the after period starts.')
    except ValueError as exc:
        return dict(status='invalid_input', error=str(exc))
    matched = sorted(set(before.product_id) & set(after.product_id))
    rows = []
    for p in matched:
        a = totals(before[before.product_id == p])
        b = totals(after[after.product_id == p])
        rows.append(dict(product_id=p, before=a, after=b,
                         change_pp=round(b['defect_rate_pct'] - a['defect_rate_pct'], 4)))
    return dict(status='ok' if rows else 'no_data', line_id=line_id,
                before=[before_start, before_end], after=[after_start, after_end],
                products=rows, unmatched_products=sorted(set(before.product_id) ^ set(after.product_id)),
                batch_ids=before.batch_id.tolist() + after.batch_id.tolist(),
                limit='Association does not establish a cause.')


@tool
def get_maintenance_events(line_id: str, start: str, end: str) -> dict:
    """Read maintenance and changeover events for a line and inclusive ISO date range.

    Events can suggest follow-up checks. They cannot establish a cause of defects.
    """
    try:
        select_batches(line_id, start, end)
    except ValueError as exc:
        return dict(status='invalid_input', error=str(exc))
    events = pd.read_csv(DATA / 'events.csv')
    rows = events[(events.line_id == line_id) & events.date.between(start, end)]
    return dict(status='ok', events=rows.to_dict('records'), line_id=line_id, start=start, end=end,
                limit='An empty result means no events in this dataset.')


TOOLS = [get_quality_summary, compare_periods, get_maintenance_events]
BY_NAME = {t.name: t for t in TOOLS}


def pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=True)
