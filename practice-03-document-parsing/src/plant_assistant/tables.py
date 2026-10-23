"""Tables: find the rows a reader recovered, and write one table in three representations.

A table from the ground truth is a list of cells (row, col, rowspan, colspan, text), as in
data/ground_truth/tables.json.
"""

from __future__ import annotations

import html
import json
import re

from .metrics import LOOKALIKES, number_value, numbers, plain

# ---------------------------------------------------------------------------------------------
# Rows in a reader output
# ---------------------------------------------------------------------------------------------
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)


def rows_of(text: str) -> list[list[str]]:
    """Every row-like thing in an output: HTML <tr>, Markdown |...| lines, and plain lines split by spaces."""
    rows = []
    for tr in _TR.findall(text or ""):
        rows.append([plain(c) for c in _CELL.findall(tr)])
    no_html = re.sub(r"<table.*?</table>", " ", text or "", flags=re.S | re.I)
    for line in no_html.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("|"):
            rows.append([plain(c) for c in s.strip("|").split("|")])
        else:
            rows.append(plain(s).split(" "))
    return rows


def row_recovered(rows: list[list[str]], key: str, values: list[str]) -> bool:
    """Is there a row with the key cell (e.g. class 'II') and the expected numbers in this order?"""
    want = [number_value(v) for v in values]
    key = key.translate(LOOKALIKES)
    for row in rows:
        if key not in (c.strip().translate(LOOKALIKES) for c in row):  # Р-101 (Cyrillic) is still the row of P-101
            continue
        got = [number_value(n) for c in row for n in numbers(c)]
        if any(got[i:i + len(want)] == want for i in range(len(got) - len(want) + 1)):
            return True
    return False


def table_score(text: str, gt_rows: list[list]) -> dict:
    """Share of data rows of one ground-truth table that a reader recovered as a row."""
    rows = rows_of(text)
    ok = [key for key, values in gt_rows if row_recovered(rows, key, values)]
    return {"rows": len(gt_rows), "recovered": len(ok), "share": len(ok) / len(gt_rows),
            "lost": [key for key, _ in gt_rows if key not in ok]}


# ---------------------------------------------------------------------------------------------
# Representations of one ground-truth table
# ---------------------------------------------------------------------------------------------
def _grid(cells, fill_spans: bool) -> list[list[str]]:
    n_rows = max(r + rs for r, _, rs, _, _ in cells)
    n_cols = max(c + cs for _, c, _, cs, _ in cells)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for r, c, rs, cs, text in cells:
        for i in range(r, r + rs):
            for j in range(c, c + cs):
                if fill_spans or (i == r and j == c):
                    grid[i][j] = text
    return grid


def to_html(cells) -> str:
    """HTML keeps merged cells (rowspan, colspan). Two to three times more tokens than Markdown."""
    by_row: dict[int, list] = {}
    for cell in sorted(cells, key=lambda x: (x[0], x[1])):
        by_row.setdefault(cell[0], []).append(cell)
    lines = ["<table>"]
    for r in sorted(by_row):
        tds = []
        for _, _, rs, cs, text in by_row[r]:
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            tds.append(f"<td{attrs}>{html.escape(text)}</td>")
        lines.append("<tr>" + "".join(tds) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def to_markdown_as_parsed(cells, header_rows: int) -> str:
    """Markdown as many converters write it: a merged cell is written once, the rest of its row shifts left.

    Markdown has no merged cells. The row of class II loses its "group" cell, so its values move one
    column to the left, under the wrong header.
    """
    n_cols = max(c + cs for _, c, _, cs, _ in cells)
    by_row: dict[int, list[str]] = {}
    for r, c, rs, cs, text in sorted(cells, key=lambda x: (x[0], x[1])):
        by_row.setdefault(r, []).append(text)
    lines = []
    for r in sorted(by_row):
        row = by_row[r] + [""] * (n_cols - len(by_row[r]))
        lines.append("| " + " | ".join(row) + " |")
        if r == 0:
            lines.append("|" + "---|" * n_cols)
    return "\n".join(lines)


def to_json_records(cells, header_rows: int) -> str:
    """One JSON object per data row; merged cells are copied into every row they cover."""
    grid = _grid(cells, fill_spans=True)
    names = []
    for j in range(len(grid[0])):
        parts = []
        for i in range(header_rows):
            if grid[i][j] and grid[i][j] not in parts:
                parts.append(grid[i][j])
        names.append(" / ".join(parts))
    records = [dict(zip(names, row)) for row in grid[header_rows:]]
    return json.dumps(records, ensure_ascii=False, indent=1)
