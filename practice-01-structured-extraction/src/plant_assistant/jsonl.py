"""Reading JSON Lines files that an editor may have reformatted.

The files are JSONL: one JSON object per line. Editors and "reformat on save" sometimes turn
them into pretty-printed JSON objects, one after another. We read both forms, so a reformatted
file does not break the practice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON Lines at line {exc.lineno}: {exc.msg}") from exc
        yield obj
