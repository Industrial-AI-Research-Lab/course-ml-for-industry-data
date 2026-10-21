"""Run readers on the scanned documents and save their outputs to data/parsed/<reader>.jsonl.

    uv run python scripts/run_readers.py                         # all readers that can run here
    uv run python scripts/run_readers.py --readers pymupdf tesseract
    uv run python scripts/run_readers.py --readers vlm:course --files R-12_rev3_scan.pdf

Readers: pymupdf, tesseract, easyocr, docling, vlm:<profile> (profiles from models.toml: local, course, ...).
vlm:dots-ocr is a document VLM with its own layout prompt: its outputs also keep the boxes of the blocks.
Calls of VLM readers are logged to data/replay/transcribe_<profile>.jsonl: the notebook replays them
when the model is not available.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_assistant import DOCS_DIR, REPLAY_DIR  # noqa: E402
from plant_assistant.config import get_profile, load_env  # noqa: E402
from plant_assistant.docs import SCANNED_FILES  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient  # noqa: E402
from plant_assistant.readers import LOCAL_READERS, read_dots_ocr, read_vlm, save_outputs  # noqa: E402

DEFAULT = ["pymupdf", "tesseract", "easyocr", "docling", "vlm:dots-ocr", "vlm:local", "vlm:course", "vlm:gpt-6.1-sol"]


def run(reader: str, files: list[str]) -> None:
    t0 = time.perf_counter()
    records = []
    if reader.startswith("vlm:"):
        profile = get_profile(reader.split(":", 1)[1])
        log = REPLAY_DIR / f"transcribe_{profile.name}.jsonl"
        log.unlink(missing_ok=True)
        client = LLMClient(profile, logger=CallLogger(log))
        read = read_dots_ocr if profile.name == "dots-ocr" else read_vlm  # a document VLM has its own prompt
        for f in files:
            records += read(DOCS_DIR / f, client, reader_name=reader, concurrency=profile.concurrency)
    else:
        for f in files:
            records += LOCAL_READERS[reader](DOCS_DIR / f)
    path = save_outputs(reader, records)
    cost = sum(r.get("cost_usd") or 0 for r in records)
    print(f"{reader:20s} {len(records):3d} pages  {time.perf_counter() - t0:7.1f} s  {cost:.3f} USD  -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", nargs="*", default=DEFAULT)
    ap.add_argument("--files", nargs="*", default=SCANNED_FILES)
    args = ap.parse_args()
    load_env()
    for reader in args.readers:
        try:
            run(reader, args.files)
        except Exception as exc:  # one reader that cannot run here must not stop the others
            print(f"{reader:20s} SKIPPED: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
