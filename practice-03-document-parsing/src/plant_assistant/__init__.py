"""Plant Engineer Assistant: demo project of the ML in Industry 2026 course (practice 3: documents)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
GT_DIR = DATA_DIR / "ground_truth"
PARSED_DIR = DATA_DIR / "parsed"
REPLAY_DIR = DATA_DIR / "replay"
RUNS_DIR = PROJECT_ROOT / "runs"
STORE_DIR = PROJECT_ROOT / "store"
REFERENCE_DIR = PROJECT_ROOT / "reference"

__all__ = ["PROJECT_ROOT", "DATA_DIR", "DOCS_DIR", "GT_DIR", "PARSED_DIR", "REPLAY_DIR", "RUNS_DIR", "STORE_DIR",
           "REFERENCE_DIR"]
