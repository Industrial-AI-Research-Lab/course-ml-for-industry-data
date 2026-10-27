"""Plant Engineer Assistant: demo project of the ML in Industry 2026 course (practice 4: retrieval)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
GT_DIR = DATA_DIR / "ground_truth"
REPLAY_DIR = DATA_DIR / "replay"
INPUTS_DIR = PROJECT_ROOT / "inputs"
RUNS_DIR = PROJECT_ROOT / "runs"
STORE_DIR = PROJECT_ROOT / "store"
REFERENCE_DIR = PROJECT_ROOT / "reference"

__all__ = ["PROJECT_ROOT", "DATA_DIR", "DOCS_DIR", "GT_DIR", "REPLAY_DIR", "INPUTS_DIR", "RUNS_DIR", "STORE_DIR",
           "REFERENCE_DIR"]
