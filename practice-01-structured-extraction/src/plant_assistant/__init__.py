"""Plant Engineer Assistant: demo project of the ML in Industry 2026 course."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
REFERENCE_DIR = PROJECT_ROOT / "reference"
FIXTURES_DIR = PROJECT_ROOT / "fixtures"

__all__ = ["PROJECT_ROOT", "DATA_DIR", "RUNS_DIR", "REFERENCE_DIR", "FIXTURES_DIR"]
