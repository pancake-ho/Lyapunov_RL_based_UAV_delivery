"""Locate the shared proposed implementation from the existing NDTVS folder."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
PROPOSED = HERE.parents[1] / "proposed"
sys.path.insert(0, str(PROPOSED))
