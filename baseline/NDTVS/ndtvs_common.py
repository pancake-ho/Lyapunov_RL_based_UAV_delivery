"""Compatibility entry point; implementation lives in the functional folder."""
import sys
from pathlib import Path

# Existing direct-file commands resolve the same canonical package as -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.NDTVS.api import *

if __name__ == "__main__":
    raise SystemExit(main())
