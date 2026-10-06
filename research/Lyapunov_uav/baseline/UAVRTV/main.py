"""Compatibility entry point; settings are in baseline/UAVRTV/config.py."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.UAVRTV.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
