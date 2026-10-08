"""Audit one episode, plot training diagnostics, or compare paired evaluations."""
from __future__ import annotations

import argparse
from pathlib import Path
from baseline.NDTVS.analysis.audit import audit, audit_run
from baseline.NDTVS.plot.learning_diagnostics import progress
from baseline.NDTVS.analysis.compare import comparable_config, compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit", "audit-run", "progress"):
        sub.add_parser(name).add_argument("directory", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("directories", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "audit":
        return audit(a.directory)
    if a.command == "progress":
        return progress(a.directory)
    if a.command == "audit-run":
        return audit_run(a.directory)
    return compare(a.directories, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
