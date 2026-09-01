"""Fail when a coverage.py JSON report is below a line-coverage target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_line_rate(report_path: Path) -> float:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    totals = report["totals"]
    statements = totals["num_statements"]
    if statements == 0:
        return 100.0
    return 100 * totals["covered_lines"] / statements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("minimum", type=float)
    args = parser.parse_args()

    rate = read_line_rate(args.report)
    print(f"Line coverage: {rate:.4f}% (required: {args.minimum:.2f}%)")
    return 0 if rate >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
