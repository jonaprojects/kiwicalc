"""Fail when a coverage.py JSON report is below a branch-coverage target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_branch_rate(report_path: Path) -> float:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    totals = report["totals"]
    branches = totals["num_branches"]
    if branches == 0:
        return 100.0
    return 100 * totals["covered_branches"] / branches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("minimum", type=float)
    args = parser.parse_args()

    rate = read_branch_rate(args.report)
    print(f"Branch coverage: {rate:.4f}% (required: {args.minimum:.2f}%)")
    return 0 if rate >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
