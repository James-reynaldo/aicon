#!/usr/bin/env python3
"""Check log files for successful job completion markers.

This script scans a directory of logs and reports which job logs do not
contain the phrase "run 9".
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional


def extract_index_from_filename(name: str) -> Optional[int]:
    """Extract a numeric index from a file name, if present."""
    matches = re.findall(r"\d+", name)
    if not matches:
        return None
    return int(matches[-1])


def find_incomplete_logs(log_dir: Path, phrase: str, recursive: bool) -> List[str]:
    """Return file names for logs that do not contain the phrase."""
    if recursive:
        files = sorted(p for p in log_dir.rglob("*") if p.is_file())
    else:
        files = sorted(p for p in log_dir.iterdir() if p.is_file())

    incomplete = []
    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue

        if phrase not in text:
            incomplete.append(str(path))
    return incomplete


def format_missing_indexes(files: Iterable[str]) -> List[str]:
    """Format missing job identifiers from file names."""
    results = []
    for path in files:
        index = extract_index_from_filename(Path(path).name)
        if index is not None:
            results.append(str(index))
        else:
            results.append(path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check log files for the phrase 'run 9' and report incomplete jobs."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path to the directory containing log files.",
    )
    parser.add_argument(
        "--phrase",
        default="run 9",
        help="Phrase to search for in each log file. Default: 'run 9'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for log files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: {log_dir} is not a directory or does not exist.")
        return 2

    incomplete_files = find_incomplete_logs(log_dir, args.phrase, args.recursive)
    missing = format_missing_indexes(incomplete_files)

    if missing:
        print("Missing completion marker in the following log files:")
        for path, idx in zip(incomplete_files, missing):
            print(f"- {path} -> {idx}")
        print()
        print("Job indexes without 'run 9':")
        print(", ".join(missing))
        return 1

    print("All log files contain the phrase 'run 9'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
