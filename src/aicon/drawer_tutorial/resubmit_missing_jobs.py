#!/usr/bin/env python3
"""Resubmit incomplete batch jobs for a given scenario.

This script scans a log directory for files that do not contain the marker
phrase (default: "run 9") and submits an sbatch job for each missing index.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_SCRIPT_BY_SCENARIO = {
    "normal": Path("Single_Sweep/Job_scripts/Normal_single_sweep.sh"),
    "noise": Path("Single_Sweep/Job_scripts/Noise_single_sweep.sh"),
    "disturbance": Path("Single_Sweep/Job_scripts/Disturbance_single_sweep.sh"),
}


def extract_index_from_filename(name: str) -> Optional[int]:
    matches = re.findall(r"\d+", name)
    if not matches:
        return None
    return int(matches[-1])


def extract_index_from_log(path: Path) -> Optional[int]:
    """Extract the job index from a log file.

    Looks for a line like:
        Running job with ID: 42
    """
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None

    match = re.search(r"Running job with ID:\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None

def find_incomplete_logs(log_dir: Path, phrase: str, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted(p for p in log_dir.rglob("*") if p.is_file())
    else:
        files = sorted(p for p in log_dir.iterdir() if p.is_file())

    incomplete: List[Path] = []
    for path in files:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue

        if phrase not in text:
            incomplete.append(path)
    return incomplete


def format_missing_indexes(files: Iterable[Path]) -> List[str]:
    results: List[str] = []
    for path in files:
        index = extract_index_from_log(path)
        if index is not None:
            results.append(str(index))
        else:
            results.append(str(path))
    return results


def resolve_script(scenario: str, script_override: Optional[Path]) -> Path:
    if script_override is not None:
        return script_override

    scenario_key = scenario.lower()
    if scenario_key == "dsturbance":
        scenario_key = "disturbance"

    script = DEFAULT_SCRIPT_BY_SCENARIO.get(scenario_key)
    if script is None:
        raise ValueError(
            f"Unsupported scenario '{scenario}'. Supported values: "
            f"{', '.join(sorted(DEFAULT_SCRIPT_BY_SCENARIO))}."
        )
    return script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find logs missing the completion marker and resubmit them via sbatch."
        )
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path to the directory containing log files.",
    )
    parser.add_argument(
        "scenario",
        choices=["disturbance", "noise", "normal"],
        help="Scenario type for job submission.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Optional explicit sbatch script path to submit instead of the default scenario mapping.",
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit sbatch commands. Without this flag, commands are shown but not executed.",
    )
    return parser.parse_args()


def submit_job(script: Path, index: str) -> subprocess.CompletedProcess[str]:
    command = ["sbatch", str(script), index]
    return subprocess.run(command, check=True, capture_output=True, text=True)


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: {log_dir} is not a directory or does not exist.")
        return 2

    script_path = resolve_script(args.scenario, args.script)
    if not script_path.exists():
        print(f"Error: sbatch script not found: {script_path}")
        return 2

    missing_files = find_incomplete_logs(log_dir, args.phrase, args.recursive)
    missing_indexes = format_missing_indexes(missing_files)

    if not missing_indexes:
        print("All log files contain the phrase 'run 9'. No resubmission necessary.")
        return 0

    print("Missing completion marker in the following log files:")
    for file_path, index in zip(missing_files, missing_indexes):
        print(f"- {file_path} -> {index}")
    print()

    print(f"Using sbatch script: {script_path}")
    print("Job indexes to submit:")
    print(", ".join(missing_indexes))
    print()

    if not args.execute:
        print("Dry run mode: sbatch commands are shown but not executed.")
        print("Re-run with --execute to submit the missing jobs.")
        for index in missing_indexes:
            print(f"sbatch {script_path} {index}")
        return 0

    for index in missing_indexes:
        print(f"Submitting job for index {index}...")
        try:
            result = submit_job(script_path, index)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to submit index {index}: {exc}")
            print(exc.stdout)
            print(exc.stderr)
            return 1
        print(result.stdout.strip())

    print("All missing jobs submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
