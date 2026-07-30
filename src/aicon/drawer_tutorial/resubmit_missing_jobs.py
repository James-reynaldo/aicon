#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

DEFAULT_SCRIPT_BY_SCENARIO = {
    "normal": Path("Single_Sweep/Job_scripts/Normal_single_sweep.sh"),
    "noise": Path("Single_Sweep/Job_scripts/Noise_single_sweep.sh"),
    "disturbance": Path("Single_Sweep/Job_scripts/Disturbance_single_sweep.sh"),
}

JOB_RE = re.compile(r"Running job with ID:\s*(\d+)")
TAIL_BYTES = 8192


def resolve_script(scenario: str, override: Optional[Path]) -> Path:
    return override if override is not None else DEFAULT_SCRIPT_BY_SCENARIO[scenario]


def scan_log(path: Path, phrase: str):
    job_id = None
    try:
        with path.open("r", errors="ignore") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                m = JOB_RE.search(line)
                if m:
                    job_id = m.group(1)
                    break

        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            tail = f.read().decode(errors="ignore")

        if phrase in tail:
            return None

        if job_id is None:
            job_id = path.stem

        return path, str(job_id)
    except OSError:
        return None


def find_incomplete(log_dir: Path, phrase: str, recursive: bool, workers: int):
    files = (
        [p for p in log_dir.rglob("*") if p.is_file()]
        if recursive
        else [p for p in log_dir.iterdir() if p.is_file()]
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [r for r in ex.map(lambda p: scan_log(p, phrase), files) if r is not None]


def submit(script: Path, idx: str):
    return subprocess.run(
        ["sbatch", str(script), idx],
        capture_output=True,
        text=True,
        check=True,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Resubmit incomplete jobs by checking log files."
    )
    p.add_argument("log_dir", type=Path)
    p.add_argument("scenario", choices=["normal", "noise", "disturbance"])
    p.add_argument("--script", type=Path)
    p.add_argument("--phrase", default="run 9")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.log_dir.is_dir():
        print(f"Error: {args.log_dir} is not a directory.")
        return 1

    script = resolve_script(args.scenario, args.script)

    if not script.exists():
        print(f"Error: cannot find {script}")
        return 1

    missing = find_incomplete(
        args.log_dir,
        args.phrase,
        args.recursive,
        args.workers,
    )

    if not missing:
        print("All jobs completed.")
        return 0

    print(f"Found {len(missing)} incomplete jobs.\n")

    for path, idx in missing:
        print(f"{path} -> {idx}")

    if not args.execute:
        print("\nDry run:\n")
        for _, idx in missing:
            print(f"sbatch {script} {idx}")
        return 0

    for _, idx in missing:
        print(f"Submitting {idx}...")
        try:
            out = submit(script, idx)
            print(out.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {idx}")
            print(e.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
