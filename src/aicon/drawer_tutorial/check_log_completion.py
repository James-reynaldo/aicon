#!/usr/bin/env python3
from pathlib import Path
import argparse
import subprocess

def find_incomplete_logs(log_dir: Path, phrase: str, tail_lines: int = 50):
    incomplete = []
    for path in sorted(log_dir.glob("*.out")):
        result = subprocess.run(
            ["tail", "-n", str(tail_lines), str(path)],
            capture_output=True,
            text=True,
            errors="ignore",
        )
        if result.returncode == 0 and phrase not in result.stdout:
            incomplete.append(str(path))
    return incomplete

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--phrase", default="run 9")
    parser.add_argument("--tail-lines", type=int, default=50)
    args = parser.parse_args()

    incomplete = find_incomplete_logs(args.log_dir, args.phrase, args.tail_lines)
    print(f"Incomplete logs: {len(incomplete)}")
    for p in incomplete:
        print(p)

if __name__ == "__main__":
    main()