#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write job indexes whose trials table does not contain exactly the required number of rows into a text file."
    )
    parser.add_argument(
        "folderpath",
        type=Path,
        help="Path to the folder containing experiment_store_job_*.db files.",
    )
    parser.add_argument(
        "--required-trials",
        type=int,
        default=10,
        help="Exact required number of rows in each database's trials table (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .txt file for the offending indexes. Defaults to <folder>/bad_trial_counts.txt.",
    )
    return parser.parse_args()


def parse_db_index(db_name: str):
    parts = db_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Could not parse index from {db_name}")
    return int(parts[3])


def count_trials_in_db(db_path: Path):
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
            return int(row[0]) if row is not None else 0
    except sqlite3.DatabaseError as exc:
        print(f"Warning: could not read {db_path.name}: {exc}")
        return None


def main():
    args = parse_args()
    folder = args.folderpath

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    if args.output is None:
        output_file = folder / "bad_trial_counts.txt"
    else:
        output_file = args.output

    bad_indexes = []
    bad_details = []

    for db in sorted(folder.glob("experiment_store_job_*.db")):
        try:
            index = parse_db_index(db.name)
        except (IndexError, ValueError):
            print(f"Skipping invalid filename: {db.name}")
            continue

        trial_count = count_trials_in_db(db)
        if trial_count is None:
            bad_indexes.append(index)
            bad_details.append((index, db.name, "DB error"))
        elif trial_count != args.required_trials:
            bad_indexes.append(index)
            bad_details.append((index, db.name, trial_count))

    bad_indexes = sorted(set(bad_indexes))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(str(i) for i in bad_indexes) + ("\n" if bad_indexes else ""))

    print(f"Found {len(bad_indexes)} index(es) with trial counts != {args.required_trials}")
    print(f"Saved to: {output_file}")

    if bad_details:
        for index, db_name, count in bad_details:
            print(f"- {db_name} (index {index}): trials={count}")
    else:
        print("No offending databases found.")


if __name__ == "__main__":
    main()