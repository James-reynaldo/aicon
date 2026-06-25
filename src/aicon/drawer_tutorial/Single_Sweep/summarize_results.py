import argparse
import csv
from pathlib import Path


def parse_bool(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "success", "s"}:
        return True
    if text in {"0", "false", "no", "n", "failed", "failure", "f"}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def parse_float(value):
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def summarize_csv_file(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return None

        normalized = [col.strip().lower() for col in header]
        required = {"success", "timesteps", "error"}
        if not required.intersection(normalized):
            return None

        try:
            success_idx = normalized.index("success")
        except ValueError:
            return None

        try:
            timesteps_idx = normalized.index("timesteps")
        except ValueError:
            return None

        error_idx = None
        for field in ("error", "err"):
            if field in normalized:
                error_idx = normalized.index(field)
                break

        total_runs = 0
        successes = 0
        sum_timesteps_success = 0.0
        sum_error = 0.0
        error_count = 0
        param_group = None
        param_name = None
        sweep_label = None
        sweep_value = None

        # Use first valid row to infer constant job metadata
        for row in reader:
            if len(row) <= max(success_idx, timesteps_idx, error_idx if error_idx is not None else 0):
                continue
            if param_group is None and "param_group" in normalized:
                param_group = row[normalized.index("param_group")].strip()
            if param_name is None and "param_name" in normalized:
                param_name = row[normalized.index("param_name")].strip()
            if sweep_label is None and "sweep_label" in normalized:
                sweep_label = row[normalized.index("sweep_label")].strip()
            if sweep_value is None and "sweep_value" in normalized:
                sweep_value = row[normalized.index("sweep_value")].strip()

            total_runs += 1
            success = parse_bool(row[success_idx])
            timesteps = parse_float(row[timesteps_idx])
            error = parse_float(row[error_idx]) if error_idx is not None else None

            if success:
                successes += 1
                if timesteps is not None:
                    sum_timesteps_success += timesteps
            if error is not None:
                sum_error += error
                error_count += 1

        if total_runs == 0:
            return None

        success_rate = successes / total_runs
        avg_timesteps_success = sum_timesteps_success / successes if successes else None
        avg_error = sum_error / total_runs if error_count else None

        return {
            "source_csv": str(csv_path),
            "job_id": csv_path.stem,
            "param_group": param_group,
            "param_name": param_name,
            "sweep_label": sweep_label,
            "sweep_value": sweep_value,
            "num_runs": total_runs,
            "success_rate": success_rate,
            "avg_timesteps_success": avg_timesteps_success,
            "avg_error": avg_error,
        }


def find_csv_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


def write_summary(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "job_id",
            "source_csv",
            "param_group",
            "param_name",
            "sweep_label",
            "sweep_value",
            "num_runs",
            "success_rate",
            "avg_timesteps_success",
            "avg_error",
        ])
        for row in rows:
            writer.writerow([
                row["job_id"],
                row["source_csv"],
                row.get("param_group", ""),
                row.get("param_name", ""),
                row.get("sweep_label", ""),
                row.get("sweep_value", ""),
                row["num_runs"],
                f"{row['success_rate']:.6f}",
                "" if row["avg_timesteps_success"] is None else f"{row['avg_timesteps_success']:.6f}",
                "" if row["avg_error"] is None else f"{row['avg_error']:.6f}",
            ])


def main():
    parser = argparse.ArgumentParser(description="Summarize all result CSVs in a Single_Sweep results folder.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="The folder containing CSV result files.",
    )
    parser.add_argument(
        "--output-file",
        default="results_summary.csv",
        help="The output summary CSV file path.",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    csv_files = find_csv_files(base_dir)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {base_dir}")

    summary = []
    for csv_path in csv_files:
        record = summarize_csv_file(csv_path)
        if record is not None:
            summary.append(record)

    if not summary:
        raise SystemExit("No valid result CSV files were summarized.")

    write_summary(Path(args.output_file), summary)
    print(f"Wrote summary for {len(summary)} files to {args.output_file}")


if __name__ == "__main__":
    main()
