import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def read_results(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return []

        normalized = [col.strip().lower() for col in header]
        field_idx = {
            name: normalized.index(name)
            for name in ("param_group", "param_name", "sweep_value", "success")
            if name in normalized
        }
        if not {"param_name", "sweep_value", "success"}.issubset(field_idx):
            return []

        rows = []
        for row in reader:
            if len(row) <= max(field_idx.values()):
                continue
            rows.append(
                {
                    "param_group": row[field_idx.get("param_group", -1)].strip() if "param_group" in field_idx else "",
                    "param_name": row[field_idx["param_name"]].strip(),
                    "sweep_value": parse_float(row[field_idx["sweep_value"]]),
                    "success": parse_bool(row[field_idx["success"]]),
                }
            )
        return rows


def build_summary(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        param_group = row.get("param_group", "")
        param_name = row["param_name"]
        sweep_value = row["sweep_value"]
        if sweep_value is None:
            continue
        grouped[(param_group, param_name)][sweep_value].append(row["success"])

    summary = {}
    for group_key, sweep_dict in grouped.items():
        summary[group_key] = []
        for sweep_value, successes in sweep_dict.items():
            summary[group_key].append(
                (sweep_value, sum(successes) / len(successes), len(successes))
            )
        summary[group_key].sort(key=lambda x: x[0])

    return summary


def plot_summary(summary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for (param_group, param_name), values in summary.items():
        sweep_values = [v[0] for v in values]
        success_rates = [v[1] for v in values]
        counts = [v[2] for v in values]

        if len(sweep_values) == 0:
            continue

        labels = [str(v) for v in sweep_values]
        x = list(range(len(labels)))

        plt.figure(figsize=(10, 5))
        plt.plot(x, success_rates, marker="o", linestyle="-", label=param_name)
        plt.title(f"Success Rate vs Sweep Value for {param_group}.{param_name}")
        plt.xlabel("Sweep Value (categorical)")
        plt.ylabel("Success Rate")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(-0.05, 1.05)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.tight_layout()

        safe_name = f"{param_group}_{param_name}".replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"success_rate_{safe_name}.png"
        plt.savefig(output_file)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot success rate vs sweep value for each param_name.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing CSV result files.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory for output plot PNGs.",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {base_dir}")

    all_rows = []
    for csv_path in csv_files:
        all_rows.extend(read_results(csv_path))

    summary = build_summary(all_rows)
    if not summary:
        raise SystemExit("No valid rows found to plot.")

    plot_summary(summary, Path(args.output_dir))
    print(f"Saved plots for {len(summary)} parameters to {args.output_dir}")


if __name__ == "__main__":
    main()
