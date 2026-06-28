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
            for name in ("param_group", "param_name", "sweep_label", "success")
            if name in normalized
        }
        if not {"param_name", "sweep_label", "success"}.issubset(field_idx):
            return []

        rows = []
        for row in reader:
            if len(row) <= max(field_idx.values()):
                continue
            rows.append(
                {
                    "param_group": row[field_idx.get("param_group", -1)].strip() if "param_group" in field_idx else "",
                    "param_name": row[field_idx["param_name"]].strip(),
                    "sweep_label": row[field_idx["sweep_label"]].strip(),
                    "success": parse_bool(row[field_idx["success"]]),
                }
            )
        return rows


def build_summary(rows, sweep_label="negative"):
    # Filter rows based on sweep_label
    if sweep_label == "positive":
        # For positive, include all rows that are not "negative" or "zero"
        filtered_rows = [row for row in rows if row.get("sweep_label") not in ("negative", "zero")]
    else:
        # For "negative" or "zero", filter exactly
        filtered_rows = [row for row in rows if row.get("sweep_label") == sweep_label]
    
    grouped = defaultdict(lambda: defaultdict(list))
    for row in filtered_rows:
        param_group = row.get("param_group", "")
        param_name = row["param_name"]
        grouped[param_group][param_name].append(row["success"])

    summary = {}
    for param_group, param_dict in grouped.items():
        summary[param_group] = []
        for param_name, successes in param_dict.items():
            success_rate = sum(successes) / len(successes) if successes else 0.0
            summary[param_group].append((param_name, success_rate))
        # Sort by success rate (ascending)
        summary[param_group].sort(key=lambda x: x[1])

    return summary


def plot_summary(summary, output_dir: Path, sweep_label="negative"):
    output_dir.mkdir(parents=True, exist_ok=True)
    for param_group, param_data in summary.items():
        if len(param_data) == 0:
            continue

        param_names = [item[0] for item in param_data]
        success_rates = [item[1] for item in param_data]

        x = list(range(len(param_names)))

        plt.figure(figsize=(12, 6))
        plt.plot(x, success_rates, marker="o", linestyle="-", color="steelblue", linewidth=2)
        plt.title(f"Success Rate by Parameter ({sweep_label.capitalize()} Sweep) - {param_group}")
        plt.xlabel("Parameter Name")
        plt.ylabel("Success Rate")
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.ylim(0, 1.05)
        plt.xticks(x, param_names, rotation=45, ha="right")
        plt.tight_layout()

        safe_name = param_group.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"success_rate_{sweep_label}_{safe_name}.png"
        plt.savefig(output_file)
        plt.close()


def plot_combined_summary(summary, output_dir: Path, sweep_label="negative"):
    """Create a combined plot showing all parameter groups together."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all data with param_group labels, preserving group blocks.
    all_data = []
    colors_map = {}
    color_palette = plt.cm.Set3(np.linspace(0, 1, len(summary)))
    
    for idx, (param_group, param_data) in enumerate(summary.items()):
        colors_map[param_group] = color_palette[idx]
        for param_name, success_rate in sorted(param_data, key=lambda x: x[1]):
            all_data.append((param_name, success_rate, param_group))
    
    param_names = [item[0] for item in all_data]
    success_rates = [item[1] for item in all_data]
    param_groups = [item[2] for item in all_data]
    colors = [colors_map[pg] for pg in param_groups]
    
    x = list(range(len(param_names)))
    
    plt.figure(figsize=(16, 8))
    plt.plot(x, success_rates, marker="o", linestyle="-", linewidth=2, color="steelblue")
    plt.title(f"Success Rate by All Parameters ({sweep_label.capitalize()} Sweep) - Combined")
    plt.xlabel("Parameter Name")
    plt.ylabel("Success Rate")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.ylim(0, 1.05)
    
    # Color code x-axis labels by param_group
    ax = plt.gca()
    for i, (label, color) in enumerate(zip(param_names, colors)):
        ax.get_xticklabels()
        ax.text(i, -0.12, label, ha="center", va="top", fontsize=8, rotation=90,
                transform=ax.get_xaxis_transform(), color=color, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(param_names))
    
    # Add legend for param groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[pg], label=pg) for pg in summary.keys()]
    plt.legend(handles=legend_elements, loc="upper left")
    
    plt.tight_layout()
    output_file = output_dir / f"success_rate_{sweep_label}_combined.png"
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
        "--sweep-label",
        choices=["negative", "zero", "positive"],
        default="negative",
        help="Which sweep values to analyze: negative, zero, or positive.",
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {base_dir}")

    all_rows = []
    for csv_path in csv_files:
        all_rows.extend(read_results(csv_path))

    summary = build_summary(all_rows, sweep_label=args.sweep_label)
    if not summary:
        raise SystemExit("No valid rows found to plot.")

    if args.sweep_label == "positive":
        out_dir = Path.cwd() / "Success_rate_comparison_plots_positive"
    elif args.sweep_label == "zero":
        out_dir = Path.cwd() / "Success_rate_comparison_plots_zero"
    else:
        out_dir = Path.cwd() / "Success_rate_comparison_plots_negative"

    plot_summary(summary, out_dir, sweep_label=args.sweep_label)
    plot_combined_summary(summary, out_dir, sweep_label=args.sweep_label)
    print(f"Saved plots for {len(summary)} parameter groups ({args.sweep_label}) to {out_dir}")


if __name__ == "__main__":
    main()
