import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import json


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


def read_results_from_db(db_path: Path):
    rows = []
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT t.metadata, t.success FROM trials t")
    for metadata_json, success_val in cur.fetchall():
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            continue

        parameter = metadata.get("parameter", "")
        if "." in parameter:
            group, name = parameter.rsplit('.', 1)
        else:
            group, name = "", parameter

        label = metadata.get("label", "")
        success = bool(success_val)
        rows.append({
            "param_group": group,
            "param_name": name,
            "sweep_label": label,
            "success": success,
        })

    conn.close()
    return rows


def build_summary(rows, sweep_label="negative"):
    # Filter rows based on sweep_label
    if sweep_label == "all":
        # Include negative, zero, and positive sweeps together.
        filtered_rows = rows
    elif sweep_label == "positive":
        # For positive, include all rows that are not "negative" or "zero"
        filtered_rows = [row for row in rows if row.get("sweep_label") not in ("negative", "zero")]
    else:
        # For "negative" or "zero", filter exactly
        filtered_rows = [row for row in rows if row.get("sweep_label") == sweep_label]
    
    # Group by param_group -> param_name -> sweep_value_label -> list of 0/1 successes
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in filtered_rows:
        param_group = row.get("param_group", "")
        param_name = row["param_name"]
        sweep_val = row.get("sweep_label", "")
        grouped[param_group][param_name][sweep_val].append(1 if row["success"] else 0)

    summary = {}
    for param_group, param_dict in grouped.items():
        summary[param_group] = []
        for param_name, sweep_map in param_dict.items():
            # compute success rate per sweep-value, then mean+var over those rates
            rates = []
            for sweep_val, trials in sweep_map.items():
                if not trials:
                    continue
                rate = float(np.sum(trials)) / float(len(trials))
                rates.append(rate)

            if rates:
                mean = float(np.mean(rates))
                var = float(np.var(rates))
                count = len(rates)
            else:
                mean = 0.0
                var = 0.0
                count = 0

            # store (param_name, mean, variance, count_of_sweep_values)
            summary[param_group].append((param_name, mean, var, count))

        # Sort by mean success rate (ascending)
        summary[param_group].sort(key=lambda x: x[1])

    return summary


def plot_summary(summary_normal, summary_noise, summary_disturbance, output_dir: Path, sweep_label="negative"):
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_normal is None:
        return

    for param_group, normal_data in summary_normal.items():
        if len(normal_data) == 0:
            continue

        # X-axis comes from the normal summary
        param_names = [item[0] for item in normal_data]
        normal_rates = [item[1] for item in normal_data]
        normal_vars = [item[2] if len(item) > 2 else 0.0 for item in normal_data]
        x = list(range(len(param_names)))

        # Map parameter name -> x index
        x_map = {name: i for i, name in enumerate(param_names)}

        plt.figure(figsize=(12, 6))

        # Plot normal (connect neighboring x points)
        plt.plot(
            x,
            normal_rates,
            marker="o",
            linestyle="-",
            color="steelblue",
            linewidth=2,
            label="Normal",
        )

        # Shade variance for positive sweep
        if sweep_label == "positive":
            stds = np.sqrt(np.array(normal_vars))
            lower = np.clip(np.array(normal_rates) - stds, 0.0, 1.0)
            upper = np.clip(np.array(normal_rates) + stds, 0.0, 1.0)
            plt.fill_between(x, lower, upper, color="steelblue", alpha=0.15)

        # Plot noise on same x-axis
        if summary_noise is not None and param_group in summary_noise:
            noise_x = []
            noise_y = []
            noise_vars = []

            for item in summary_noise[param_group]:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                if name in x_map:
                    noise_x.append(x_map[name])
                    noise_y.append(rate)
                    noise_vars.append(var)

            # sort by x so lines connect neighbors only
            if noise_x:
                pairs = sorted(zip(noise_x, noise_y, noise_vars), key=lambda t: t[0])
                nx, ny, nvar = zip(*pairs)
                plt.plot(
                    list(nx),
                    list(ny),
                    marker="x",
                    linestyle="-",
                    color="orange",
                    linewidth=2,
                    label="Noise",
                )
                if sweep_label == "positive":
                    nstd = np.sqrt(np.array(nvar))
                    lower = np.clip(np.array(ny) - nstd, 0.0, 1.0)
                    upper = np.clip(np.array(ny) + nstd, 0.0, 1.0)
                    plt.fill_between(list(nx), lower, upper, color="orange", alpha=0.12)

        # Plot disturbance on same x-axis
        if summary_disturbance is not None and param_group in summary_disturbance:
            disturbance_x = []
            disturbance_y = []
            disturbance_vars = []

            for item in summary_disturbance[param_group]:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                if name in x_map:
                    disturbance_x.append(x_map[name])
                    disturbance_y.append(rate)
                    disturbance_vars.append(var)

            if disturbance_x:
                pairs = sorted(zip(disturbance_x, disturbance_y, disturbance_vars), key=lambda t: t[0])
                dx, dy, dvar = zip(*pairs)
                plt.plot(
                    list(dx),
                    list(dy),
                    marker="s",
                    linestyle="-",
                    color="green",
                    linewidth=2,
                    label="Disturbance",
                )
                if sweep_label == "positive":
                    dstd = np.sqrt(np.array(dvar))
                    lower = np.clip(np.array(dy) - dstd, 0.0, 1.0)
                    upper = np.clip(np.array(dy) + dstd, 0.0, 1.0)
                    plt.fill_between(list(dx), lower, upper, color="green", alpha=0.12)

        plt.title(f"Success Rate by Parameter ({sweep_label.capitalize()} Sweep) - {param_group}")
        plt.xlabel("Parameter Name")
        plt.ylabel("Success Rate")
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.ylim(0, 1.05)
        plt.xticks(x, param_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        safe_name = param_group.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"success_rate_{sweep_label}_{safe_name}.png"
        plt.savefig(output_file)
        plt.close()


def summary_lookup(summary):
    lookup = {}
    if summary is None:
        return lookup

    for param_group, param_data in summary.items():
        for item in param_data:
            # item is (param_name, mean, var, count)
            if not item:
                continue
            param_name = item[0]
            success_rate = item[1] if len(item) > 1 else 0.0
            lookup[(param_group, param_name)] = success_rate

    return lookup


def find_stable_all_points(summary_all, summary_negative, summary_zero, summary_positive, tolerance=0.10):
    stable_points = set()
    all_lookup = summary_lookup(summary_all)
    negative_lookup = summary_lookup(summary_negative)
    zero_lookup = summary_lookup(summary_zero)
    positive_lookup = summary_lookup(summary_positive)

    for key, all_rate in all_lookup.items():
        if key not in negative_lookup or key not in zero_lookup or key not in positive_lookup:
            continue

        rates = [
            all_rate,
            negative_lookup[key],
            zero_lookup[key],
            positive_lookup[key],
        ]
        if max(rates) - min(rates) <= tolerance:
            stable_points.add(key)

    return stable_points


def plot_combined_summary(
    summary_normal,
    summary_noise,
    summary_disturbance,
    output_dir: Path,
    sweep_label="negative",
    stable_normal=None,
    stable_noise=None,
    stable_disturbance=None,
):
    """Create a combined plot showing all parameter groups together."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_normal is None:
        return
    
    # Collect all data with param_group labels, preserving group blocks.
    all_data = []
    colors_map = {}
    color_palette = plt.cm.Set3(np.linspace(0, 1, len(summary_normal)))
    
    for idx, (param_group, param_data) in enumerate(summary_normal.items()):
        colors_map[param_group] = color_palette[idx]
        for item in sorted(param_data, key=lambda x: x[1]):
            # item: (param_name, mean, var, count)
            param_name = item[0]
            success_rate = item[1] if len(item) > 1 else 0.0
            var = item[2] if len(item) > 2 else 0.0
            all_data.append((param_name, success_rate, var, param_group))

    if len(all_data) == 0:
        return
    
    param_names = [item[0] for item in all_data]
    success_rates = [item[1] for item in all_data]
    success_vars = [item[2] for item in all_data]
    param_groups = [item[3] for item in all_data]
    colors = [colors_map[pg] for pg in param_groups]

    x = list(range(len(param_names)))
    x_map = {(param_groups[i], param_names[i]): i for i in range(len(all_data))}

    plt.figure(figsize=(16, 8))
    plt.plot(
        x,
        success_rates,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="steelblue",
        label="Normal",
    )
    if sweep_label == "positive":
        stds = np.sqrt(np.array(success_vars))
        lower = np.clip(np.array(success_rates) - stds, 0.0, 1.0)
        upper = np.clip(np.array(success_rates) + stds, 0.0, 1.0)
        plt.fill_between(x, lower, upper, color="steelblue", alpha=0.12)
    stable_normal = stable_normal or set()
    stable_noise = stable_noise or set()
    stable_disturbance = stable_disturbance or set()
    fully_stable_points = set()
    if sweep_label == "all":
        fully_stable_points = stable_normal & stable_noise & stable_disturbance

    # Add dashed separators between parameter-group blocks (always)
    group_boundaries = []
    if param_groups:
        last_group = param_groups[0]
        for i, pg in enumerate(param_groups):
            if pg != last_group:
                # boundary at index i (start of new group)
                group_boundaries.append(i)
                last_group = pg

    ax = plt.gca()
    for boundary in group_boundaries:
        ax.axvline(boundary - 0.5, linestyle='--', color='black', alpha=0.9, linewidth=1.2, zorder=10)
    if sweep_label == "all":
        stable_normal_x = []
        stable_normal_y = []
        for param_name, success_rate, param_group in all_data:
            if (param_group, param_name) in stable_normal:
                stable_normal_x.append(x_map[(param_group, param_name)])
                stable_normal_y.append(success_rate)

        if stable_normal_x:
            plt.plot(
                stable_normal_x,
                stable_normal_y,
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor="steelblue",
                markeredgecolor="black",
                markeredgewidth=2,
                color="steelblue",
                label="_nolegend_",
            )

    if summary_noise is not None:
        noise_x = []
        noise_y = []
        noise_vars = []
        stable_noise_x = []
        stable_noise_y = []
        for param_group, param_data in summary_noise.items():
            for item in param_data:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                key = (param_group, name)
                if key in x_map:
                    noise_x.append(x_map[key])
                    noise_y.append(rate)
                    noise_vars.append(var)
                    if sweep_label == "all" and key in stable_noise:
                        stable_noise_x.append(x_map[key])
                        stable_noise_y.append(rate)

        if noise_x:
            pairs = sorted(zip(noise_x, noise_y, noise_vars), key=lambda t: t[0])
            nx, ny, nvar = zip(*pairs)
            plt.plot(
                list(nx),
                list(ny),
                marker="x",
                linestyle="-",
                linewidth=2,
                color="orange",
                label="Noise",
            )
            if sweep_label == "positive":
                nstd = np.sqrt(np.array(nvar))
                lower = np.clip(np.array(ny) - nstd, 0.0, 1.0)
                upper = np.clip(np.array(ny) + nstd, 0.0, 1.0)
                plt.fill_between(list(nx), lower, upper, color="orange", alpha=0.10)
        if stable_noise_x:
            plt.plot(
                stable_noise_x,
                stable_noise_y,
                marker="x",
                linestyle="None",
                markersize=10,
                markeredgewidth=3,
                color="black",
                label="_nolegend_",
            )

    if summary_disturbance is not None:
        disturbance_x = []
        disturbance_y = []
        disturbance_vars = []
        stable_disturbance_x = []
        stable_disturbance_y = []
        for param_group, param_data in summary_disturbance.items():
            for item in param_data:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                key = (param_group, name)
                if key in x_map:
                    disturbance_x.append(x_map[key])
                    disturbance_y.append(rate)
                    disturbance_vars.append(var)
                    if sweep_label == "all" and key in stable_disturbance:
                        stable_disturbance_x.append(x_map[key])
                        stable_disturbance_y.append(rate)

        if disturbance_x:
            pairs = sorted(zip(disturbance_x, disturbance_y, disturbance_vars), key=lambda t: t[0])
            dx, dy, dvar = zip(*pairs)
            plt.plot(
                list(dx),
                list(dy),
                marker="s",
                linestyle="-",
                linewidth=2,
                color="green",
                label="Disturbance",
            )
            if sweep_label == "positive":
                dstd = np.sqrt(np.array(dvar))
                lower = np.clip(np.array(dy) - dstd, 0.0, 1.0)
                upper = np.clip(np.array(dy) + dstd, 0.0, 1.0)
                plt.fill_between(list(dx), lower, upper, color="green", alpha=0.10)
        if stable_disturbance_x:
            plt.plot(
                stable_disturbance_x,
                stable_disturbance_y,
                marker="s",
                linestyle="None",
                markersize=9,
                markerfacecolor="green",
                markeredgecolor="black",
                markeredgewidth=2,
                color="green",
                label="_nolegend_",
            )

    plt.title(f"Success Rate by All Parameters ({sweep_label.capitalize()} Sweep) - Combined")
    plt.xlabel("Parameter Name")
    plt.ylabel("Success Rate")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.ylim(0, 1.05)
    
    # Identify parameters with above-average standard deviations (for positive sweep)
    high_std_params = set()
    if sweep_label == "positive":
        param_stds = {}  # (param_group, param_name) -> list of stds
        
        # Collect all std values across all 3 scenarios
        for param_group, param_data in summary_normal.items():
            for item in param_data:
                param_name = item[0]
                var = item[2] if len(item) > 2 else 0.0
                key = (param_group, param_name)
                if key not in param_stds:
                    param_stds[key] = []
                param_stds[key].append(np.sqrt(var))
        
        if summary_noise is not None:
            for param_group, param_data in summary_noise.items():
                for item in param_data:
                    param_name = item[0]
                    var = item[2] if len(item) > 2 else 0.0
                    key = (param_group, param_name)
                    if key not in param_stds:
                        param_stds[key] = []
                    param_stds[key].append(np.sqrt(var))
        
        if summary_disturbance is not None:
            for param_group, param_data in summary_disturbance.items():
                for item in param_data:
                    param_name = item[0]
                    var = item[2] if len(item) > 2 else 0.0
                    key = (param_group, param_name)
                    if key not in param_stds:
                        param_stds[key] = []
                    param_stds[key].append(np.sqrt(var))
        
        # Calculate average std per parameter
        param_avg_stds = {}
        all_stds_for_avg = []
        for key, stds in param_stds.items():
            avg_std = float(np.mean(stds))
            param_avg_stds[key] = avg_std
            all_stds_for_avg.extend(stds)
        
        # Calculate overall average std
        overall_avg_std = float(np.mean(all_stds_for_avg)) if all_stds_for_avg else 0.0
        
        # Identify parameters with above-average std
        for key, avg_std in param_avg_stds.items():
            if avg_std > overall_avg_std:
                high_std_params.add(key)
    
    # Color code x-axis labels by param_group
    ax = plt.gca()
    for i, (label, color) in enumerate(zip(param_names, colors)):
        ax.get_xticklabels()
        label_key = (param_groups[i], label)
        label_box = None
        label_weight = "bold"
        if label_key in fully_stable_points:
            label_weight = "heavy"
            label_box = {
                "boxstyle": "round,pad=0.2",
                "facecolor": "lemonchiffon",
                "edgecolor": "black",
                "linewidth": 1.2,
            }
        elif sweep_label == "positive" and label_key in high_std_params:
            label_weight = "bold"
            label_box = {
                "boxstyle": "round,pad=0.2",
                "facecolor": "none",
                "edgecolor": "black",
                "linewidth": 1.5,
            }
        ax.text(
            i,
            -0.12,
            label,
            ha="center",
            va="top",
            fontsize=8,
            rotation=90,
            transform=ax.get_xaxis_transform(),
            color=color,
            weight=label_weight,
            bbox=label_box,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(param_names))
    
    # Add legend for param groups
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    dataset_legend = [
        Line2D([0], [0], marker="o", color="steelblue", linestyle="None", label="Normal"),
        Line2D([0], [0], marker="x", color="orange", linestyle="None", label="Noise"),
        Line2D([0], [0], marker="s", color="green", linestyle="None", label="Disturbance"),
    ]
    if sweep_label == "all":
        dataset_legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=2,
                color="black",
                linestyle="None",
                label="<=10% variation across all/negative/zero/positive",
            )
        )
        if fully_stable_points:
            dataset_legend.append(
                Patch(
                    facecolor="lemonchiffon",
                    edgecolor="black",
                    label="X label: Normal, Noise, and Disturbance all stable",
                )
            )
    elif sweep_label == "positive" and high_std_params:
        dataset_legend.append(
            Patch(
                facecolor="none",
                edgecolor="black",
                label="X label: Above-average std deviation across all scenarios",
            )
        )
    group_legend = [Patch(facecolor=colors_map[pg], label=pg) for pg in summary_normal.keys()]
    dataset_legend_artist = ax.legend(handles=dataset_legend, loc="upper left")
    ax.add_artist(dataset_legend_artist)
    ax.legend(handles=group_legend, loc="upper right", title="Parameter Groups")
    
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
        "--db-path-normal",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--db-path-noise",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--db-path-disturbance",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--sweep-label",
        choices=["negative", "zero", "positive", "all"],
        default="negative",
        help="Which sweep values to analyze: negative, zero, positive, or all.",
    )
    args = parser.parse_args()

    all_rows_normal = []
    if args.db_path_normal:
        all_rows_normal = read_results_from_db(Path(args.db_path_normal))
    all_rows_noise = []
    if args.db_path_noise:
        all_rows_noise = read_results_from_db(Path(args.db_path_noise))
    all_rows_disturbance = []
    if args.db_path_disturbance:
        all_rows_disturbance = read_results_from_db(Path(args.db_path_disturbance))
    # else:
    #     base_dir = Path(args.results_dir)
    #     csv_files = sorted(base_dir.glob("*.csv"))
    #     if not csv_files:
    #         raise SystemExit(f"No CSV files found in {base_dir}")
    #     for csv_path in csv_files:
    #         all_rows.extend(read_results(csv_path))

    summary_normal = build_summary(all_rows_normal, sweep_label=args.sweep_label)
    summary_noise = build_summary(all_rows_noise, sweep_label=args.sweep_label)
    summary_disturbance = build_summary(all_rows_disturbance, sweep_label=args.sweep_label)
    if not summary_normal:
        raise SystemExit("No valid rows found to plot.")

    stable_normal = set()
    stable_noise = set()
    stable_disturbance = set()
    if args.sweep_label == "all":
        stable_normal = find_stable_all_points(
            summary_normal,
            build_summary(all_rows_normal, sweep_label="negative"),
            build_summary(all_rows_normal, sweep_label="zero"),
            build_summary(all_rows_normal, sweep_label="positive"),
        )
        stable_noise = find_stable_all_points(
            summary_noise,
            build_summary(all_rows_noise, sweep_label="negative"),
            build_summary(all_rows_noise, sweep_label="zero"),
            build_summary(all_rows_noise, sweep_label="positive"),
        )
        stable_disturbance = find_stable_all_points(
            summary_disturbance,
            build_summary(all_rows_disturbance, sweep_label="negative"),
            build_summary(all_rows_disturbance, sweep_label="zero"),
            build_summary(all_rows_disturbance, sweep_label="positive"),
        )

    if args.sweep_label == "positive":
        out_dir = Path.cwd() / "Success_rate_comparison_plots_positive"
    elif args.sweep_label == "zero":
        out_dir = Path.cwd() / "Success_rate_comparison_plots_zero"
    elif args.sweep_label == "all":
        out_dir = Path.cwd() / "Success_rate_comparison_plots_all"
    else:
        out_dir = Path.cwd() / "Success_rate_comparison_plots_negative"

    plot_summary(summary_normal, summary_noise, summary_disturbance, out_dir, sweep_label=args.sweep_label)
    plot_combined_summary(
        summary_normal,
        summary_noise,
        summary_disturbance,
        out_dir,
        sweep_label=args.sweep_label,
        stable_normal=stable_normal,
        stable_noise=stable_noise,
        stable_disturbance=stable_disturbance,
    )
    print(f"Saved plots for {len(summary_normal)} parameter groups ({args.sweep_label}) to {out_dir}")


if __name__ == "__main__":
    main()
