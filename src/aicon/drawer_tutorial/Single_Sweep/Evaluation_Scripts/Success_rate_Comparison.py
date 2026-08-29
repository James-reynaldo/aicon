import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import json

CONSIDER_ONLY_SUCCESSFUL = True  # If True, only consider successful trials for metrics like avg_timesteps, avg_error, etc.


def normalize_sweep_label(label):
    if label is None:
        return ""
    text = str(label).strip()
    if text == "":
        return ""
    if text in {"standard", "negative", "zero", "x1000"}:
        return text
    if text.startswith("x") and text[1:] in {"0.2", "0.5", "2", "5", "1000"}:
        return "x1000" if text[1:] == "1000" else f"{text[1:]}x"
    if text.endswith("x") and text[:-1] in {"0.2", "0.5", "2", "5", "1000"}:
        return "x1000" if text[:-1] == "1000" else f"{text[:-1]}x"
    if text in {"0.2x", "0.5x", "2x", "5x", "x1000"}:
        return text
    return text


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


def filter_rows_by_sweep_label(rows, sweep_label):
    if sweep_label == "all":
        # Include negative, zero, and positive sweeps together.
        return rows
    if sweep_label == "positive":
        # For positive, include all rows that are not negative, zero, or x1000,
        # regardless of whether the raw database label is stored as x2/x5 or 2x/5x.
        return [
            row for row in rows
            if normalize_sweep_label(row.get("sweep_label")) not in ("negative", "zero", "x1000","x0.001")
        ]
    if sweep_label == "Non_reasonable":
        # For Non_reasonable, include all rows that are not negative, zero, or x1000,
        # regardless of whether the raw database label is stored as x2/x5 or 2x/5x.
        return [
            row for row in rows
            if normalize_sweep_label(row.get("sweep_label")) in ("negative", "zero", "x1000","x0.001")
        ]
    # For "negative", "zero", or "x1000", filter exactly after normalization.
    return [
        row
        for row in rows
        if normalize_sweep_label(row.get("sweep_label")) == sweep_label
    ]


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
    cur.execute("SELECT t.metadata, t.success, t.timesteps, t.error, t.anchor_error, t.kinematic_axis_error FROM trials t")
    for metadata_json, success_val, timesteps_val, error_val, anchor_err_val, kinematic_err_val in cur.fetchall():
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            continue

        parameter = metadata.get("parameter", "")
        if "." in parameter:
            group, name = parameter.rsplit('.', 1)
        else:
            group, name = "", parameter

        label = normalize_sweep_label(metadata.get("label", ""))
        success = parse_bool(success_val)
        experiment_type = metadata.get("experiment_type", "")
        rows.append({
            "param_group": group,
            "param_name": name,
            "sweep_label": label,
            "success": success,
            "timesteps": timesteps_val,
            "error": error_val,
            "anchor_error": anchor_err_val,
            "kinematic_axis_error": kinematic_err_val,
            "experiment_type": experiment_type,
        })

    conn.close()
    return rows


def build_summary(rows, sweep_label="negative", metric="success_rate", include_failed_trials: bool = False, debug_param_group: str = None):
    # Filter rows based on sweep_label
    filtered_rows = filter_rows_by_sweep_label(rows, sweep_label)
    # Compute a global "standard" baseline (if present) to use as fallback
    standard_baseline = np.nan
    standard_vals = []
    for r in filtered_rows:
        if r.get("param_group") == "standard" and r.get("param_name") == "standard" and r.get("sweep_label") == "standard":
            if metric == "success_rate":
                standard_vals.append(1.0 if r.get("success") else 0.0)
            elif metric == "avg_timesteps":
                if include_failed_trials or r.get("success"):
                    if r.get("timesteps") is not None:
                        try:
                            standard_vals.append(float(r.get("timesteps")))
                        except Exception:
                            pass
            elif metric == "avg_anchor_error":
                if include_failed_trials or r.get("success"):
                    if r.get("anchor_error") is not None:
                        try:
                            standard_vals.append(float(r.get("anchor_error")))
                        except Exception:
                            pass
            elif metric == "avg_kinematic_axis_error":
                if include_failed_trials or r.get("success"):
                    if r.get("kinematic_axis_error") is not None:
                        try:
                            standard_vals.append(float(r.get("kinematic_axis_error")))
                        except Exception:
                            pass
            elif metric == "avg_error":
                if include_failed_trials or r.get("success"):
                    if r.get("error") is not None:
                        try:
                            standard_vals.append(float(r.get("error")))
                        except Exception:
                            pass
    if standard_vals:
        standard_baseline = float(np.mean(np.asarray(standard_vals, dtype=float)))

    # Group by param_group -> param_name -> sweep_value_label -> list of rows
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in filtered_rows:
        # Only include rows from 1d_sweep experiments for 1D summaries
        if row.get("experiment_type") and row.get("experiment_type") != "1d_sweep":
            continue
        param_group = row.get("param_group", "")
        param_name = row["param_name"]
        sweep_val = row.get("sweep_label", "")
        grouped[param_group][param_name][sweep_val].append(row)

    summary = {}
    # For each parameter, compute metric per sweep label then aggregate (mean+var)
    for param_group, param_dict in grouped.items():
        summary[param_group] = []
        for param_name, sweep_map in param_dict.items():
            # Requested semantics for the 1D sweep summary:
            # - y value = mean of the per-label means across the 5 sweep labels
            # - shade = S_i = sqrt(var(rates)), where rates is the list of 5 label means
            rates = []
            included_labels = []
            for sweep_val, rows_list in sweep_map.items():
                if not rows_list:
                    continue
                if metric == "success_rate":
                    rate = float(np.sum([1 if r.get("success") else 0 for r in rows_list])) / float(len(rows_list))
                    rates.append(rate)
                    included_labels.append(sweep_val)
                elif metric == "avg_anchor_error":
                    if include_failed_trials:
                        anchor_errors = [r.get("anchor_error") for r in rows_list if r.get("anchor_error") is not None]
                    else:
                        anchor_errors = [r.get("anchor_error") for r in rows_list if r.get("success") and r.get("anchor_error") is not None]
                    if anchor_errors:
                        rates.append(float(np.mean(anchor_errors)))
                        included_labels.append(sweep_val)
                elif metric == "avg_timesteps":
                    if include_failed_trials:
                        success_timesteps = [r.get("timesteps") for r in rows_list if r.get("timesteps") is not None]
                    else:
                        success_timesteps = [r.get("timesteps") for r in rows_list if r.get("success") and r.get("timesteps") is not None]
                    if success_timesteps:
                        rates.append(float(np.mean(success_timesteps)))
                        included_labels.append(sweep_val)
                elif metric == "avg_kinematic_axis_error":
                    if include_failed_trials:
                        kinematic_errors = [r.get("kinematic_axis_error") for r in rows_list if r.get("kinematic_axis_error") is not None]
                    else:
                        kinematic_errors = [r.get("kinematic_axis_error") for r in rows_list if r.get("success") and r.get("kinematic_axis_error") is not None]
                    if kinematic_errors:
                        rates.append(float(np.mean(kinematic_errors)))
                        included_labels.append(sweep_val)
                elif metric == "avg_error":
                    if include_failed_trials:
                        errors = [r.get("error") for r in rows_list if r.get("error") is not None]
                    else:
                        errors = [r.get("error") for r in rows_list if r.get("success") and r.get("error") is not None]
                    if errors:
                        rates.append(float(np.mean(errors)))
                        included_labels.append(sweep_val)

            # If the per-parameter data lacks a 'standard' label but a global standard baseline exists,
            # include it in the rates so S_i accounts for the baseline.
            if "standard" not in included_labels and not np.isnan(standard_baseline):
                rates.append(standard_baseline)
                included_labels.append("standard")

            if rates:
                rates_array = np.asarray(rates, dtype=float)
                mean = float(np.mean(rates_array))
                # Build a mapping from included label -> mean value (order preserved)
                try:
                    label_mean_map = {lab: float(rates_array[i]) for i, lab in enumerate(included_labels)}
                except Exception:
                    label_mean_map = {lab: None for lab in included_labels}
                var = float(np.var(rates_array))
                count = len(rates_array)
                si = float(np.sqrt(var))
                # Detailed debug output: label->mean mapping, raw rates list, computed S_i, mean, count
                if debug_param_group is not None and debug_param_group == param_group:
                    print(
                        f"DEBUG_SI: param_group={param_group}, param_name={param_name}, label_means={label_mean_map}, rates={rates_array.tolist()}, S_i={si:.6f}, mean={mean}, count={count}"
                    )
            else:
                mean = 0.0
                var = 0.0
                count = 0

            if param_group != "standard":
                summary[param_group].append((param_name, mean, var, count))

        # Keep a metric-independent order so the x-axis stays stable across metrics.
        summary[param_group].sort(key=lambda x: x[0])

    return summary


def build_anchor_error_outcome_summary(rows, sweep_label="negative", metric="avg_anchor_error"):
    filtered_rows = filter_rows_by_sweep_label(rows, sweep_label)

    grouped = defaultdict(lambda: defaultdict(lambda: {"success": [], "failed": []}))
    for row in filtered_rows:
        param_group = row.get("param_group", "")
        param_name = row.get("param_name", "")
        outcome_key = "success" if row.get("success") else "failed"
        grouped[param_group][param_name][outcome_key].append(row)

    summary = {}
    for param_group, param_dict in grouped.items():
        summary[param_group] = []
        for param_name, outcome_map in param_dict.items():
            if metric == "avg_anchor_error":
                success_errors = [r.get("anchor_error") for r in outcome_map["success"] if r.get("anchor_error") is not None]
                failed_errors = [r.get("anchor_error") for r in outcome_map["failed"] if r.get("anchor_error") is not None]
            elif metric == "avg_error":
                success_errors = [r.get("error") for r in outcome_map["success"] if r.get("error") is not None]
                failed_errors = [r.get("error") for r in outcome_map["failed"] if r.get("error") is not None]
            elif metric == "avg_kinematic_axis_error":
                success_errors = [r.get("kinematic_axis_error") for r in outcome_map["success"] if r.get("kinematic_axis_error") is not None]
                failed_errors = [r.get("kinematic_axis_error") for r in outcome_map["failed"] if r.get("kinematic_axis_error") is not None]
            success_mean = float(np.mean(success_errors)) if success_errors else None
            success_var = float(np.var(success_errors)) if success_errors else None
            failed_mean = float(np.mean(failed_errors)) if failed_errors else None
            failed_var = float(np.var(failed_errors)) if failed_errors else None

            summary[param_group].append(
                (
                    param_name,
                    success_mean,
                    success_var,
                    len(success_errors),
                    failed_mean,
                    failed_var,
                    len(failed_errors),
                )
            )

        summary[param_group].sort(key=lambda x: x[0])

    return summary


def plot_summary(summary_normal, summary_noise, summary_disturbance, summary_bad_prior, output_dir: Path, sweep_label="negative", metric="success_rate"):
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

        # Shade variance for the plotted metric
        stds = np.sqrt(np.array(normal_vars))
        if metric == "success_rate":
            lower = np.clip(np.array(normal_rates) - stds, 0.0, 1.0)
            upper = np.clip(np.array(normal_rates) + stds, 0.0, 1.0)
        else:
            lower = np.array(normal_rates) - stds
            upper = np.array(normal_rates) + stds
        plt.fill_between(x, lower, upper, color="steelblue", alpha=0.15, label="±1 std")

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
                if nvar:
                    nstd = np.sqrt(np.array(nvar))
                    if metric == "success_rate":
                        lower = np.clip(np.array(ny) - nstd, 0.0, 1.0)
                        upper = np.clip(np.array(ny) + nstd, 0.0, 1.0)
                    else:
                        lower = np.array(ny) - nstd
                        upper = np.array(ny) + nstd
                    plt.fill_between(list(nx), lower, upper, color="orange", alpha=0.18, label="Noise ±1 std")

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
                if dvar:
                    dstd = np.sqrt(np.array(dvar))
                    if metric == "success_rate":
                        lower = np.clip(np.array(dy) - dstd, 0.0, 1.0)
                        upper = np.clip(np.array(dy) + dstd, 0.0, 1.0)
                    else:
                        lower = np.array(dy) - dstd
                        upper = np.array(dy) + dstd
                    plt.fill_between(list(dx), lower, upper, color="green", alpha=0.16, label="Disturbance ±1 std")

        # Plot bad prior on same x-axis
        if summary_bad_prior is not None and param_group in summary_bad_prior:
            bad_prior_x = []
            bad_prior_y = []
            bad_prior_vars = []

            for item in summary_bad_prior[param_group]:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                if name in x_map:
                    bad_prior_x.append(x_map[name])
                    bad_prior_y.append(rate)
                    bad_prior_vars.append(var)

            if bad_prior_x:
                pairs = sorted(zip(bad_prior_x, bad_prior_y, bad_prior_vars), key=lambda t: t[0])
                bx, by, bvar = zip(*pairs)
                plt.plot(
                    list(bx),
                    list(by),
                    marker="^",
                    linestyle="-",
                    color="firebrick",
                    linewidth=2,
                    label="Bad Prior",
                )
                if bvar:
                    bstd = np.sqrt(np.array(bvar))
                    if metric == "success_rate":
                        lower = np.clip(np.array(by) - bstd, 0.0, 1.0)
                        upper = np.clip(np.array(by) + bstd, 0.0, 1.0)
                    else:
                        lower = np.array(by) - bstd
                        upper = np.array(by) + bstd
                    plt.fill_between(list(bx), lower, upper, color="firebrick", alpha=0.14, label="Bad Prior ±1 std")
        # plt.ylim(0, 1.05)
        #Make adjustable YLim
        # plt.ylim(bottom=0)
        plt.xticks(x, param_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        safe_name = param_group.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"{metric}_{sweep_label}_{safe_name}.png"
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
    summary_bad_prior,
    output_dir: Path,
    sweep_label="negative",
    metric="success_rate",
    stable_normal=None,
    stable_noise=None,
    stable_disturbance=None,
    stable_bad_prior=None,
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
        for item in sorted(param_data, key=lambda x: x[0]):
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
    if success_vars:
        stds = np.sqrt(np.array(success_vars))
        print("stds:", stds,"len", len(stds))
        # Print parameter names with their corresponding stds for debugging
        print("Parameter stds mapping:")
        for i, (param_name, std_val) in enumerate(zip(param_names, stds)):
            marker = " <-- initial_uncertainty_depth" if "initial_uncertainty_depth" in param_name else ""
            print(f"  [{i}] {param_name}: {std_val:.6f}{marker}")
        if metric == "success_rate":
            lower = np.clip(np.array(success_rates) - stds, 0.0, 1.0)
            upper = np.clip(np.array(success_rates) + stds, 0.0, 1.0)
        else:
            lower = np.array(success_rates) - stds
            upper = np.array(success_rates) + stds
        plt.fill_between(x, lower, upper, color="steelblue", alpha=0.15, label="Normal ±1 std")
    stable_normal = stable_normal or set()
    stable_noise = stable_noise or set()
    stable_disturbance = stable_disturbance or set()
    stable_bad_prior = stable_bad_prior or set()
    fully_stable_points = set()
    if sweep_label == "all":
        fully_stable_points = stable_normal & stable_noise & stable_disturbance & stable_bad_prior

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
            if nvar:
                nstd = np.sqrt(np.array(nvar))
                if metric == "success_rate":
                    lower = np.clip(np.array(ny) - nstd, 0.0, 1.0)
                    upper = np.clip(np.array(ny) + nstd, 0.0, 1.0)
                else:
                    lower = np.array(ny) - nstd
                    upper = np.array(ny) + nstd
                plt.fill_between(list(nx), lower, upper, color="orange", alpha=0.12, label="Noise ±1 std")
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
            if dvar:
                dstd = np.sqrt(np.array(dvar))
                if metric == "success_rate":
                    lower = np.clip(np.array(dy) - dstd, 0.0, 1.0)
                    upper = np.clip(np.array(dy) + dstd, 0.0, 1.0)
                else:
                    lower = np.array(dy) - dstd
                    upper = np.array(dy) + dstd
                plt.fill_between(list(dx), lower, upper, color="green", alpha=0.10, label="Disturbance ±1 std")
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

    if summary_bad_prior is not None:
        bad_prior_x = []
        bad_prior_y = []
        bad_prior_vars = []
        bad_prior_names = []
        stable_bad_prior_x = []
        stable_bad_prior_y = []
        for param_group, param_data in summary_bad_prior.items():
            for item in param_data:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                key = (param_group, name)
                if key in x_map:
                    bad_prior_x.append(x_map[key])
                    bad_prior_y.append(rate)
                    bad_prior_vars.append(var)
                    bad_prior_names.append(name)
                    if sweep_label == "all" and key in stable_bad_prior:
                        stable_bad_prior_x.append(x_map[key])
                        stable_bad_prior_y.append(rate)

        if bad_prior_x:
            pairs = sorted(zip(bad_prior_x, bad_prior_y, bad_prior_vars, bad_prior_names), key=lambda t: t[0])
            bx, by, bvar, bnames = zip(*pairs)
            plt.plot(
                list(bx),
                list(by),
                marker="^",
                linestyle="-",
                linewidth=2,
                color="firebrick",
                label="Bad Prior",
            )
            if bvar:
                bstd = np.sqrt(np.array(bvar))
                print("bstd:", bstd, "len", len(bstd))
                # Print bad_prior parameter names with their corresponding stds
                print("Bad Prior parameter stds mapping:")
                for i, (param_name, std_val) in enumerate(zip(bnames, bstd)):
                    marker = " <-- initial_uncertainty_depth" if "initial_uncertainty_depth" in param_name else ""
                    print(f"  [{i}] {param_name}: {std_val:.6f}{marker}")
                if metric == "success_rate":
                    lower = np.clip(np.array(by) - bstd, 0.0, 1.0)
                    upper = np.clip(np.array(by) + bstd, 0.0, 1.0)
                else:
                    lower = np.array(by) - bstd
                    upper = np.array(by) + bstd
                plt.fill_between(list(bx), lower, upper, color="firebrick", alpha=0.10, label="Bad Prior ±1 std")
        if stable_bad_prior_x:
            plt.plot(
                stable_bad_prior_x,
                stable_bad_prior_y,
                marker="^",
                linestyle="None",
                markersize=9,
                markerfacecolor="firebrick",
                markeredgecolor="black",
                markeredgewidth=2,
                color="firebrick",
                label="_nolegend_",
            )

    title_map = {
        "success_rate": "Success Rate",
        "avg_timesteps": "Average Timesteps (successes)",
        "avg_error": "Average Error",
        "avg_anchor_error": "Average Anchor Error",
        "avg_kinematic_axis_error": "Average Kinematic Axis Error",
    }
    ylabel = title_map.get(metric, metric)
    plt.title(f"{ylabel} by All Parameters ({sweep_label.capitalize()} Sweep) - Combined")
    plt.xlabel("Parameter Name")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    # plt.ylim(bottom=0)
    
    # Identify parameters with above-average standard deviations (for positive sweep)
    high_std_params = set()
    if sweep_label == "positive":
        param_stds = {}  # (param_group, param_name) -> list of stds
        
        # Collect all std values across all scenarios
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

        if summary_bad_prior is not None:
            for param_group, param_data in summary_bad_prior.items():
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
        Line2D([0], [0], marker="^", color="firebrick", linestyle="None", label="Bad Prior"),
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
                    label="X label: Normal, Noise, Disturbance, and Bad Prior all stable",
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
    output_file = output_dir / f"{metric}_{sweep_label}_combined.png"
    plt.savefig(output_file)
    plt.close()


def plot_anchor_error_outcome_summary(summary_bad_prior, output_dir: Path, sweep_label="negative", metric="avg_anchor_error"):
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_bad_prior is None:
        return

    for param_group, param_data in summary_bad_prior.items():
        if len(param_data) == 0:
            continue

        param_names = [item[0] for item in param_data]
        x = list(range(len(param_names)))

        success_means = [item[1] for item in param_data]
        success_vars = [item[2] if len(item) > 2 else None for item in param_data]
        failed_means = [item[4] for item in param_data]
        failed_vars = [item[5] if len(item) > 5 else None for item in param_data]

        plt.figure(figsize=(12, 6))

        def plot_series(values, variances, color, marker, label):
            points = [
                (idx, value, variance)
                for idx, (value, variance) in enumerate(zip(values, variances))
                if value is not None
            ]
            if not points:
                return

            xs, ys, vs = zip(*points)
            plt.plot(
                list(xs),
                list(ys),
                marker=marker,
                linestyle="-",
                color=color,
                linewidth=2,
                label=label,
            )

            if any(variance is not None for variance in vs):
                stds = np.sqrt(np.array([0.0 if variance is None else variance for variance in vs]))
                lower = np.array(ys) - stds
                upper = np.array(ys) + stds
                plt.fill_between(list(xs), lower, upper, color=color, alpha=0.15, label=f"{label} ±1 std")

        plot_series(success_means, success_vars, "steelblue", "o", "Successful trials")
        plot_series(failed_means, failed_vars, "firebrick", "x", "Failed trials")

        plt.xticks(x, param_names, rotation=45, ha="right")
        plt.title(f"{{metric.capitalize()}} by Trial Outcome ({sweep_label.capitalize()} Sweep) - {param_group}")
        plt.xlabel("Parameter Name")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.grid(True, linestyle="--", alpha=0.5, axis="y")
        plt.legend()
        plt.tight_layout()

        safe_name = param_group.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f"{metric}_{sweep_label}_{safe_name}_success_vs_failure.png"
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
        "--db-path-bad-prior",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--sweep-label",
        choices=["negative", "zero", "positive","x1000", "all","Non_reasonable"],
        default="negative",
        help="Which sweep values to analyze: negative, zero, positive, or all.",
    )
    parser.add_argument(
        "--metric",
        choices=["success_rate", "avg_timesteps", "avg_error", "avg_anchor_error", "avg_kinematic_axis_error"],
        default="success_rate",
        help="Metric to compute and plot.",
    )
    parser.add_argument(
        "--compare-success-failure",
        action="store_true",
        help="Compare anchor error for successful versus failed trials in the bad_prior database.",
    )
    parser.add_argument(
        "--print-si",
        action="store_true",
        help="Print single-sweep standard deviations (S_i) for drawer_position.initial_uncertainty_depth and _scale from bad_prior DB and exit.",
    )
    parser.add_argument(
        "--print-top-std",
        action="store_true",
        help="Print the top-10 1D sweep standard deviations for bad_prior timesteps and exit.",
    )
    parser.add_argument(
        "--include-failed-trials",
        action="store_true",
        help="Include failed trials when computing per-label means (affects avg_* metrics and S_i).",
    )
    args = parser.parse_args()

    if args.compare_success_failure and args.metric != "avg_anchor_error" and args.metric != "avg_error" and args.metric != "avg_kinematic_axis_error":
        raise SystemExit("--compare-success-failure is only supported with --metric avg_anchor_error, avg_error, or avg_kinematic_axis_error")

    all_rows_normal = []
    if args.db_path_normal:
        all_rows_normal = read_results_from_db(Path(args.db_path_normal))
    all_rows_noise = []
    if args.db_path_noise:
        all_rows_noise = read_results_from_db(Path(args.db_path_noise))
    all_rows_disturbance = []
    if args.db_path_disturbance:
        all_rows_disturbance = read_results_from_db(Path(args.db_path_disturbance))
    all_rows_bad_prior = []
    if args.db_path_bad_prior:
        all_rows_bad_prior = read_results_from_db(Path(args.db_path_bad_prior))

    # Optionally print S_i values for specific drawer_position parameters from the bad_prior DB
    if args.print_si:
        # Load the emergent interactions reader dynamically to reuse its read_single_sweep_series
        try:
            from importlib.util import spec_from_file_location, module_from_spec
            mod_path = Path(__file__).resolve().parents[1] / "Pairwise_Sweep" / "Evaluation_Scripts" / "emergent_interactions_single_Scenario.py"
            spec = spec_from_file_location("eis", str(mod_path))
            eis = module_from_spec(spec)
            spec.loader.exec_module(eis)
        except Exception as exc:
            raise SystemExit(f"Failed to load emergent interactions reader: {exc}")

        # Determine DB path (use provided bad-prior DB if given, otherwise default location)
        if args.db_path_bad_prior:
            db_path = Path(args.db_path_bad_prior)
        else:
            db_path = Path(__file__).resolve().parents[1] / "data" / "bad_prior" / "experiment_store.db"

        si_values, one_d_successes_by_parameter, standard_baseline = eis.read_single_sweep_series(
            db_path, metrics="timesteps", include_failed_trials=args.include_failed_trials
        )
        targets = [
            "drawer_position.initial_uncertainty_depth",
            "drawer_position.initial_uncertainty_scale",
        ]
        print(f"Single-sweep standard deviations (S_i) for bad_prior (timesteps, include_failed_trials={args.include_failed_trials}):")
        for t in targets:
            print(f" - {t}: S_i = {si_values.get(t, float('nan'))}, 1D means = {one_d_successes_by_parameter.get(t)})")
        return

    if args.print_top_std:
        if not args.db_path_bad_prior:
            db_path = Path(__file__).resolve().parents[1] / "data" / "bad_prior" / "experiment_store.db"
        else:
            db_path = Path(args.db_path_bad_prior)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT t.timesteps, t.metadata FROM trials t WHERE t.metadata IS NOT NULL").fetchall()
        param_data = {}
        for ts, md in rows:
            try:
                metadata = json.loads(md)
            except Exception:
                continue
            if metadata.get("experiment_type") != "1d_sweep":
                continue
            parameter = metadata.get("parameter")
            label = normalize_sweep_label(metadata.get("label"))
            if label not in {"0.2x", "0.5x", "2x", "5x", "standard"}:
                continue
            if ts is None:
                continue
            try:
                value = float(ts)
            except (TypeError, ValueError):
                continue
            param_data.setdefault(parameter, {}).setdefault(label, []).append(value)
        conn.close()

        stds = []
        for parameter, grouped in param_data.items():
            label_means = {label: float(np.mean(values)) for label, values in sorted(grouped.items())}
            if len(label_means) > 1:
                stds.append((parameter, float(np.std(list(label_means.values()))), label_means))

        print("Top 10 1D sweep std devs for bad_prior (timesteps):")
        for parameter, std_dev, label_means in sorted(stds, key=lambda x: x[1], reverse=True)[:10]:
            print(f" - {parameter}: std dev = {std_dev}, label means = {label_means}")
        return

    if args.compare_success_failure:
        if not args.db_path_bad_prior:
            raise SystemExit("--compare-success-failure requires --db-path-bad-prior")

        outcome_summary = build_anchor_error_outcome_summary(all_rows_bad_prior, sweep_label=args.sweep_label, metric=args.metric)
        if not outcome_summary:
            raise SystemExit("No valid bad_prior rows found to plot.")

        if args.metric == "avg_anchor_error":
            out_dir = Path.cwd() / f"avg_anchor_error_bad_prior_success_failure_comparison_plots_{args.sweep_label}"
        elif args.metric == "avg_error":
            out_dir = Path.cwd() / f"avg_error_bad_prior_success_failure_comparison_plots_{args.sweep_label}"
        elif args.metric == "avg_kinematic_axis_error":
            out_dir = Path.cwd() / f"avg_kinematic_axis_error_bad_prior_success_failure_comparison_plots_{args.sweep_label}"
        plot_anchor_error_outcome_summary(outcome_summary, out_dir, sweep_label=args.sweep_label, metric=args.metric)
        print(f"Saved success/failure {args.metric.replace('_', ' ')} plots to {out_dir}")
        return
    # else:
    #     base_dir = Path(args.results_dir)
    #     csv_files = sorted(base_dir.glob("*.csv"))
    #     if not csv_files:
    #         raise SystemExit(f"No CSV files found in {base_dir}")
    #     for csv_path in csv_files:
    #         all_rows.extend(read_results(csv_path))

    summary_normal = build_summary(all_rows_normal, sweep_label=args.sweep_label, metric=args.metric, include_failed_trials=args.include_failed_trials)
    summary_noise = build_summary(all_rows_noise, sweep_label=args.sweep_label, metric=args.metric, include_failed_trials=args.include_failed_trials)
    summary_disturbance = build_summary(all_rows_disturbance, sweep_label=args.sweep_label, metric=args.metric, include_failed_trials=args.include_failed_trials)
    summary_bad_prior = build_summary(all_rows_bad_prior, sweep_label=args.sweep_label, metric=args.metric, include_failed_trials=args.include_failed_trials, debug_param_group="bad_prior")
    if not summary_normal:
        raise SystemExit("No valid rows found to plot.")

    stable_normal = set()
    stable_noise = set()
    stable_disturbance = set()
    stable_bad_prior = set()
    if args.sweep_label == "all":
        stable_normal = find_stable_all_points(
            summary_normal,
            build_summary(all_rows_normal, sweep_label="negative", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_normal, sweep_label="zero", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_normal, sweep_label="positive", metric=args.metric, include_failed_trials=args.include_failed_trials),
        )
        stable_noise = find_stable_all_points(
            summary_noise,
            build_summary(all_rows_noise, sweep_label="negative", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_noise, sweep_label="zero", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_noise, sweep_label="positive", metric=args.metric, include_failed_trials=args.include_failed_trials),
        )
        stable_disturbance = find_stable_all_points(
            summary_disturbance,
            build_summary(all_rows_disturbance, sweep_label="negative", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_disturbance, sweep_label="zero", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_disturbance, sweep_label="positive", metric=args.metric, include_failed_trials=args.include_failed_trials),
        )
        stable_bad_prior = find_stable_all_points(
            summary_bad_prior,
            build_summary(all_rows_bad_prior, sweep_label="negative", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_bad_prior, sweep_label="zero", metric=args.metric, include_failed_trials=args.include_failed_trials),
            build_summary(all_rows_bad_prior, sweep_label="positive", metric=args.metric, include_failed_trials=args.include_failed_trials),
        )

    # output folder includes metric name
    metric_suffix = args.metric
    if args.sweep_label == "positive":
        out_dir = Path.cwd() / f"{metric_suffix}_comparison_plots_positive"
    elif args.sweep_label == "zero":
        out_dir = Path.cwd() / f"{metric_suffix}_comparison_plots_zero"
    elif args.sweep_label == "x1000":
        out_dir = Path.cwd() / f"{metric_suffix}_comparison_plots_x1000"
    elif args.sweep_label == "all":
        out_dir = Path.cwd() / f"{metric_suffix}_comparison_plots_all"
    else:
        out_dir = Path.cwd() / f"{metric_suffix}_comparison_plots_negative"

    plot_summary(
        summary_normal,
        summary_noise,
        summary_disturbance,
        summary_bad_prior,
        out_dir,
        sweep_label=args.sweep_label,
        metric=args.metric,
    )
    plot_combined_summary(
        summary_normal,
        summary_noise,
        summary_disturbance,
        summary_bad_prior,
        out_dir,
        sweep_label=args.sweep_label,
        metric=args.metric,
        stable_normal=stable_normal,
        stable_noise=stable_noise,
        stable_disturbance=stable_disturbance,
        stable_bad_prior=stable_bad_prior,
    )
    print(f"Saved plots for {len(summary_normal)} parameter groups ({args.sweep_label}, metric={args.metric}) to {out_dir}")


if __name__ == "__main__":
    main()