import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from .utils import (
    load_sweep_plan,
    value_index_to_label,
    short_label,
    normalize_one_d_label,
    parse_pair,
    _filter_rows_by_trial_window,
    _get_matplotlib_pyplot,
    resolve_parameter_selection
)
from .anova_analysis import calculate_ANOVA_interaction_score, compute_per_parameter_stats
from .heatmap_generation import heatmap_ss_ab
from .early_late_comparison import plot_early_vs_late_scatter, compute_window_metrics_summary
from .residual_plots import plot_si_vs_residuals

SI_SEPERATE = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}


def read_pairwise_job_series(db_path, metrics="success", include_failed_trials=False, trial_window="all"):
    conn = sqlite3.connect(db_path)
    rows = []
    if metrics == "success":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    elif metrics == "anchor_error":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.anchor_error, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    elif metrics == "timesteps":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.timesteps, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    elif metrics == "epistemicity":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.trace_ids, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    else:
        conn.close()
        raise ValueError(f"Unsupported metrics: {metrics}")

    rows = _filter_rows_by_trial_window(rows, trial_window=trial_window)
    print("rows", len(rows))
    # If computing epistemicity, preload trace_catalog to reduce DB queries
    trace_catalog_map = None
    if metrics == "epistemicity":
        try:
            trace_catalog_map = {}
            for trace_id, trace_text in conn.execute("SELECT trace_id, trace_text FROM trace_catalog").fetchall():
                try:
                    trace_catalog_map[int(trace_id)] = trace_text
                except Exception:
                    continue
        except Exception:
            trace_catalog_map = None

    # If computing epistemicity, preload trace_catalog to reduce DB queries
    trace_catalog_map = None
    if metrics == "epistemicity":
        try:
            trace_catalog_map = {}
            for trace_id, trace_text in conn.execute("SELECT trace_id, trace_text FROM trace_catalog").fetchall():
                try:
                    trace_catalog_map[int(trace_id)] = trace_text
                except Exception:
                    continue
        except Exception:
            trace_catalog_map = None

    # Group pairwise data by (pair, label_a, label_b) to preserve label information
    by_pair_labels = defaultdict(lambda: defaultdict(list))
    by_job = defaultdict(list)

    for row in rows:
        if metrics == "success":
            try:
                _, _, metric_value, metadata_json = row
            except Exception:
                continue
        else:
            # rows are (config_id, seed, metric, success, metadata)
            try:
                if metrics == "epistemicity":
                    _, _, trace_ids_json, success_flag, metadata_json = row
                else:
                    _, _, metric_value, success_flag, metadata_json = row
            except Exception:
                continue
            # Optionally skip failed trials
            if not include_failed_trials and not success_flag:
                continue
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            continue

        if metadata.get("experiment_type") != "pairwise_sweep":
            continue

        parameters = parse_pair(metadata)
        if parameters is None or len(parameters) != 2:
            continue

        pair = tuple(sorted(parameters))
        try:
            if metrics == "epistemicity":
                metric_val = _compute_epistemicity_from_trace_ids(conn, trace_ids_json, trace_catalog_map=(trace_catalog_map if 'trace_catalog_map' in locals() else None))
            else:
                metric_val = float(metric_value)
        except (TypeError, ValueError):
            continue
        # Try to extract label information from metadata and normalize labels
        sweep_labels = metadata.get("sweep_labels", "")
        if sweep_labels:
            raw_parts = [part.strip() for part in str(sweep_labels).split(",") if part.strip()]
            if len(raw_parts) == 2:
                # Convert value_X format to actual labels
                label_a = value_index_to_label(raw_parts[0])
                label_b = value_index_to_label(raw_parts[1])
                label_a = normalize_one_d_label(label_a)
                label_b = normalize_one_d_label(label_b)
                # Only include known labels
                if label_a in ONE_D_LABELS_TO_INCLUDE and label_b in ONE_D_LABELS_TO_INCLUDE:
                    # Normalize label order to match sorted parameter order
                    if parameters[0] == pair[1]:
                        label_a, label_b = label_b, label_a
                    by_pair_labels[pair][(label_a, label_b)].append(metric_val)
        
        # Also keep the old grouping for backward compatibility
        job_key = (
            pair,
            str(metadata.get("sweep_labels", "")),
            str(metadata.get("sweep_values", "")),
        )
        by_job[job_key].append(metric_val)
        

    # Build pair_vectors with label information
    pair_vectors = {}
    for pair in by_pair_labels:
        pair_vectors[pair] = {}
        for (label_a, label_b), successes in by_pair_labels[pair].items():
            if successes:
                pair_vectors[pair][(label_a, label_b)] = float(np.mean(np.asarray(successes, dtype=float)))

    print(f"Read {len(pair_vectors)} parameter pairs with label combinations from database.")
    parameters = sorted({param for pair in pair_vectors for param in pair})
    conn.close()
    return parameters, pair_vectors


def read_single_sweep_series(db_path, metrics="success", include_failed_trials=False, trial_window="all"):
    """
    Read 1d sweep trials and compute Si values.
    Returns both Si values and organized success data by parameter and label.
    """
    conn = sqlite3.connect(db_path)
    rows = []
    if metrics == "success":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    elif metrics == "anchor_error":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.anchor_error, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    elif metrics == "timesteps":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.timesteps, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    elif metrics == "epistemicity":
        rows = conn.execute(
            """
            SELECT t.config_id, t.seed, t.trace_ids, t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    else:
        conn.close()
        raise ValueError(f"Unsupported metrics: {metrics}")

    rows = _filter_rows_by_trial_window(rows, trial_window=trial_window)

    # If computing epistemicity, preload trace_catalog to reduce DB queries
    trace_catalog_map = None
    if metrics == "epistemicity":
        try:
            trace_catalog_map = {}
            for trace_id, trace_text in conn.execute("SELECT trace_id, trace_text FROM trace_catalog").fetchall():
                try:
                    trace_catalog_map[int(trace_id)] = trace_text
                except Exception:
                    continue
        except Exception:
            trace_catalog_map = None

    # Group by parameter -> label -> list of successes
    by_param_label = defaultdict(lambda: defaultdict(list))
    by_param_sweep = defaultdict(lambda: defaultdict(list))
    standard_baseline_successes = []

    for row in rows:
        if metrics == "success":
            try:
                _, _, metric_value, metadata_json = row
            except Exception:
                continue
        else:
            try:
                if metrics == "epistemicity":
                    _, _, trace_ids_json, success_flag, metadata_json = row
                else:
                    _, _, metric_value, success_flag, metadata_json = row
            except Exception:
                continue
            if not include_failed_trials and not success_flag:
                continue
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            continue

        if metadata.get("experiment_type") != "1d_sweep":
            continue

        parameter = metadata.get("parameter")
        if not parameter:
            continue

        sweep_value = metadata.get("sweep_value")
        label = normalize_one_d_label(metadata.get("label", ""))
        
        # Filter to only included labels
        if label not in ONE_D_LABELS_TO_INCLUDE:
            continue
        
        # Standard baseline trials may not have a sweep value.
        if sweep_value is None and label != "standard":
            continue

        try:
            if metrics == "epistemicity":
                metric_val = _compute_epistemicity_from_trace_ids(conn, trace_ids_json, trace_catalog_map=(trace_catalog_map if 'trace_catalog_map' in locals() else None))
            else:
                metric_val = float(metric_value)
        except (TypeError, ValueError):
            # Skip rows with non-numeric metric values
            continue

        if parameter == "standard.standard" and label == "standard":
            standard_baseline_successes.append(metric_val)
            continue

        by_param_label[parameter][label].append(metric_val)
        by_param_sweep[parameter][(sweep_value, label)].append(metric_val)

    # Compute mean success per parameter per label for use in Hi computation
    one_d_successes_by_parameter = {}
    for parameter in by_param_label:
        one_d_successes_by_parameter[parameter] = {}
        for label in by_param_label[parameter]:
            successes = by_param_label[parameter][label]
            if successes:
                one_d_successes_by_parameter[parameter][label] = float(np.mean(np.asarray(successes, dtype=float)))

    if standard_baseline_successes:
        standard_baseline = float(np.mean(np.asarray(standard_baseline_successes, dtype=float)))
        for parameter in one_d_successes_by_parameter:
            one_d_successes_by_parameter[parameter].setdefault("standard", standard_baseline)
    else:
        standard_baseline = np.nan

    # For each parameter, compute S_i from the mean success rates per label.
    si_values = {}
    if metrics == "epistemicity":
        # SI should be the standard deviation of trial-level epistemicity values
        for parameter, label_map in by_param_label.items():
            vals = []
            for label_vals in label_map.values():
                vals.extend([float(v) for v in label_vals if not np.isnan(v)])
            if vals:
                si_values[parameter] = float(np.std(np.asarray(vals, dtype=float)))
            else:
                si_values[parameter] = 0.0
    else:
        for parameter, label_means in one_d_successes_by_parameter.items():
            rates = [float(value) for value in label_means.values() if not np.isnan(value)]
            if len(rates) > 1:
                si_values[parameter] = float(np.sqrt(np.var(np.asarray(rates, dtype=float))))
            elif rates:
                # If only one label mean is available, variance is 0
                si_values[parameter] = 0.0

    conn.close()

    return si_values, one_d_successes_by_parameter, standard_baseline


def _trace_text_has_uncertainty_component(trace_text):
    if trace_text is None:
        return False
    text = str(trace_text).strip()
    if text in {"", "[]", "null", "None", "NaN", "nan"}:
        return False
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = text

    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            token = str(item).lower()
            if "uncertainty" in token:
                return True
        return False

    if isinstance(parsed, str):
        return "uncertainty" in parsed.lower()

    return bool(parsed)


def _compute_epistemicity_from_trace_ids(conn, trace_ids, trace_catalog_map=None):
    if not trace_ids:
        return 0.0
    try:
        trace_ids = json.loads(trace_ids) if isinstance(trace_ids, str) else list(trace_ids)
    except Exception:
        return 0.0

    if not trace_ids:
        return 0.0

    total_steps = len(trace_ids)
    gradient_steps = 0

    # If a preloaded trace_catalog_map is provided, use it to avoid a DB roundtrip per trace id
    if trace_catalog_map is not None:
        for trace_id in trace_ids:
            try:
                trace_id_int = int(trace_id)
            except (TypeError, ValueError):
                continue
            trace_text = trace_catalog_map.get(trace_id_int)
            if _trace_text_has_uncertainty_component(trace_text):
                gradient_steps += 1
    else:
        for trace_id in trace_ids:
            try:
                trace_id_int = int(trace_id)
            except (TypeError, ValueError):
                continue
            row = conn.execute("SELECT trace_text FROM trace_catalog WHERE trace_id = ?", (trace_id_int,)).fetchone()
            trace_text = row[0] if row is not None else None
            if _trace_text_has_uncertainty_component(trace_text):
                gradient_steps += 1
    if total_steps == 0:
        return 0.0
    return float(gradient_steps) / float(total_steps)

def compute_hi(parameters, pair_vectors, one_d_successes_by_parameter=None, standard_baseline=None):
    """
    Compute Hi values incorporating both pairwise and 1D sweep data.
    For each parameter pair and label combination, use pairwise data if available,
    otherwise fall back to 1D data. Creates 25 values per pair (5x5 labels minus duplicate standard).
    """
    if one_d_successes_by_parameter is None:
        one_d_successes_by_parameter = {}
    
    labels = ["0.2x", "0.5x", "2x", "5x", "standard"]
    
    # Build augmented pair vectors that combine pairwise and 1D data
    augmented_pair_vectors = {}
    augmented_pair_diagnostics = {}
    for param_a, param_b in [(p1, p2) for p1 in parameters for p2 in parameters if p1 != p2]:
        pair_key = tuple(sorted([param_a, param_b]))
        all_values = []
        present_label_pairs = []
        missing_label_pairs = []
        
        # Get pairwise data for this pair if it exists (dict of label combinations)
        pairwise_by_labels = pair_vectors.get(pair_key, {})
        
        # Build 5x5 grid of label combinations (25 total)
        for label_a in labels:
            for label_b in labels:
                # Try to get pairwise data for this specific label combination
                pairwise_value = pairwise_by_labels.get((label_a, label_b), np.nan)
                
                if not np.isnan(pairwise_value):
                    # Use pairwise data for this specific label combination
                    all_values.append(pairwise_value)
                    present_label_pairs.append((label_a, label_b, "pairwise"))
                else:
                    # Only allow fallback to 1D when at least one label is 'standard'.
                    # If neither label is 'standard' and pairwise data is missing,
                    # do not use 1D fallbacks for this label combination.
                    if label_a == "standard" and label_b == "standard":
                        all_values.append(standard_baseline)
                        present_label_pairs.append((label_a, label_b, "standard_baseline"))
                    elif label_a == "standard" and label_b != "standard":
                        val_b = one_d_successes_by_parameter.get(param_b, {}).get(label_b, np.nan)
                        if not np.isnan(val_b):
                            all_values.append(float(val_b))
                            present_label_pairs.append((label_a, label_b, f"1d:{short_label(param_b)}"))
                        else:
                            all_values.append(np.nan)
                            missing_label_pairs.append((label_a, label_b, short_label(param_b)))
                    elif label_b == "standard" and label_a != "standard":
                        val_a = one_d_successes_by_parameter.get(param_a, {}).get(label_a, np.nan)
                        if not np.isnan(val_a):
                            all_values.append(float(val_a))
                            present_label_pairs.append((label_a, label_b, f"1d:{short_label(param_a)}"))
                        else:
                            all_values.append(np.nan)
                            missing_label_pairs.append((label_a, label_b, short_label(param_a)))
                    else:
                        all_values.append(np.nan)
                        missing_label_pairs.append((label_a, label_b, "pairwise_only"))
                    
        if all_values:
            augmented_pair_vectors[pair_key] = np.asarray(all_values, dtype=float)
            augmented_pair_diagnostics[pair_key] = {
                "present": present_label_pairs,
                "missing": missing_label_pairs,
                "value_count": len(all_values),
                "valid_value_count": sum(not np.isnan(value) for value in all_values),
            }

    baseline_values = {}
    for parameter in parameters:
        collected = []
        for pair, values in augmented_pair_vectors.items():
            if parameter in pair:
                collected.extend(values.tolist())
        baseline_values[parameter] = float(np.mean(collected)) if collected else np.nan

    hi_values = {}
    variation_table = {}
    raw_variation_table = {}
    for parameter in parameters:
        partner_variances = []
        variation_table[parameter] = {}
        raw_variation_table[parameter] = {}
        for partner in parameters:
            if partner == parameter:
                continue

            pair_key = tuple(sorted([parameter, partner]))
            series = augmented_pair_vectors.get(pair_key)
            if series is None or series.size == 0:
                continue

            partner_baseline = baseline_values.get(partner, np.nan)
            if np.isnan(partner_baseline):
                continue

            deviations = series - partner_baseline
            variance = float(np.var(deviations))
            raw_variance = float(np.var(series))
            partner_variances.append(variance)
            variation_table[parameter][partner] = variance
            raw_variation_table[parameter][partner] = raw_variance

        hi_values[parameter] = float(np.mean(partner_variances)) if partner_variances else np.nan

    return baseline_values, hi_values, variation_table, raw_variation_table, augmented_pair_vectors, augmented_pair_diagnostics

from .anova_analysis import calculate_ANOVA_interaction_score, compute_per_parameter_stats


def plot_si_per_parameter(si_values, output_path):
    plt = _get_matplotlib_pyplot()
    parameters = sorted(si_values.keys())
    si_vals = [si_values[param] for param in parameters]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(parameters)), 6.0))
    ax.bar(range(len(parameters)), si_vals, color="skyblue", alpha=0.7)
    ax.set_xticks(range(len(parameters)))
    ax.set_xticklabels([short_label(param) for param in parameters], rotation=90, fontsize=8)
    ax.set_ylabel("S_i")
    ax.set_title("S_i per Parameter")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def scatter_plot_interaction_vs_total(min_ss_total, parameters, ss_total_table, ss_ab_table, output_path):
    plt = _get_matplotlib_pyplot()
    x_vals = []
    y_vals = []
    labels = []
    pairs = []
    for pair in ss_total_table:
        total = ss_total_table[pair]
        interaction = ss_ab_table.get(pair, np.nan)
        if not np.isnan(total) and not np.isnan(interaction):
            x_vals.append(total)
            y_vals.append(interaction)
            labels.append(f"{pair[0]} & {pair[1]}")
            pairs.append(pair)

    residuals_by_pair = {}
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_vals, y_vals, alpha=0.7)

    if len(x_vals) >= 2:
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        y_fit = slope * x_arr + intercept
        sort_idx = np.argsort(x_arr)
        ax.plot(x_arr[sort_idx], y_fit[sort_idx], color="red", linestyle="--", label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")

        residuals = y_arr - y_fit
        abs_residuals = np.abs(residuals)
        ranked = np.argsort(-abs_residuals)
        top_count = min(20, len(abs_residuals))

        short_labels = [f"{short_label(pair.split(' & ')[0])} & {short_label(pair.split(' & ')[1])}" for pair in labels]
        for idx in ranked[:top_count]:
            ax.annotate(
                short_labels[idx],
                (x_arr[idx], y_arr[idx]),
                fontsize=7,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ss_res = np.sum((y_arr - y_fit) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.legend()
    else:
        for idx, pair in enumerate(pairs):
            residuals_by_pair[pair] = np.nan

    ax.set_xlabel("Total Sum of Squares (SS_total)")
    ax.set_ylabel("Interaction Sum of Squares (SS_AB)")
    ax.set_title("Interaction vs Total Sum of Squares")
    ax.grid(True)
    # Plot minimum SS_total threshold
    if min_ss_total is not None and min_ss_total > 0:
        ax.axvline(
            x=min_ss_total,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Min SS_total = {min_ss_total:.3e}"
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return residuals_by_pair





     

def main():
    parser = argparse.ArgumentParser(description="Plot emergent interactions from a single experiment database.")
    parser.add_argument("--db-path", type=Path, required=False, help="SQLite database path containing pairwise and single-sweep trials.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to pairwise_emergent_interaction_hi.png next to the database.",
    )
    parser.add_argument(
        "--debug-success-rates",
        action="store_true",
        help="Print raw 1D and pairwise metric value breakdowns and exit.",
    )
    parser.add_argument(
        "--debug-parameter-a",
        type=str,
        default=None,
        help="First parameter name for pairwise success-rate debug.",
    )
    parser.add_argument(
        "--debug-parameter-b",
        type=str,
        default=None,
        help="Second parameter name for pairwise success-rate debug.",
    )
    parser.add_argument(
        "--highlight-group",
        action="append",
        default=[],
        metavar="P1,P2,...",
        help="Comma-separated parameter list to circle on the scatter plot. Repeat the flag for multiple groups.",
    )
    parser.add_argument(
        "--x-threshold",
        type=float,
        default=None,
        help="X-axis (SI) threshold for auto-highlighting. If both x and y thresholds are provided, parameters above both are highlighted.",
    )
    parser.add_argument(
        "--y-threshold",
        type=float,
        default=None,
        help="Y-axis (residual) threshold for auto-highlighting. If both x and y thresholds are provided, parameters above both are highlighted.",
    )
    parser.add_argument(
        "--y-max-threshold",
        type=float,
        default=None,
        help="Y-axis threshold for Max residual labeling. Controls which Max residual points get labeled (separate from mean residual threshold).",
    )
    parser.add_argument(
        "--metric",
        choices=["success", "anchor_error", "timesteps", "grasp", "epistemicity"],
        default="success",
        help="Metric to analyze: success (default), anchor_error, timesteps, grasp, or epistemicity.",
    )
    parser.add_argument(
        "--include-failed-trials",
        action="store_true",
        help="When analyzing timesteps, include failed trials (otherwise only successful trials are considered).",
    )
    args = parser.parse_args()

    if args.db_path is None or not args.db_path.exists():
        db_path = Path(__file__).parent.parent.parent.parent / "Ignored_files" / "data" / "noise" / "experiment_store.db"
        print(f"Database path not provided or does not exist. Using default: {db_path}")
        # raise SystemExit(f"Database not found: {args.db_path}")
    else:
        db_path = args.db_path.resolve()


    # Which metric to analyze
    print(f"Using metric: {args.metric}")

    # Determine output directory: use --output if provided (file or directory),
    # otherwise write to project_root/metrics/<metric>/
    project_root = Path(__file__).parent.parent.parent.parent / "Ignored_files"
    if args.output:
        out_arg = Path(args.output)
        # If user passed a directory, use it. If a file, use its parent.
        if str(out_arg).endswith(('/', '\\')) or out_arg.exists() and out_arg.is_dir():
            output_dir = out_arg
        else:
            output_dir = out_arg.parent
    else:
        output_dir = db_path.parent / "metrics" / args.metric

    output_dir.mkdir(parents=True, exist_ok=True)

    parameters, pairwise_series = read_pairwise_job_series(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
    )
    print(f"Read {len(parameters)} parameters and {len(pairwise_series)} pairwise series from database.")

    # Read single sweep data for Si values and 1D parameter data
    si_values, one_d_successes_by_parameter, standard_baseline = read_single_sweep_series(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
    )
    print(f"Read {len(si_values)} single-sweep Si values and 1D parameter data from database.")
    _, _, _, _, augmented_pair_vectors, augmented_pair_diagnostics = compute_hi(
        parameters, pairwise_series, one_d_successes_by_parameter, standard_baseline
    )
    # Build a consolidated list of parameter pairs that are missing any label combination values, missing means 0% success for at least one label combination
    missing_pairs = [
        pair for pair, diag in (augmented_pair_diagnostics or {}).items()
        if diag and diag.get("missing")
    ]
    if missing_pairs:
        print(f"Pairs missing label combinations (count={len(missing_pairs)}):")
        for p in missing_pairs:
            print(f" - {p}: missing {augmented_pair_diagnostics[p].get('missing')}")
        # Save to metric-specific output folder for easier inspection
        try:
            out_path = output_dir / "missing_pairs.txt"
            with open(out_path, "w", encoding="utf-8") as fh:
                for p in missing_pairs:
                    fh.write(f"{p}: {augmented_pair_diagnostics[p].get('missing')}\n")
            print(f"Wrote missing pairs to: {out_path}")
        except Exception:
            pass
    _, ss_total_table, ss_ab_table, ss_a_table, ss_b_table = calculate_ANOVA_interaction_score(
        parameters, augmented_pair_vectors, augmented_pair_diagnostics
    )

    print("number of pairs:", len(ss_total_table))
    print("number of exact zeros ss_total:", sum(1 for v in ss_total_table.values() if v == 0))

    min_ss_total = 1e-6 # just as long as not zero, to avoid division by zero in interaction score calculation

    # scatter_plot_interaction_vs_total(min_ss_total, parameters, ss_total_table, ss_ab_table, output_dir / "interaction_vs_total.png")

    # if file already exists don't overwrite it, just print a message and skip
    heatmap_path = output_dir / "ss_ab_heatmap.pdf"
    normalized_ssab_heatmap_path = output_dir / "ss_ab_heatmap_normalized_ssab.pdf"
    normalized_cohen_heatmap_path = output_dir / "ss_ab_heatmap_normalized_cohen.pdf"
    if heatmap_path.exists() or normalized_ssab_heatmap_path.exists() or normalized_cohen_heatmap_path.exists():
        print(f"Heatmap files already exist. Skipping heatmap generation.")
    else:
        try:
            # Generate normalized SS_AB and normalized Cohen heatmaps
            heatmap_ss_ab(ss_ab_table, heatmap_path, max_value=None, ss_a_table=ss_a_table, ss_b_table=ss_b_table, ss_total_table=ss_total_table)
        except ModuleNotFoundError as exc:
            if exc.name == "scipy":
                print("SciPy not installed; skipping SS_AB heatmap generation.")
            else:
                raise

    # Compute per-parameter max and mean residuals (raw SS_AB)
    max_ssab_s = {}
    mean_ssab_s = {}
    for (pa, pb), interaction in ss_ab_table.items():
        if np.isnan(interaction):
            continue
        max_ssab_s[pa] = max(max_ssab_s.get(pa, -np.inf), interaction)
        max_ssab_s[pb] = max(max_ssab_s.get(pa, -np.inf), interaction)
        mean_ssab_s[pa] = mean_ssab_s.get(pa, []) + [max(0, interaction)] # Store only non-negative residuals for mean computation
        mean_ssab_s[pb] = mean_ssab_s.get(pb, []) + [max(0, interaction)] # Store only non-negative residuals for mean computation
    mean_ssab_s = {p: float(np.mean(vals)) for p, vals in mean_ssab_s.items()} if mean_ssab_s else {}

    # Compute per-parameter max and mean SS_AB values
    max_ssab_s, mean_ssab_s, max_f2_s, mean_f2_s = compute_per_parameter_stats(ss_ab_table, ss_a_table, ss_b_table)

    # Sort from highest to lowest SS_ab before computing derivatives.
    mean_ssab_values = sorted(mean_ssab_s.values(), reverse=True)
    max_ssab_values = sorted(max_ssab_s.values(), reverse=True)

    # Early-vs-late comparison for sensitivity and residual magnitude.
    early_parameters, early_si_values, early_max_ssab_s, early_mean_ssab_s, early_max_f2_s, early_mean_f2_s = compute_window_metrics_summary(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
        trial_window="first5",
    )
    late_parameters, late_si_values, late_max_ssab_s, late_mean_ssab_s, late_max_f2_s, late_mean_f2_s = compute_window_metrics_summary(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
        trial_window="last5",
    )

    # Disabled: early vs late scatter plots
    # comparison_parameters = sorted(set(early_parameters) | set(late_parameters))
    # si_early = {parameter: early_si_values.get(parameter, np.nan) for parameter in comparison_parameters}
    # si_late = {parameter: late_si_values.get(parameter, np.nan) for parameter in comparison_parameters}
    # residual_early = {parameter: early_max_f2_s.get(parameter, np.nan) for parameter in comparison_parameters}
    # residual_late = {parameter: late_max_f2_s.get(parameter, np.nan) for parameter in comparison_parameters}
    #
    # plot_early_vs_late_scatter(
    #     [si_early.get(param, np.nan) for param in comparison_parameters],
    #     [si_late.get(param, np.nan) for param in comparison_parameters],
    #     comparison_parameters,
    #     "Visible",
    #     "Not visible",
    #     "Sensitivity: Visible vs Not visible",
    #     output_dir / "sensitivity_early_vs_late.png",
    # )
    # plot_early_vs_late_scatter(
    #     [residual_early.get(param, np.nan) for param in comparison_parameters],
    #     [residual_late.get(param, np.nan) for param in comparison_parameters],
    #     comparison_parameters,
    #     "Visible",
    #     "Not visible",
    #     "Residual (Cohen's f²): Early vs Late",
    #     output_dir / "residual_early_vs_late.png",
    # )

    highlight_groups = [
        [item.strip() for item in group.split(",") if item.strip()]
        for group in args.highlight_group
        if any(item.strip() for item in group.split(","))
    ]

    # Only plot si vs mean residuals (normalized and rooted SS_AB)
    plot_si_vs_residuals(
        parameters,
        si_values,
        max_ssab_s,
        output_dir / "si_vs_mean_residuals.png",
        highlight_groups=highlight_groups,
        metrics=args.metric,
        x_threshold=args.x_threshold,
        y_threshold=args.y_threshold,
        y_max_threshold=args.y_max_threshold,
        residual_series=[("Mean residual (SS_AB)", mean_ssab_s)],
        ss_ab_table=ss_ab_table,
    )

    #plot also number of zeros in the mean_ssab_s for each parameter
    num_zeros_per_param = {p: 0 for p in parameters}
    for (pa, pb), interaction in ss_ab_table.items():
        if np.isnan(interaction) or interaction <= 0:
            num_zeros_per_param[pa] += 1
            num_zeros_per_param[pb] += 1

if __name__ == "__main__":
    main()
