import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

SI_SEPERATE = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

def _get_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting functions. Install it or run with --debug-success-rates to inspect rates without plotting."
        ) from exc
    return plt

ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}


def load_sweep_plan():
    """Load sweep plan from JSON to map value indices to multiplier labels."""
    # Try multiple possible paths to locate the JSON file
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "Sweep_files" / "sweep_plan_actual_values.json",
        Path(__file__).parent.parent.parent.parent.parent / "Sweep_files" / "sweep_plan_actual_values.json",
        Path("/scratch/aldo/aicon/src/aicon/drawer_tutorial/Sweep_files/sweep_plan_actual_values.json"),
    ]
    
    for json_path in possible_paths:
        try:
            with open(json_path) as f:
                data = json.load(f)
            # Build a mapping from multiplier to label
            multiplier_to_label = {}
            sweep_multipliers = data.get("sweep_multipliers", [])
            for idx, multiplier in enumerate(sweep_multipliers):
                if multiplier == -1:
                    label = "standard"
                elif multiplier == 0:
                    label = "0"
                else:
                    label = f"{multiplier}x"
                multiplier_to_label[idx] = label
            return multiplier_to_label
        except Exception:
            continue
    
    print(f"[WARNING] Could not load sweep plan from any path")
    return {}


SWEEP_PLAN_MULTIPLIERS = load_sweep_plan()


def value_index_to_label(value_str):
    """Convert 'value_X' format to actual multiplier label."""
    try:
        if value_str.startswith("value_"):
            idx = int(value_str.split("_")[1])
            label = SWEEP_PLAN_MULTIPLIERS.get(idx, value_str)
            return label
    except Exception:
        pass
    return value_str


def short_label(label):
    return label.split(".")[-1]


def normalize_one_d_label(label):
    if label is None:
        return ""
    label = str(label).strip()
    if label == "standard":
        return "standard"
    if label.startswith("x"):
        normalized = label[1:]
        if normalized in {"0.2", "0.5", "2", "5"}:
            return f"{normalized}x"
    if label.endswith("x"):
        normalized = label[:-1]
        if normalized in {"0.2", "0.5", "2", "5"}:
            return label
    return label


def parse_pair(metadata):
    if not metadata:
        return None

    raw_parameters = metadata.get("parameters")
    if not raw_parameters:
        return None

    sweep_labels = metadata.get("sweep_labels", "")
    pair_size = len([part for part in str(sweep_labels).split(",") if part.strip()])
    if pair_size <= 0:
        pair_size = 2

    if "." in raw_parameters:
        group_part, param_part = raw_parameters.rsplit(".", 1)
        groups = [part.strip() for part in group_part.split(",")]
        params = [part.strip() for part in param_part.split(",")]
        if len(params) == pair_size:
            if len(groups) == len(params):
                return tuple(f"{group}.{param}" for group, param in zip(groups, params))
            return tuple(params)

    params = [part.strip() for part in raw_parameters.split(",")]
    if len(params) == pair_size:
        return tuple(params)

    return None


def _filter_rows_by_trial_window(rows, trial_window="all"):
    """Restrict each run to all / first-5 / last-5 trials by seed ordering."""
    if trial_window == "all":
        return rows

    grouped = defaultdict(list)
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) < 2:
            continue
        config_id = row[0]
        seed = row[1]
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            continue
        grouped[str(config_id)].append((seed_value, row))

    filtered = []
    for entries in grouped.values():
        entries.sort(key=lambda item: item[0])
        if trial_window == "first5":
            selected = entries[:5]
        elif trial_window == "last5":
            selected = entries[-5:]
        else:
            selected = entries
        filtered.extend(row for _, row in selected)
    return filtered


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

def calculate_ANOVA_interaction_score(parameters, augmented_pair_vectors, augmented_pair_diagnostics=None):
    """
    Calculate an ANOVA-like interaction score for each parameter based on the augmented pair vectors.
    This score reflects the degree of interaction effects relative to main effects.
    """
    interaction_scores = {}
    ss_ab_table = {}
    ss_total_table = {}

    for pair, values in augmented_pair_vectors.items():
        try:
            matrix = values.reshape((5, 5))  # Assuming 5 levels for each parameter
        except ValueError:
            print(f"Warning: Could not reshape values for pair {pair} into 5x5 matrix.")
            print(f"Values: {values}")
            if augmented_pair_diagnostics is not None:
                diagnostics = augmented_pair_diagnostics.get(pair, {})
                print(f"Present label pairs for {pair}: {diagnostics.get('present', [])}")
                print(f"Missing label pairs for {pair}: {diagnostics.get('missing', [])}")
                print(f"Value count for {pair}: {diagnostics.get('value_count', len(values))}")
                print(f"Valid value count for {pair}: {diagnostics.get('valid_value_count', np.sum(~np.isnan(values)))}")
            # import sys
            # sys.exit(1)
            # Skip pairs that cannot be reshaped into a 5x5 matrix
            continue

        if not np.any(np.isfinite(matrix)):
            continue

        param_a, param_b = pair
        if param_a == "connection.DistGraspHandConnection.gripper_activation_threshold" and param_b == "grasp_likelihood.gripper_activation_threshold":
            pass
        mean_a = np.nanmean(matrix, axis=1, keepdims=True)
        mean_b = np.nanmean(matrix, axis=0, keepdims=True)
        mean_value = np.nanmean(values)

        #Calculate interaction residual
        interaction_residuals = matrix - mean_a - mean_b + mean_value

        #Compute sum of squares per interaction
        ss_interaction = np.nansum(interaction_residuals ** 2)

        ss_total = np.nansum((matrix - mean_value) ** 2)

        ss_a = np.nansum((mean_a - mean_value) ** 2) * matrix.shape[1]

        ss_b = np.nansum((mean_b - mean_value) ** 2) * matrix.shape[0]

        interaction_scores[pair] = (
            ss_interaction / ss_total
            if ss_total > 0 else np.nan
        )

        ss_total_table[pair] = ss_total
        ss_ab_table[pair] = ss_interaction

    return interaction_scores, ss_total_table, ss_ab_table


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


def compute_window_metrics_summary(db_path, metrics="success", include_failed_trials=False, trial_window="all"):
    """Compute the standard per-parameter SI and residual summaries for a given trial window."""
    parameters, pairwise_series = read_pairwise_job_series(
        db_path,
        metrics=metrics,
        include_failed_trials=include_failed_trials,
        trial_window=trial_window,
    )
    si_values, one_d_successes_by_parameter, standard_baseline = read_single_sweep_series(
        db_path,
        metrics=metrics,
        include_failed_trials=include_failed_trials,
        trial_window=trial_window,
    )
    _, _, _, _, augmented_pair_vectors, augmented_pair_diagnostics = compute_hi(
        parameters, pairwise_series, one_d_successes_by_parameter, standard_baseline
    )
    interaction_scores, ss_total_table, ss_ab_table = calculate_ANOVA_interaction_score(
        parameters, augmented_pair_vectors, augmented_pair_diagnostics
    )

    max_ssab_s = {}
    mean_ssab_s = {}
    for (pa, pb), interaction in ss_ab_table.items():
        if np.isnan(interaction):
            continue
        max_ssab_s[pa] = max(max_ssab_s.get(pa, -np.inf), interaction)
        max_ssab_s[pb] = max(max_ssab_s.get(pb, -np.inf), interaction)
        mean_ssab_s[pa] = mean_ssab_s.get(pa, []) + [max(0, interaction)]
        mean_ssab_s[pb] = mean_ssab_s.get(pb, []) + [max(0, interaction)]
    mean_ssab_s = {p: float(np.mean(vals)) for p, vals in mean_ssab_s.items()} if mean_ssab_s else {}

    # --------------------------------------------------
    # Hub-parameter interaction summary and CSV export
    # --------------------------------------------------
    close_param = "connection.DistGraspHandConnection.close_dist_threshold"
    uncertainty_param = "connection.DistGraspHandConnection.uncertainty_dist_threshold"

    # Sum of all unique SS_AB values (ignore NaN and None)
    total_SS_AB = sum(float(v) for v in ss_ab_table.values() if v is not None and not np.isnan(v))

    # Accumulated SS_AB for each hub parameter (sum over pairs involving the parameter)
    close_SS_AB = sum(
        float(v)
        for (pa, pb), v in ss_ab_table.items()
        if v is not None and not np.isnan(v) and (pa == close_param or pb == close_param)
    )
    uncertainty_SS_AB = sum(
        float(v)
        for (pa, pb), v in ss_ab_table.items()
        if v is not None and not np.isnan(v) and (pa == uncertainty_param or pb == uncertainty_param)
    )

    # SS_AB between the two hub parameters (if present)
    ss_between = 0.0
    if (close_param, uncertainty_param) in ss_ab_table:
        val = ss_ab_table[(close_param, uncertainty_param)]
        ss_between = float(val) if val is not None and not np.isnan(val) else 0.0
    elif (uncertainty_param, close_param) in ss_ab_table:
        val = ss_ab_table[(uncertainty_param, close_param)]
        ss_between = float(val) if val is not None and not np.isnan(val) else 0.0

    # Combined interaction strength (avoid double-counting the direct interaction)
    combined_SS_AB = close_SS_AB + uncertainty_SS_AB - ss_between
    percentage = (combined_SS_AB / total_SS_AB * 100.0) if total_SS_AB and total_SS_AB > 0 else 0.0

    # Write results to CSV (place next to DB under metrics/<metric>/)
    import csv
    output_dir = Path(db_path).parent / "metrics" / metrics
    csv_path = output_dir / "hub_parameters_interaction_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_SS_AB", f"{total_SS_AB:.12g}"])
        writer.writerow(["close_SS_AB", f"{close_SS_AB:.12g}"])
        writer.writerow(["uncertainty_SS_AB", f"{uncertainty_SS_AB:.12g}"])
        writer.writerow(["SS_AB_between", f"{ss_between:.12g}"])
        writer.writerow(["combined_SS_AB", f"{combined_SS_AB:.12g}"])
        writer.writerow(["percentage", f"{percentage:.6g}"])

    print(f"Saved hub interaction summary to {csv_path}")

    return parameters, si_values, max_ssab_s, mean_ssab_s, ss_ab_table, ss_total_table


def plot_early_vs_late_scatter(
    x_values,
    y_values,
    labels,
    x_label,
    y_label,
    title,
    output_path,
):
    plt = _get_matplotlib_pyplot()
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6, 6))

    xs = np.asarray(list(x_values), dtype=float)
    ys = np.asarray(list(y_values), dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    labels = [label for label, keep in zip(labels, finite) if keep]

    catalog_rows = []
    for index, (label, x_val, y_val) in enumerate(zip(labels, xs, ys), start=1):
        catalog_rows.append({"point_number": index, "parameter": label, x_label: float(x_val), y_label: float(y_val)})

    if xs.size:
        max_extent = float(np.nanmax(np.maximum(np.abs(xs), np.abs(ys)))) if xs.size else 1.0
        pad = max_extent * 0.1 if max_extent > 0 else 0.1
        lower = min(0.0, float(np.nanmin(np.concatenate([xs, ys])))) - pad
        upper = max(0.0, float(np.nanmax(np.concatenate([xs, ys])))) + pad
        if upper <= lower:
            upper = lower + 1.0
    else:
        lower, upper = 0.0, 1.0

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="gray", linewidth=1.0, alpha=0.8)

    group_names = [label.split(".", 1)[0] if "." in label else label for label in labels]
    colour_map = {group_name: plt.get_cmap("tab10")(idx % 10) for idx, group_name in enumerate(dict.fromkeys(group_names))}
    for idx, (label, x_val, y_val) in enumerate(zip(labels, xs, ys), start=1):
        group_name = label.split(".", 1)[0] if "." in label else label
        ax.scatter(
            [x_val],
            [y_val],
            color=colour_map[group_name],
            edgecolors="black",
            linewidth=0.5,
            s=60,
            zorder=3,
        )
        ax.annotate(str(idx), (x_val, y_val), xytext=(5, 5), textcoords="offset points", fontsize=8, color="black")

    handles = [
        Line2D([0], [0], marker='o', color='w', label=group_name,
               markerfacecolor=colour_map[group_name], markeredgecolor='black', markersize=8)
        for group_name in dict.fromkeys(group_names)
    ]
    if handles:
        ax.legend(handles=handles, title="Group", loc="lower right", fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    catalog_path = output_path.with_name(f"{output_path.stem}_catalog.csv")
    with open(catalog_path, "w", encoding="utf-8", newline="") as fh:
        import csv
        fieldnames = ["point_number", "parameter", x_label, y_label]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog_rows)


def plot_si_vs_residuals(
    parameters,
    si_values,
    max_ssabiduals_per_parameter,
    output_path,
    highlight_groups=None,
    cluster_labels=None,
    metrics="success",
    x_threshold=None,
    y_threshold=None,
    y_max_threshold=None,
    residual_series=None,
    ss_ab_table=None,
):
    """Plot Si values against residuals per parameter on a shared scatter plot.
    Optional parameter groups can be highlighted with circles.
    """
    plt = _get_matplotlib_pyplot()
    from matplotlib.patches import Ellipse

    if residual_series is None:
        residual_series = [("Max residual", max_ssabiduals_per_parameter)]
    elif isinstance(residual_series, dict):
        residual_series = [("Residual", residual_series)]
    else:
        residual_series = list(residual_series)

    x_vals = []
    y_vals = []
    labels_list = []
    point_clusters = []
    point_parameters = []

    normalized_groups = []
    for group_index, group in enumerate(highlight_groups or [], start=1):
        if isinstance(group, str):
            members = [item.strip() for item in group.split(",") if item.strip()]
        else:
            members = [str(item).strip() for item in group if str(item).strip()]
        if members:
            normalized_groups.append((f"Group {group_index}", members))

    all_series_points = []
    for series_name, residuals_by_parameter in residual_series:
        for param in parameters:
            si = si_values.get(param, np.nan)
            residual = residuals_by_parameter.get(param, np.nan)
            if not np.isnan(si) and not np.isnan(residual):
                x_vals.append(si)
                y_vals.append(residual)
                labels_list.append(short_label(param))
                point_clusters.append(cluster_labels.get(param) if cluster_labels else None)
                point_parameters.append(param)
                all_series_points.append((series_name, param, si, residual))

    fig, ax = plt.subplots(figsize=(10, 8))
    x_arr = np.asarray(x_vals, dtype=float) if x_vals else np.asarray([], dtype=float)
    y_arr = np.asarray(y_vals, dtype=float) if y_vals else np.asarray([], dtype=float)

    # Determine plotting bounds
    if x_arr.size:
        x_min, x_max = float(np.nanmin(x_arr)), float(np.nanmax(x_arr))
    else:
        x_min, x_max = 0.0, 1.0
    if y_arr.size:
        y_min, y_max = float(np.nanmin(y_arr)), float(np.nanmax(y_arr))
    else:
        y_min, y_max = 0.0, 1.0

    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else max(0.1, abs(x_min) * 0.1)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(0.1, abs(y_min) * 0.1)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    if metrics == "timesteps":
        x_lo = 0
        x_hi = 300
        y_lo = 0
        y_hi = 500000
    elif metrics == "success":
        x_lo = 0
        x_hi = 0.5
        y_lo = 0
        y_hi = 3.0
    elif metrics == "anchor_error":
        x_lo = 0
        x_hi = 0.15
        y_lo = 0
        y_hi = 0.3
    elif metrics == "epistemicity":
        x_lo = 0
        x_hi = 1.0
        y_lo = 0
        y_hi = 1.0
    else:
        pass

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    highlighted_parameters = set()
    single_series_name = residual_series[0][0] if len(residual_series) == 1 else None

    if x_threshold is not None and y_threshold is not None:
        threshold_members = []
        for param in parameters:
            x_val = si_values.get(param, np.nan)
            y_val = max(
                (residuals_by_parameter.get(param, np.nan) for _, residuals_by_parameter in residual_series),
                default=np.nan,
            )
            if not np.isnan(x_val) and not np.isnan(y_val):
                if x_val > x_threshold and y_val > y_threshold:
                    threshold_members.append(param)
        if threshold_members:
            normalized_groups = [("Above Thresholds", threshold_members)]

    base_group_colour_map = {
        (p.split('.', 1)[0] if '.' in p else p): plt.get_cmap('tab10')(i % 10)
        for i, p in enumerate(dict.fromkeys(
            (param.split('.', 1)[0] if '.' in param else param) for param in parameters
        ))
    }

    if normalized_groups:
        colour_map = {
            group_name: plt.get_cmap("tab10")(index % 10)
            for index, (group_name, _) in enumerate(normalized_groups)
        }
        for index, (group_name, members) in enumerate(normalized_groups):
            resolved_members = resolve_parameter_selection(members, parameters)
            group_points = []
            for param in resolved_members:
                si = si_values.get(param, np.nan)
                y_value = max(
                    (residuals_by_parameter.get(param, np.nan) for _, residuals_by_parameter in residual_series),
                    default=np.nan,
                )
                if not np.isnan(si) and not np.isnan(y_value):
                    group_points.append((param, si, y_value))
            if not group_points:
                continue
            highlighted_parameters.update(point[0] for point in group_points)
            # Keep threshold-outliers in their parameter-group color and use a round
            # marker so the mean-only scatter remains visually consistent.
            for param, si_val, y_val in group_points:
                if single_series_name == "Mean residual":
                    group_key = param.split('.', 1)[0] if '.' in param else param
                    draw_color = base_group_colour_map.get(group_key, '#808080')
                    draw_marker = "o"
                else:
                    max_v = max((m.get(param, np.nan) for _, m in residual_series), default=np.nan)
                    is_max = not np.isnan(max_v) and np.isfinite(max_v) and np.isclose(y_val, float(max_v))
                    if is_max:
                        draw_color = "#ff7f0e"
                        draw_marker = "s"
                    else:
                        group_key = param.split('.', 1)[0] if '.' in param else param
                        draw_color = base_group_colour_map.get(group_key, '#808080')
                        draw_marker = "o"

                ax.scatter(
                    [si_val],
                    [y_val],
                    alpha=0.95,
                    s=120,
                    color=draw_color,
                    edgecolors="black",
                    linewidth=0.8,
                    marker=draw_marker,
                    zorder=3,
                )

    # Color points by their parameter 'group' (prefix before the first dot)
    groups = [p.split('.', 1)[0] if '.' in p else p for p in parameters]
    unique_groups = list(dict.fromkeys(groups))
    group_colour_map = {g: plt.get_cmap('tab10')(i % 10) for i, g in enumerate(unique_groups)}

    # Marker/alpha/size per series name (shape -> series type)
    # Swapped so that Max residual is less transparent (higher alpha) and drawn on top.
    series_markers = {
        "Mean residual": {'marker': 'o', 'alpha': 0.95, 's': 80, 'z': 2},
        "Max residual": {'marker': 's', 'alpha': 0.95, 's': 120, 'z': 3},
    }

    # Plot each series per-parameter so color encodes group and marker encodes series type
    series_seen = set()
    group_seen = set()
    for series_name, residuals_by_parameter in residual_series:
        props = series_markers.get(series_name, {'marker': 'o', 'alpha': 0.8, 's': 80, 'z': 2})
        for param in parameters:
            si = si_values.get(param, np.nan)
            residual = residuals_by_parameter.get(param, np.nan)
            if np.isnan(si) or np.isnan(residual):
                continue
            group = param.split('.', 1)[0] if '.' in param else param
            colour = group_colour_map.get(group, '#808080')
            ax.scatter(
                [si],
                [residual],
                alpha=props['alpha'],
                s=props['s'],
                color=colour,
                edgecolors='black',
                linewidth=0.4,
                marker=props['marker'],
                zorder=props['z'],
            )
        series_seen.add(series_name)

    # Build legends: one for groups (colors) and one for series (marker shapes)
    from matplotlib.lines import Line2D
    group_handles = [
        Line2D([0], [0], marker='o', color='w', label=g,
               markerfacecolor=group_colour_map[g], markeredgecolor='black', markersize=8)
        for g in unique_groups
    ]
    series_handles = []
    for name in series_seen:
        m = series_markers.get(name, {'marker': 'o'})['marker']
        series_handles.append(Line2D([0], [0], marker=m, color='k', label=name,
                                     markerfacecolor='gray', markeredgecolor='black', markersize=8))
    # Place group (color) legend in the upper-left as requested
    if group_handles:
        ax.legend(handles=group_handles, title='Group', loc='upper left', fontsize=8)
    # Place series (marker) legend in the upper-right without hiding the group legend
    # if series_handles:
    #     ax.add_artist(ax.legend(handles=series_handles, title='Series', loc='upper right', fontsize=8))

    if cluster_labels:
        from matplotlib.lines import Line2D

        cluster_ids = sorted({cluster for cluster in point_clusters if cluster is not None})
        colour_map = {
            cluster: plt.get_cmap("tab10")(index % 10)
            for index, cluster in enumerate(cluster_ids)
        }
        point_colours = [colour_map.get(cluster, "#808080") for cluster in point_clusters]
        ax.scatter(
            x_arr, y_arr, c=point_colours, alpha=0.2, s=1,
            edgecolors="none",
        )
        cluster_legend = [
            Line2D(
                [0], [0], marker="o", color="w", label=f"Cluster {cluster}",
                markerfacecolor=colour_map[cluster], markeredgecolor="black", markersize=8,
            )
            for cluster in cluster_ids
        ]
        if cluster_legend:
            if ax.legend_ is not None:
                ax.add_artist(ax.legend_)
            ax.legend(handles=cluster_legend, loc="upper right", fontsize=8, title="Clusters")

    # Use Mean residual mapping for annotation positions and threshold checks.
    mean_residuals = None
    max_residuals = None
    for name, mapping in residual_series:
        if name == "Mean residual":
            mean_residuals = mapping
        if name == "Max residual":
            max_residuals = mapping
    # If no explicit "Max residual" series, compute per-param max across series.
    if max_residuals is None:
        # Compute per-parameter maximum across all residual series robustly.
        max_residuals = {}
        for param in parameters:
            vals = [mapping.get(param, np.nan) for _, mapping in residual_series]
            if vals:
                max_v = np.nanmax(vals)
                max_residuals[param] = float(max_v) if not np.isnan(max_v) else np.nan
            else:
                max_residuals[param] = np.nan

    # Annotate parameters exceeding x OR y thresholds for Mean residual labels.
    annotate_mean_params = set()
    if x_threshold is not None or y_threshold is not None:
        for param in parameters:
            x_val = si_values.get(param, np.nan)
            y_val_mean = mean_residuals.get(param, np.nan) if mean_residuals is not None else np.nan
            above = False
            if x_threshold is not None and not np.isnan(x_val) and x_val > x_threshold:
                above = True
            if y_threshold is not None and not np.isnan(y_val_mean) and y_val_mean > y_threshold:
                above = True
            if above:
                annotate_mean_params.add(param)

    # Annotate parameters exceeding x OR y_max thresholds for Max residual labels.
    annotate_max_params = set()
    if x_threshold is not None or y_max_threshold is not None:
        for param in parameters:
            x_val = si_values.get(param, np.nan)
            y_val_max = max_residuals.get(param, np.nan) if max_residuals is not None else np.nan
            above_max = False
            if x_threshold is not None and not np.isnan(x_val) and x_val > x_threshold:
                above_max = True
            if y_max_threshold is not None and not np.isnan(y_val_max) and y_val_max > y_max_threshold:
                above_max = True
            if above_max:
                annotate_max_params.add(param)

    # Do not annotate highlighted groups here; annotations are strictly
    # controlled by x/y thresholds per user request.

    # Annotate only the active residual series when the plot is restricted to
    # a single metric; otherwise both mean and max labels are meaningful.
    active_series_names = {series_name for series_name, _ in residual_series}
    if len(residual_series) == 1:
        active_name = next(iter(active_series_names))
        if active_name == "Mean residual":
            annotate_mean_params = set(annotate_mean_params)
            annotate_max_params = set()
        elif active_name == "Max residual":
            annotate_mean_params = set()
            annotate_max_params = set(annotate_max_params)

    # For mean-only / max-only plots, keep only the threshold-passing subset in
    # the label list; do not annotate every point on the chart.
    if len(residual_series) == 1:
        if active_name == "Mean residual":
            annotate_mean_params = {
                param for param in annotate_mean_params
                if not np.isnan(si_values.get(param, np.nan)) and not np.isnan(mean_residuals.get(param, np.nan) if mean_residuals is not None else np.nan)
            }
        elif active_name == "Max residual":
            annotate_max_params = {
                param for param in annotate_max_params
                if not np.isnan(si_values.get(param, np.nan)) and not np.isnan(max_residuals.get(param, np.nan) if max_residuals is not None else np.nan)
            }

    # Annotate Mean residual points
    for param in sorted(annotate_mean_params):
        si = si_values.get(param, np.nan)
        mean_r = mean_residuals.get(param, np.nan) if mean_residuals is not None else np.nan
        if np.isnan(si) or np.isnan(mean_r):
            continue
        ax.annotate(
            short_label(param),
            (si, mean_r),
            fontsize=7,
            alpha=0.9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Annotate Max residual points (separate threshold control)
    for param in sorted(annotate_max_params):
        si = si_values.get(param, np.nan)
        max_r = max_residuals.get(param, np.nan) if max_residuals is not None else np.nan
        if np.isnan(si) or np.isnan(max_r):
            continue
        ax.annotate(
            short_label(param),
            (si, max_r),
            fontsize=7,
            alpha=0.9,
            xytext=(5, -8),
            textcoords="offset points",
        )

    ax.set_xlabel("S_i (Self-Interaction std_dev)")
    ax.set_ylabel("Residual per Parameter")
    if len(residual_series) == 1:
        single_name = residual_series[0][0]
        if single_name == "Mean residual":
            ax.set_title("SI vs Mean Residual")
        elif single_name == "Max residual":
            ax.set_title("SI vs Max Interaction")
        else:
            ax.set_title(f"SI vs {single_name}")
    else:
        ax.set_title("SI vs Residuals (Mean and Max)")
    ax.grid(True, alpha=0.3)
    if series_seen:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    mean_residuals = None
    max_residuals = None
    for name, mapping in residual_series:
        if name == "Mean residual":
            mean_residuals = mapping
        if name == "Max residual":
            max_residuals = mapping
    if max_residuals is None:
        max_residuals = {}
        for param in parameters:
            vals = [mapping.get(param, np.nan) for _, mapping in residual_series]
            if vals:
                max_v = np.nanmax(vals)
                max_residuals[param] = float(max_v) if not np.isnan(max_v) else np.nan
            else:
                max_residuals[param] = np.nan

    # Compute strongest partner per parameter from provided ss_ab_table
    strongest_pair_by_param = {}
    ss_ab_table_local = ss_ab_table or {}
    for (param_a, param_b), interaction in ss_ab_table_local.items():
        if np.isnan(interaction):
            continue
        for param in (param_a, param_b):
            current = strongest_pair_by_param.get(param)
            pair_key = tuple(sorted((param_a, param_b)))
            if current is None or interaction > current[0]:
                strongest_pair_by_param[param] = (float(interaction), pair_key)

    summary_rows = []
    for param in parameters:
        si = si_values.get(param, np.nan)
        mean_interaction = mean_residuals.get(param, np.nan) if mean_residuals is not None else np.nan
        max_interaction = max_residuals.get(param, np.nan) if max_residuals is not None else np.nan
        if np.isnan(si) and np.isnan(mean_interaction) and np.isnan(max_interaction):
            continue

        partner_pair = strongest_pair_by_param.get(param)
        if partner_pair is not None:
            pair_a, pair_b = partner_pair[1]
            max_interaction_pair = f"{pair_a} | {pair_b}"
        else:
            max_interaction_pair = ""

        summary_rows.append({
            "parameter": param,
            "single_sensitivity": float(si) if not np.isnan(si) else np.nan,
            "mean_interaction": float(mean_interaction) if not np.isnan(mean_interaction) else np.nan,
            "max_interaction": float(max_interaction) if not np.isnan(max_interaction) else np.nan,
            "max_interaction_pair": max_interaction_pair,
        })

    csv_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["parameter", "single_sensitivity", "mean_interaction", "max_interaction", "max_interaction_pair"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        f"Group {index + 1}": [member for member in members if member in parameters]
        for index, (_, members) in enumerate(normalized_groups)
    }
     

def heatmap_ss_ab(ss_ab_table, output_path, max_value= None):
    plt = _get_matplotlib_pyplot()

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    import numpy as np

    # --------------------------------------------------
    # Build interaction matrix
    # --------------------------------------------------
    parameters = sorted({param for pair in ss_ab_table for param in pair})

    matrix = np.full(
        (len(parameters), len(parameters)),
        np.nan
    )

    param_index = {
        param: idx for idx, param in enumerate(parameters)
    }

    for (param_a, param_b), ss_ab in ss_ab_table.items():
        idx_a = param_index[param_a]
        idx_b = param_index[param_b]

        matrix[idx_a, idx_b] = ss_ab
        matrix[idx_b, idx_a] = ss_ab  # Symmetric


    # --------------------------------------------------
    # Hierarchical clustering (normal; no pinned parameters)
    # --------------------------------------------------
    # Cluster all parameters normally using the full matrix.
    if len(parameters) > 1:
        indices = [param_index[p] for p in parameters]
        cluster_matrix = np.nan_to_num(matrix[np.ix_(indices, indices)], nan=0.0)
        distance = pdist(cluster_matrix, metric="correlation")
        linkage_matrix = linkage(distance, method="average")
        order = leaves_list(linkage_matrix)
        parameters = [parameters[i] for i in order]
        new_param_index = {param: idx for idx, param in enumerate(parameters)}
        old_indices = [param_index[p] for p in parameters]
        matrix = matrix[np.ix_(old_indices, old_indices)]
    else:
        new_param_index = {param: idx for idx, param in enumerate(parameters)}


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")   # NaN values

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.2 * len(parameters)),
            max(8.0, 1.2 * len(parameters))
        )
    )

    if max_value is None:
        im = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=0
        )
    else:
        im = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=max_value
        )


    # Show parameter names on axes
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xticklabels([short_label(p) for p in parameters], fontsize=8, rotation=90)
    ax.set_yticklabels([short_label(p) for p in parameters], fontsize=8)

    # No cell gridlines between heatmap tiles.
    ax.tick_params(which="both", length=0)

    # Remove outer border
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def resolve_parameter_selection(selection, parameters):
    """Resolve user-provided parameter names against full names and short labels."""
    if selection is None:
        return []

    resolved = []
    seen = set()
    for item in selection:
        candidate = str(item).strip()
        if not candidate:
            continue

        matches = [
            parameter
            for parameter in parameters
            if parameter == candidate
            or short_label(parameter) == candidate
            or parameter.endswith(f".{candidate}")
        ]
        for parameter in matches:
            if parameter not in seen:
                resolved.append(parameter)
                seen.add(parameter)

    return resolved

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
    _, ss_total_table, ss_ab_table = calculate_ANOVA_interaction_score(
        parameters, augmented_pair_vectors, augmented_pair_diagnostics
    )

    print("number of pairs:", len(ss_total_table))
    print("number of exact zeros ss_total:", sum(1 for v in ss_total_table.values() if v == 0))

    min_ss_total = 1e-6 # just as long as not zero, to avoid division by zero in interaction score calculation

    # scatter_plot_interaction_vs_total(min_ss_total, parameters, ss_total_table, ss_ab_table, output_dir / "interaction_vs_total.png")

    # if file already exists don't overwrite it, just print a message and skip
    heatmap_path = output_dir / "ss_ab_heatmap.pdf"
    if heatmap_path.exists():
        print(f"Heatmap file already exists: {heatmap_path}. Skipping heatmap generation.")
    else:
        try:
            heatmap_ss_ab(ss_ab_table, heatmap_path, max_value=2.0)
        except ModuleNotFoundError as exc:
            if exc.name == "scipy":
                print("SciPy not installed; skipping SS_AB heatmap generation.")
            else:
                raise

    # Compute per-parameter max and mean residuals
    max_ssab_s = {}
    mean_ssab_s = {}
    for (pa, pb), interaction in ss_ab_table.items():
        if np.isnan(interaction):
            continue
        max_ssab_s[pa] = max(max_ssab_s.get(pa, -np.inf), interaction)
        max_ssab_s[pb] = max(max_ssab_s.get(pb, -np.inf), interaction)
        mean_ssab_s[pa] = mean_ssab_s.get(pa, []) + [max(0, interaction)] # Store only non-negative residuals for mean computation
        mean_ssab_s[pb] = mean_ssab_s.get(pb, []) + [max(0, interaction)] # Store only non-negative residuals for mean computation
    mean_ssab_s = {p: float(np.mean(vals)) for p, vals in mean_ssab_s.items()} if mean_ssab_s else {}

    # Sort from highest to lowest SS_ab before computing derivatives.
    mean_ssab_values = sorted(mean_ssab_s.values(), reverse=True)
    max_ssab_values = sorted(max_ssab_s.values(), reverse=True)

    # Early-vs-late comparison for sensitivity and residual magnitude.
    early_parameters, early_si_values, early_max_ssab_s, early_mean_ssab_s, _, _ = compute_window_metrics_summary(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
        trial_window="first5",
    )
    late_parameters, late_si_values, late_max_ssab_s, late_mean_ssab_s, _, _ = compute_window_metrics_summary(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
        trial_window="last5",
    )

    comparison_parameters = sorted(set(early_parameters) | set(late_parameters))
    si_early = {parameter: early_si_values.get(parameter, np.nan) for parameter in comparison_parameters}
    si_late = {parameter: late_si_values.get(parameter, np.nan) for parameter in comparison_parameters}
    residual_early = {parameter: early_max_ssab_s.get(parameter, np.nan) for parameter in comparison_parameters}
    residual_late = {parameter: late_max_ssab_s.get(parameter, np.nan) for parameter in comparison_parameters}

    plot_early_vs_late_scatter(
        [si_early.get(param, np.nan) for param in comparison_parameters],
        [si_late.get(param, np.nan) for param in comparison_parameters],
        comparison_parameters,
        "Visible",
        "Not visible",
        "Sensitivity: Visible vs Not visible",
        output_dir / "sensitivity_early_vs_late.png",
    )
    plot_early_vs_late_scatter(
        [residual_early.get(param, np.nan) for param in comparison_parameters],
        [residual_late.get(param, np.nan) for param in comparison_parameters],
        comparison_parameters,
        "Visible",
        "Not visible",
        "Residual: Early vs Late",
        output_dir / "residual_early_vs_late.png",
    )

    highlight_groups = [
        [item.strip() for item in group.split(",") if item.strip()]
        for group in args.highlight_group
        if any(item.strip() for item in group.split(","))
    ]

    plot_si_vs_residuals(
        parameters,
        si_values,
        max_ssab_s,
        output_dir / "si_vs_residuals.png",
        highlight_groups=highlight_groups,
        metrics=args.metric,
        x_threshold=args.x_threshold,
        y_threshold=args.y_threshold,
        y_max_threshold=args.y_max_threshold,
        residual_series=[
            ("Mean residual", mean_ssab_s),
            ("Max residual", max_ssab_s),
        ],
        ss_ab_table=ss_ab_table,
    )
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
        residual_series=[("Mean residual", mean_ssab_s)],
        ss_ab_table=ss_ab_table,
    )
    plot_si_vs_residuals(
        parameters,
        si_values,
        max_ssab_s,
        output_dir / "si_vs_max_interaction.png",
        highlight_groups=highlight_groups,
        metrics=args.metric,
        x_threshold=args.x_threshold,
        y_threshold=args.y_threshold,
        y_max_threshold=args.y_max_threshold,
        residual_series=[("Max residual", max_ssab_s)],
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
