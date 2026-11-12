"""Data reading functions for pairwise and single sweep analysis."""
from collections import defaultdict
from pathlib import Path
import json
import sqlite3
import numpy as np

from .utils import (
    load_sweep_plan,
    value_index_to_label,
    short_label,
    normalize_one_d_label,
    parse_pair,
    _filter_rows_by_trial_window,
    ONE_D_LABELS_TO_INCLUDE,
)


def read_pairwise_job_series(db_path, metrics="success", include_failed_trials=False, trial_window="all"):
    """Read pairwise job series from database."""
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
                metric_val = _compute_epistemicity_from_trace_ids(conn, trace_ids_json, trace_catalog_map=trace_catalog_map)
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
                metric_val = _compute_epistemicity_from_trace_ids(conn, trace_ids_json, trace_catalog_map=trace_catalog_map)
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
