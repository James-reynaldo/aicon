import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

CONSIDER_ONLY_SUCCESSFUL = True


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
        return rows
    if sweep_label == "positive":
        return [
            row for row in rows
            if normalize_sweep_label(row.get("sweep_label")) not in ("negative", "zero", "x1000", "x0.001")
        ]
    if sweep_label == "Non_reasonable":
        return [
            row for row in rows
            if normalize_sweep_label(row.get("sweep_label")) in ("negative", "zero", "x1000", "x0.001")
        ]
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


def _trace_text_has_uncertainty_component(trace_text):
    """The stored trace catalog records component names in order, not numeric payloads."""
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


def _compute_epistemicity_from_trace_ids(trace_text_by_id, trace_ids):
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
    for trace_id in trace_ids:
        try:
            trace_id_int = int(trace_id)
        except (TypeError, ValueError):
            continue
        trace_text = trace_text_by_id.get(trace_id_int)
        if _trace_text_has_uncertainty_component(trace_text):
            gradient_steps += 1
    if total_steps == 0:
        return 0.0
    return float(gradient_steps) / float(total_steps)


def read_results_from_db(db_path: Path, compute_epistemicity: bool = False):
    rows = []
    conn = sqlite3.connect(str(db_path))
    trace_text_by_id = {}
    if compute_epistemicity:
        trace_text_by_id = {
            int(trace_id): trace_text
            for trace_id, trace_text in conn.execute("SELECT trace_id, trace_text FROM trace_catalog").fetchall()
            if trace_id is not None
        }

    cur = conn.cursor()
    cur.execute("SELECT t.metadata, t.success, t.timesteps, t.error, t.anchor_error, t.kinematic_axis_error, t.trace_ids FROM trials t")
    for metadata_json, success_val, timesteps_val, error_val, anchor_err_val, kinematic_err_val, trace_ids_json in cur.fetchall():
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
        epistemicity = _compute_epistemicity_from_trace_ids(trace_text_by_id, trace_ids_json) if compute_epistemicity else 0.0
        rows.append({
            "param_group": group,
            "param_name": name,
            "sweep_label": label,
            "success": success,
            "timesteps": timesteps_val,
            "error": error_val,
            "anchor_error": anchor_err_val,
            "kinematic_axis_error": kinematic_err_val,
            "epistemicity": epistemicity,
            "experiment_type": experiment_type,
        })

    conn.close()
    return rows


def build_summary(rows, sweep_label="negative", metric="success_rate", include_failed_trials: bool = False, debug_param_group: str = None):
    filtered_rows = filter_rows_by_sweep_label(rows, sweep_label)
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
            elif metric == "epistemicity":
                trial_epistemicity = r.get("epistemicity")
                if trial_epistemicity is not None:
                    try:
                        standard_vals.append(float(trial_epistemicity))
                    except Exception:
                        pass
    if standard_vals:
        standard_baseline = float(np.mean(np.asarray(standard_vals, dtype=float)))

    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in filtered_rows:
        if row.get("experiment_type") and row.get("experiment_type") != "1d_sweep":
            continue
        param_group = row.get("param_group", "")
        param_name = row["param_name"]
        sweep_val = row.get("sweep_label", "")
        grouped[param_group][param_name][sweep_val].append(row)

    summary = {}
    for param_group, param_dict in grouped.items():
        summary[param_group] = []
        for param_name, sweep_map in param_dict.items():
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
                elif metric == "epistemicity":
                    if include_failed_trials:
                        epistemic_values = [r.get("epistemicity") for r in rows_list if r.get("epistemicity") is not None]
                    else:
                        epistemic_values = [r.get("epistemicity") for r in rows_list if r.get("success") and r.get("epistemicity") is not None]
                    if epistemic_values:
                        rates.append(float(np.mean(epistemic_values)))
                        included_labels.append(sweep_val)

            if "standard" not in included_labels and not np.isnan(standard_baseline):
                rates.append(standard_baseline)
                included_labels.append("standard")

            if rates:
                rates_array = np.asarray(rates, dtype=float)
                mean = float(np.mean(rates_array))
                try:
                    label_mean_map = {lab: float(rates_array[i]) for i, lab in enumerate(included_labels)}
                except Exception:
                    label_mean_map = {lab: None for lab in included_labels}
                var = float(np.var(rates_array))
                count = len(rates_array)
                si = float(np.sqrt(var))
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


def summary_lookup(summary):
    lookup = {}
    if summary is None:
        return lookup

    for param_group, param_data in summary.items():
        for item in param_data:
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


def filter_rows_for_sweep(all_rows_normal, all_rows_noise, all_rows_disturbance, all_rows_bad_prior, sweep_label):
    """Filter all loaded experiment rows to a single sweep family before aggregation."""
    return {
        "normal": filter_rows_by_sweep_label(all_rows_normal, sweep_label),
        "noise": filter_rows_by_sweep_label(all_rows_noise, sweep_label),
        "disturbance": filter_rows_by_sweep_label(all_rows_disturbance, sweep_label),
        "bad_prior": filter_rows_by_sweep_label(all_rows_bad_prior, sweep_label),
    }


def aggregate_metrics_for_plot(filtered_rows, include_failed_trials: bool = False, sweep_label="negative"):
    """Build the metric summaries and stable-point sets used for plotting."""
    return build_metric_grid_inputs(
        filtered_rows["normal"],
        filtered_rows["noise"],
        filtered_rows["disturbance"],
        filtered_rows["bad_prior"],
        sweep_label=sweep_label,
        include_failed_trials=include_failed_trials,
    )


def build_metric_grid_inputs(
    all_rows_normal,
    all_rows_noise,
    all_rows_disturbance,
    all_rows_bad_prior,
    sweep_label="negative",
    include_failed_trials: bool = False,
):
    """Build the four metric summaries and stable-point dictionaries expected by the 1x4 grid plot."""
    metric_names = [
        "success_rate",
        "avg_timesteps",
        "avg_anchor_error",
        "epistemicity",
    ]

    metric_summaries = {}
    stable_by_metric = {}

    for metric_name in metric_names:
        summary_normal = build_summary(
            all_rows_normal,
            sweep_label=sweep_label,
            metric=metric_name,
            include_failed_trials=include_failed_trials,
        )
        summary_noise = build_summary(
            all_rows_noise,
            sweep_label=sweep_label,
            metric=metric_name,
            include_failed_trials=include_failed_trials,
        )
        summary_disturbance = build_summary(
            all_rows_disturbance,
            sweep_label=sweep_label,
            metric=metric_name,
            include_failed_trials=include_failed_trials,
        )
        summary_bad_prior = build_summary(
            all_rows_bad_prior,
            sweep_label=sweep_label,
            metric=metric_name,
            include_failed_trials=include_failed_trials,
            debug_param_group="bad_prior",
        )

        stable_normal = set()
        stable_noise = set()
        stable_disturbance = set()
        stable_bad_prior = set()
        if sweep_label == "all":
            stable_normal = find_stable_all_points(
                summary_normal,
                build_summary(all_rows_normal, sweep_label="negative", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_normal, sweep_label="zero", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_normal, sweep_label="positive", metric=metric_name, include_failed_trials=include_failed_trials),
            )
            stable_noise = find_stable_all_points(
                summary_noise,
                build_summary(all_rows_noise, sweep_label="negative", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_noise, sweep_label="zero", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_noise, sweep_label="positive", metric=metric_name, include_failed_trials=include_failed_trials),
            )
            stable_disturbance = find_stable_all_points(
                summary_disturbance,
                build_summary(all_rows_disturbance, sweep_label="negative", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_disturbance, sweep_label="zero", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_disturbance, sweep_label="positive", metric=metric_name, include_failed_trials=include_failed_trials),
            )
            stable_bad_prior = find_stable_all_points(
                summary_bad_prior,
                build_summary(all_rows_bad_prior, sweep_label="negative", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_bad_prior, sweep_label="zero", metric=metric_name, include_failed_trials=include_failed_trials),
                build_summary(all_rows_bad_prior, sweep_label="positive", metric=metric_name, include_failed_trials=include_failed_trials),
            )

        metric_summaries[metric_name] = {
            "normal": summary_normal,
            "noise": summary_noise,
            "disturbance": summary_disturbance,
            "bad_prior": summary_bad_prior,
        }
        stable_by_metric[metric_name] = {
            "normal": stable_normal,
            "noise": stable_noise,
            "disturbance": stable_disturbance,
            "bad_prior": stable_bad_prior,
        }

    return metric_summaries, stable_by_metric
