import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt


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
    except Exception:
        return False


def parse_float(value):
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def read_results(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return rows
        normalized = [c.strip().lower() for c in header]

        # required: param_name, sweep_label, success (sweep_value optional)
        try:
            idx_name = normalized.index("param_name")
            idx_label = normalized.index("sweep_label")
            idx_success = normalized.index("success")
        except ValueError:
            return rows

        idx_group = normalized.index("param_group") if "param_group" in normalized else None
        idx_value = normalized.index("sweep_value") if "sweep_value" in normalized else None
        idx_timesteps = normalized.index("timesteps") if "timesteps" in normalized else None

        for row in reader:
            if len(row) <= max(i for i in [idx_name, idx_label, idx_success, idx_value, idx_timesteps] if i is not None):
                continue
            group = row[idx_group].strip() if idx_group is not None else ""
            name = row[idx_name].strip()
            label = row[idx_label].strip()
            success = parse_bool(row[idx_success])
            value = parse_float(row[idx_value]) if idx_value is not None else None
            timesteps = parse_float(row[idx_timesteps]) if idx_timesteps is not None else None
            rows.append({"group": group, "name": name, "label": label, "success": success, "sweep_value": value, "timesteps": timesteps})
    return rows


def _extract_baseline_value(params, group, name):
    if params is None:
        return None
    if group.startswith("connection."):
        _, connection_group = group.split(".", 1)
        # Support two possible shapes in stored params:
        # 1) connection parameters nested under "connection_params":
        #    { "connection_params": { "DistGraspHandConnection": { ... } } }
        # 2) connection parameters stored at top-level keyed by connection group:
        #    { "DistGraspHandConnection": { ... } }
        val = None
        try:
            val = params.get("connection_params", {}).get(connection_group, {}).get(name)
        except Exception:
            val = None
        if val is None:
            try:
                val = params.get(connection_group, {}).get(name)
            except Exception:
                val = None
        return val
    return params.get(group, {}).get(name)


def read_results_from_db(db_path: Path):
    rows = []
    baseline_params = None
    baseline_trials = []

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT t.metadata, t.success, t.timesteps, c.params FROM trials t JOIN configs c ON t.config_id = c.config_id")
    for metadata_json, success_val, timesteps, params_json in cur.fetchall():
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            continue

        parameter = metadata.get("parameter", "")
        label = metadata.get("label", "")
        sweep_value = metadata.get("sweep_value")

        if parameter == "standard.standard" and label == "standard":
            if baseline_params is None:
                try:
                    baseline_params = json.loads(params_json)
                except Exception:
                    baseline_params = None
                else:
                    # Debug: print connection_params keys and the DistGraspHandConnection dict
                    try:
                        if isinstance(baseline_params, dict):
                            conn_params = baseline_params.get("connection_params")
                            conn_keys = list(conn_params.keys()) if isinstance(conn_params, dict) else None
                            print(f"DEBUG connection_params keys: {conn_keys}")
                            if isinstance(conn_params, dict) and "DistGraspHandConnection" in conn_params:
                                print("DEBUG DistGraspHandConnection:")
                                try:
                                    print(json.dumps(conn_params["DistGraspHandConnection"], indent=2))
                                except Exception:
                                    print(conn_params["DistGraspHandConnection"])
                    except Exception:
                        pass
            baseline_trials.append({"success": bool(success_val), "timesteps": timesteps})
            continue

        if "." in parameter:
            group, name = parameter.rsplit('.', 1)
        else:
            group, name = "", parameter

        # Handle connection parameters that use a slash between connection group and param,
        # e.g. "connection.DistGraspHandConnection/uncertainty_dist_threshold"
        if group == "connection" and "/" in name:
            conn_group, param_name = name.split("/", 1)
            group = f"connection.{conn_group}"
            name = param_name

        if sweep_value is None and label == "standard":
            sweep_value = metadata.get("standard_value")
            if sweep_value is None:
                sweep_value = _extract_baseline_value(baseline_params, group, name)

        rows.append({"group": group, "name": name, "label": label, "success": bool(success_val), "sweep_value": sweep_value, "timesteps": timesteps})

    if baseline_params and baseline_trials:
        all_groups = {(r["group"], r["name"]) for r in rows if not (r["group"] == "standard" and r["name"] == "standard")}
        for group, name in all_groups:
            baseline_value = _extract_baseline_value(baseline_params, group, name)
            print(f"Baseline value for {group}/{name}: {baseline_value}")
            # If baseline missing for connection params, try a fallback: find any config
            # that contains the connection group and use its value as a sensible default.
            if baseline_value is None and group.startswith("connection."):
                try:
                    _, conn_group = group.split('.', 1)
                    # search configs table for any params containing the connection group
                    cur.execute("SELECT params FROM configs WHERE params LIKE ? LIMIT 1", (f"%{conn_group}%",))
                    row = cur.fetchone()
                    if row:
                        try:
                            cand = json.loads(row[0])
                            # try both shapes
                            if isinstance(cand, dict):
                                baseline_value = cand.get("connection_params", {}).get(conn_group, {}).get(name)
                                if baseline_value is None:
                                    baseline_value = cand.get(conn_group, {}).get(name)
                                if baseline_value is not None:
                                    print(f"DEBUG fallback baseline from configs for {group}/{name}: {baseline_value}")
                        except Exception:
                            pass
                except Exception:
                    pass
            if baseline_value is None:
                continue
            for trial in baseline_trials:
                rows.append({
                    "group": group,
                    "name": name,
                    "label": "standard",
                    "success": trial["success"],
                    "sweep_value": baseline_value,
                    "timesteps": trial["timesteps"],
                })

    conn.close()
    return rows


def aggregate(rows):
    # data[group][name][label] -> dict with success and timesteps lists
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"success": [], "timesteps": []})))
    label_values = defaultdict(lambda: defaultdict(list))  # label_values[group][label] -> list of numeric values for sort ordering
    for r in rows:
        g = r["group"]
        n = r["name"]
        l = r["label"]
        cell = data[g][n][l]
        cell["success"].append(1.0 if r["success"] else 0.0)
        if r["timesteps"] is not None and r["success"]:
            cell["timesteps"].append(r["timesteps"])
        if r["sweep_value"] is not None:
            label_values[g][l].append(r["sweep_value"])

    # compute mean success and mean timesteps per cell
    result = {}
    for g, names in data.items():
        labels = set()
        for n, labdict in names.items():
            labels.update(labdict.keys())
        labels = list(labels)

        label_to_sort_value = {}
        for l in labels:
            vals = label_values[g].get(l)
            if vals:
                label_to_sort_value[l] = float(np.median(vals))
        if label_to_sort_value:
            labels.sort(key=lambda x: label_to_sort_value.get(x, math.inf))
        else:
            labels.sort()

        name_list = sorted(names.keys())
        success_mat = np.full((len(name_list), len(labels)), np.nan)
        timesteps_mat = np.full((len(name_list), len(labels)), np.nan)
        counts = np.zeros((len(name_list), len(labels)), dtype=int)
        importances = np.zeros(len(name_list), dtype=float)

        for i, n in enumerate(name_list):
            successes = []
            for j, l in enumerate(labels):
                cell = names[n].get(l)
                if cell is None:
                    continue
                if cell["success"]:
                    success_mat[i, j] = float(np.mean(cell["success"]))
                    counts[i, j] = len(cell["success"])
                    successes.append(success_mat[i, j])
                if cell["timesteps"]:
                    timesteps_mat[i, j] = float(np.mean(cell["timesteps"]))
            importances[i] = float(np.mean(successes)) if successes else np.nan

        # sort rows by importance descending
        order = np.argsort(-importances, kind="mergesort")
        name_list = [name_list[i] for i in order]
        success_mat = success_mat[order, :]
        timesteps_mat = timesteps_mat[order, :]
        counts = counts[order, :]
        importances = importances[order]

        result[g] = {
            "names": name_list,
            "labels": labels,
            "success_mat": success_mat,
            "timesteps_mat": timesteps_mat,
            "counts": counts,
            "importances": importances,
            "group_importance": float(np.nanmean(importances)) if np.any(~np.isnan(importances)) else float('nan'),
        }

    # sort groups by importance descending
    result = dict(sorted(result.items(), key=lambda item: item[1]["group_importance"] if not math.isnan(item[1]["group_importance"]) else -math.inf, reverse=True))
    return result


def plot_group_heatmap(group_key, group_data, out_dir: Path, metric: str = "success", annotate: bool = True):
    names = group_data["names"]
    labels = group_data["labels"]
    counts = group_data["counts"]
    if metric == "success":
        mat = group_data["success_mat"]
        cmap = plt.get_cmap("RdYlGn")
        colorbar_label = "Success rate"
        title_metric = "Success Rate"
        vmin, vmax = 0, 1
    else:
        mat = group_data["timesteps_mat"]
        cmap = plt.get_cmap("YlGnBu")
        colorbar_label = "Average timesteps"
        title_metric = "Average Timesteps"
        vmin, vmax = 0, 1000

    if mat.size == 0:
        return None

    plt.figure(figsize=(max(6, len(labels)*0.6), max(4, len(names)*0.4)))
    masked = np.ma.masked_invalid(mat)
    im = plt.imshow(masked, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)

    y_positions = np.arange(len(names))
    plt.yticks(y_positions, names)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Sweep label")
    plt.ylabel("Param name")
    plt.title(f"{title_metric} for {group_key}")

    # Add importance values to the right side
    importance_values = group_data.get("importances")
    if importance_values is not None:
        for i, imp in enumerate(importance_values):
            if math.isnan(imp):
                imp_text = "n/a"
            else:
                imp_text = f"{imp:.2f}"
            plt.text(len(labels) + 0.1, i, imp_text, va='center', ha='left', color='black', fontsize=8)
        plt.xlim(-0.5, len(labels) + 1.5)

    plt.tight_layout()

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if math.isnan(val):
                    txt = "-"
                    plt.text(j, i, txt, ha='center', va='center', color='gray', fontsize=8)
                else:
                    if metric == "success":
                        txt = f"{val:.2f}\n({counts[i,j]})"
                    else:
                        txt = f"({counts[i,j]})"
                    color = 'black' if val < (0.5 if metric == "success" else (vmin + vmax) / 2) else 'white'
                    plt.text(j, i, txt, ha='center', va='center', color=color, fontsize=8)

    safe_group = str(group_key).replace('/', '_').replace(' ', '_')
    out_file = out_dir / f"heatmap_{safe_group}_{metric}.png"
    plt.savefig(out_file)
    plt.close()
    return out_file


def plot_combined_heatmap(agg, out_dir: Path, metric: str = "success", annotate: bool = True):
    all_labels =['negative', 'zero', 'x0.001','x0.2','x0.5','standard','x2','x5','x1000']
    print(all_labels)
    if not all_labels:
        return None

    row_labels = []
    label_index = {label: idx for idx, label in enumerate(all_labels)}
    total_rows = sum(len(data["names"]) for data in agg.values())
    mat = np.full((total_rows, len(all_labels)), np.nan)
    counts = np.zeros((total_rows, len(all_labels)), dtype=int)
    group_boundaries = []
    row_to_group = []

    row = 0
    for group_key, group_data in agg.items():
        group_boundaries.append(row)
        source_mat = group_data["success_mat"] if metric == "success" else group_data["timesteps_mat"]
        source_counts = group_data["counts"]
        for local_row, name in enumerate(group_data["names"]):
            row_labels.append(f"{group_key}/{name}")
            row_to_group.append(group_key)
            for label, j in label_index.items():
                if label in group_data["labels"]:
                    label_idx = group_data["labels"].index(label)
                    if label_idx < source_mat.shape[1]:
                        mat[row, j] = source_mat[local_row, label_idx]
                        counts[row, j] = source_counts[local_row, label_idx]
            row += 1

    if mat.size == 0:
        return None

    if metric == "success":
        cmap = plt.get_cmap("RdYlGn")
        colorbar_label = "Success rate"
        title_metric = "Success Rate"
        vmin, vmax = 0, 1
    else:
        cmap = plt.get_cmap("YlGnBu")
        colorbar_label = "Average timesteps"
        title_metric = "Average Timesteps"
        vmin, vmax = 0, 1000

    plt.figure(figsize=(max(8, len(all_labels)*0.5), max(6, total_rows*0.25)))
    masked = np.ma.masked_invalid(mat)
    im = plt.imshow(masked, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    y_positions = np.arange(len(row_labels))
    plt.yticks(y_positions, row_labels)
    plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha="right")
    plt.xlabel("Sweep label")
    plt.ylabel("Group/Param name")
    plt.title(f"{title_metric} for all groups")

    for boundary in group_boundaries[1:]:
        plt.axhline(boundary - 0.5, color='black', linewidth=0.5)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if math.isnan(val):
                    txt = "-"
                    plt.text(j, i, txt, ha='center', va='center', color='gray', fontsize=6)
                else:
                    if metric == "success":
                        txt = f"{val:.2f}\n({counts[i,j]})"
                    else:
                        txt = f"({counts[i,j]})"
                    color = 'black' if val < (0.5 if metric == "success" else (vmin + vmax) / 2) else 'white'
                    plt.text(j, i, txt, ha='center', va='center', color=color, fontsize=6)

    safe_name = "all_groups"
    out_file = out_dir / f"heatmap_{safe_name}_{metric}.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Create heatmaps per param_group (param_name x sweep_label).")
    parser.add_argument('--results-dir', default='results', help='Results CSV folder')
    parser.add_argument('--db-path', default='', help='SQLite experiment database path to read results from')
    parser.add_argument('--output-dir', default='heatmaps', help='Output folder for heatmaps')
    parser.add_argument('--metric', choices=['success', 'timesteps', 'both'], default='both', help='Which metric to plot')
    parser.add_argument('--combined', action='store_true', help='Create a single heatmap that includes all groups in one figure')
    parser.add_argument('--no-annotate', action='store_true', help='Do not write text annotations in cells')
    args = parser.parse_args()

    rows = []
    if args.db_path:
        rows = read_results_from_db(Path(args.db_path))
    else:
        base = Path(args.results_dir)
        for p in sorted(base.glob('*.csv')):
            rows.extend(read_results(p))

    if not rows:
        raise SystemExit('No rows found in results')

    agg = aggregate(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['success', 'timesteps'] if args.metric == 'both' else [args.metric]
    created = []
    if args.combined:
        for metric in metrics:
            f = plot_combined_heatmap(agg, out_dir, metric=metric, annotate=not args.no_annotate)
            if f:
                created.append(str(f))
    else:
        for g, d in agg.items():
            for metric in metrics:
                f = plot_group_heatmap(g, d, out_dir, metric=metric, annotate=not args.no_annotate)
                if f:
                    created.append(str(f))

    print(f"Wrote {len(created)} heatmaps to {out_dir}")


if __name__ == '__main__':
    main()
