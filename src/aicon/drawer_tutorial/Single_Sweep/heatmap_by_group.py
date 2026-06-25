import argparse
import csv
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
        }
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


def main():
    parser = argparse.ArgumentParser(description="Create heatmaps per param_group (param_name x sweep_label).")
    parser.add_argument('--results-dir', default='results', help='Results CSV folder')
    parser.add_argument('--output-dir', default='heatmaps', help='Output folder for heatmaps')
    parser.add_argument('--metric', choices=['success', 'timesteps', 'both'], default='both', help='Which metric to plot')
    parser.add_argument('--no-annotate', action='store_true', help='Do not write text annotations in cells')
    args = parser.parse_args()

    rows = []
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
    for g, d in agg.items():
        for metric in metrics:
            f = plot_group_heatmap(g, d, out_dir, metric=metric, annotate=not args.no_annotate)
            if f:
                created.append(str(f))

    print(f"Wrote {len(created)} heatmaps to {out_dir}")


if __name__ == '__main__':
    main()
