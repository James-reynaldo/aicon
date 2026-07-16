import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist
except ImportError:
    leaves_list = None
    linkage = None
    pdist = None


ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}


def parse_pair(metadata):
    """Return the two swept parameter labels from pairwise sweep metadata."""
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


def merge_one_d_successes(successes_by_pair, one_d_successes_by_parameter):
    """Add relevant 1d sweep successes to each pairwise success group."""
    merged_successes = {}
    included_one_d_trials = 0

    for pair, pairwise_successes in successes_by_pair.items():
        successes = list(pairwise_successes)
        for parameter in pair:
            one_d_successes = one_d_successes_by_parameter.get(parameter, [])
            successes.extend(one_d_successes)
            included_one_d_trials += len(one_d_successes)
        merged_successes[pair] = successes

    return merged_successes, included_one_d_trials


def pair_statistics(successes):
    if not successes:
        return np.nan, np.nan

    values = np.asarray(successes, dtype=float)
    return float(np.mean(values)), float(np.std(values))


def read_successes(db_path):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT t.success, t.metadata
            FROM trials t
            WHERE t.metadata IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    successes_by_pair = defaultdict(list)
    one_d_successes_by_parameter = defaultdict(list)
    skipped = 0

    for success, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            skipped += 1
            continue

        experiment_type = metadata.get("experiment_type")

        if experiment_type == "1d_sweep":
            if metadata.get("label") not in ONE_D_LABELS_TO_INCLUDE:
                continue

            parameter = metadata.get("parameter")
            if not parameter:
                skipped += 1
                continue

            one_d_successes_by_parameter[parameter].append(float(success))
            continue

        if experiment_type == "pairwise_sweep":
            pair = parse_pair(metadata)
            if pair is None or len(pair) != 2:
                skipped += 1
                continue

            successes_by_pair[tuple(sorted(pair))].append(float(success))
            continue

    successes_by_pair, included_one_d_trials = merge_one_d_successes(
        successes_by_pair, one_d_successes_by_parameter
    )

    return successes_by_pair, skipped, included_one_d_trials


def build_matrices(successes_by_pair):
    parameters = sorted({param for pair in successes_by_pair for param in pair})
    index = {param: i for i, param in enumerate(parameters)}
    mean_matrix = np.full((len(parameters), len(parameters)), np.nan)
    std_matrix = np.full((len(parameters), len(parameters)), np.nan)

    for (param_a, param_b), successes in successes_by_pair.items():
        success_rate, std_dev = pair_statistics(successes)
        if np.isnan(success_rate):
            continue
        i = index[param_a]
        j = index[param_b]
        mean_matrix[i, j] = success_rate
        mean_matrix[j, i] = success_rate
        std_matrix[i, j] = std_dev
        std_matrix[j, i] = std_dev

    return parameters, mean_matrix, std_matrix


def cluster_order_for_matrix(matrix):
    """Return row/column order from hierarchical clustering."""
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0])
    if leaves_list is None or linkage is None or pdist is None:
        raise SystemExit(
            "Hierarchical clustering requires scipy. Install it with: pip install scipy"
        )

    filled = matrix.copy()
    global_mean = np.nanmean(filled)
    if np.isnan(global_mean):
        global_mean = 0.0

    row_means = np.nanmean(filled, axis=1)
    row_means = np.where(np.isnan(row_means), global_mean, row_means)
    nan_rows, nan_cols = np.where(np.isnan(filled))
    filled[nan_rows, nan_cols] = row_means[nan_rows]

    condensed = pdist(filled, metric="euclidean")
    if condensed.size == 0 or np.allclose(condensed, 0.0):
        return np.arange(matrix.shape[0])

    return leaves_list(linkage(condensed, method="average"))


def reorder_matrix(parameters, matrix, order):
    ordered_parameters = [parameters[i] for i in order]
    ordered_matrix = matrix[np.ix_(order, order)]
    return ordered_parameters, ordered_matrix


def short_label(label):
    return label.split(".")[-1]


def row_label_highlights(matrix):
    """Return row label highlight colors based on row means vs the table mean."""
    table_mean = np.nanmean(matrix)
    row_means = np.nanmean(matrix, axis=1)
    highlights = []

    boundary = 0.05
    for row_mean in row_means:
        if np.isnan(row_mean) or np.isnan(table_mean):
            highlights.append(None)
        elif row_mean >= (1 + boundary) * table_mean:
            highlights.append("#fff176")
        elif row_mean <= (1 - boundary) * table_mean:
            highlights.append("#ef5350")
        else:
            highlights.append(None)

    return highlights


def plot_matrix(parameters, matrix, output_path, title, colorbar_label, vmax, row_sums=None, row_sum_label=None):
    size = max(7.0, 0.45 * len(parameters) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(masked, vmin=0.0, vmax=vmax, cmap=cmap)

    labels = [short_label(param) for param in parameters]
    ax.set_xticks(np.arange(len(parameters)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(parameters)), labels=labels)
    for tick_label, highlight in zip(ax.get_yticklabels(), row_label_highlights(matrix)):
        if highlight is None:
            continue
        tick_label.set_bbox(
            {
                "boxstyle": "square,pad=0.2",
                "facecolor": highlight,
                "edgecolor": "none",
            }
        )
    ax.set_title(title)

    # If requested, make room on the right for row-sum labels
    n = len(parameters)
    if row_sums is not None:
        # extend x-limits to make space; image spans -0.5..n-0.5
        ax.set_xlim(-0.5, n - 0.5 + 1.2)

    ax.set_xticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(parameters)):
        for j in range(len(parameters)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            text_color = "white" if value < (0.5 * vmax) else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    # Draw row-sum labels to the right of the matrix when provided
    if row_sums is not None:
        x_pos = n - 0.5 + 0.6
        for i, s in enumerate(row_sums):
            if s is None or (isinstance(s, float) and np.isnan(s)):
                txt = "n/a"
            else:
                try:
                    txt = f"{float(s):.2f}"
                except Exception:
                    txt = str(s)
            ax.text(
                x_pos,
                i,
                txt,
                ha="left",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none"},
                clip_on=False,
            )
        # header for row sums
        header_y = -0.8
        label = row_sum_label or "Sum"
        ax.text(x_pos, header_y, label, ha="left", va="bottom", fontsize=9, weight="bold", clip_on=False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a square table of average success rates for pairwise sweep "
            "parameter pairs from an experiment_store.db file."
        )
    )
    parser.add_argument("--db-path", type=Path, help="Path to experiment_store.db")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to pairwise_success_rate_table.png next to the database.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"Database not found: {args.db_path}")

    output_path = args.output or args.db_path.with_name("pairwise_success_rate_table.png")
    std_output_path = output_path.with_name("pairwise_success_std_table.png")
    successes_by_pair, skipped, included_one_d_trials = read_successes(args.db_path)
    if not successes_by_pair:
        raise SystemExit("No pairwise_sweep trials found in the database.")

    parameters, mean_matrix, std_matrix = build_matrices(successes_by_pair)
    plot_matrix(
        parameters,
        mean_matrix,
        output_path,
        "Pairwise Sweep Mean Success Rate",
        "Mean success rate",
        1.0,
    )
    std_order = cluster_order_for_matrix(std_matrix)
    clustered_parameters, clustered_std_matrix = reorder_matrix(
        parameters, std_matrix, std_order
    )
    # compute row sums for the clustered std matrix and show them on the right
    clustered_row_sums = [float(np.nansum(row)) for row in clustered_std_matrix]
    plot_matrix(
        clustered_parameters,
        clustered_std_matrix,
        std_output_path,
        "Pairwise Sweep Standard Deviation of Success (Hierarchical Clustering)",
        "Standard deviation",
        0.5,
        row_sums=clustered_row_sums,
        row_sum_label="Row sum",
    )

    total_trials = sum(len(successes) for successes in successes_by_pair.values())
    print(f"Saved {output_path}")
    print(f"Saved {std_output_path}")
    print(f"Averaged {total_trials} trials across {len(successes_by_pair)} parameter pairs.")
    print(f"Included {included_one_d_trials} /x0.2/x0.5/x2/x5 1d_sweep trials in pair averages.")
    if skipped:
        print(f"Skipped {skipped} malformed rows.")


if __name__ == "__main__":
    main()
