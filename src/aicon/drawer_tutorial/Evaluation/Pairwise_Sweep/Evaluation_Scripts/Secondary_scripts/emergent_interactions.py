import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}


def short_label(label):
    return label.split(".")[-1]


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


def read_pairwise_job_series(db_path):
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

    # Group pairwise data by (pair, label_a, label_b) to preserve label information
    by_pair_labels = defaultdict(lambda: defaultdict(list))
    by_job = defaultdict(list)

    for success, metadata_json in rows:
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
        success_val = float(success)
        
        # Try to extract label information from metadata
        sweep_labels = metadata.get("sweep_labels", "")
        if sweep_labels:
            labels = [part.strip() for part in str(sweep_labels).split(",")]
            if len(labels) == 2:
                label_a, label_b = labels
                # Normalize label order to match sorted parameter order
                if parameters[0] == pair[1]:
                    label_a, label_b = label_b, label_a
                by_pair_labels[pair][(label_a, label_b)].append(success_val)
        
        # Also keep the old grouping for backward compatibility
        job_key = (
            pair,
            str(metadata.get("sweep_labels", "")),
            str(metadata.get("sweep_values", "")),
        )
        by_job[job_key].append(success_val)

    # Build pair_vectors with label information
    pair_vectors = {}
    for pair in by_pair_labels:
        pair_vectors[pair] = {}
        for (label_a, label_b), successes in by_pair_labels[pair].items():
            if successes:
                pair_vectors[pair][(label_a, label_b)] = float(np.mean(np.asarray(successes, dtype=float)))

    parameters = sorted({param for pair in pair_vectors for param in pair})
    
    return parameters, pair_vectors


def read_single_sweep_series(db_path):
    """
    Read 1d sweep trials and compute Si values.
    Returns both Si values and organized success data by parameter and label.
    """
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

    # Group by parameter -> label -> list of successes
    by_param_label = defaultdict(lambda: defaultdict(list))
    by_param_sweep = defaultdict(lambda: defaultdict(list))

    for success, metadata_json in rows:
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
        label = metadata.get("label", "")
        
        # Filter to only included labels
        if label not in ONE_D_LABELS_TO_INCLUDE:
            continue
        
        if sweep_value is None:
            continue

        success_val = float(success)
        by_param_label[parameter][label].append(success_val)
        by_param_sweep[parameter][(sweep_value, label)].append(success_val)

    # Compute mean success per parameter per label for use in Hi computation
    one_d_successes_by_parameter = {}
    for parameter in by_param_label:
        one_d_successes_by_parameter[parameter] = {}
        for label in by_param_label[parameter]:
            successes = by_param_label[parameter][label]
            if successes:
                one_d_successes_by_parameter[parameter][label] = float(np.mean(np.asarray(successes, dtype=float)))

    # For each parameter, compute success rates per sweep value and then variance for Si
    si_values = {}
    for parameter, sweep_data in by_param_sweep.items():
        positive_rates = []
        
        for (sweep_value, label), successes in sweep_data.items():
            if successes:
                success_rate = float(np.mean(np.asarray(successes, dtype=float)))
                positive_rates.append(success_rate)
        
        # Compute variance of success rates across positive sweeps
        if positive_rates and len(positive_rates) > 1:
            si_values[parameter] = float(np.var(np.asarray(positive_rates, dtype=float)))
        elif positive_rates:
            # If only one positive sweep value, variance is 0
            si_values[parameter] = 0.0

    return si_values, one_d_successes_by_parameter


def compute_hi(parameters, pair_vectors, one_d_successes_by_parameter=None):
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
    for param_a, param_b in [(p1, p2) for p1 in parameters for p2 in parameters if p1 != p2]:
        pair_key = tuple(sorted([param_a, param_b]))
        all_values = []
        
        # Get pairwise data for this pair if it exists (dict of label combinations)
        pairwise_by_labels = pair_vectors.get(pair_key, {})
        
        # Build 5x5 grid of label combinations (25 total, excluding duplicate standard)
        for label_a in labels:
            for label_b in labels:
                # Skip duplicate standard-standard combination for different params
                if label_a == "standard" and label_b == "standard" and param_a != param_b:
                    continue
                
                # Try to get pairwise data for this specific label combination
                pairwise_value = pairwise_by_labels.get((label_a, label_b), np.nan)
                
                if not np.isnan(pairwise_value):
                    # Use pairwise data for this specific label combination
                    all_values.append(pairwise_value)
                else:
                    # Fall back to 1D data for this label combination
                    val_a = one_d_successes_by_parameter.get(param_a, {}).get(label_a, np.nan)
                    val_b = one_d_successes_by_parameter.get(param_b, {}).get(label_b, np.nan)
                    available = [v for v in [val_a, val_b] if not np.isnan(v)]
                    if available:
                        all_values.append(float(np.mean(available)))
        
        if all_values:
            augmented_pair_vectors[pair_key] = np.asarray(all_values, dtype=float)

    baseline_values = {}
    for parameter in parameters:
        collected = []
        for pair, values in augmented_pair_vectors.items():
            if parameter in pair:
                collected.extend(values.tolist())
        baseline_values[parameter] = float(np.mean(collected)) if collected else np.nan

    hi_values = {}
    variation_table = {}
    for parameter in parameters:
        partner_variances = []
        variation_table[parameter] = {}
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
            partner_variances.append(variance)
            variation_table[parameter][partner] = variance

        hi_values[parameter] = float(np.mean(partner_variances)) if partner_variances else np.nan

    return baseline_values, hi_values, variation_table


def normalize_to_01(values_dict):
    """Normalize a dictionary of values to [0, 1] range."""
    valid_values = [v for v in values_dict.values() if not np.isnan(v) and not np.isinf(v)]
    if not valid_values:
        return {k: 0.0 for k in values_dict}
    
    min_val = min(valid_values)
    max_val = max(valid_values)
    range_val = max_val - min_val
    
    normalized = {}
    for k, v in values_dict.items():
        if np.isnan(v) or np.isinf(v):
            normalized[k] = 0.0
        elif range_val == 0:
            normalized[k] = 0.5
        else:
            normalized[k] = (v - min_val) / range_val
    
    return normalized


def print_deviation_variation_table(parameters, variation_table):
    rows = []
    for parameter in parameters:
        for partner, variance in sorted(variation_table.get(parameter, {}).items()):
            rows.append((parameter, partner, variance))

    if not rows:
        print("No deviation variation data available.")
        return

    col1 = max(len("Parameter"), max(len(row[0]) for row in rows))
    col2 = max(len("Partner"), max(len(row[1]) for row in rows))
    col3 = len("Deviation Var")

    print("\nVariation of deviations by parameter and partner:")
    print(f"{ 'Parameter'.ljust(col1) }  { 'Partner'.ljust(col2) }  { 'Deviation Var'.rjust(col3) }")
    print(f"{ '-' * col1 }  { '-' * col2 }  { '-' * col3 }")
    for parameter, partner, variance in rows:
        print(f"{ parameter.ljust(col1) }  { partner.ljust(col2) }  {variance:>{col3}.6f}")


def _nan_rmse(a, b, ignore_indices=None):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if ignore_indices is not None:
        ignore = np.zeros_like(mask, dtype=bool)
        ignore[list(ignore_indices)] = True
        mask &= ~ignore
    if not np.any(mask):
        return np.nan
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def _compute_parameter_distance_matrix(matrix):
    n = matrix.shape[0]
    dist = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d_row = _nan_rmse(matrix[i, :], matrix[j, :], ignore_indices={i, j})
            d_col = _nan_rmse(matrix[:, i], matrix[:, j], ignore_indices={i, j})
            ds = [d for d in (d_row, d_col) if not np.isnan(d)]
            if ds:
                dist[i, j] = dist[j, i] = float(np.mean(ds))
    finite = dist[np.isfinite(dist)]
    if finite.size:
        max_val = np.nanmax(finite)
        dist[np.isnan(dist)] = max_val * 10.0 + 1.0
    else:
        dist[np.isnan(dist)] = 1.0
    np.fill_diagonal(dist, np.inf)
    return dist


def _hierarchical_average_linkage_order(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 1:
        return list(range(n))

    clusters = [[i] for i in range(n)]
    sizes = [1] * n
    dist = distance_matrix.copy()
    while len(clusters) > 1:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if j <= i:
            i, j = j, i

        new_cluster = clusters[i] + clusters[j]
        new_size = sizes[i] + sizes[j]

        keep = [k for k in range(len(clusters)) if k not in (i, j)]
        new_dist = np.full((len(keep) + 1, len(keep) + 1), np.inf, dtype=float)
        for a, ka in enumerate(keep):
            for b, kb in enumerate(keep):
                new_dist[a, b] = dist[ka, kb]
        for a, ka in enumerate(keep):
            new_dist[a, -1] = new_dist[-1, a] = (
                dist[ka, i] * sizes[i] + dist[ka, j] * sizes[j]
            ) / float(sizes[i] + sizes[j])

        clusters = [clusters[k] for k in keep] + [new_cluster]
        sizes = [sizes[k] for k in keep] + [new_size]
        dist = new_dist

    return clusters[0]


def plot_variation_heatmap(parameters, variation_table, output_path):
    n = len(parameters)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i, parameter in enumerate(parameters):
        for j, partner in enumerate(parameters):
            if parameter == partner:
                continue
            matrix[i, j] = variation_table.get(parameter, {}).get(partner, np.nan)

    order = _hierarchical_average_linkage_order(_compute_parameter_distance_matrix(matrix))
    ordered_parameters = [parameters[i] for i in order]
    ordered_matrix = matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * n + 3.0), max(8.0, 0.5 * n + 3.0)))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(ordered_matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([short_label(p) for p in ordered_parameters], rotation=90)
    ax.set_yticklabels([short_label(p) for p in ordered_parameters])
    ax.set_xlabel("Partner")
    ax.set_ylabel("Parameter")
    ax.set_title("Deviation variance heatmap (clustered)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Variance")

    outlier_cells = []
    if np.any(~np.isnan(ordered_matrix)):
        max_val = np.nanmax(ordered_matrix)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                value = ordered_matrix[i, j]
                if np.isnan(value):
                    continue

                # Determine whether this value stands out from its immediate neighbors.
                neighbors = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni = i + di
                        nj = j + dj
                        if di == 0 and dj == 0:
                            continue
                        if ni < 0 or nj < 0 or ni >= n or nj >= n:
                            continue
                        neighbor_value = ordered_matrix[ni, nj]
                        if not np.isnan(neighbor_value):
                            neighbors.append(neighbor_value)

                if neighbors:
                    neighbor_mean = float(np.mean(neighbors))
                    neighbor_std = float(np.std(neighbors))
                    diff = abs(value - neighbor_mean)
                    if neighbor_std > 0:
                        if diff > 3.5 * neighbor_std or diff > 0.40 * max(1.0, neighbor_mean):
                            outlier_cells.append((i, j))
                    elif diff > 0:
                        outlier_cells.append((i, j))

                color = "white" if value > max_val / 2 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=6)

    from matplotlib.patches import Rectangle
    for i, j in outlier_cells:
        rect = Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor="red", linewidth=1.5)
        ax.add_patch(rect)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_emergent_interaction(parameters, hi_values, output_path):
    size = max(10.0, 0.42 * len(parameters) + 4.0)
    fig, ax = plt.subplots(figsize=(size, 6.0))

    x = np.arange(len(parameters))
    values = [hi_values.get(parameter, np.nan) for parameter in parameters]
    bars = ax.bar(x, values, color="steelblue", edgecolor="black", linewidth=0.8)
    ax.set_ylabel("H_i")
    ax.set_title("Interaction-only signal by parameter")
    ax.set_xticks(x, labels=[short_label(param) for param in parameters], rotation=90)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_hi_si_comparison(parameters, hi_values_norm, si_values_norm, output_path):
    """Plot normalized Hi and Si values with their ratio."""
    size = max(12.0, 0.42 * len(parameters) + 4.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(size, 10.0))

    x = np.arange(len(parameters))
    width = 0.35

    # Plot 1: Normalized Hi and Si values
    hi_data = [hi_values_norm.get(param, 0.0) for param in parameters]
    si_data = [si_values_norm.get(param, 0.0) for param in parameters]

    bars1 = ax1.bar(x - width / 2, hi_data, width, label="H_i (normalized)", color="steelblue", edgecolor="black", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, si_data, width, label="S_i (normalized)", color="coral", edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("Normalized Value [0, 1]")
    ax1.set_title("Normalized H_i and S_i by parameter")
    ax1.set_xticks(x, labels=[short_label(param) for param in parameters], rotation=90)
    ax1.legend()
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.set_ylim(0, 1.1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Plot 2: Hi/Si ratio
    ratio_data = []
    for param in parameters:
        hi_norm = hi_values_norm.get(param, 0.0)
        si_norm = si_values_norm.get(param, 0.0)
        if si_norm > 0:
            ratio = hi_norm / si_norm
        else:
            ratio = 0.0 if hi_norm == 0 else float('inf')
        ratio_data.append(ratio)

    # Clip infinite values for visualization
    ratio_data_clipped = [min(r, 10.0) if r != float('inf') else 10.0 for r in ratio_data]

    bars3 = ax2.bar(x, ratio_data_clipped, color="green", edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("H_i / S_i")
    ax2.set_title("Interaction-to-Self Variance Ratio (H_i / S_i)")
    ax2.set_xticks(x, labels=[short_label(param) for param in parameters], rotation=90)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars3, ratio_data)):
        height = bar.get_height()
        label_text = f"{ratio:.2f}" if ratio != float('inf') else "∞"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot emergent interactions from a single experiment database.")
    parser.add_argument("--db-path", type=Path, required=True, help="SQLite database path containing pairwise and single-sweep trials.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to pairwise_emergent_interaction_hi.png next to the database.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"Database not found: {args.db_path}")

    parameters, pairwise_series = read_pairwise_job_series(args.db_path)
    if not parameters:
        raise SystemExit("No pairwise sweep rows found in the database.")

    # Read single sweep data for Si values and 1D parameter data
    si_values, one_d_successes_by_parameter = read_single_sweep_series(args.db_path)
    
    _, hi_values, variation_table = compute_hi(parameters, pairwise_series, one_d_successes_by_parameter)
    print_deviation_variation_table(parameters, variation_table)
    si_values_aligned = {param: si_values.get(param, 0.0) for param in parameters}
    
    # Normalize both Hi and Si to [0, 1]
    hi_values_norm = normalize_to_01(hi_values)
    si_values_norm = normalize_to_01(si_values_aligned)
    
    print("\n" + "="*60)
    print("Normalized H_i and S_i values:")
    print("="*60)
    print(f"{'Parameter':<40} {'H_i (norm)':<15} {'S_i (norm)':<15} {'H_i/S_i':<15}")
    print("-"*60)
    for param in parameters:
        hi_norm = hi_values_norm.get(param, 0.0)
        si_norm = si_values_norm.get(param, 0.0)
        if si_norm > 0:
            ratio = hi_norm / si_norm
            ratio_str = f"{ratio:.4f}"
        else:
            ratio_str = "∞" if hi_norm > 0 else "0.0000"
        print(f"{short_label(param):<40} {hi_norm:<15.4f} {si_norm:<15.4f} {ratio_str:<15}")
    print("="*60)

    output_path = args.output or args.db_path.with_name("pairwise_emergent_interaction_hi.png")
    plot_emergent_interaction(parameters, hi_values, output_path)
    print(f"Saved {output_path}")

    heatmap_path = args.db_path.with_name("pairwise_deviation_variation_heatmap.png")
    plot_variation_heatmap(parameters, variation_table, heatmap_path)
    print(f"Saved {heatmap_path}")
    
    # Plot Hi/Si comparison
    comparison_path = args.db_path.with_name("pairwise_hi_si_comparison.png")
    plot_hi_si_comparison(parameters, hi_values_norm, si_values_norm, comparison_path)
    print(f"Saved {comparison_path}")


if __name__ == "__main__":
    main()
