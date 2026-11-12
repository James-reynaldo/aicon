"""Early vs late comparison for pairwise sweep analysis."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .utils import short_label, _filter_rows_by_trial_window, _get_matplotlib_pyplot
from .anova_analysis import calculate_ANOVA_interaction_score, compute_hi


def compute_window_metrics_summary(db_path, metrics="success", include_failed_trials=False, trial_window="all"):
    """Compute the standard per-parameter SI and residual summaries for a given trial window."""
    from .data_reading import read_pairwise_job_series, read_single_sweep_series
    from .anova_analysis import compute_hi
    
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

    # Compute per-parameter max and mean SS_AB values
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

    # Write results to CSV (place next to DB under metrics/<metric>/)
    import csv
    from pathlib import Path
    output_dir = Path(db_path).parent / "metrics" / metrics
    csv_path = output_dir / "hub_parameters_interaction_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Hub parameter analysis
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

    # Note: CSV export removed per user request
    # print(f"Saved hub interaction summary to {csv_path}")

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
    """Plot scatter plot comparing early vs late window values."""
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

    # Note: CSV export removed per user request
    # catalog_path = output_path.with_name(f"{output_path.stem}_catalog.csv")
    # with open(catalog_path, "w", encoding="utf-8", newline="") as fh:
    #     fieldnames = ["point_number", "parameter", x_label, y_label]
    #     writer = csv.DictWriter(fh, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(catalog_rows)
