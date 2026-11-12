"""Residual plots for pairwise sweep analysis."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .utils import short_label, _get_matplotlib_pyplot, resolve_parameter_selection

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
        y_hi = 0.5
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

    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        f"Group {index + 1}": [member for member in members if member in parameters]
        for index, (_, members) in enumerate(normalized_groups)
    }
