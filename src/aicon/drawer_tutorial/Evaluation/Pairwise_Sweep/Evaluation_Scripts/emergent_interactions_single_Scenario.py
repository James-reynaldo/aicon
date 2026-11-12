"""Main script for plotting emergent interactions from a single experiment database."""
import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Matplotlib configuration
plt.rcParams.update({
    # Font family and LaTeX-like math
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    # Consistent typography
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.unicode_minus": False,
})

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
from .data_reading import read_pairwise_job_series, read_single_sweep_series
from .anova_analysis import compute_hi, calculate_ANOVA_interaction_score
from .heatmap_generation import heatmap_ss_ab
from .residual_plots import plot_si_vs_residuals
from .early_late_comparison import plot_early_vs_late_scatter, compute_window_metrics_summary

SI_SEPERATE = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}

# Scenario configurations for 2x2 subplot
SCENARIO_CONFIGS = {
    "Normal": {
        "db_path": "data_combine/normal/experiment_store.db",
        "row": 0, "col": 0,
        "title": "Normal"
    },
    "Noise": {
        "db_path": "data_combine/noise/experiment_store.db",
        "row": 0, "col": 1,
        "title": "Noise"
    },
    "Disturbance": {
        "db_path": "data_combine/disturbance/experiment_store.db",
        "row": 1, "col": 0,
        "title": "Disturbance"
    },
    "Bad Prior": {
        "db_path": "data_combine/bad_prior/experiment_store.db",
        "row": 1, "col": 1,
        "title": "Bad Prior"
    }
}


def get_scenario_from_db_path(db_path):
    """Extract scenario name from database path.
    
    Args:
        db_path: Path to the experiment database
        
    Returns:
        Scenario name: 'Normal', 'Noise', 'Disturbance', or 'Bad Prior'
    """
    db_str = str(db_path).lower()
    
    if "normal" in db_str:
        return "Normal"
    elif "noise" in db_str:
        return "Noise"
    elif "disturbance" in db_str:
        return "Disturbance"
    elif "bad_prior" in db_str or "badprior" in db_str:
        return "Bad Prior"
    else:
        # Default to Normal if no match found
        return "Normal"


def plot_si_vs_mean_residuals_2x2(scenario_data, output_dir, metrics, highlight_groups, x_threshold, y_threshold, y_max_threshold):
    """Plot Si vs Mean residuals for all 4 scenarios in a 2x2 subplot.
    
    Args:
        scenario_data: dict mapping scenario name to (parameters, si_values, mean_ssab_s, ss_ab_table)
        output_dir: Output directory for the plot
        metrics: Metric being analyzed
        highlight_groups: Groups to highlight
        x_threshold, y_threshold, y_max_threshold: Threshold values
    """
    plt = _get_matplotlib_pyplot()
    
    # Create 2x2 subplot with shared axes
    # A4 portrait: 210mm x 297mm (8.27in x 11.69in)
    # Usable area with margins: ~160mm x 240mm (6.30in x 9.45in)
    # 2x2 grid with 4 subplots, each ~65mm x 65mm for good aspect ratio
    # Total plot area: ~130mm x 130mm (5.12in x 5.12in)
    # Adding space for titles, labels, and legend: total figure ~5.5in x 7.5in
    fig_width = 5.5  # inches (139.7mm) - fits within A4 portrait width
    fig_height = 7.5  # inches (190.5mm) - fits within A4 portrait height
    
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), 
                             sharex='col', sharey='row',
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.25})
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Common settings
    x_limits = {}
    y_limits = {}
    
    # First pass: determine common limits for each column/row
    for scenario_name, (parameters, si_values, mean_ssab_s, ss_ab_table) in scenario_data.items():
        if scenario_name not in SCENARIO_CONFIGS:
            continue
        
        config = SCENARIO_CONFIGS[scenario_name]
        row, col = config["row"], config["col"]
        
        # Compute residuals
        max_ssab_s = {}
        for (pa, pb), interaction in ss_ab_table.items():
            if np.isnan(interaction):
                continue
            max_ssab_s[pa] = max(max_ssab_s.get(pa, -np.inf), interaction)
            max_ssab_s[pb] = max(max_ssab_s.get(pb, -np.inf), interaction)
        
        # Collect data points
        x_vals = []
        y_vals = []
        for param in parameters:
            si = si_values.get(param, np.nan)
            residual = mean_ssab_s.get(param, np.nan)
            if not np.isnan(si) and not np.isnan(residual):
                x_vals.append(si)
                y_vals.append(residual)
        
        # Update limits for this column/row
        if col not in x_limits:
            x_limits[col] = [float('inf'), float('-inf')]
        if row not in y_limits:
            y_limits[row] = [float('inf'), float('-inf')]
        
        if x_vals:
            x_min, x_max = min(x_vals), max(x_vals)
            x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 0.1
            x_limits[col][0] = min(x_limits[col][0], x_min - x_pad)
            x_limits[col][1] = max(x_limits[col][1], x_max + x_pad)
        
        if y_vals:
            y_min, y_max = min(y_vals), max(y_vals)
            y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
            y_limits[row][0] = min(y_limits[row][0], y_min - y_pad)
            y_limits[row][1] = max(y_limits[row][1], y_max + y_pad)
    
    # Set default limits if empty
    if not x_limits:
        x_limits = {0: [0, 0.5], 1: [0, 0.5]}
    if not y_limits:
        y_limits = {0: [0, 0.5], 1: [0, 0.5]}
    
    # Second pass: plot each scenario
    first_subplot_done = False
    all_groups = []
    for scenario_name, (parameters, si_values, mean_ssab_s, ss_ab_table) in scenario_data.items():
        if scenario_name not in SCENARIO_CONFIGS:
            continue
        
        config = SCENARIO_CONFIGS[scenario_name]
        row, col = config["row"], config["col"]
        ax = axes_flat[row * 2 + col]
        
        # Compute residuals
        max_ssab_s = {}
        for (pa, pb), interaction in ss_ab_table.items():
            if np.isnan(interaction):
                continue
            max_ssab_s[pa] = max(max_ssab_s.get(pa, -np.inf), interaction)
            max_ssab_s[pb] = max(max_ssab_s.get(pb, -np.inf), interaction)
        
        # Collect groups for global legend
        groups = [p.split('.', 1)[0] if '.' in p else p for p in parameters]
        # Replace group names for legend
        groups = ['visibility connection' if g == 'visible connection' else g for g in groups]
        groups = ['visibility estimator' if g == 'visible' else g for g in groups]
        groups = ['grasp connection' if g == 'connection' else g for g in groups]
        for g in groups:
            if g not in all_groups:
                all_groups.append(g)
        
        # Plot using the existing function's logic but adapted for subplot
        plot_si_vs_residuals_subplot(
            ax, parameters, si_values, max_ssab_s, mean_ssab_s, ss_ab_table,
            highlight_groups, metrics, x_threshold, y_threshold, y_max_threshold,
            x_limits[col], y_limits[row],
            title=config["title"],
            add_legend=False
        )
        first_subplot_done = True
    
    # Add common axis labels with LaTeX-style subscripts
    fig.text(0.5, 0.03, r'Individual Sensitivity $S_{p}$', ha='center', fontsize=9)
    fig.text(0.0, 0.5, r'Mean RMS Interaction $I_{AB}$', va='center', rotation='vertical', fontsize=9)
    
    # Add global group legend at bottom in two rows
    group_colour_map = {g: plt.get_cmap('tab10')(i % 10) for i, g in enumerate(all_groups)}
    group_handles = [
        Line2D([0], [0], marker='o', color='w', label=g,
               markerfacecolor=group_colour_map[g], markeredgecolor='black', markersize=6)
        for g in all_groups
    ]
    
    # Add visibility connection with grey dot if not already in legend
    if 'visibility connection' not in all_groups:
        group_handles.append(
            Line2D([0], [0], marker='o', color='w', label='visibility connection',
                   markerfacecolor='grey', markeredgecolor='black', markersize=6)
        )
    
    if group_handles:
        fig.legend(handles=group_handles, title='Group', loc='lower center',
                   fontsize=7, ncol=5, bbox_to_anchor=(0.5, -0.06))
    
    # Save the plot
    output_path = output_dir / "si_vs_mean_residuals_2x2.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PDF with high quality for thesis
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    # Print figure dimensions for verification
    print(f"2x2 Si vs Mean residuals plot saved to: {output_path}")
    print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches ({fig_width*25.4:.1f} x {fig_height*25.4:.1f} mm)")
    print(f"Font sizes: axis labels=9pt, legend=7pt, titles=10pt")
    print(f"Designed for A4 portrait page (~140mm width)")
    
    # Also save PNG preview
    png_path = str(output_path).replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    print(f"PNG preview saved to: {png_path}")
    
    plt.close(fig)


def plot_si_vs_residuals_subplot(ax, parameters, si_values, max_ssab_s, mean_ssab_s, ss_ab_table,
                                  highlight_groups, metrics, x_threshold, y_threshold, y_max_threshold,
                                  x_limits, y_limits, title="", add_legend=True):
    """Plot Si vs residuals on a single subplot axis."""
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    
    # Determine plotting bounds
    x_lo, x_hi = x_limits
    y_lo, y_hi = y_limits
    
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title, fontsize=10)
    
    # Color points by their parameter 'group' (prefix before the first dot)
    groups = [p.split('.', 1)[0] if '.' in p else p for p in parameters]
    # Replace group names for legend
    groups = ['visibility connection' if g == 'visible connection' else g for g in groups]
    groups = ['visibility estimator' if g == 'visible' else g for g in groups]
    groups = ['grasp connection' if g == 'connection' else g for g in groups]
    unique_groups = list(dict.fromkeys(groups))
    group_colour_map = {g: plt.get_cmap('tab10')(i % 10) for i, g in enumerate(unique_groups)}
    
    # Marker/alpha/size per series name (reduced for smaller subplots)
    series_markers = {
        "Mean residual": {'marker': 'o', 'alpha': 0.95, 's': 60, 'z': 2},
    }
    
    # Plot each series
    series_seen = set()
    target_param = "connection.DistGraspHandConnection.close_dist_threshold"
    for series_name, residuals_by_parameter in [("Mean residual", mean_ssab_s)]:
        props = series_markers.get(series_name, {'marker': 'o', 'alpha': 0.95, 's': 80, 'z': 2})
        for param in parameters:
            si = si_values.get(param, np.nan)
            residual = residuals_by_parameter.get(param, np.nan)
            if np.isnan(si) or np.isnan(residual):
                continue
            group = param.split('.', 1)[0] if '.' in param else param
            # Replace group names for legend
            group = 'visibility connection' if group == 'visible connection' else group
            group = 'visibility estimator' if group == 'visible' else group
            group = 'grasp connection' if group == 'connection' else group
            colour = group_colour_map.get(group, '#808080')
            # Print greyed out parameters
            if colour == '#808080':
                print(f"Grey parameter: {param} (group: {group})")
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
            # Label only the close_dist_threshold parameter
            if param == target_param:
                ax.annotate(
                    "close dist threshold",
                    (si, residual),
                    fontsize=7,
                    alpha=0.9,
                    xytext=(-55, -15),
                    textcoords="offset points",
                    zorder=10,
                )
        series_seen.add(series_name)
    

    # Add threshold lines if specified
    if x_threshold is not None and y_threshold is not None:
        ax.axvline(x=x_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=y_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)


def plot_all_scenarios_2x2(args):
    """Process all 4 scenarios and create a 2x2 subplot for si_vs_mean_residuals."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent / "Ignored_files"
    
    # Collect data from all scenarios
    scenario_data = {}
    
    for scenario_name, config in SCENARIO_CONFIGS.items():
        db_path = project_root / config["db_path"]
        
        if not db_path.exists():
            print(f"Database not found for {scenario_name}: {db_path}")
            continue
        
        print(f"Processing {scenario_name} scenario...")
        
        # Read data for this scenario
        parameters, pairwise_series = read_pairwise_job_series(
            db_path,
            metrics=args.metric,
            include_failed_trials=args.include_failed_trials,
        )
        
        si_values, one_d_successes_by_parameter, standard_baseline = read_single_sweep_series(
            db_path,
            metrics=args.metric,
            include_failed_trials=args.include_failed_trials,
        )
        
        _, _, _, _, augmented_pair_vectors, augmented_pair_diagnostics = compute_hi(
            parameters, pairwise_series, one_d_successes_by_parameter, standard_baseline
        )
        
        _, _, ss_ab_table = calculate_ANOVA_interaction_score(
            parameters, augmented_pair_vectors, augmented_pair_diagnostics
        )
        
        # Compute mean residuals
        mean_ssab_s = {}
        for (pa, pb), interaction in ss_ab_table.items():
            if np.isnan(interaction):
                continue
            mean_ssab_s[pa] = mean_ssab_s.get(pa, []) + [max(0, interaction)]
            mean_ssab_s[pb] = mean_ssab_s.get(pb, []) + [max(0, interaction)]
        mean_ssab_s = {p: float(np.mean(vals)) for p, vals in mean_ssab_s.items()}
        
        scenario_data[scenario_name] = (parameters, si_values, mean_ssab_s, ss_ab_table)
    
    if not scenario_data:
        print("No scenario data found. Exiting.")
        return
    
    # Determine output directory
    if args.output:
        out_arg = Path(args.output)
        if str(out_arg).endswith(('/', '\\')) or out_arg.exists() and out_arg.is_dir():
            output_dir = out_arg
        else:
            output_dir = out_arg.parent
    else:
        # Use first scenario's output directory
        first_db = list(scenario_data.values())[0][3].keys() if scenario_data else []
        if first_db:
            output_dir = project_root / "metrics" / args.metric
        else:
            output_dir = Path(".")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse highlight groups
    highlight_groups = [
        [item.strip() for item in group.split(",") if item.strip()]
        for group in args.highlight_group
        if any(item.strip() for item in group.split(","))
    ]
    
    # Create 2x2 plot
    plot_si_vs_mean_residuals_2x2(
        scenario_data, output_dir, args.metric,
        highlight_groups, args.x_threshold, args.y_threshold, args.y_max_threshold
    )


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
    parser.add_argument(
        "--two-by-two",
        action="store_true",
        help="Create 2x2 subplot with all scenarios (Normal, Noise, Disturbance, Bad Prior).",
    )
    args = parser.parse_args()

    # Handle 2x2 mode - process all scenarios
    if args.two_by_two:
        plot_all_scenarios_2x2(args)
        return

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
    _, normalized_interaction_ss, ss_ab_table = calculate_ANOVA_interaction_score(
        parameters, augmented_pair_vectors, augmented_pair_diagnostics
    )

    print("number of pairs:", len(normalized_interaction_ss))
    print("number of exact zeros ss_total:", sum(1 for v in normalized_interaction_ss.values() if v == 0))

    min_ss_total = 1e-6 # just as long as not zero, to avoid division by zero in interaction score calculation

    # scatter_plot_interaction_vs_total(min_ss_total, parameters, normalized_interaction_ss, ss_ab_table, output_dir / "interaction_vs_total.png")

    # if file already exists don't overwrite it, just print a message and skip
    # Detect scenario from database path for file naming
    scenario = get_scenario_from_db_path(db_path)
    scenario_filename = scenario.replace(" ", "_")  # Convert "Bad Prior" to "Bad_Prior"
    heatmap_path_normalized = output_dir / f"ss_ab_heatmap_normalized_{scenario_filename}.pdf"
    heatmap_path_ssab = output_dir / f"ss_ab_heatmap_normalized_ssab_{scenario_filename}.pdf"
    
    # Check if either heatmap already exists
    if heatmap_path_normalized.exists() or heatmap_path_ssab.exists():
        print(f"Heatmap files already exist: {heatmap_path_normalized}, {heatmap_path_ssab}. Skipping heatmap generation.")
    else:
        # Define parameters to highlight for heatmap labeling
        highlight_params = {
            "connection.DistGraspHandConnection.close_dist_threshold",
            "connection.DistGraspHandConnection.uncertainty_dist_threshold",
        }
        try:
            heatmap_ss_ab(ss_ab_table, heatmap_path_ssab, max_value=0.3, highlight_params=highlight_params, scenario=scenario)
            # heatmap_ss_ab(normalized_interaction_ss, heatmap_path_normalized, highlight_params=highlight_params)
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
