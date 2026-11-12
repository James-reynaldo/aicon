import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

from .sweep_analysis_utils import (
    aggregate_metrics_for_plot,
    build_anchor_error_outcome_summary,
    build_metric_grid_inputs,
    build_summary,
    filter_rows_by_sweep_label,
    filter_rows_for_sweep,
    normalize_sweep_label,
    read_results_from_db,
)

def main():
    parser = argparse.ArgumentParser(description="Plot success rate vs sweep value for each param_name.")
    parser.add_argument(
        "--db-path-normal",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--db-path-noise",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--db-path-disturbance",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--db-path-bad-prior",
        default="",
        help="SQLite experiment database path to read results from",
    )
    parser.add_argument(
        "--sweep-label",
        choices=["negative", "zero", "positive","x1000", "all","Non_reasonable"],
        default="negative",
        help="Which sweep values to analyze: negative, zero, positive, or all.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=["success_rate", "avg_timesteps", "avg_error", "avg_anchor_error", "avg_kinematic_axis_error", "epistemicity"],
        default=["success_rate"],
        help="Metric(s) to compute and plot. Can specify multiple metrics.",
    )
    parser.add_argument(
        "--include-failed-trials",
        action="store_true",
        help="Include failed trials when computing per-label means (affects avg_* metrics and S_i).",
    )
    args = parser.parse_args()

    need_epistemicity = True

    all_rows_normal = []
    if args.db_path_normal:
        all_rows_normal = read_results_from_db(Path(args.db_path_normal), compute_epistemicity=need_epistemicity)
    all_rows_noise = []
    if args.db_path_noise:
        all_rows_noise = read_results_from_db(Path(args.db_path_noise), compute_epistemicity=need_epistemicity)
    all_rows_disturbance = []
    if args.db_path_disturbance:
        all_rows_disturbance = read_results_from_db(Path(args.db_path_disturbance), compute_epistemicity=need_epistemicity)
    all_rows_bad_prior = []
    if args.db_path_bad_prior:
        all_rows_bad_prior = read_results_from_db(Path(args.db_path_bad_prior), compute_epistemicity=need_epistemicity)

    filtered_rows = filter_rows_for_sweep(
        all_rows_normal,
        all_rows_noise,
        all_rows_disturbance,
        all_rows_bad_prior,
        args.sweep_label,
    )

    metric_summaries, stable_by_metric = aggregate_metrics_for_plot(
        filtered_rows,
        include_failed_trials=args.include_failed_trials,
        sweep_label=args.sweep_label,
    )

    # Create two 2x1 figures with metrics split
    # Figure 1: Success Rate & Avg Timesteps
    # Figure 2: Avg Anchor Error & Epistemicity
    metric_names = ["success_rate", "avg_timesteps", "avg_anchor_error", "epistemicity"]
    
    # Determine output directory based on sweep label
    # Use the specified output directory
    out_dir = Path("/scratch/aldo/aicon/src/aicon/drawer_tutorial/Ignored_files/data/Single/Plots")

    # A4 landscape: 297 × 210 mm
    # Usable width: 277 mm (20 mm margins on each side)
    # Usable height: ~135 mm per figure (accounting for caption space and vertical spacing)
    # Convert to inches: width=10.9, height=5.3
    fig_width = 10.9
    fig_height = 5.3
    
    # Collect all parameter names from the first metric (success_rate)
    summary_normal = metric_summaries["success_rate"]["normal"]
    all_data = []
    for param_group, param_data in summary_normal.items():
        for item in sorted(param_data, key=lambda x: x[0]):
            param_name = item[0]
            success_rate = item[1] if len(item) > 1 else 0.0
            var = item[2] if len(item) > 2 else 0.0
            all_data.append((param_name, success_rate, var, param_group))
    
    param_names = [item[0] for item in all_data]
    param_groups = [item[3] for item in all_data]
    x = list(range(len(param_names)))
    x_map = {(param_groups[i], param_names[i]): i for i in range(len(all_data))}
    
    # Define parameters to highlight (full name format: group.param_name)
    highlight_params = {
        "connection.DistGraspHandConnection.close_dist_threshold",
        "connection.DistGraspHandConnection.gripper_activation_threshold",
        "connection.DistGraspHandConnection.uncertainty_bias",
        "drawer_position.forward_noise_base",
        "drawer_position.forward_noise_grasped_coeff",
        "ee_pose.proprio_update_noise",
    }
    
    # Define parameters to label on x-axis for figure 1 (Success Rate & Avg Timesteps)
    # Only show close_dist_threshold and gripper_activation_threshold
    highlight_params_fig1 = {
        "connection.DistGraspHandConnection.close_dist_threshold",
        "connection.DistGraspHandConnection.gripper_activation_threshold",
    }
    
    # Create filtered lists for x-axis labels (only highlighted params)
    # Build a mapping from (group, name) to index for quick lookup
    all_x_map = {(param_groups[i], param_names[i]): i for i in range(len(all_data))}
    
    # Create labels that only show highlighted parameter names
    x_labels = []
    for i, (pn, pg) in enumerate(zip(param_names, param_groups)):
        full_param_name = f"{pg}.{pn}" if pg else pn
        if full_param_name in highlight_params:
            x_labels.append(pn)  # Show only the parameter name (not group)
        else:
            x_labels.append("")  # Empty label for non-highlighted params
    
    # For figure 1, only show labels for close_dist_threshold and gripper_activation_threshold
    x_labels_fig1 = []
    for i, (pn, pg) in enumerate(zip(param_names, param_groups)):
        full_param_name = f"{pg}.{pn}" if pg else pn
        if full_param_name in highlight_params_fig1:
            x_labels_fig1.append(pn)  # Show only the parameter name (not group)
        else:
            x_labels_fig1.append("")  # Empty label for non-highlighted params
    
    # Debug: print actual parameter names
    print("Actual parameter names in database:")
    for i, (pn, pg) in enumerate(zip(param_names, param_groups)):
        full_param_name = f"{pg}.{pn}" if pg else pn
        print(f"  {i}: '{full_param_name}'")
    
    # Find group boundaries for vertical dashed lines
    group_boundaries = []
    for i in range(len(param_groups) - 1):
        if param_groups[i] != param_groups[i + 1]:
            group_boundaries.append(i + 0.5)  # +0.5 to place line between indices
    
    # Extract unique group names in order and calculate group centers for labels
    group_names = []
    group_centers = []
    if param_groups:
        first_group = param_groups[0].split('.')[-1] if '.' in param_groups[0] else param_groups[0]
        # Add line breaks to long group names
        if first_group == "DistGraspHandConnection":
            first_group = "Grasp\nConnection"
        elif first_group == "VisibleConnection":
            first_group = "Visibility\nConnection"
        elif first_group == "grasp_likelihood":
            first_group = "grasp\nlikelihood"
        elif first_group == "VisibleEEDrawerConnection":
            first_group = "Visibility\nConnection"
        elif first_group == "visible":
            first_group = "Visibility\nEstimator"
        group_names.append(first_group)
        group_start = 0
        for i, pg in enumerate(param_groups):
            pg_display = pg.split('.')[-1] if '.' in pg else pg
            # Add line breaks to long group names
            if pg_display == "DistGraspHandConnection":
                pg_display = "Grasp\nConnection"
            elif pg_display == "VisibleConnection":
                pg_display = "Visibility\nConnection"
            elif pg_display == "grasp_likelihood":
                pg_display = "grasp\nlikelihood"
            elif pg_display == "VisibleEEDrawerConnection":
                pg_display = "Visibility\nConnection"
            elif pg_display == "visible":
                pg_display = "Visibility\nEstimator"
            if pg_display != group_names[-1]:
                # End of previous group, calculate center
                group_centers.append((group_start + i - 1) / 2.0)
                group_names.append(pg_display)
                group_start = i
        # Don't forget the last group
        group_centers.append((group_start + len(param_groups) - 1) / 2.0)
    
    # Helper function to plot a single scenario on a given axis
    def plot_scenario_on_axis(ax, summary, label, color, marker, metric, stable_set):
        """Plot a single scenario on the given axis."""
        scenario_x = []
        scenario_y = []
        scenario_vars = []
        
        for pg, param_data in summary.items():
            for item in param_data:
                name = item[0]
                rate = item[1] if len(item) > 1 else 0.0
                var = item[2] if len(item) > 2 else 0.0
                key = (pg, name)
                if key in x_map:
                    scenario_x.append(x_map[key])
                    scenario_y.append(rate)
                    scenario_vars.append(var)
        
        if not scenario_x:
            return ax
        
        pairs = sorted(zip(scenario_x, scenario_y, scenario_vars), key=lambda t: t[0])
        sx, sy, svar = zip(*pairs)
        
        ax.plot(
            list(sx),
            list(sy),
            marker=marker,
            linestyle="-",
            linewidth=2,
            color=color,
            label=label,
        )
        
        # Add fill for variance if available
        if svar:
            stds = np.sqrt(np.array(svar))
            if metric == "success_rate":
                lower = np.clip(np.array(sy) - stds, 0.0, 1.0)
                upper = np.clip(np.array(sy) + stds, 0.0, 1.0)
            else:
                lower = np.array(sy) - stds
                upper = np.array(sy) + stds
            ax.fill_between(list(sx), lower, upper, color=color, alpha=0.15, label=rf"{label} $\pm$1 std")
        
        # Highlight specific parameters
        highlight_x = []
        highlight_y = []
        for i, (pn, pg) in enumerate(zip(param_names, param_groups)):
            full_param_name = f"{pg}.{pn}" if pg else pn
            if full_param_name in highlight_params and i in list(sx):
                highlight_x.append(i)
                highlight_y.append(sy[list(sx).index(i)])

        # Removed highlighting circles and legend
        
        # Plot stable points if available
        if stable_set:
            stable_x = []
            stable_y = []
            for i, (pn, pg) in enumerate(zip(param_names, param_groups)):
                if (pg, pn) in stable_set and i in list(sx):
                    stable_x.append(i)
                    stable_y.append(sy[list(sx).index(i)])
            
            if stable_x:
                ax.scatter(
                    stable_x,
                    stable_y,
                    s=100,
                    marker='*',
                    color='black',
                    label='_nolegend_',
                    zorder=10,
                )
        
        # Add dashed vertical lines between groups
        for boundary in group_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        return ax
    
    # Create Figure 1: Success Rate & Avg Timesteps
    fig1, axes1 = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    
    # Subplot 1: Success Rate - all scenarios
    axes1[0].set_title(r"Success Rate")
    plot_scenario_on_axis(axes1[0], metric_summaries["success_rate"]["normal"], "Normal", "steelblue", "o", "success_rate", stable_by_metric["success_rate"]["normal"])
    if metric_summaries["success_rate"]["noise"]:
        plot_scenario_on_axis(axes1[0], metric_summaries["success_rate"]["noise"], "Noise", "orange", "x", "success_rate", stable_by_metric["success_rate"]["noise"])
    if metric_summaries["success_rate"]["disturbance"]:
        plot_scenario_on_axis(axes1[0], metric_summaries["success_rate"]["disturbance"], "Disturbance", "green", "s", "success_rate", stable_by_metric["success_rate"]["disturbance"])
    if metric_summaries["success_rate"]["bad_prior"]:
        plot_scenario_on_axis(axes1[0], metric_summaries["success_rate"]["bad_prior"], "Bad Prior", "firebrick", "^", "success_rate", stable_by_metric["success_rate"]["bad_prior"])
    axes1[0].set_ylabel(r"Success Rate")
    axes1[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Subplot 2: Avg Timesteps - all scenarios
    axes1[1].set_title(r"Avg Timesteps (successes only)")
    plot_scenario_on_axis(axes1[1], metric_summaries["avg_timesteps"]["normal"], "Normal", "steelblue", "o", "avg_timesteps", stable_by_metric["avg_timesteps"]["normal"])
    if metric_summaries["avg_timesteps"]["noise"]:
        plot_scenario_on_axis(axes1[1], metric_summaries["avg_timesteps"]["noise"], "Noise", "orange", "x", "avg_timesteps", stable_by_metric["avg_timesteps"]["noise"])
    if metric_summaries["avg_timesteps"]["disturbance"]:
        plot_scenario_on_axis(axes1[1], metric_summaries["avg_timesteps"]["disturbance"], "Disturbance", "green", "s", "avg_timesteps", stable_by_metric["avg_timesteps"]["disturbance"])
    if metric_summaries["avg_timesteps"]["bad_prior"]:
        plot_scenario_on_axis(axes1[1], metric_summaries["avg_timesteps"]["bad_prior"], "Bad Prior", "firebrick", "^", "avg_timesteps", stable_by_metric["avg_timesteps"]["bad_prior"])
    axes1[1].set_ylabel(r"Avg Timesteps")
    axes1[1].set_xlabel(r"Highlighted Parameter Name")
    axes1[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Create a single global legend at the bottom for figure 1
    handles1, labels1 = axes1[0].get_legend_handles_labels()
    bottom_position = -0.30
    fig1.legend(handles1, labels1, loc='lower center', ncol=8, bbox_to_anchor=(0.5, bottom_position))
    
    # Set x-ticks for all subplots in figure 1
    for ax in axes1:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels_fig1, rotation=45, ha="right")

    # Add parameter group labels above the top subplot for figure 1
    if group_names and group_centers:
        # Use axis coordinates for x (data) and y (above the axis)
        for name, center in zip(group_names, group_centers):
            axes1[0].text(
                center,
                1.12,
                name,
                transform=axes1[0].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                linespacing=1.2,
            )

    fig1.tight_layout()
    fig1.subplots_adjust(bottom=0.12)  # Adjust bottom to make room for legend
    fig1.subplots_adjust(top=0.90)  # Adjust top to make room for group labels
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file1 = out_dir / f"metrics_success_rate_avg_timesteps_{args.sweep_label}.png"
    fig1.savefig(output_file1, dpi=300, bbox_inches="tight")
    
    # Also save as PDF for LaTeX inclusion
    pdf_output1 = out_dir / f"metrics_success_rate_avg_timesteps_{args.sweep_label}.pdf"
    fig1.savefig(pdf_output1, dpi=300, bbox_inches="tight")
    print(f"Saved 2x1 plot (Success Rate & Avg Timesteps) ({args.sweep_label}) to {output_file1}")
    print(f"Saved PDF version for LaTeX to {pdf_output1}")
    
    # Create Figure 2: Avg Anchor Error & Epistemicity
    fig2, axes2 = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    
    # Subplot 1: Avg Anchor Error - all scenarios
    axes2[0].set_title(r"Avg Anchor Error")
    plot_scenario_on_axis(axes2[0], metric_summaries["avg_anchor_error"]["normal"], "Normal", "steelblue", "o", "avg_anchor_error", stable_by_metric["avg_anchor_error"]["normal"])
    if metric_summaries["avg_anchor_error"]["noise"]:
        plot_scenario_on_axis(axes2[0], metric_summaries["avg_anchor_error"]["noise"], "Noise", "orange", "x", "avg_anchor_error", stable_by_metric["avg_anchor_error"]["noise"])
    if metric_summaries["avg_anchor_error"]["disturbance"]:
        plot_scenario_on_axis(axes2[0], metric_summaries["avg_anchor_error"]["disturbance"], "Disturbance", "green", "s", "avg_anchor_error", stable_by_metric["avg_anchor_error"]["disturbance"])
    if metric_summaries["avg_anchor_error"]["bad_prior"]:
        plot_scenario_on_axis(axes2[0], metric_summaries["avg_anchor_error"]["bad_prior"], "Bad Prior", "firebrick", "^", "avg_anchor_error", stable_by_metric["avg_anchor_error"]["bad_prior"])
    axes2[0].set_ylabel(r"Avg Anchor Error")
    axes2[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Subplot 2: Epistemicity - all scenarios
    axes2[1].set_title(r"Epistemicity")
    plot_scenario_on_axis(axes2[1], metric_summaries["epistemicity"]["normal"], "Normal", "steelblue", "o", "epistemicity", stable_by_metric["epistemicity"]["normal"])
    if metric_summaries["epistemicity"]["noise"]:
        plot_scenario_on_axis(axes2[1], metric_summaries["epistemicity"]["noise"], "Noise", "orange", "x", "epistemicity", stable_by_metric["epistemicity"]["noise"])
    if metric_summaries["epistemicity"]["disturbance"]:
        plot_scenario_on_axis(axes2[1], metric_summaries["epistemicity"]["disturbance"], "Disturbance", "green", "s", "epistemicity", stable_by_metric["epistemicity"]["disturbance"])
    if metric_summaries["epistemicity"]["bad_prior"]:
        plot_scenario_on_axis(axes2[1], metric_summaries["epistemicity"]["bad_prior"], "Bad Prior", "firebrick", "^", "epistemicity", stable_by_metric["epistemicity"]["bad_prior"])
    axes2[1].set_xlabel(r"Highlighted Parameter Name")
    axes2[1].set_ylabel(r"Epistemicity")
    axes2[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Create a single global legend at the bottom for figure 2
    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='lower center', ncol=8, bbox_to_anchor=(0.5, bottom_position))
    
    # Set x-ticks for all subplots in figure 2
    for ax in axes2:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Add parameter group labels above the top subplot for figure 2
    if group_names and group_centers:
        # Use axis coordinates for x (data) and y (above the axis)
        for name, center in zip(group_names, group_centers):
            axes2[0].text(
                center,
                1.12,
                name,
                transform=axes2[0].get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                linespacing=1.2,
            )

    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.12)  # Adjust bottom to make room for legend
    fig2.subplots_adjust(top=0.90)  # Adjust top to make room for group labels
    output_file2 = out_dir / f"metrics_avg_anchor_error_epistemicity_{args.sweep_label}.png"
    fig2.savefig(output_file2, dpi=300, bbox_inches="tight")
    
    # Also save as PDF for LaTeX inclusion
    pdf_output2 = out_dir / f"metrics_avg_anchor_error_epistemicity_{args.sweep_label}.pdf"
    fig2.savefig(pdf_output2, dpi=300, bbox_inches="tight")
    print(f"Saved 2x1 plot (Avg Anchor Error & Epistemicity) ({args.sweep_label}) to {output_file2}")
    print(f"Saved PDF version for LaTeX to {pdf_output2}")


if __name__ == "__main__":
    main()