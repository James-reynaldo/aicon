"""Identify and visualize parameter pairs with high raw and normalized interaction scores.

This script filters parameter pairs based on both:
1. Raw interaction sum of squares (SS_ab)
2. Normalized interaction score (SS_ab / SS_total)

Pairs are retained only if they meet BOTH threshold criteria.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Matplotlib configuration for A4 thesis
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

# Add current directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from .utils import short_label
from .data_reading import read_pairwise_job_series, read_single_sweep_series
from .anova_analysis import compute_hi


def calculate_ANOVA_interaction_score_with_raw_values(parameters, augmented_pair_vectors, augmented_pair_diagnostics=None):
    """
    Calculate ANOVA-like interaction scores with access to raw SS values.
    
    Returns:
        interaction_scores: dict of (SS_ab / SS_total) for each pair
        normalized_interaction_ss: dict of sqrt(SS_ab / SS_total) for each pair
        ss_ab_table: dict of sqrt(SS_ab / 25) for each pair
        raw_ss_ab: dict of raw SS_ab for each pair
        raw_ss_total: dict of raw SS_total for each pair
    """
    interaction_scores = {}
    ss_ab_table = {}
    normalized_interaction_ss = {}
    raw_ss_ab = {}
    raw_ss_total = {}

    for pair, values in augmented_pair_vectors.items():
        try:
            matrix = values.reshape((5, 5))
        except ValueError:
            if augmented_pair_diagnostics is not None:
                diagnostics = augmented_pair_diagnostics.get(pair, {})
                print(f"Warning: Could not reshape values for pair {pair}. Valid count: {diagnostics.get('valid_value_count', 0)}")
            continue

        if not np.any(np.isfinite(matrix)):
            continue

        param_a, param_b = pair
        
        mean_a = np.nanmean(matrix, axis=1, keepdims=True)
        mean_b = np.nanmean(matrix, axis=0, keepdims=True)
        mean_value = np.nanmean(values)

        # Calculate interaction residual
        interaction_residuals = matrix - mean_a - mean_b + mean_value

        # Compute sum of squares
        ss_interaction = float(np.nansum(interaction_residuals ** 2))
        ss_total = float(np.nansum((matrix - mean_value) ** 2))
        ss_a = float(np.nansum((mean_a - mean_value) ** 2) * matrix.shape[1])
        ss_b = float(np.nansum((mean_b - mean_value) ** 2) * matrix.shape[0])

        # Store raw values
        raw_ss_ab[pair] = ss_interaction
        raw_ss_total[pair] = ss_total

        # Calculate normalized scores
        if ss_total > 0:
            interaction_scores[pair] = ss_interaction / ss_total
            normalized_interaction_ss[pair] = ss_interaction / ss_total  # SS_ab / SS_total (no sqrt)
        else:
            interaction_scores[pair] = np.nan
            normalized_interaction_ss[pair] = np.nan

        # SS_ab normalization (for compatibility with existing code)
        n_cells = 25
        ss_ab_table[pair] = np.sqrt(ss_interaction / n_cells)

    return interaction_scores, normalized_interaction_ss, ss_ab_table, raw_ss_ab, raw_ss_total


def main():
    parser = argparse.ArgumentParser(
        description="Filter and visualize parameter pairs with high interaction scores."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=False,
        help="SQLite database path containing pairwise and single-sweep trials.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to metrics/high_interaction_pairs next to the database.",
    )
    parser.add_argument(
        "--metric",
        choices=["success", "anchor_error", "timesteps", "grasp", "epistemicity"],
        default="success",
        help="Metric to analyze (default: success).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="",
        help="Scenario name to include in the title (default: empty).",
    )
    parser.add_argument(
        "--include-failed-trials",
        action="store_true",
        help="Include failed trials when analyzing timesteps.",
    )
    parser.add_argument(
        "--min-ss-ab",
        type=float,
        default=None,
        help="Minimum RMS interaction residual (sqrt(SS_ab/25)) threshold. If None, uses data-driven threshold.",
    )
    parser.add_argument(
        "--min-normalized",
        type=float,
        default=None,
        help="Minimum normalized interaction score (SS_ab / SS_total) threshold. If None, uses data-driven threshold.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="If set, select top N pairs by normalized score (overrides threshold-based selection).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output (default: 300).",
    )
    parser.add_argument(
        "--label-top",
        type=int,
        default=10,
        help="Number of top pairs to label on the plot (default: 10).",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Minimum x-axis value (RMS interaction residual). If None, uses data-driven minimum.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Maximum x-axis value (RMS interaction residual). If None, uses data-driven maximum.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Minimum y-axis value (normalized score). If None, uses data-driven minimum.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum y-axis value (normalized score). If None, uses data-driven maximum.",
    )

    args = parser.parse_args()

    # Determine database path
    if args.db_path is None or not args.db_path.exists():
        db_path = Path(__file__).parent.parent.parent.parent / "Ignored_files" / "data" / "noise" / "experiment_store.db"
        print(f"Database path not provided or does not exist. Using default: {db_path}")
    else:
        db_path = args.db_path.resolve()

    # Determine output directory
    project_root = Path(__file__).parent.parent.parent.parent / "Ignored_files"
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = db_path.parent / "metrics" / "high_interaction_pairs"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using metric: {args.metric}")
    print(f"Database: {db_path}")
    print(f"Output directory: {output_dir}")

    # Load data
    parameters, pairwise_series = read_pairwise_job_series(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
    )
    print(f"Loaded {len(parameters)} parameters and {len(pairwise_series)} pairwise series.")

    si_values, one_d_successes_by_parameter, standard_baseline = read_single_sweep_series(
        db_path,
        metrics=args.metric,
        include_failed_trials=args.include_failed_trials,
    )
    print(f"Loaded {len(si_values)} single-sweep Si values.")

    # Compute Hi values and augmented pair vectors
    _, _, _, _, augmented_pair_vectors, augmented_pair_diagnostics = compute_hi(
        parameters, pairwise_series, one_d_successes_by_parameter, standard_baseline
    )

    # Calculate ANOVA interaction scores with raw values
    interaction_scores, normalized_interaction_ss, ss_ab_table, raw_ss_ab, raw_ss_total = \
        calculate_ANOVA_interaction_score_with_raw_values(
            parameters, augmented_pair_vectors, augmented_pair_diagnostics
        )

    # Extract all valid pairs
    all_pairs = list(normalized_interaction_ss.keys())
    normalized_scores = np.array([normalized_interaction_ss[pair] for pair in all_pairs])
    ss_ab_values = np.array([ss_ab_table[pair] for pair in all_pairs])
    
    # Filter out non-finite values
    valid_mask = np.isfinite(normalized_scores) & np.isfinite(ss_ab_values)
    valid_pairs = [all_pairs[i] for i in range(len(all_pairs)) if valid_mask[i]]
    valid_normalized = normalized_scores[valid_mask]
    valid_ss_ab = ss_ab_values[valid_mask]

    print(f"\nTotal parameter pairs: {len(all_pairs)}")
    print(f"Valid pairs (finite scores): {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("No valid pairs found. Exiting.")
        return

    # Determine thresholds
    if args.top_n is not None:
        # Select top N pairs by normalized score
        threshold_normalized = np.percentile(valid_normalized, 100 - (100 * args.top_n / len(valid_pairs)))
        threshold_ss_ab = None  # No SS_ab threshold when using top-N
        print(f"\nSelecting top {args.top_n} pairs by normalized score.")
    else:
        # Use data-driven thresholds if not specified
        if args.min_normalized is None:
            # Use 75th percentile as default
            threshold_normalized = np.percentile(valid_normalized, 75)
        else:
            threshold_normalized = args.min_normalized
            
        if args.min_ss_ab is None:
            # Use 75th percentile of SS_ab as default
            threshold_ss_ab = np.percentile(valid_ss_ab, 75)
        else:
            threshold_ss_ab = args.min_ss_ab

    print(f"\nThresholds:")
    print(f"  Minimum normalized interaction score: {threshold_normalized:.6f}")
    print(f"  Minimum SS_ab: {threshold_ss_ab:.6f}")

    # Filter pairs meeting BOTH criteria
    retained_mask = (normalized_scores >= threshold_normalized) & (ss_ab_values >= threshold_ss_ab)
    retained_pairs = [all_pairs[i] for i in range(len(all_pairs)) if retained_mask[i]]
    
    # Also get filtered-out pairs for visualization
    filtered_mask = ~retained_mask & valid_mask
    filtered_pairs = [all_pairs[i] for i in range(len(all_pairs)) if filtered_mask[i]]

    print(f"\nResults:")
    print(f"  Pairs meeting BOTH thresholds: {len(retained_pairs)}")
    print(f"  Pairs filtered out: {len(filtered_pairs)}")

    # Automatically highlight all pairs containing uncertainty_dist_threshold
    highlight_pairs = []
    for pair in all_pairs:
        param_a, param_b = pair
        if "uncertainty_dist_threshold" in param_a.lower() or "uncertainty_dist_threshold" in param_b.lower():
            highlight_pairs.append(pair)
    
    print(f"  Pairs to highlight (uncertainty_dist_threshold): {len(highlight_pairs)}")

    # Function to determine color based on parameter names
    def get_pair_color(pair):
        param_a, param_b = pair
        has_uncertainty = "uncertainty_dist_threshold" in param_a.lower() or "uncertainty_dist_threshold" in param_b.lower()
        has_close = "close_dist_threshold" in param_a.lower() or "close_dist_threshold" in param_b.lower()
        
        if has_uncertainty and has_close:
            return "#1F77B4"  # Blue - both
        elif has_uncertainty:
            return "#2CA02C"  # Green - uncertainty only
        elif has_close:
            return "#FF7F0E"  # Orange - close only
        else:
            return "#999999"  # Grey - neither

    # Create diagnostic plot
    # Use default style but preserve our font configuration
    plt.style.use("default")
    # Re-apply font configuration after style use
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    })
    # Half A4 page landscape: ~14cm x 10cm for the figure area
    fig_width = 14.0  # cm
    fig_height = 10.0  # cm
    fig, ax = plt.subplots(figsize=(fig_width / 2.54, fig_height / 2.54), dpi=args.dpi)

    # Plot filtered-out pairs with color coding
    filtered_ss_ab = []
    filtered_normalized = []
    filtered_colors = []
    
    for i, pair in enumerate(all_pairs):
        if filtered_mask[i]:
            filtered_ss_ab.append(ss_ab_values[i])
            filtered_normalized.append(normalized_scores[i])
            filtered_colors.append(get_pair_color(pair))
    
    filtered_ss_ab = np.array(filtered_ss_ab)
    filtered_normalized = np.array(filtered_normalized)
    filtered_colors = np.array(filtered_colors)
    
    if len(filtered_ss_ab) > 0:
        ax.scatter(filtered_ss_ab, filtered_normalized, 
                  color=filtered_colors, alpha=0.4, s=30, 
                  label=f"Filtered out ({len(filtered_ss_ab)})")

    # Plot retained pairs with color coding
    retained_ss_ab = []
    retained_normalized = []
    retained_colors = []
    
    for i, pair in enumerate(all_pairs):
        if retained_mask[i]:
            retained_ss_ab.append(ss_ab_values[i])
            retained_normalized.append(normalized_scores[i])
            retained_colors.append(get_pair_color(pair))
    
    retained_ss_ab = np.array(retained_ss_ab)
    retained_normalized = np.array(retained_normalized)
    retained_colors = np.array(retained_colors)
    
    if len(retained_ss_ab) > 0:
        ax.scatter(retained_ss_ab, retained_normalized, 
                  color=retained_colors, alpha=0.9, s=60, 
                  label=f"Retained ({len(retained_ss_ab)})")

    # Plot highlighted pairs with color coding
    if highlight_pairs:
        highlight_ss_ab = []
        highlight_normalized = []
        highlight_colors = []
        
        for pair in highlight_pairs:
            if pair in all_pairs:
                idx = all_pairs.index(pair)
                if valid_mask[idx]:
                    highlight_ss_ab.append(ss_ab_values[idx])
                    highlight_normalized.append(normalized_scores[idx])
                    highlight_colors.append(get_pair_color(pair))
        
        highlight_ss_ab = np.array(highlight_ss_ab)
        highlight_normalized = np.array(highlight_normalized)
        highlight_colors = np.array(highlight_colors)
        
        if len(highlight_ss_ab) > 0:
            ax.scatter(highlight_ss_ab, highlight_normalized, 
                      color=highlight_colors, alpha=1.0, s=80, marker="o", 
                      label=f"Highlighted ({len(highlight_ss_ab)})",
                      zorder=10, edgecolors="black", linewidths=0.5)

    # Draw threshold lines
    if threshold_ss_ab is not None:
        ax.axvline(x=threshold_ss_ab, color="#FF6B6B", linestyle="--", linewidth=1.5, 
                   label=f"SS_ab threshold = {threshold_ss_ab:.6f}")
    ax.axhline(y=threshold_normalized, color="#4ECDC4", linestyle="--", linewidth=1.5, 
               label=f"Normalized threshold = {threshold_normalized:.6f}")

    # Styling
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")

    ax.set_xlabel(r"RMS interaction residual $I_{AB}$")
    ax.set_ylabel(r"Normalized interaction score ($SS_{ab}$ / $SS_{total}$)")
    if args.scenario:
        ax.set_title(f"Pairwise Interaction Strength and Normalized Scores, {args.scenario} scenario")
    else:
        ax.set_title("Pairwise Interaction Strength and Normalized Scores")
    
    # Set axis limits if specified
    if args.x_min is not None or args.x_max is not None:
        x_lo = args.x_min if args.x_min is not None else ax.get_xlim()[0]
        x_hi = args.x_max if args.x_max is not None else ax.get_xlim()[1]
        ax.set_xlim(x_lo, x_hi)
    if args.y_min is not None or args.y_max is not None:
        y_lo = args.y_min if args.y_min is not None else ax.get_ylim()[0]
        y_hi = args.y_max if args.y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(y_lo, y_hi)
    
    # Custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2CA02C', markersize=8, label='uncertainty_dist_threshold pairs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7F0E', markersize=8, label='close_dist_threshold pairs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1F77B4', markersize=8, label='uncertainty/close dist threshold'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=8, alpha=0.4, label='Other pairs'),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    plt.tight_layout()

    # Save figure
    png_path = output_dir / "high_interaction_pairs.png"
    pdf_path = output_dir / "high_interaction_pairs.pdf"
    
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save CSV with retained pairs
    csv_path = output_dir / "high_interaction_pairs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter A", "Parameter B", "SS_ab", "Normalized Score", "SS_ab", "SS_total", "Highlighted", "Color"])
        
        # Sort by normalized score descending
        sorted_indices = np.argsort(-retained_normalized)
        for idx in sorted_indices:
            pair = retained_pairs[idx]
            ss_ab_val = retained_ss_ab[idx]
            normalized = retained_normalized[idx]
            ss_ab = raw_ss_ab[pair]
            ss_total = raw_ss_total[pair]
            is_highlighted = pair in highlight_pairs
            color = get_pair_color(pair)
            color_name = "blue" if color == "#1F77B4" else ("green" if color == "#2CA02C" else ("orange" if color == "#FF7F0E" else "grey"))
            writer.writerow([
                pair[0], pair[1], 
                f"{ss_ab_val:.8f}", f"{normalized:.8f}",
                f"{ss_ab:.8f}", f"{ss_total:.8f}",
                "Yes" if is_highlighted else "No",
                color_name
            ])

    print(f"\nOutput files:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  CSV: {csv_path}")
    
    # Save ss_ab_table to JSON for heatmap generation
    ss_ab_json_path = output_dir / "ss_ab_table.json"
    import json
    # Convert tuple keys to strings for JSON serialization
    ss_ab_table_json = {f"{pair[0]}|{pair[1]}": float(val) for pair, val in ss_ab_table.items()}
    with open(ss_ab_json_path, "w") as f:
        json.dump(ss_ab_table_json, f, indent=2)
    print(f"  SS_AB table: {ss_ab_json_path}")
    
    # Print retained pairs summary
    if len(retained_pairs) > 0:
        print(f"\nRetained parameter pairs (sorted by normalized score):")
        for idx in sorted_indices[:10]:  # Show top 10
            pair = retained_pairs[idx]
            is_highlighted = pair in highlight_pairs
            color = get_pair_color(pair)
            color_name = "blue" if color == "#1F77B4" else ("green" if color == "#2CA02C" else ("orange" if color == "#FF7F0E" else "grey"))
            highlight_marker = " *H*" if is_highlighted else ""
            print(f"  {short_label(pair[0])}, {short_label(pair[1])}: SS_ab={retained_ss_ab[idx]:.6f}, "
                  f"normalized={retained_normalized[idx]:.6f}, color={color_name}{highlight_marker}")
        
        # Print highlighted pairs that are NOT in retained pairs
        non_retained_highlighted = [p for p in highlight_pairs if p not in retained_pairs]
        if non_retained_highlighted:
            print(f"\nHighlighted pairs (not in retained set):")
            for pair in non_retained_highlighted:
                idx = all_pairs.index(pair)
                if valid_mask[idx]:
                    color = get_pair_color(pair)
                    color_name = "blue" if color == "#1F77B4" else ("green" if color == "#2CA02C" else ("orange" if color == "#FF7F0E" else "grey"))
                    print(f"  {short_label(pair[0])}, {short_label(pair[1])}: SS_ab={ss_ab_values[idx]:.6f}, "
                          f"normalized={normalized_scores[idx]:.6f}, color={color_name}")


if __name__ == "__main__":
    main()
