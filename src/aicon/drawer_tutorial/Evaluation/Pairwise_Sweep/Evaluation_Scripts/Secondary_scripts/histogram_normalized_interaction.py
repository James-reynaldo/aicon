"""Generate a publication-quality histogram of normalized pairwise interaction scores.

This script creates a minimalist histogram showing the distribution of normalized
interaction scores (SS_ab / SS_total) across all parameter pairs for a given scenario.

The normalized interaction score is calculated as:
    normalized_score = sqrt(SS_interaction / SS_total)

where SS_interaction quantifies the interaction effect and SS_total quantifies the
total variation in the response surface.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..utils import resolve_parameter_selection
from ..data_reading import read_pairwise_job_series, read_single_sweep_series
from ..anova_analysis import compute_hi, calculate_ANOVA_interaction_score


def main():
    parser = argparse.ArgumentParser(
        description="Generate histogram of normalized pairwise interaction scores."
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
        help="Output directory. Defaults to metrics/normalized_interaction_histogram next to the database.",
    )
    parser.add_argument(
        "--metric",
        choices=["success", "anchor_error", "timesteps", "grasp", "epistemicity"],
        default="success",
        help="Metric to analyze (default: success).",
    )
    parser.add_argument(
        "--include-failed-trials",
        action="store_true",
        help="Include failed trials when analyzing timesteps.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Number of bins for histogram. If None, uses Freedman-Diaconis rule.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#4C72B0",
        help="Color for histogram bars (default: #4C72B0 - muted blue).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output (default: 300).",
    )

    args = parser.parse_args()

    # Determine database path
    if args.db_path is None or not args.db_path.exists():
        db_path = Path(__file__).parent.parent.parent.parent.parent / "Ignored_files" / "data" / "noise" / "experiment_store.db"
        print(f"Database path not provided or does not exist. Using default: {db_path}")
    else:
        db_path = args.db_path.resolve()

    # Determine output directory
    project_root = Path(__file__).parent.parent.parent.parent.parent / "Ignored_files"
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = db_path.parent / "metrics" / "normalized_interaction_histogram"

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

    # Calculate ANOVA interaction scores
    _, normalized_interaction_ss, _ = calculate_ANOVA_interaction_score(
        parameters, augmented_pair_vectors, augmented_pair_diagnostics
    )

    # Extract normalized interaction scores, filtering out NaN and non-finite values
    scores = np.array(list(normalized_interaction_ss.values()))
    valid_scores = scores[np.isfinite(scores)]

    print(f"Total parameter pairs: {len(scores)}")
    print(f"Valid parameter pairs (finite scores): {len(valid_scores)}")
    print(f"Score range: [{valid_scores.min():.6f}, {valid_scores.max():.6f}]")
    print(f"Mean: {valid_scores.mean():.6f}, Median: {np.median(valid_scores):.6f}")

    # Determine number of bins using Freedman-Diaconis rule if not specified
    if args.n_bins is None:
        iqr = np.percentile(valid_scores, 75) - np.percentile(valid_scores, 25)
        if iqr > 0:
            bin_width = 2 * iqr / (len(valid_scores) ** (1/3))
            n_bins = int((valid_scores.max() - valid_scores.min()) / bin_width)
            n_bins = max(10, min(n_bins, 50))  # Clamp between 10 and 50
        else:
            n_bins = 20
    else:
        n_bins = args.n_bins

    print(f"Using {n_bins} bins for histogram.")

    # Create publication-quality histogram
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=args.dpi)

    # Plot histogram with minimalist styling
    n, bins, patches = ax.hist(
        valid_scores,
        bins=n_bins,
        color=args.color,
        edgecolor="none",
        alpha=0.8,
        range=(valid_scores.min(), valid_scores.max()),
    )

    # Styling
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Gridlines
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xlabel("Normalized interaction score (SS_ab / SS_total)", fontsize=12)
    ax.set_ylabel("Number of parameter pairs", fontsize=12)
    ax.set_title("Distribution of Normalized Pairwise Interaction Scores", fontsize=14)

    # Tick labels
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save figures
    png_path = output_dir / "normalized_interaction_histogram.png"
    pdf_path = output_dir / "normalized_interaction_histogram.pdf"

    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"\nHistogram saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"\nGenerated with {len(valid_scores)} valid parameter pairs using {n_bins} bins.")


if __name__ == "__main__":
    main()
