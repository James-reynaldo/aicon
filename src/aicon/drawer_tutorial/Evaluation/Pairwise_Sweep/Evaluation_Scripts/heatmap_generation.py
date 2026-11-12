"""Heatmap generation for pairwise sweep analysis."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from .utils import _get_matplotlib_pyplot, short_label

# Matplotlib configuration
# Design for LaTeX thesis with 12pt main font and \textwidth figure width
# When LaTeX includes figure with \includegraphics[width=\textwidth], it scales to text width
# We design at final physical size to avoid font size discrepancies
plt.rcParams.update({
    # Font family and LaTeX-like math
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    # Font sizes designed for 12pt LaTeX document with \textwidth figure
    # These sizes match LaTeX's 12pt baseline
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,  # Numeric labels 1, 2
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 12,
    "axes.unicode_minus": False,
})


def heatmap_ss_ab(ss_ab_table, output_path, max_value=None, highlight_params=None, scenario="Drawer"):
    """Generate SS_AB interaction heatmap with hierarchical clustering.
    
    Args:
        ss_ab_table: dict of (param_a, param_b) -> ss_ab_value
        output_path: Path to save the heatmap
        max_value: Optional max value for color scale
        highlight_params: Set of parameter names to always label (if None, label all)
        scenario: Name of the scenario for the title (default: "Drawer")
    """
    plt = _get_matplotlib_pyplot()

    # --------------------------------------------------
    # Build interaction matrix
    # --------------------------------------------------
    # Always include ALL parameters in the matrix
    all_parameters = sorted({param for pair in ss_ab_table for param in pair})
    parameters = all_parameters
    
    # Determine which parameters to label
    if highlight_params is not None and len(highlight_params) > 0:
        # Only label the highlighted parameters, others get empty labels
        labels_to_show = {p for p in parameters if p in highlight_params}
    else:
        # Label all parameters
        labels_to_show = set(parameters)

    matrix = np.full(
        (len(parameters), len(parameters)),
        np.nan
    )

    param_index = {
        param: idx for idx, param in enumerate(parameters)
    }

    for (param_a, param_b), ss_ab in ss_ab_table.items():
        idx_a = param_index[param_a]
        idx_b = param_index[param_b]

        matrix[idx_a, idx_b] = ss_ab
        matrix[idx_b, idx_a] = ss_ab  # Symmetric


    # --------------------------------------------------
    # Hierarchical clustering (normal; no pinned parameters)
    # --------------------------------------------------
    # Cluster all parameters normally using the full matrix.
    if len(parameters) > 1:
        indices = [param_index[p] for p in parameters]
        cluster_matrix = np.nan_to_num(matrix[np.ix_(indices, indices)], nan=0.0)
        distance = pdist(cluster_matrix, metric="correlation")
        linkage_matrix = linkage(distance, method="average")
        order = leaves_list(linkage_matrix)
        parameters = [parameters[i] for i in order]
        new_param_index = {param: idx for idx, param in enumerate(parameters)}
        old_indices = [param_index[p] for p in parameters]
        matrix = matrix[np.ix_(old_indices, old_indices)]
    else:
        new_param_index = {param: idx for idx, param in enumerate(parameters)}


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")   # NaN values

    # Design figure at final physical size for LaTeX thesis
    # Standard thesis text width is ~6-7 inches
    # We use 7 inches to provide more space for labels while still fitting in text width
    # This ensures font sizes appear correct without LaTeX scaling adjustments
    fig_width = 7.0  # inches (slightly wider for better label spacing)
    fig_height = 7.0  # square for heatmap

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        dpi=100  # Lower DPI for PDF, higher for screen preview
    )

    if max_value is None:
        im = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=0
        )
    else:
        im = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=max_value
        )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label("RMS Interaction ($I_{AB}$)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # Add axis labels
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Parameters", fontsize=12)

    # Show parameter names on axes
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    
    # Build tick labels: only highlight_params get numeric labels, others are empty
    x_labels = []
    y_labels = []
    
    # Create mapping from parameter name to numeric label (consistent numbering)
    param_to_label = {}
    label_counter = 1
    for param in parameters:
        if param in highlight_params:
            param_to_label[param] = str(label_counter)
            label_counter += 1
    
    for p in parameters:
        if p in param_to_label:
            x_labels.append(param_to_label[p])
            y_labels.append(param_to_label[p])
        else:
            x_labels.append("")
            y_labels.append("")
    
    ax.set_xticklabels(x_labels, rotation=90, fontsize=9)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=9)

    # Adjust tick label padding to prevent overlapping
    ax.tick_params(axis='x', which='major', pad=5)
    ax.tick_params(axis='y', which='major', pad=5)

    # No cell gridlines between heatmap tiles.
    ax.tick_params(which="both", length=0)

    # Add title
    ax.set_title(f"Interaction Score in {scenario} Scenario", fontsize=12, pad=15)

    # Remove outer border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add parameter key below the heatmap
    # Reserve space for parameter key (reduced from 0.15 to 0.12)
    fig.subplots_adjust(bottom=0.12)
    
    # Add parameter key as text below the heatmap
    param_key_text = "Parameter Key:\n"
    label_counter = 1
    for param in parameters:
        if param in highlight_params:
            short_name = param.split(".")[-1]  # Get just the parameter name
            param_key_text += f"{label_counter}: {short_name}\n"
            label_counter += 1
    
    # Parameter key font size matches LaTeX 12pt baseline
    fig.text(0.5, 0.10, param_key_text.rstrip("\n"), 
             ha="center", va="top", fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save PDF with high quality for thesis
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05
    )
    
    # Print figure dimensions for verification
    print(f"Heatmap saved to: {output_path}")
    print(f"Figure size: {fig_width:.2f} x {fig_height:.2f} inches")
    print(f"Font sizes configured: title=12pt, labels=12pt, tick labels=12pt, parameter key=11pt")
    
    # Also save PNG preview
    png_path = str(output_path).replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"PNG preview saved to: {png_path}")

    plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate SS_AB interaction heatmap with hierarchical clustering."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to ss_ab_table.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the heatmap PNG"
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Maximum value for color scale"
    )
    parser.add_argument(
        "--highlight-params",
        type=str,
        nargs="+",
        default=None,
        help="Parameter names to highlight (label with ★). Can specify multiple."
    )
    
    args = parser.parse_args()
    
    # Load ss_ab_table
    with open(args.input) as f:
        ss_ab_table_json = json.load(f)
    
    # Convert string keys back to tuple keys
    ss_ab_table = {}
    for key, value in ss_ab_table_json.items():
        param_a, param_b = key.split("|")
        ss_ab_table[(param_a, param_b)] = value
    
    # Convert highlight params to set
    highlight_params = set(args.highlight_params) if args.highlight_params else None
    
    # Generate heatmap
    heatmap_ss_ab(ss_ab_table, args.output, max_value=args.max_value, highlight_params=highlight_params)
    
    print(f"Heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
