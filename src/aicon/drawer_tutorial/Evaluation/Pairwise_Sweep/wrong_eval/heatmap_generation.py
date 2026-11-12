"""Heatmap generation for pairwise sweep analysis."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from .utils import _get_matplotlib_pyplot


def heatmap_ss_ab(ss_ab_table, output_path, max_value=None, ss_a_table=None, ss_b_table=None, ss_total_table=None):
    """Generate heatmaps of SS_AB interaction effects with hierarchical clustering.
    
    Args:
        ss_ab_table: Dictionary of (param_a, param_b) -> SS_AB values
        output_path: Path to save the output figures (without extension)
        max_value: Maximum value for color scale (auto-computed if None)
        ss_a_table: Optional dictionary of (param_a, param_b) -> SS_A values
        ss_b_table: Optional dictionary of (param_a, param_b) -> SS_B values
        ss_total_table: Optional dictionary of (param_a, param_b) -> SS_total values
    
    Generates two heatmaps:
        1. Normalized and rooted SS_AB: (SS_AB / 25)^0.5
        2. Normalized interaction effect: SS_AB / SS_residual (where SS_residual = SS_total - SS_AB)
    """
    plt = _get_matplotlib_pyplot()

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    import numpy as np

    # --------------------------------------------------
    # Build interaction matrix
    # --------------------------------------------------
    parameters = sorted({param for pair in ss_ab_table for param in pair})

    param_index = {
        param: idx for idx, param in enumerate(parameters)
    }

    # Build normalized SS_AB and normalized interaction matrices
    normalized_ssab_matrix = np.full(
        (len(parameters), len(parameters)),
        np.nan
    )
    normalized_interaction_matrix = np.full(
        (len(parameters), len(parameters)),
        np.nan
    )

    for (param_a, param_b), ss_ab in ss_ab_table.items():
        idx_a = param_index[param_a]
        idx_b = param_index[param_b]

        # Normalized and rooted SS_AB: (SS_AB / 25)^0.5
        normalized = ss_ab / 25.0
        normalized_ssab_matrix[idx_a, idx_b] = normalized ** 0.5
        normalized_ssab_matrix[idx_b, idx_a] = normalized ** 0.5  # Symmetric

        # Normalized interaction effect: SS_AB / SS_residual
        # where SS_residual = SS_total - SS_AB (interaction variance / error variance)
        if ss_total_table is not None:
            ss_total = ss_total_table.get((param_a, param_b), ss_total_table.get((param_b, param_a), np.nan))
            if ss_total is not None and not np.isnan(ss_total) and ss_total > ss_ab:
                ss_residual = ss_total - ss_ab
                interaction_effect = ss_ab / ss_residual if ss_residual > 0 else np.nan
                normalized_interaction_matrix[idx_a, idx_b] = interaction_effect
                normalized_interaction_matrix[idx_b, idx_a] = interaction_effect  # Symmetric
            else:
                normalized_interaction_matrix[idx_a, idx_b] = np.nan
                normalized_interaction_matrix[idx_b, idx_a] = np.nan
        else:
            normalized_interaction_matrix[idx_a, idx_b] = np.nan
            normalized_interaction_matrix[idx_b, idx_a] = np.nan

    # --------------------------------------------------
    # Plot normalized SS_AB heatmap
    # --------------------------------------------------

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")   # NaN values

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.2 * len(parameters)),
            max(8.0, 1.2 * len(parameters))
        )
    )

    im = ax.imshow(
        normalized_ssab_matrix,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=0
    )

    # Show parameter names on axes
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameters, fontsize=8, rotation=90)
    ax.set_yticklabels(parameters, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Root Mean Square Error (RMSE) - Normalized SS_AB / 25", fontsize=10)

    # Add title
    ax.set_title("Interaction Effects Heatmap (Normalized SS_AB)", fontsize=12)

    # Save normalized SS_AB heatmap
    normalized_ssab_output_path = output_path.parent / (output_path.stem + "_normalized_ssab.pdf")
    plt.savefig(normalized_ssab_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # Plot normalized interaction effect heatmap
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.2 * len(parameters)),
            max(8.0, 1.2 * len(parameters))
        )
    )

    im = ax.imshow(
        normalized_interaction_matrix,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=0
    )

    # Show parameter names on axes
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameters, fontsize=8, rotation=90)
    ax.set_yticklabels(parameters, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Interaction Effect (SS_AB / SS_residual)", fontsize=10)

    # Add title
    ax.set_title("Interaction Effects Heatmap (Normalized Interaction Effect)", fontsize=12)

    # Save normalized interaction heatmap
    normalized_interaction_output_path = output_path.parent / (output_path.stem + "_normalized_interaction.pdf")
    plt.savefig(normalized_interaction_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)