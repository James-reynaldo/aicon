"""Module for pairwise sweep analysis and visualization."""
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
from .anova_analysis import calculate_ANOVA_interaction_score
from .heatmap_generation import heatmap_ss_ab
from .residual_plots import plot_si_vs_residuals
from .early_late_comparison import plot_early_vs_late_scatter, compute_window_metrics_summary
from .data_reading import read_pairwise_job_series, read_single_sweep_series
from .anova_analysis import compute_hi

__all__ = [
    'load_sweep_plan',
    'value_index_to_label',
    'short_label',
    'normalize_one_d_label',
    'parse_pair',
    '_filter_rows_by_trial_window',
    '_get_matplotlib_pyplot',
    'resolve_parameter_selection',
    'calculate_ANOVA_interaction_score',
    'heatmap_ss_ab',
    'plot_si_vs_residuals',
    'plot_early_vs_late_scatter',
    'compute_window_metrics_summary',
    'read_pairwise_job_series',
    'read_single_sweep_series',
    'compute_hi',
]
