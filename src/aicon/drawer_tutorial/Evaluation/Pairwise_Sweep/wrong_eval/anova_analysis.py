"""ANOVA interaction score computation for pairwise sweep analysis."""
import numpy as np


def calculate_ANOVA_interaction_score(parameters, pair_vectors, pair_diagnostics=None):
    """Compute ANOVA-like interaction scores for parameter pairs.
    
    Args:
        parameters: List of parameter names
        pair_vectors: Dictionary of (param_a, param_b) -> array of values
        pair_diagnostics: Optional dictionary with pair diagnostics
        
    Returns:
        Tuple of (interaction_scores, ss_total_table, ss_ab_table, ss_a_table, ss_b_table)
        where interaction_scores is a dict of (param_a, param_b) -> SS_AB value
    """
    interaction_scores = {}
    ss_total_table = {}
    ss_ab_table = {}
    ss_a_table = {}
    ss_b_table = {}
    
    for (param_a, param_b), values in pair_vectors.items():
        if values is None or values.size == 0:
            continue
            
        # Compute means
        n = len(values)
        if n < 2:
            continue
            
        grand_mean = np.mean(values)
        
        # Reshape values into 5x5 grid (for 5 labels per parameter)
        # Values should be in order: standard, 0.2x, 0.5x, 2x, 5x
        try:
            values_2d = values.reshape(5, 5)
        except ValueError:
            # If we can't reshape, skip this pair
            continue
        
        # Compute marginal means
        row_means = np.mean(values_2d, axis=1)  # Mean for each level of param_a
        col_means = np.mean(values_2d, axis=0)  # Mean for each level of param_b
        
        # Compute SS_A (main effect of param_a)
        ss_a = 5 * np.sum((row_means - grand_mean) ** 2)
        
        # Compute SS_B (main effect of param_b)
        ss_b = 5 * np.sum((col_means - grand_mean) ** 2)
        
        # Compute SS_total
        ss_total = np.sum((values_2d - grand_mean) ** 2)
        
        # Compute SS_AB (interaction) = SS_total - SS_A - SS_B
        # But we need to account for the fact that SS_total includes all variation
        # The interaction is what's left after removing main effects
        
        # For a 2-way ANOVA with interaction:
        # SS_total = SS_A + SS_B + SS_AB + SS_error
        # But with only one observation per cell, we can't estimate SS_error separately
        
        # Using the formula: SS_AB = sum((row_mean[i] + col_mean[j] - grand_mean - value[i,j])^2)
        ss_ab = 0.0
        for i in range(5):
            for j in range(5):
                expected = row_means[i] + col_means[j] - grand_mean
                ss_ab += (values_2d[i, j] - expected) ** 2
        
        # Store results
        interaction_scores[(param_a, param_b)] = ss_ab
        ss_total_table[(param_a, param_b)] = ss_total
        ss_ab_table[(param_a, param_b)] = ss_ab
        ss_a_table[(param_a, param_b)] = ss_a
        ss_b_table[(param_a, param_b)] = ss_b
        
        # Also store the reverse pair for symmetry
        interaction_scores[(param_b, param_a)] = ss_ab
        ss_total_table[(param_b, param_a)] = ss_total
        ss_ab_table[(param_b, param_a)] = ss_ab
        ss_a_table[(param_b, param_a)] = ss_a
        ss_b_table[(param_b, param_a)] = ss_b
    
    return interaction_scores, ss_total_table, ss_ab_table, ss_a_table, ss_b_table


def compute_per_parameter_stats(ss_ab_table, ss_a_table=None, ss_b_table=None):
    """Compute per-parameter statistics from SS_AB table.
    
    Args:
        ss_ab_table: Dictionary of (param_a, param_b) -> SS_AB values
        ss_a_table: Optional dictionary of (param_a, param_b) -> SS_A values
        ss_b_table: Optional dictionary of (param_a, param_b) -> SS_B values
        
    Returns:
        Tuple of (max_ssab_s, mean_ssab_s, max_f2_s, mean_f2_s)
        where f2 dicts are empty (Cohen's f² was removed)
    """
    max_ssab_s = {}
    mean_ssab_s = {}
    
    for (param_a, param_b), ss_ab in ss_ab_table.items():
        if ss_ab is None or np.isnan(ss_ab):
            continue
            
        # Update max SS_AB for each parameter
        if param_a not in max_ssab_s or ss_ab > max_ssab_s[param_a]:
            max_ssab_s[param_a] = ss_ab
        if param_b not in max_ssab_s or ss_ab > max_ssab_s[param_b]:
            max_ssab_s[param_b] = ss_ab
            
        # Accumulate SS_AB values for mean computation
        if param_a not in mean_ssab_s:
            mean_ssab_s[param_a] = []
        if param_b not in mean_ssab_s:
            mean_ssab_s[param_b] = []
            
        mean_ssab_s[param_a].append(ss_ab)
        mean_ssab_s[param_b].append(ss_ab)
    
    # Compute means
    mean_ssab_s = {p: float(np.mean(vals)) for p, vals in mean_ssab_s.items()} if mean_ssab_s else {}
    
    # Empty dicts for Cohen's f² (was removed from codebase)
    max_f2_s = {}
    mean_f2_s = {}
    
    return max_ssab_s, mean_ssab_s, max_f2_s, mean_f2_s
