"""Hi computation and ANOVA interaction score calculation."""
import numpy as np

from .utils import short_label
from .data_reading import _compute_epistemicity_from_trace_ids


def compute_hi(parameters, pair_vectors, one_d_successes_by_parameter=None, standard_baseline=None):
    """
    Compute Hi values incorporating both pairwise and 1D sweep data.
    For each parameter pair and label combination, use pairwise data if available,
    otherwise fall back to 1D data. Creates 25 values per pair (5x5 labels minus duplicate standard).
    """
    if one_d_successes_by_parameter is None:
        one_d_successes_by_parameter = {}
    
    labels = ["0.2x", "0.5x", "2x", "5x", "standard"]
    
    # Build augmented pair vectors that combine pairwise and 1D data
    augmented_pair_vectors = {}
    augmented_pair_diagnostics = {}
    for param_a, param_b in [(p1, p2) for p1 in parameters for p2 in parameters if p1 != p2]:
        pair_key = tuple(sorted([param_a, param_b]))
        all_values = []
        present_label_pairs = []
        missing_label_pairs = []
        
        # Get pairwise data for this pair if it exists (dict of label combinations)
        pairwise_by_labels = pair_vectors.get(pair_key, {})
        
        # Build 5x5 grid of label combinations (25 total)
        for label_a in labels:
            for label_b in labels:
                # Try to get pairwise data for this specific label combination
                pairwise_value = pairwise_by_labels.get((label_a, label_b), np.nan)
                
                if not np.isnan(pairwise_value):
                    # Use pairwise data for this specific label combination
                    all_values.append(pairwise_value)
                    present_label_pairs.append((label_a, label_b, "pairwise"))
                else:
                    # Only allow fallback to 1D when at least one label is 'standard'.
                    # If neither label is 'standard' and pairwise data is missing,
                    # do not use 1D fallbacks for this label combination.
                    if label_a == "standard" and label_b == "standard":
                        all_values.append(standard_baseline)
                        present_label_pairs.append((label_a, label_b, "standard_baseline"))
                    elif label_a == "standard" and label_b != "standard":
                        val_b = one_d_successes_by_parameter.get(param_b, {}).get(label_b, np.nan)
                        if not np.isnan(val_b):
                            all_values.append(float(val_b))
                            present_label_pairs.append((label_a, label_b, f"1d:{short_label(param_b)}"))
                        else:
                            all_values.append(np.nan)
                            missing_label_pairs.append((label_a, label_b, short_label(param_b)))
                    elif label_b == "standard" and label_a != "standard":
                        val_a = one_d_successes_by_parameter.get(param_a, {}).get(label_a, np.nan)
                        if not np.isnan(val_a):
                            all_values.append(float(val_a))
                            present_label_pairs.append((label_a, label_b, f"1d:{short_label(param_a)}"))
                        else:
                            all_values.append(np.nan)
                            missing_label_pairs.append((label_a, label_b, short_label(param_a)))
                    else:
                        all_values.append(np.nan)
                        missing_label_pairs.append((label_a, label_b, "pairwise_only"))
                    
        if all_values:
            augmented_pair_vectors[pair_key] = np.asarray(all_values, dtype=float)
            augmented_pair_diagnostics[pair_key] = {
                "present": present_label_pairs,
                "missing": missing_label_pairs,
                "value_count": len(all_values),
                "valid_value_count": sum(not np.isnan(value) for value in all_values),
            }

    baseline_values = {}
    for parameter in parameters:
        collected = []
        for pair, values in augmented_pair_vectors.items():
            if parameter in pair:
                collected.extend(values.tolist())
        baseline_values[parameter] = float(np.mean(collected)) if collected else np.nan

    hi_values = {}
    variation_table = {}
    raw_variation_table = {}
    for parameter in parameters:
        partner_variances = []
        variation_table[parameter] = {}
        raw_variation_table[parameter] = {}
        for partner in parameters:
            if partner == parameter:
                continue

            pair_key = tuple(sorted([parameter, partner]))
            series = augmented_pair_vectors.get(pair_key)
            if series is None or series.size == 0:
                continue

            partner_baseline = baseline_values.get(partner, np.nan)
            if np.isnan(partner_baseline):
                continue

            deviations = series - partner_baseline
            variance = float(np.var(deviations))
            raw_variance = float(np.var(series))
            partner_variances.append(variance)
            variation_table[parameter][partner] = variance
            raw_variation_table[parameter][partner] = raw_variance

        hi_values[parameter] = float(np.mean(partner_variances)) if partner_variances else np.nan

    return baseline_values, hi_values, variation_table, raw_variation_table, augmented_pair_vectors, augmented_pair_diagnostics


def calculate_ANOVA_interaction_score(parameters, augmented_pair_vectors, augmented_pair_diagnostics=None):
    """
    Calculate an ANOVA-like interaction score for each parameter based on the augmented pair vectors.
    This score reflects the degree of interaction effects relative to main effects.
    """
    interaction_scores = {}
    ss_ab_table = {}
    normalized_interaction_ss = {}

    for pair, values in augmented_pair_vectors.items():
        try:
            matrix = values.reshape((5, 5))  # Assuming 5 levels for each parameter
        except ValueError:
            print(f"Warning: Could not reshape values for pair {pair} into 5x5 matrix.")
            print(f"Values: {values}")
            if augmented_pair_diagnostics is not None:
                diagnostics = augmented_pair_diagnostics.get(pair, {})
                print(f"Present label pairs for {pair}: {diagnostics.get('present', [])}")
                print(f"Missing label pairs for {pair}: {diagnostics.get('missing', [])}")
                print(f"Value count for {pair}: {diagnostics.get('value_count', len(values))}")
                print(f"Valid value count for {pair}: {diagnostics.get('valid_value_count', np.sum(~np.isnan(values)))}")
            # import sys
            # sys.exit(1)
            # Skip pairs that cannot be reshaped into a 5x5 matrix
            continue

        if not np.any(np.isfinite(matrix)):
            continue

        param_a, param_b = pair
        if param_a == "connection.DistGraspHandConnection.gripper_activation_threshold" and param_b == "grasp_likelihood.gripper_activation_threshold":
            pass
        mean_a = np.nanmean(matrix, axis=1, keepdims=True)
        mean_b = np.nanmean(matrix, axis=0, keepdims=True)
        mean_value = np.nanmean(values)

        #Calculate interaction residual
        interaction_residuals = matrix - mean_a - mean_b + mean_value

        #Compute sum of squares per interaction
        ss_interaction = np.nansum(interaction_residuals ** 2)

        ss_total = np.nansum((matrix - mean_value) ** 2)

        ss_a = np.nansum((mean_a - mean_value) ** 2) * matrix.shape[1]

        ss_b = np.nansum((mean_b - mean_value) ** 2) * matrix.shape[0]

        interaction_scores[pair] = (
            ss_interaction / ss_total
            if ss_total > 0 else np.nan
        )

        # Normalize by number of cells (25) and take square root
        n_cells = 25
        # Output the normalized interaction score (SS_interaction / SS_total, no sqrt)
        normalized_interaction_ss[pair] = ss_interaction / ss_total if ss_total > 0 else np.nan
        ss_ab_table[pair] = np.sqrt(ss_interaction / n_cells)

    return interaction_scores, normalized_interaction_ss, ss_ab_table
