import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

SI_SEPERATE = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

def get_functional_taxonomy():
    return {
        # ========================================================================
        # 1. INITIALIZATION - Starting beliefs for all estimator components
        # ========================================================================
        "initialization_priors": {

            # Visibility Estimator
            "visible.initial_likelihood_prior",                      # DistGraspEstimator.visible
            
            # Grasp Likelihood Estimator
            "grasp_likelihood.initial_likelihood",                   # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.initially_grasped_likelihood",         # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.initial_time_since_hand_change",       # DistGraspEstimator.grasp_likelihood

            # Kinematic Joint Estimator
            "kinematic_joint.initial_azimuth_default",               # DistGraspEstimator.kinematic_joint
            "kinematic_joint.initial_elevation_default",             # DistGraspEstimator.kinematic_joint
            
            # Drawer Position Estimator
            "drawer_position.depth_prior",                           # DistGraspEstimator.drawer_position
            
        },

        "initialization_uncertainty": {

            # End-Effector Pose Estimator
            "ee_pose.initial_uncertainty_scale",                     # DistGraspEstimator.end_effector_pose
            "drawer_position.initial_uncertainty_scale",             # DistGraspEstimator.drawer_position
            "drawer_position.initial_uncertainty_xy",               # DistGraspEstimator.drawer_position
            "drawer_position.initial_uncertainty_depth",            # DistGraspEstimator.drawer_position
            "drawer_position.initial_uncertainty_xy_none",           # DistGraspEstimator.drawer_position
            "drawer_position.initial_uncertainty_depth_none",        # DistGraspEstimator.drawer_position
            
        },
        
        # ========================================================================
        # 2. UNCERTAINTY MODELLING - Confidence levels in estimated states
        # ========================================================================
        "uncertainty_modelling": {
            # Drawer Position Estimator
            "drawer_position.meas_noise_factor",                     # DistGraspEstimator.drawer_position
            "drawer_position.R_add_scale",                           # DistGraspEstimator.drawer_position *
            "drawer_position.grasped_update_R_scale",                # DistGraspEstimator.drawer_position
            "drawer_position.grasped_uncertainty_threshold",         # DistGraspEstimator.drawer_position
            "drawer_position.missed_absent_measurement_uncertainty_coeff",  # DistGraspEstimator.drawer_position
            "drawer_position.measurement_nan_reject_scale",          # DistGraspEstimator.drawer_position
            "drawer_position.sample_init_mean_uncertainty_multiplier",  # DistGraspEstimator.drawer_position
            
            # Grasp Likelihood Estimator
            "grasp_likelihood.uncertainty_dist_threshold",           # DistGraspEstimator.grasp_likelihood
            
        },
        
        # ========================================================================
        # 3. DYNAMICS AND PREDICTION - State evolution between observations
        # ========================================================================
        "dynamics_and_prediction": {
            # End-Effector Pose Estimator
            "ee_pose.action_process_noise",                # DistGraspEstimator.end_effector_pose
            "ee_pose.proprio_update_noise",                # DistGraspEstimator.end_effector_pose
            
            # Drawer Position Estimator
            "drawer_position.forward_noise_grasped_coeff",           # DistGraspEstimator.drawer_position
            "drawer_position.forward_noise_base",                    # DistGraspEstimator.drawer_position
            
            # Kinematic Joint Estimator
            "kinematic_joint.grasped_noise",                         # DistGraspEstimator.kinematic_joint
            "kinematic_joint.ungrasped_noise",                       # DistGraspEstimator.kinematic_joint
            "kinematic_joint.axis_azimuth_process_noise",            # DistGraspEstimator.kinematic_joint
            "kinematic_joint.axis_elevation_process_noise",          # DistGraspEstimator.kinematic_joint
            "kinematic_joint.joint_process_noise",                   # DistGraspEstimator.kinematic_joint
            "kinematic_joint.anchor_process_noise",                  # DistGraspEstimator.kinematic_joint
            
            # Grasp Likelihood Estimator
            "grasp_likelihood.force_time_scale",                     # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.force_time_max",                       # DistGraspEstimator.grasp_likelihood
            
            # Distance Grasp Hand Connection
            "connection.DistGraspHandConnection.dist_decay",         # DistGraspHandConnection
            "connection.DistGraspHandConnection.time_since_hand_change_threshold",  # DistGraspHandConnection
        },
        
        # ========================================================================
        # 4. MEASUREMENT AND LIKELIHOOD - Observation interpretation
        # ========================================================================
        "measurement_and_likelihood": {
            # Visibility Estimator
            "visible.update_gain",                                # DistGraspEstimator.visible
            
            # Drawer Position Estimator
            "drawer_position.visual_likelihood_steepness",           # DistGraspEstimator.drawer_position
            "drawer_position.sample_init_mean_likelihood_threshold",  # DistGraspEstimator.drawer_position
            "drawer_position.measurement_existence_threshold",       # DistGraspEstimator.drawer_position
            
            # Grasp Likelihood Estimator
            "grasp_likelihood.baseline_measurement_likelihood",      # DistGraspEstimator.grasp_likelihood
            
            # Distance Grasp Hand Connection
            "connection.DistGraspHandConnection.small_likelihood_value", # DistGraspHandConnection
        },
        
        # ========================================================================
        # 5. DECISION GATING - Branch activation logic and thresholds
        # ========================================================================
        "decision_gating": {
            # Visibility Estimator
            "visible.initial_clip_min",                             # DistGraspEstimator.visible
            "visible.initial_clip_max",                             # DistGraspEstimator.visible
            
            # Grasp Likelihood Estimator
            "grasp_likelihood.gripper_activation_threshold",         # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.close_distance_threshold",             # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.hand_change_time_threshold",           # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.low_likelihood_threshold",             # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.negative_innovation_threshold",        # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.negative_innovation_scale",            # DistGraspEstimator.grasp_likelihood
            
            # Drawer Position Estimator
            "drawer_position.grasped_outlier_rejection_threshold",   # DistGraspEstimator.drawer_position
            "drawer_position.sample_init_mean_distance_threshold",  # DistGraspEstimator.drawer_position
            "drawer_position.hand_change_recovery_time",             # DistGraspEstimator.drawer_position 
            
            # Kinematic Joint Estimator
            "kinematic_joint.grasp_threshold",                       # DistGraspEstimator.kinematic_joint
            "kinematic_joint.grasp_floor",                           # DistGraspEstimator.kinematic_joint
            "kinematic_joint.outlier_rejection_treshold",            # DistGraspEstimator.kinematic_joint *
            "kinematic_joint.shift_clip_min",                        # DistGraspEstimator.kinematic_joint

            "grasp_likelihood.initial_clip_min",                     # DistGraspEstimator.grasp_likelihood
            "grasp_likelihood.initial_clip_max",                     # DistGraspEstimator.grasp_likelihood
        },

        "connection_thresholds": {
            # Distance Grasp Hand Connection
            "connection.DistGraspHandConnection.close_dist_threshold",  # DistGraspHandConnection *
            "connection.DistGraspHandConnection.gripper_activation_threshold",  # DistGraspHandConnection *
            "connection.DistGraspHandConnection.low_likelihood_threshold",  # DistGraspHandConnection
            "connection.DistGraspHandConnection.uncertainty_dist_threshold",  # DistGraspHandConnection *
    
            # Distance Grasp Hand Connection (tuning-specific parameters)
            "connection.DistGraspHandConnection.ft_tresh_multiplier",  # DistGraspHandConnection
            "connection.DistGraspHandConnection.ft_tresh_cap",       # DistGraspHandConnection

        },

        "connection_likelihood": {
            "connection.DistGraspHandConnection.low_likelihood_threshold",  # DistGraspHandConnection
            "connection.DistGraspHandConnection.dist_sigmoid_scale",  # DistGraspHandConnection *
            # Distance Grasp Hand Connection
            "connection.DistGraspHandConnection.uncertainty_bias",       # DistGraspHandConnection *
            "connection.DistGraspHandConnection.uncertainty_scale_uncertainty",  # DistGraspHandConnection
            "connection.DistGraspHandConnection.uncertainty_scale_relevance",     # DistGraspHandConnection
        },
        
        # ========================================================================
        # 6. TUNING AND SCALING - Strength adjustment of existing mechanisms
        # ========================================================================
        # "tuning_and_scaling": {

        #     # Distance Grasp Hand Connection (tuning-specific parameters)
        #     "connection.DistGraspHandConnection.ft_noise_offset",    # DistGraspHandConnection
        #     "connection.DistGraspHandConnection.ft_tresh_multiplier",  # DistGraspHandConnection
        #     "connection.DistGraspHandConnection.ft_tresh_cap",       # DistGraspHandConnection
        #     "connection.DistGraspHandConnection.dist_sigmoid_scale",  # DistGraspHandConnection *

        #     # Miscellaneous Practical Parameters
        #     "drawer_position.tf_lookup_timeout",                    # DistGraspEstimator (practical)
        # }
    }

def _get_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting functions. Install it or run with --debug-success-rates to inspect rates without plotting."
        ) from exc
    return plt

ONE_D_LABELS_TO_INCLUDE = {"0.2x", "0.5x", "2x", "5x", "standard"}


def short_label(label):
    return label.split(".")[-1]


def normalize_one_d_label(label):
    if label is None:
        return ""
    label = str(label).strip()
    if label == "standard":
        return "standard"
    if label.startswith("x"):
        normalized = label[1:]
        if normalized in {"0.2", "0.5", "2", "5"}:
            return f"{normalized}x"
    if label.endswith("x"):
        normalized = label[:-1]
        if normalized in {"0.2", "0.5", "2", "5"}:
            return label
    return label


def parse_pair(metadata):
    if not metadata:
        return None

    raw_parameters = metadata.get("parameters")
    if not raw_parameters:
        return None

    sweep_labels = metadata.get("sweep_labels", "")
    pair_size = len([part for part in str(sweep_labels).split(",") if part.strip()])
    if pair_size <= 0:
        pair_size = 2

    if "." in raw_parameters:
        group_part, param_part = raw_parameters.rsplit(".", 1)
        groups = [part.strip() for part in group_part.split(",")]
        params = [part.strip() for part in param_part.split(",")]
        if len(params) == pair_size:
            if len(groups) == len(params):
                return tuple(f"{group}.{param}" for group, param in zip(groups, params))
            return tuple(params)

    params = [part.strip() for part in raw_parameters.split(",")]
    if len(params) == pair_size:
        return tuple(params)

    return None


def read_pairwise_job_series(db_path, metrics="success", include_failed_trials=False):
    conn = sqlite3.connect(db_path)
    try:
        if metrics == "success":
            rows = conn.execute(
                """
                SELECT t.success, t.metadata
                FROM trials t
                WHERE t.metadata IS NOT NULL
                """
            ).fetchall()
        elif metrics == "anchor_error":
            rows = conn.execute(
                """
                SELECT t.anchor_error, t.success, t.metadata
                FROM trials t
                WHERE t.metadata IS NOT NULL
                """
            ).fetchall()
        elif metrics == "timesteps":
            rows = conn.execute(
                """
                SELECT t.timesteps, t.success, t.metadata
                FROM trials t
                WHERE t.metadata IS NOT NULL
                """
            ).fetchall()
    finally:
        conn.close()

    # Group pairwise data by (pair, label_a, label_b) to preserve label information
    by_pair_labels = defaultdict(lambda: defaultdict(list))
    by_job = defaultdict(list)

    for row in rows:
        if metrics != "success":
            # rows are (timesteps, success, metadata)
            try:
                metric_value, success_flag, metadata_json = row
            except Exception:
                continue
            # Optionally skip failed trials
            if not include_failed_trials and not success_flag:
                continue
        else:
            try:
                metric_value, metadata_json = row
            except Exception:
                continue
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            continue

        if metadata.get("experiment_type") != "pairwise_sweep":
            continue

        parameters = parse_pair(metadata)
        if parameters is None or len(parameters) != 2:
            continue

        pair = tuple(sorted(parameters))
        try:
            metric_val = float(metric_value)
        except (TypeError, ValueError):
            continue
        
        # Try to extract label information from metadata and normalize labels
        sweep_labels = metadata.get("sweep_labels", "")
        if sweep_labels:
            raw_parts = [part.strip() for part in str(sweep_labels).split(",") if part.strip()]
            if len(raw_parts) == 2:
                label_a, label_b = normalize_one_d_label(raw_parts[0]), normalize_one_d_label(raw_parts[1])
                # Only include known labels
                if label_a in ONE_D_LABELS_TO_INCLUDE and label_b in ONE_D_LABELS_TO_INCLUDE:
                    # Normalize label order to match sorted parameter order
                    if parameters[0] == pair[1]:
                        label_a, label_b = label_b, label_a
                    by_pair_labels[pair][(label_a, label_b)].append(metric_val)
        
        # Also keep the old grouping for backward compatibility
        job_key = (
            pair,
            str(metadata.get("sweep_labels", "")),
            str(metadata.get("sweep_values", "")),
        )
        by_job[job_key].append(metric_val)

    # Build pair_vectors with label information
    pair_vectors = {}
    for pair in by_pair_labels:
        pair_vectors[pair] = {}
        for (label_a, label_b), successes in by_pair_labels[pair].items():
            if successes:
                pair_vectors[pair][(label_a, label_b)] = float(np.mean(np.asarray(successes, dtype=float)))

    parameters = sorted({param for pair in pair_vectors for param in pair})

    return parameters, pair_vectors


def read_single_sweep_series(db_path, metrics="success", include_failed_trials=False):
    """
    Read 1d sweep trials and compute Si values.
    Returns both Si values and organized success data by parameter and label.
    """
    conn = sqlite3.connect(db_path)
    try:
        if metrics == "success":
            rows = conn.execute(
                """
                SELECT t.success, t.metadata
                FROM trials t
                WHERE t.metadata IS NOT NULL
                """
            ).fetchall()
        elif metrics == "anchor_error":
            rows = conn.execute(
                """
                SELECT t.anchor_error,t.success, t.metadata
                FROM trials t
                WHERE t.metadata IS NOT NULL
                """
            ).fetchall()
        elif metrics == "timesteps":
            rows = conn.execute(
                """
                SELECT t.timesteps, t.success, t.metadata
                FROM trials t
                WHERE t.metadata IS NOT NULL
                """
            ).fetchall()
        else:
            raise ValueError(f"Unsupported metrics: {metrics}")
    finally:
        conn.close()

    # Group by parameter -> label -> list of successes
    by_param_label = defaultdict(lambda: defaultdict(list))
    by_param_sweep = defaultdict(lambda: defaultdict(list))
    standard_baseline_successes = []

    for row in rows:
        if metrics != "success":
            try:
                metric_value, success_flag, metadata_json = row
            except Exception:
                continue
            if not include_failed_trials and not success_flag:
                continue
        else:
            try:
                metric_value, metadata_json = row
            except Exception:
                continue
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            continue

        if metadata.get("experiment_type") != "1d_sweep":
            continue

        parameter = metadata.get("parameter")
        if not parameter:
            continue

        sweep_value = metadata.get("sweep_value")
        label = normalize_one_d_label(metadata.get("label", ""))
        
        # Filter to only included labels
        if label not in ONE_D_LABELS_TO_INCLUDE:
            continue
        
        # Standard baseline trials may not have a sweep value.
        if sweep_value is None and label != "standard":
            continue

        try:
            metric_val = float(metric_value)
        except (TypeError, ValueError):
            # Skip rows with non-numeric metric values
            continue

        if parameter == "standard.standard" and label == "standard":
            standard_baseline_successes.append(metric_val)
            continue

        by_param_label[parameter][label].append(metric_val)
        by_param_sweep[parameter][(sweep_value, label)].append(metric_val)

    # Compute mean success per parameter per label for use in Hi computation
    one_d_successes_by_parameter = {}
    for parameter in by_param_label:
        one_d_successes_by_parameter[parameter] = {}
        for label in by_param_label[parameter]:
            successes = by_param_label[parameter][label]
            if successes:
                one_d_successes_by_parameter[parameter][label] = float(np.mean(np.asarray(successes, dtype=float)))

    if standard_baseline_successes:
        standard_baseline = float(np.mean(np.asarray(standard_baseline_successes, dtype=float)))
        for parameter in one_d_successes_by_parameter:
            one_d_successes_by_parameter[parameter].setdefault("standard", standard_baseline)
    else:
        standard_baseline = np.nan

    # For each parameter, compute S_i from the mean success rates per label.
    si_values = {}
    for parameter, label_means in one_d_successes_by_parameter.items():
        rates = [float(value) for value in label_means.values() if not np.isnan(value)]
        if len(rates) > 1:
            si_values[parameter] = float(np.sqrt(np.var(np.asarray(rates, dtype=float))))
        elif rates:
            # If only one label mean is available, variance is 0
            si_values[parameter] = 0.0

    return si_values, one_d_successes_by_parameter, standard_baseline

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
    ss_total_table = {}

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

        ss_total_table[pair] = ss_total
        ss_ab_table[pair] = ss_interaction

    return interaction_scores, ss_total_table, ss_ab_table


def plot_si_per_parameter(si_values, output_path):
    plt = _get_matplotlib_pyplot()
    parameters = sorted(si_values.keys())
    si_vals = [si_values[param] for param in parameters]

    fig, ax = plt.subplots(figsize=(max(10.0, 1.2 * len(parameters)), 6.0))
    ax.bar(range(len(parameters)), si_vals, color="skyblue", alpha=0.7)
    ax.set_xticks(range(len(parameters)))
    ax.set_xticklabels([short_label(param) for param in parameters], rotation=90, fontsize=8)
    ax.set_ylabel("S_i")
    ax.set_title("S_i per Parameter")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def scatter_plot_interaction_vs_total(min_ss_total, parameters, ss_total_table, ss_ab_table, output_path):
    plt = _get_matplotlib_pyplot()
    x_vals = []
    y_vals = []
    labels = []
    pairs = []
    for pair in ss_total_table:
        total = ss_total_table[pair]
        interaction = ss_ab_table.get(pair, np.nan)
        if not np.isnan(total) and not np.isnan(interaction):
            x_vals.append(total)
            y_vals.append(interaction)
            labels.append(f"{pair[0]} & {pair[1]}")
            pairs.append(pair)

    residuals_by_pair = {}
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_vals, y_vals, alpha=0.7)

    if len(x_vals) >= 2:
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        y_fit = slope * x_arr + intercept
        sort_idx = np.argsort(x_arr)
        ax.plot(x_arr[sort_idx], y_fit[sort_idx], color="red", linestyle="--", label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")

        residuals = y_arr - y_fit
        abs_residuals = np.abs(residuals)
        ranked = np.argsort(-abs_residuals)
        top_count = min(20, len(abs_residuals))

        short_labels = [f"{short_label(pair.split(' & ')[0])} & {short_label(pair.split(' & ')[1])}" for pair in labels]
        for idx in ranked[:top_count]:
            ax.annotate(
                short_labels[idx],
                (x_arr[idx], y_arr[idx]),
                fontsize=7,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ss_res = np.sum((y_arr - y_fit) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.legend()
    else:
        for idx, pair in enumerate(pairs):
            residuals_by_pair[pair] = np.nan

    ax.set_xlabel("Total Sum of Squares (SS_total)")
    ax.set_ylabel("Interaction Sum of Squares (SS_AB)")
    ax.set_title("Interaction vs Total Sum of Squares")
    ax.grid(True)
    # Plot minimum SS_total threshold
    if min_ss_total is not None and min_ss_total > 0:
        ax.axvline(
            x=min_ss_total,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Min SS_total = {min_ss_total:.3e}"
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return residuals_by_pair


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
        y_hi = 800000
    elif metrics == "success":
        x_lo = 0
        x_hi = 0.5
        y_lo = 0
        y_hi = 3.0
    elif metrics == "anchor_error":
        x_lo = 0
        x_hi = 0.15
        y_lo = 0
        y_hi = 0.3
    else:
        pass

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    highlighted_parameters = set()

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
            # Draw each highlighted point individually so Max residual points
            # keep their square marker and orange colour instead of being
            # overridden by the group colour/marker.
            for param, si_val, y_val in group_points:
                # compute per-param max across all residual series
                max_v = max((m.get(param, np.nan) for _, m in residual_series), default=np.nan)
                is_max = not np.isnan(max_v) and np.isfinite(max_v) and np.isclose(y_val, float(max_v))
                if is_max:
                    draw_color = "#ff7f0e"
                    draw_marker = "s"
                else:
                    draw_color = colour_map[group_name]
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

    plot_colours = {
        "Mean residual": "#1f77b4",
        "Max residual": "#ff7f0e",
    }
    plot_markers = {
        "Mean residual": "o",
        "Max residual": "s",
    }
    series_seen = set()
    for series_name, residuals_by_parameter in residual_series:
        series_x = []
        series_y = []
        for param in parameters:
            si = si_values.get(param, np.nan)
            residual = residuals_by_parameter.get(param, np.nan)
            if not np.isnan(si) and not np.isnan(residual):
                series_x.append(si)
                series_y.append(residual)
        if not series_x:
            continue
        ax.scatter(
            series_x,
            series_y,
            alpha=0.8,
            s=80,
            color=plot_colours.get(series_name, "#808080"),
            edgecolors="black",
            linewidth=0.4,
            marker=plot_markers.get(series_name, "o"),
            label=series_name,
            zorder=2,
        )
        series_seen.add(series_name)

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

    # Annotate parameters exceeding x OR y thresholds for Mean residual labels.
    annotate_mean_params = set()
    if x_threshold is not None or y_threshold is not None:
        for param in parameters:
            x_val = si_values.get(param, np.nan)
            y_val_mean = mean_residuals.get(param, np.nan) if mean_residuals is not None else np.nan
            above = False
            if x_threshold is not None and not np.isnan(x_val) and x_val > x_threshold:
                above = True
            if y_threshold is not None and not np.isnan(y_val_mean) and y_val_mean > y_threshold:
                above = True
            if above:
                annotate_mean_params.add(param)

    # Annotate parameters exceeding x OR y_max thresholds for Max residual labels.
    annotate_max_params = set()
    if x_threshold is not None or y_max_threshold is not None:
        for param in parameters:
            x_val = si_values.get(param, np.nan)
            y_val_max = max_residuals.get(param, np.nan) if max_residuals is not None else np.nan
            above_max = False
            if x_threshold is not None and not np.isnan(x_val) and x_val > x_threshold:
                above_max = True
            if y_max_threshold is not None and not np.isnan(y_val_max) and y_val_max > y_max_threshold:
                above_max = True
            if above_max:
                annotate_max_params.add(param)

    # Do not annotate highlighted groups here; annotations are strictly
    # controlled by x/y thresholds per user request.

    # Annotate Mean residual points
    for param in sorted(annotate_mean_params):
        si = si_values.get(param, np.nan)
        mean_r = mean_residuals.get(param, np.nan) if mean_residuals is not None else np.nan
        if np.isnan(si) or np.isnan(mean_r):
            continue
        ax.annotate(
            short_label(param),
            (si, mean_r),
            fontsize=7,
            alpha=0.9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Annotate Max residual points (separate threshold control)
    for param in sorted(annotate_max_params):
        si = si_values.get(param, np.nan)
        max_r = max_residuals.get(param, np.nan) if max_residuals is not None else np.nan
        if np.isnan(si) or np.isnan(max_r):
            continue
        ax.annotate(
            short_label(param),
            (si, max_r),
            fontsize=7,
            alpha=0.9,
            xytext=(5, -8),
            textcoords="offset points",
        )

    ax.set_xlabel("S_i (Self-Interaction std_dev)")
    ax.set_ylabel("Residual per Parameter")
    ax.set_title("SI vs Residuals (Mean and Max)")
    ax.grid(True, alpha=0.3)
    if series_seen:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        f"Group {index + 1}": [member for member in members if member in parameters]
        for index, (_, members) in enumerate(normalized_groups)
    }
     

def heatmap_ss_ab(ss_ab_table, output_path, max_value= None):
    plt = _get_matplotlib_pyplot()

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    import numpy as np

    # --------------------------------------------------
    # Build interaction matrix
    # --------------------------------------------------
    parameters = sorted({param for pair in ss_ab_table for param in pair})

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
    # Hierarchical clustering with pinned first 3 parameters
    # --------------------------------------------------

    # Find indices of the three target parameters
    target_labels = ["close_dist_threshold", "uncertainty_dist_threshold", "gripper_activation_threshold"]
    target_params = []
    for param in parameters:
        if short_label(param) in target_labels and len(target_params) < 3:
            target_params.append(param)
    
    # Ensure we have exactly 3 target params
    target_params = target_params[:3]
    remaining_params = [p for p in parameters if p not in target_params]
    
    # Apply clustering only to remaining parameters
    if len(remaining_params) > 1:
        remaining_indices = [param_index[p] for p in remaining_params]
        remaining_matrix = matrix[np.ix_(remaining_indices, remaining_indices)]
        cluster_matrix = np.nan_to_num(remaining_matrix, nan=0.0)
        distance = pdist(cluster_matrix, metric="correlation")
        linkage_matrix = linkage(distance, method="average")
        order = leaves_list(linkage_matrix)
        clustered_remaining = [remaining_params[i] for i in order]
    else:
        clustered_remaining = remaining_params
    
    # Combine: pinned params first, then clustered params
    parameters = target_params + clustered_remaining
    new_param_index = {param: idx for idx, param in enumerate(parameters)}
    
    # Reorder matrix
    old_indices = [param_index[p] for p in parameters]
    matrix = matrix[np.ix_(old_indices, old_indices)]


    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")   # NaN values

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.2 * len(parameters)),
            max(8.0, 1.2 * len(parameters))
        )
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


    # Axis labels
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))

    ax.set_xticklabels(
        [short_label(p) for p in parameters],
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        [short_label(p) for p in parameters],
        fontsize=8
    )


    # --------------------------------------------------
    # Cell boundaries
    # --------------------------------------------------

    ax.set_xticks(
        np.arange(-0.5, len(parameters), 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, len(parameters), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=0.4
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax.tick_params(
        which="major",
        length=0
    )


    # Remove outer border
    for spine in ax.spines.values():
        spine.set_visible(False)


    ax.set_title(
        "Hierarchically Clustered Interaction Sum of Squares ($SS_{AB}$)",
        fontsize=14
    )


    # Colorbar
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.02
    )

    cbar.set_label(
        r"$SS_{AB}$",
        fontsize=11
    )


    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def heatmap_pair_means(augmented_pair_vectors, output_path, max_value=None):
    plt = _get_matplotlib_pyplot()

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    import numpy as np

    parameters = sorted({param for pair in augmented_pair_vectors for param in pair})

    matrix = np.full((len(parameters), len(parameters)), np.nan)
    param_index = {param: idx for idx, param in enumerate(parameters)}

    for (param_a, param_b), values in augmented_pair_vectors.items():
        idx_a = param_index[param_a]
        idx_b = param_index[param_b]
        matrix[idx_a, idx_b] = float(np.nanmean(values))
        matrix[idx_b, idx_a] = float(np.nanmean(values))

    # Find indices of the three target parameters
    target_labels = ["close_dist_threshold", "uncertainty_dist_threshold", "gripper_activation_threshold"]
    target_params = []
    for param in parameters:
        if short_label(param) in target_labels and len(target_params) < 3:
            target_params.append(param)
    
    # Ensure we have exactly 3 target params
    target_params = target_params[:3]
    remaining_params = [p for p in parameters if p not in target_params]
    
    # Apply clustering only to remaining parameters
    if len(remaining_params) > 1:
        remaining_indices = [param_index[p] for p in remaining_params]
        remaining_matrix = matrix[np.ix_(remaining_indices, remaining_indices)]
        cluster_matrix = np.nan_to_num(remaining_matrix, nan=0.0)
        distance = pdist(cluster_matrix, metric="correlation")
        linkage_matrix = linkage(distance, method="average")
        order = leaves_list(linkage_matrix)
        clustered_remaining = [remaining_params[i] for i in order]
    else:
        clustered_remaining = remaining_params
    
    # Combine: pinned params first, then clustered params
    parameters = target_params + clustered_remaining
    new_param_index = {param: idx for idx, param in enumerate(parameters)}
    
    # Reorder matrix
    old_indices = [param_index[p] for p in parameters]
    matrix = matrix[np.ix_(old_indices, old_indices)]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 1.2 * len(parameters)),
            max(8.0, 1.2 * len(parameters))
        )
    )

    if max_value is None:
        im = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
    else:
        im = ax.imshow(
            matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=max_value,
        )

    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xticklabels([short_label(p) for p in parameters], rotation=90, fontsize=8)
    ax.set_yticklabels([short_label(p) for p in parameters], fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "Hierarchically Clustered Mean Pair Values",
        fontsize=14
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(
        "Mean Pair Value",
        fontsize=11
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_histogram(data, title, xlabel, ylabel, num_zeros, output_path):
    plt = _get_matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    #add text of number of ignored zeros in the histogram
    if num_zeros > 0:
        ax.text(
            0.95,
            0.95,
            f"Ignored zeros: {num_zeros}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def sorted_value_plot(data_dict, title, ylabel, output_path, knee=None, log_scale=False):
    plt = _get_matplotlib_pyplot()

    values = sorted(data_dict.values(), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(values)), values, linewidth=2)

    ax.set_xlabel("Pair rank")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel} (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if knee is not None:
        ax.axvline(knee, color="red", linestyle="--", label=f"Knee at rank {knee}")
        ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def shorten_param_name(name):
    parts = name.split(".")
    if len(parts) == 1:
        return name

    prefix = ".".join("".join(word[0] for word in part.split("_")) for part in parts[:-1])
    return f"{prefix}.{parts[-1]}"

def wrap_text(text, width=25):
    words = text.split(".")
    lines = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= width:
            current += ("" if current == "" else ".") + word
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return "\n".join(lines)


def resolve_parameter_selection(selection, parameters):
    """Resolve user-provided parameter names against full names and short labels."""
    if selection is None:
        return []

    resolved = []
    seen = set()
    for item in selection:
        candidate = str(item).strip()
        if not candidate:
            continue

        matches = [
            parameter
            for parameter in parameters
            if parameter == candidate
            or short_label(parameter) == candidate
            or parameter.endswith(f".{candidate}")
        ]
        for parameter in matches:
            if parameter not in seen:
                resolved.append(parameter)
                seen.add(parameter)

    return resolved

def create_table(table_data, output_path):
    """Render each selected parameter, its strongest interaction partner, and score."""
    sorted_data = sorted(table_data, key=lambda row: row[2], reverse=True)
    display_data = [
        (wrap_text(parameter), wrap_text(partner), f"{score:.4g}")
        for parameter, partner, score in sorted_data
    ]

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.35 * len(display_data) + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=display_data,
        colLabels=["Parameter", "Partner Parameter", "Max SS_AB"],
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 2.0)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_quadrant_parameter_table(quadrants, ss_ab_table, output_path):
    """Plot non-insensitive parameters in quadrant columns with strongest-pair links."""
    plt = _get_matplotlib_pyplot()
    from matplotlib.patches import FancyArrowPatch

    quadrants = {
        name: parameters_in_quadrant
        for name, parameters_in_quadrant in quadrants.items()
        if name != "Insensitive"
    }
    quadrant_colours = {
        "Sensitive by itself": "#bdd7e7",
        "Mainly interaction-sensitive": "#fdd0a2",
        "Sensitive both individually and through interactions": "#c7e9c0",
    }
    column_names = list(quadrants)
    column_parameters = {
        name: list(parameters_in_quadrant)
        for name, parameters_in_quadrant in quadrants.items()
    }
    displayed_parameters = {
        parameter
        for parameters_in_quadrant in column_parameters.values()
        for parameter in parameters_in_quadrant
    }
    parameter_columns = {
        parameter: column_name
        for column_name, parameters_in_quadrant in column_parameters.items()
        for parameter in parameters_in_quadrant
    }

    # Retain only strongest-pair links whose two parameters are displayed.
    links = []
    linked_pairs = set()
    for parameter in displayed_parameters:
        best_match = max(
            (
                (score, parameter_b if parameter_a == parameter else parameter_a)
                for (parameter_a, parameter_b), score in ss_ab_table.items()
                if parameter in (parameter_a, parameter_b) and not np.isnan(score)
            ),
            default=None,
            key=lambda match: match[0],
        )
        if best_match is None:
            continue
        _, partner = best_match
        pair_key = frozenset((parameter, partner))
        if partner in displayed_parameters and pair_key not in linked_pairs:
            linked_pairs.add(pair_key)
            links.append((parameter, partner))

    # Reorder each column by the mean row of its linked neighbours. Repeating
    # left-to-right and right-to-left passes reduces crossings without changing
    # quadrant membership.
    neighbours = {parameter: [] for parameter in displayed_parameters}
    for parameter, partner in links:
        neighbours[parameter].append(partner)
        neighbours[partner].append(parameter)

    def parameter_ranks():
        return {
            parameter: row
            for parameters_in_quadrant in column_parameters.values()
            for row, parameter in enumerate(parameters_in_quadrant)
        }

    for _ in range(8):
        for column_name in column_names + column_names[::-1]:
            ranks = parameter_ranks()
            original_positions = {
                parameter: row
                for row, parameter in enumerate(column_parameters[column_name])
            }
            column_parameters[column_name].sort(
                key=lambda parameter: (
                    float(np.mean([
                        ranks[neighbour]
                        for neighbour in neighbours[parameter]
                        if parameter_columns[neighbour] != column_name
                    ]))
                    if any(
                        parameter_columns[neighbour] != column_name
                        for neighbour in neighbours[parameter]
                    )
                    else original_positions[parameter],
                    original_positions[parameter],
                )
            )

    max_rows = max((len(values) for values in column_parameters.values()), default=0)
    row_spacing = 1.5
    top_y = (max_rows - 1) * row_spacing
    fig_height = max(4.0, 0.58 * max_rows + 2.6)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    positions = {}
    for column, column_name in enumerate(column_names):
        ax.text(
            column,
            top_y + 1.25,
            column_name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            wrap=True,
            bbox=dict(
                facecolor=quadrant_colours[column_name],
                edgecolor="black",
                boxstyle="round,pad=0.4",
            ),
            zorder=4,
        )
        for row, parameter in enumerate(column_parameters[column_name]):
            positions[parameter] = (column, top_y - row * row_spacing)

    for edge_index, (parameter, partner) in enumerate(links):
        curvature = 0.2 if edge_index % 2 == 0 else -0.2
        ax.add_patch(
            FancyArrowPatch(
                positions[parameter],
                positions[partner],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={curvature}",
                color="#4c78a8",
                linewidth=1.2,
                alpha=0.75,
                zorder=1,
            )
        )

    for parameter, (x_position, y_position) in positions.items():
        ax.text(
            x_position,
            y_position,
            short_label(parameter),
            ha="center",
            va="center",
            fontsize=8,
            zorder=3,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.28"),
        )

    ax.set_xlim(-0.55, len(column_names) - 0.45)
    ax.set_ylim(-row_spacing, top_y + 2.0)
    ax.set_title("Parameters by Sensitivity Quadrant (curved lines: strongest displayed SS_AB pair)")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
        choices=["success", "anchor_error", "timesteps", "grasp"],
        default="success",
        help="Metric to analyze: success (default), anchor_error, timesteps, or grasp.",
    )
    parser.add_argument(
        "--include-failed-trials",
        action="store_true",
        help="When analyzing timesteps, include failed trials (otherwise only successful trials are considered).",
    )
    args = parser.parse_args()

    if args.db_path is None or not args.db_path.exists():
        db_path = Path(__file__).parent.parent.parent / "data" / "noise" / "experiment_store.db"
        print(f"Database path not provided or does not exist. Using default: {db_path}")
        # raise SystemExit(f"Database not found: {args.db_path}")
    else:
        db_path = args.db_path.resolve()


    # Which metric to analyze
    print(f"Using metric: {args.metric}")

    # Determine output directory: use --output if provided (file or directory),
    # otherwise write to project_root/metrics/<metric>/
    project_root = Path(__file__).parent.parent.parent
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
        db_path, metrics=args.metric, include_failed_trials=args.include_failed_trials
    )


    # Read single sweep data for Si values and 1D parameter data
    si_values, one_d_successes_by_parameter, standard_baseline = read_single_sweep_series(
        db_path, metrics=args.metric, include_failed_trials=args.include_failed_trials
    )
    # Print single-sweep standard deviations (S_i) for key parameters for debugging
    target_params = [
        "drawer_position.initial_uncertainty_depth",
        "drawer_position.initial_uncertainty_scale",
    ]
    print("Single-sweep standard deviations (S_i) from 1D sweeps:")
    for tp in target_params:
        si_val = si_values.get(tp, np.nan)
        means = one_d_successes_by_parameter.get(tp, {})
        print(f" - {tp}: S_i = {si_val}, 1D means = {means}")
    
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
    interaction_scores, ss_total_table, ss_ab_table = calculate_ANOVA_interaction_score(
        parameters, augmented_pair_vectors, augmented_pair_diagnostics
    )

    print("number of pairs:", len(ss_total_table))
    print("number of exact zeros ss_total:", sum(1 for v in ss_total_table.values() if v == 0))

    min_ss_total = 1e-6 # just as long as not zero, to avoid division by zero in interaction score calculation

    scatter_plot_interaction_vs_total(min_ss_total, parameters, ss_total_table, ss_ab_table, output_dir / "interaction_vs_total.png")

    # if file already exists don't overwrite it, just print a message and skip
    heatmap_path = output_dir / "ss_ab_heatmap.png"
    if heatmap_path.exists():
        print(f"Heatmap file already exists: {heatmap_path}. Skipping heatmap generation.")
    else:
        heatmap_ss_ab(ss_ab_table, heatmap_path)

    mean_heatmap_path = output_dir / "pair_mean_heatmap.png"
    if mean_heatmap_path.exists():
        print(f"Heatmap file already exists: {mean_heatmap_path}. Skipping mean heatmap generation.")
    else:
        heatmap_pair_means(augmented_pair_vectors, mean_heatmap_path)

    valid_pairs = {
        pair: score
        for pair, score in ss_ab_table.items()
        if ss_total_table[pair] > min_ss_total
    }
 

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
    mean_ssab_deriv = np.diff(mean_ssab_values)
    max_ssab_deriv = np.diff(max_ssab_values)


    #calculte rolling average
    window_size = 5
    mean_ssab_deriv_rolling = np.convolve(mean_ssab_deriv, np.ones(window_size)/window_size, mode='valid')
    max_ssab_deriv_rolling = np.convolve(max_ssab_deriv, np.ones(window_size)/window_size, mode='valid')
    
    from scipy.stats import median_abs_deviation

    # --------------------------------------------------
    # Estimate epsilon from the plateau (last 50 points)
    # --------------------------------------------------
    tail = np.abs(max_ssab_deriv_rolling[-50:])

    eps = np.median(tail) + 3 * median_abs_deviation(tail)

    print(f"Epsilon = {eps:.5f}")

    # --------------------------------------------------
    # Find first sustained plateau
    # --------------------------------------------------
    plateau_index = None
    for i in range(len(max_ssab_deriv_rolling) - window_size):
        if np.all(np.abs(max_ssab_deriv_rolling[i:i+window_size]) < eps):
            plateau_index = i
            break

    print(f"Plateau starts at rank {plateau_index}")

    # --------------------------------------------------
    # Find SI plateau index if SI_SEPERATE is True
    # --------------------------------------------------
    plateau_index_si = None
    plt = _get_matplotlib_pyplot()
    if SI_SEPERATE:
        # Get SI values sorted in descending order
        si_sorted_values = sorted(si_values.values(), reverse=True)
        
        # Filter out zeros and near-zero values (keep only values > 1e-6)
        si_nonzero_values = np.array([v for v in si_sorted_values if v > 1e-6])
        
        if len(si_nonzero_values) > window_size:
            # Calculate derivatives of the non-zero SI values
            si_deriv = np.diff(si_nonzero_values)
            si_deriv_rolling = np.convolve(si_deriv, np.ones(window_size)/window_size, mode='valid')
            
            # Estimate epsilon for SI plateau detection using last 20% of non-zero values
            tail_si = np.abs(si_deriv_rolling[-10:])  # Last 10 points of the rolling derivative
            if len(tail_si) > 0:
                eps_si = np.median(tail_si) + 3 * median_abs_deviation(tail_si) if len(tail_si) > 1 else np.median(tail_si)
            else:
                eps_si = np.median(np.abs(si_deriv_rolling)) + 3 * median_abs_deviation(np.abs(si_deriv_rolling))
            
            print(f"SI Epsilon = {eps_si:.5f}")
            
            # Find first sustained plateau in non-zero region
            for i in range(len(si_deriv_rolling) - window_size):
                if np.all(np.abs(si_deriv_rolling[i:i+window_size]) < eps_si):
                    plateau_index_si = i
                    break
            
            print(f"SI Plateau starts at rank {plateau_index_si}")
            
            # Plot SI derivative
            fig_si, ax_si = plt.subplots(figsize=(10, 5))
            ax_si.plot(range(len(si_deriv)), si_deriv, label="SI Derivative", color="green")
            ax_si.plot(range(len(si_deriv_rolling)), si_deriv_rolling, label="SI Derivative (Rolling Avg)", color="green", linestyle="--")
            ax_si.axhline(0, color="black", linestyle="--", linewidth=1.0)
            if plateau_index_si is not None:
                ax_si.axvline(plateau_index_si, color="red", linestyle="--", label=f"SI Plateau starts at rank {plateau_index_si}")
                y_min, y_max = ax_si.get_ylim()
                y_middle = (y_min + y_max) / 2
                ax_si.text(
                    plateau_index_si,
                    y_middle,
                    f"SI Plateau\nrank = {plateau_index_si}",
                    color="red",
                    rotation=90,
                    verticalalignment="top",
                    horizontalalignment="right",
                    fontsize=10,
                    bbox=dict(alpha=0.3)
                )
            ax_si.set_xlabel("Non-Zero SI Rank")
            ax_si.set_ylabel("Derivative of SI")
            ax_si.set_title("Derivative of SI (Non-Zero Values)")
            ax_si.legend()
            fig_si.tight_layout()
            fig_si.savefig(output_dir / "si_derivative.png", dpi=200)

    #plot the derivative of max_ssab_s as a line plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(range(len(mean_ssab_deriv)), mean_ssab_deriv, label="Mean ssab Derivative", color="blue")
    ax.plot(range(len(max_ssab_deriv)), max_ssab_deriv, label="Max ssab Derivative", color="orange")
    # ax.plot(range(len(mean_ssab_deriv_rolling)), mean_ssab_deriv_rolling, label="Mean ssab Derivative (Rolling Avg)", color="blue", linestyle="--")
    ax.plot(range(len(max_ssab_deriv_rolling)), max_ssab_deriv_rolling, label="Max ssab Derivative (Rolling Avg)", color="orange", linestyle="--")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    if plateau_index is not None:
        ax.axvline(plateau_index, color="red", linestyle="--", label=f"Plateau starts at rank {plateau_index}")
        y_min, y_max = ax.get_ylim()
        y_middle = (y_min + y_max) / 2
        # Print plateau index value on the plot
        ax.text(
            plateau_index,
            y_middle,
            f"Plateau\nrank = {plateau_index}",
            color="red",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(alpha=0.3)
        )
    ax.set_xlabel("Parameter Rank")
    ax.set_ylabel("Derivative of ssab")
    ax.set_title("Derivative of Max ssab per Parameter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ssab_derivative.png", dpi=200)

    sorted_value_plot(
            max_ssab_s,
            title="Sorted Maximum SS_ab per Parameter",
            ylabel="Max SS_ab",
            output_path=output_dir / "max_ssab_sorted.png",
            log_scale=True,
            knee=None
        )

    sorted_value_plot(
            mean_ssab_s,
            title="Sorted Mean SS_ab per Parameter",
            ylabel="Mean SS_ab",
            output_path=output_dir / "mean_ssab_sorted.png",
            log_scale=True,
            knee=None
        )

    if SI_SEPERATE:
        sorted_value_plot(
                si_values,
                title="Sorted SI per Parameter",
                ylabel="SI (Sensitivity Index)",
                output_path=output_dir / "si_sorted.png",
                log_scale=True,
                knee=plateau_index_si
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
    )

    #get top "plateau index" number of parameters by max_ssab_s
    top_parameters = sorted(parameters, key=lambda p: max_ssab_s.get(p, -np.inf), reverse=True)[:plateau_index]

    # Find the strongest interaction partner for every selected parameter.
    top_parameter_pairs = []
    for parameter in top_parameters:
        best_match = max(
            (
                (interaction, pb if pa == parameter else pa)
                for (pa, pb), interaction in ss_ab_table.items()
                if parameter in (pa, pb) and not np.isnan(interaction)
            ),
            default=None,
            key=lambda match: match[0],
        )
        if best_match is not None:
            score, partner = best_match
            top_parameter_pairs.append((parameter, partner, score))

    # Output each parameter, its strongest partner, and interaction score.
    create_table(top_parameter_pairs, output_dir / "high_interaction_pairs.png")

    #plot also number of zeros in the mean_ssab_s for each parameter
    num_zeros_per_param = {p: 0 for p in parameters}
    for (pa, pb), interaction in ss_ab_table.items():
        if np.isnan(interaction) or interaction <= 0:
            num_zeros_per_param[pa] += 1
            num_zeros_per_param[pb] += 1
    plot_histogram(
        list(num_zeros_per_param.values()),
        title="Histogram of Number of Zero Interaction Residuals per Parameter",
        xlabel="Number of Zero Interaction Residuals",
        ylabel="Frequency",
        num_zeros=0,
        output_path=output_dir / "num_zero_interaction_residuals_histogram.png"
    )

if __name__ == "__main__":
    main()
