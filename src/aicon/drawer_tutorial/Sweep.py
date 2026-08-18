import time
import csv
import copy
import random
import inspect
from pathlib import Path

import lovely_tensors as lt
import numpy as np
import torch
from robosuite.utils.transform_utils import quat2mat

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.drawer_tutorial.connections import DistGraspHandConnection
from aicon.drawer_tutorial.estimators import (
    DrawerPositionEstimator,
    EEPoseEstimator,
    GraspedEstimator,
    KinematicJointEstimator,
    VisibleEstimator,
)
from aicon.middleware.python_sequential import build_components, run_component_sequence
from aicon.drawer_tutorial.run_demo import (
    DEFAULT_INITIAL_PANDA_QPOS,
    create_demo_visualizers,
    render_frame,
    run_trial as run_demo_trial,
    setup_env as setup_demo_env,
    update_demo_visualizers,
)


DEFAULT_VISUALIZE_KINEMATICS = True
DEFAULT_VISUALIZE_EE = True
DEFAULT_VISUALIZE_GRASP_DIAGNOSTICS = True
# Demo configuration: set mode to either "sweep" or "default_loop"
# - "sweep": run the parameter sweep (existing behavior)
# - "default_loop": run the environment repeatedly with default params
DEMO_MODE = "sweep"  # options: "sweep", "default_loop"
# Delay between default-loop trials (seconds)
DEFAULT_LOOP_DELAY = 1.0
# Set to an integer for reproducible runs, or None to disable explicit seeding.
DEMO_SEED = None
# If True, reseed before each trial so each trial starts from the same RNG state.
RESEED_EACH_TRIAL = False
# If True, only run the single parameter selected below.
RUN_ONLY_SINGLE_PARAMETER = True
# Selected single-parameter sweep target.
SINGLE_SWEEP_GROUP = "connection.DistGraspHandConnection"
SINGLE_SWEEP_PARAM = "ft_noise_offset"


def set_global_seed(seed):
    """Seed Python / NumPy / Torch RNGs used by the demo and robosuite resets."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_env(initial_qpos=None):
    """Use the same renderer, camera, and visualization indicators as ``run_demo``."""
    env, _ = setup_demo_env("keyboard", initial_qpos=initial_qpos)
    return env


_NON_TUNABLE_INIT_PARAMS = {
    "self", "name", "connections", "goals", "dtype", "device", "mockbuild",
    "no_differentiation", "prevent_loops_in_differentiation",
    "max_length_differentiation_trace",
}

# These are the parameters intentionally included in the sweep. Their values are
# read from the constructors, so no numeric defaults are duplicated here.
_ESTIMATOR_SWEEP_PARAMS = {
    "ee_pose": (
        "initial_uncertainty_scale", "action_process_noise", "proprio_update_noise",
    ),
    "visible": (
        "initial_likelihood_prior", "initial_clip_min", "initial_clip_max", "update_gain",
    ),
    "grasp_likelihood": (
        # "initial_likelihood", 
        # "initially_grasped_likelihood",

        "baseline_measurement_likelihood", 
        "initial_clip_min", 
        "initial_clip_max",
        "initial_time_since_hand_change", 

        # "gripper_activation_threshold",
        # "close_distance_threshold", 
        # "uncertainty_dist_threshold",
        # "hand_change_time_threshold", 
        # "force_time_scale", 
        # "force_time_max",
        # "low_likelihood_threshold", 
        
        "negative_innovation_threshold",
        "negative_innovation_scale",
    ),
    "drawer_position": (
        # "depth_prior", 
        
        "initial_uncertainty_scale", "initial_uncertainty_xy",
        "initial_uncertainty_depth", "initial_uncertainty_xy_none",
        "initial_uncertainty_depth_none", 
        
        # "sample_init_mean_likelihood_threshold",
        # "sample_init_mean_distance_threshold",
        # "sample_init_mean_uncertainty_multiplier", 
        
        "meas_noise_factor",
        "visual_likelihood_steepness", "R_add_scale", "measurement_nan_reject_scale",
        "forward_noise_grasped_coeff", "forward_noise_base", "grasped_update_R_scale",
        "grasped_outlier_rejection_threshold", "measurement_existence_threshold",
        "grasped_uncertainty_threshold", "missed_absent_measurement_uncertainty_coeff",
        "hand_change_recovery_time", 
        
        # "tf_lookup_timeout",
    ),
    "kinematic_joint": (
        # "initial_azimuth_default", 
        # "initial_elevation_default", 
        "initial_uncertainty_scale",
        "initial_elevation_uncertainty_scale",
        "initial_azimuth_uncertainty_scale",
        "joint_initial_uncertainty_scale",
        
        "grasped_noise",
        "ungrasped_noise", "axis_azimuth_process_noise",
        "axis_elevation_process_noise", "joint_process_noise", "anchor_process_noise",
        "grasp_threshold", "grasp_floor", "outlier_rejection_treshold", 
        # "shift_clip_min",
    ),
}


def _get_constructor_defaults(component_class):
    """Return the tunable keyword defaults declared by a component constructor."""
    return {
        parameter.name: parameter.default
        for parameter in inspect.signature(component_class.__init__).parameters.values()
        if parameter.name not in _NON_TUNABLE_INIT_PARAMS
        and parameter.default is not inspect.Parameter.empty
    }


def _get_selected_constructor_defaults(component_class, parameter_names):
    defaults = _get_constructor_defaults(component_class)
    return {parameter_name: defaults[parameter_name] for parameter_name in parameter_names}


def get_default_estimator_params():
    """Get estimator defaults from the constructors used by the experiment builder."""
    return {
        "ee_pose": _get_selected_constructor_defaults(
            EEPoseEstimator, _ESTIMATOR_SWEEP_PARAMS["ee_pose"]
        ),
        "visible": _get_selected_constructor_defaults(
            VisibleEstimator, _ESTIMATOR_SWEEP_PARAMS["visible"]
        ),
        "grasp_likelihood": _get_selected_constructor_defaults(
            GraspedEstimator, _ESTIMATOR_SWEEP_PARAMS["grasp_likelihood"]
        ),
        "drawer_position": _get_selected_constructor_defaults(
            DrawerPositionEstimator, _ESTIMATOR_SWEEP_PARAMS["drawer_position"]
        ),
        "kinematic_joint": _get_selected_constructor_defaults(
            KinematicJointEstimator, _ESTIMATOR_SWEEP_PARAMS["kinematic_joint"]
        ),
    }


def get_default_connection_params():
    """Get connection defaults from the constructors used by the experiment builder."""
    return {
        "DistGraspHandConnection": _get_constructor_defaults(DistGraspHandConnection),
    }

def get_sweep_values(standard_value):
    return [
        ("negative", -standard_value),
        ("zero", 0.0),
        ("x0.001", 0.001 * standard_value),
        ("x0.2", 0.2 * standard_value),
        ("x0.5", 0.5 * standard_value),
        ("x2", 2.0 * standard_value),
        ("x5", 5.0 * standard_value),
        ("x1000", 1000.0 * standard_value),
    ]

# def get_sweep_values(standard_value):
#     return [
#         ("x0.2", 0.2 * standard_value),
#         ("x5", 5.0 * standard_value),
#     ]


def get_standard_job(base_params):
    return {
        "group_name": "standard",
        "param_name": "standard",
        "standard_value": None,
        "sweep_label": "standard",
        "sweep_value": None,
        "trial_params": copy.deepcopy(base_params),
    }


def generate_single_parameter_sweeps(base_params):
    for group_name in sorted(base_params.keys()):
        for param_name in sorted(base_params[group_name].keys()):
            print(f"Generating sweep for {group_name}.{param_name}")
            standard_value = base_params[group_name][param_name]
            for sweep_label, sweep_value in get_sweep_values(standard_value):
                trial_params = copy.deepcopy(base_params)
                trial_params[group_name][param_name] = sweep_value
                yield {
                    "group_name": group_name,
                    "param_name": param_name,
                    "standard_value": standard_value,
                    "sweep_label": sweep_label,
                    "sweep_value": sweep_value,
                    "trial_params": trial_params,
                }


def filter_single_parameter_sweeps(sweep_jobs, group_name, param_name):
    return [job for job in sweep_jobs if job["group_name"] == group_name and job["param_name"] == param_name]


def run_trial(env, estimator_params, max_timesteps=None, sweep_label="", group_name="",
              param_name="", sweep_value=None, stop_on_done=True, reset_on_start=True,
              random_init_time=0.0, random_std=0.1, disturbance=None, noise_scale=1.0, prior_noise_std=0.01, prior_noise_std_kinematic=0.0):
    """Sweep-compatible adapter around the shared demo trial runner."""
    status_label = f"{group_name}.{param_name}={sweep_label}:{sweep_value}"
    return run_demo_trial(
        env,
        device=None,
        estimator_params=estimator_params,
        max_timesteps=max_timesteps,
        stop_on_done=stop_on_done,
        reset_on_start=reset_on_start,
        random_init_time=random_init_time,
        random_std=random_std,
        random_disturbance=disturbance,
        noise_scale=noise_scale,
        prior_noise_std=prior_noise_std,
        prior_noise_std_kinematic=prior_noise_std_kinematic,
        # Sweep historically creates its diagnostic plots independently of MuJoCo rendering.
        status_label=status_label,
    )


def save_sweep_results(records, save_path):
    fieldnames = [
        "trial_index",
        "param_group",
        "param_name",
        "standard_value",
        "sweep_label",
        "sweep_value",
        "success",
        "timesteps",
        "timeout_timesteps",
        "stop_joint_error_est_minus_true",
        "stop_grasp_estimate",
        "stop_actual_grasped",
        "error",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_sweep_summary(summary, save_path):
    """Save summary of success rates per parameter."""
    fieldnames = [
        "param_group",
        "param_name",
        "standard_value",
        "sweep_label",
        "sweep_value",
        "successes",
        "total_runs",
        "success_rate",
        "average_timesteps",
        "average_stop_joint_error",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main(env, max_timesteps_per_trial=500, rec_save_path=None,
         visualize_kinematic_angles=DEFAULT_VISUALIZE_KINEMATICS,
         visualize_ee=DEFAULT_VISUALIZE_EE,
         visualize_grasp_diagnostics=DEFAULT_VISUALIZE_GRASP_DIAGNOSTICS):
    env.reset()

    # Print debug information about the drawer
    env.env.print_debug_info()

    # Get the observation using the original method
    env_obs = env._get_observations()


    base_params = get_default_estimator_params()
    base_conn_params = get_default_connection_params()

    # estimator sweeps
    sweep_jobs_est = list(generate_single_parameter_sweeps(base_params))
    # connection sweeps (create trials that override connection params while keeping estimator params default)
    sweep_jobs_conn = list(generate_single_parameter_sweeps(base_conn_params))
    for job in sweep_jobs_conn:
        # rename group for clarity
        job["group_name"] = f"connection.{job['group_name']}"
        # extract the connection trial params and store under top-level connection_params in trial_params
        conn_trial = job.pop("trial_params")
        trial_est_params = copy.deepcopy(base_params)
        trial_est_params["connection_params"] = conn_trial
        job["trial_params"] = trial_est_params

    sweep_jobs = sweep_jobs_est + sweep_jobs_conn
    if RUN_ONLY_SINGLE_PARAMETER:
        sweep_jobs = filter_single_parameter_sweeps(sweep_jobs, SINGLE_SWEEP_GROUP, SINGLE_SWEEP_PARAM)
        if not sweep_jobs:
            raise ValueError(f"No sweep jobs matched {SINGLE_SWEEP_GROUP}.{SINGLE_SWEEP_PARAM}")
    results = []
    summary = []
    
    num_runs_per_param = 5

    print(f"\nStarting sweep: {len(sweep_jobs)} parameters, {num_runs_per_param} runs per parameter, timeout={max_timesteps_per_trial} timesteps per trial")

    for param_index, job in enumerate(sweep_jobs, start=1):
        group_name = job["group_name"]
        param_name = job["param_name"]
        sweep_label = job["sweep_label"]
        sweep_value = job["sweep_value"]
        print(
            f"Currently sweeping [{param_index:03d}/{len(sweep_jobs):03d}]: "
            f"{group_name}.{param_name} -> {sweep_label} ({sweep_value})",
            flush=True,
        )
        
        successes = 0
        total_timesteps = 0
        total_stop_joint_error = 0.0
        stop_joint_error_count = 0
        # Run this parameter 5 times
        for run_num in range(1, num_runs_per_param + 1):
            trial_index = (param_index - 1) * num_runs_per_param + run_num
            
            if DEMO_SEED is not None and RESEED_EACH_TRIAL:
                set_global_seed(DEMO_SEED)

            error_text = ""
            success = False
            timesteps = max_timesteps_per_trial
            stop_joint_error = float("nan")
            stop_grasp_estimate = float("nan")
            stop_actual_grasped = False
            try:
                success, timesteps, stop_joint_error, stop_grasp_estimate, stop_actual_grasped = run_trial(
                    env,
                    estimator_params=job["trial_params"],
                    max_timesteps=max_timesteps_per_trial,
                    sweep_label=sweep_label,
                    group_name=group_name,
                    param_name=param_name,
                    sweep_value=sweep_value,
                    visualize_kinematic_angles=visualize_kinematic_angles,
                    visualize_ee=visualize_ee,
                    visualize_grasp_diagnostics=visualize_grasp_diagnostics,
                )
            except Exception as exc:
                error_text = str(exc)
                success = False
                timesteps = max_timesteps_per_trial
                stop_joint_error = float("nan")
                stop_grasp_estimate = float("nan")
                stop_actual_grasped = False

            if success:
                successes += 1
            if not np.isnan(timesteps):
                total_timesteps += timesteps
            if not np.isnan(stop_joint_error):
                total_stop_joint_error += stop_joint_error
                stop_joint_error_count += 1
            
            results.append(
                {
                    "trial_index": trial_index,
                    "param_group": group_name,
                    "param_name": param_name,
                    "standard_value": job["standard_value"],
                    "sweep_label": sweep_label,
                    "sweep_value": sweep_value,
                    "success": int(success),
                    "timesteps": timesteps,
                    "timeout_timesteps": max_timesteps_per_trial,
                    "stop_joint_error_est_minus_true": stop_joint_error,
                    "stop_grasp_estimate": stop_grasp_estimate,
                    "stop_actual_grasped": int(stop_actual_grasped),
                    "error": error_text,
                }
            )
            print(f"    Run {run_num}/{num_runs_per_param}: {'Success' if success else 'Failed'}", flush=True)
        
        # Calculate and report success rate for this parameter
        success_rate = successes / num_runs_per_param
        average_timesteps = total_timesteps / num_runs_per_param
        average_stop_joint_error = total_stop_joint_error / stop_joint_error_count if stop_joint_error_count > 0 else float("nan")
        summary.append(
            {
                "param_group": group_name,
                "param_name": param_name,
                "standard_value": job["standard_value"],
                "sweep_label": sweep_label,
                "sweep_value": sweep_value,
                "successes": successes,
                "total_runs": num_runs_per_param,
                "success_rate": success_rate,
                "average_timesteps": average_timesteps,
                "average_stop_joint_error": average_stop_joint_error,
            }
        )
        print(
            f"  Success rate: {successes}/{num_runs_per_param} = {success_rate*100:.1f}% | "
            f"avg timesteps: {average_timesteps:.1f} | "
            f"avg stop joint err: {average_stop_joint_error:.6f}\n",
            flush=True,
        )

    if rec_save_path is None:
        rec_save_path = str(Path(__file__).with_name("sweep_results.csv"))
    save_sweep_results(results, rec_save_path)
    
    # Save summary with success rates
    summary_save_path = str(Path(__file__).with_name("sweep_summary.csv"))
    save_sweep_summary(summary, summary_save_path)

    total_success = sum(r["success"] for r in results)
    print(f"\nSweep complete. Successes: {total_success}/{len(results)}")
    print(f"Results saved to: {rec_save_path}")
    print(f"Summary saved to: {summary_save_path}")


if __name__ == "__main__":
    set_global_seed(DEMO_SEED)
    env = setup_env(initial_qpos=DEFAULT_INITIAL_PANDA_QPOS)
    main(env, max_timesteps_per_trial=500)
