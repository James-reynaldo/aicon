import time
import csv
import copy
import random
from pathlib import Path

import lovely_tensors as lt
import numpy as np
import robosuite as suite
import torch
from robosuite.utils.transform_utils import quat2mat
from robosuite.wrappers import VisualizationWrapper

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.drawer_tutorial.connections import (
    # expose DistGraspHandConnection defaults for sweeping
    # keep these names in sync with the constructor defaults
    FT_ROTATION,
    FT_COM,
    FT_BIAS,
    EE_MASS,
)

# Import our custom environment
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.middleware.python_sequential import build_components, run_component_sequence

DEFAULT_RENDER_SWEEP = True
# Demo configuration: set mode to either "sweep" or "default_loop"
# - "sweep": run the parameter sweep (existing behavior)
# - "default_loop": run the environment repeatedly with default params
DEMO_MODE = "default_loop"  # options: "sweep", "default_loop"
# Delay between default-loop trials (seconds)
DEFAULT_LOOP_DELAY = 1.0
# Set to an integer for reproducible runs, or None to disable explicit seeding.
DEMO_SEED = 123
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


def setup_env(initial_qpos=None, render=False):
    # Create our custom environment
    env = DrawerOpenEnv(
        robots="Panda",
        has_renderer=render,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="agentview",
        horizon=100,
        control_freq=30,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSITION"),
        initial_qpos=initial_qpos,
    )

    env = VisualizationWrapper(env)
    env.reset()
    return env


def get_default_estimator_params():
    return {
        "ee_pose": {
            "action_process_noise": 0.01, # Seem to not matter so much, just not negative (fail)
            "proprio_update_noise": 0.01, # Seem to not matter so much, just not negative (fail)
            "initial_uncertainty_scale": 0.001, # Seem to not matter at all
        },
        "visible_likelihood": {
            "initial_likelihood": 0.01,
        },
        "grasp_likelihood": {
            "initial_likelihood": 0.01, # Seem to not matter at all
        },
        "drawer_position": {
            "forward_noise_grasped_scale": 0.15, # When negative it will fail, else does not seem to matter much
            "forward_noise_base_scale": 0.005, # Seem to not matter at all
            "grasp_update_noise": 0.01, # When negative, timestep is longer but it will still often succeed; when positive it does not seem to matter much
            "direct_measure_noise": 0.01, # When negative there is a chance to fail
            "grasp_outlier_threshold": 0.05, # Seem to not matter at all
            "initial_uncertainty_scale": 0.001, # When value too high, estimation error is big
        },
        "kinematic_joint": {
            "grasped_noise": 0.002, # When value too high, estimation error quite big
            "ungrasped_noise": 0.001, # Seem to not matter at all
            "axis_azimuth_process_noise": 0.001, # Does not seem to matter
            "axis_elevation_process_noise": 0.001, # Does not seem to matter
            "joint_process_noise": 0.2, # When negative, take a long time and high estimation error, when zero, high estimation error
            "anchor_process_noise": 1e-6, # Does not seem to matter
            "grasp_threshold": 0.2, # Does not matter much, just not zero
            "grasp_floor": 0.2, # When value too high, estimation error quite big
            "covariance_alpha": 1.0, # When negative, has tendency to fail
        },
    }


def get_sweep_values(standard_value):
    return [
        ("standard", standard_value),
        ("negative", -standard_value),
        ("zero", 0.0),
        ("x0.001", 0.001 * standard_value),
        ("x0.5", 0.5 * standard_value),
        ("x2", 2.0 * standard_value),
        ("x1000", 1000.0 * standard_value),
    ]


def generate_single_parameter_sweeps(base_params):
    for group_name in sorted(base_params.keys()):
        for param_name in sorted(base_params[group_name].keys()):
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


def get_default_connection_params():
    # Defaults mirror the hard-coded values in DistGraspHandConnection
    return {
        "DistGraspHandConnection": {
            "dist_decay": 4.0,
            "close_dist_threshold": 0.05,
            "force_threshold": 10.0,
            "ft_noise_offset": 5.0,
            "low_likelihood_threshold": 0.1,
            "gripper_activation_threshold": 0.5,
            "uncertainty_dist_threshold": 0.25,
            "uncertainty_dist_offset": 0.2,
            "uncertainty_dist_scale": 5.0,
            "uncertainty_dist_relevance_scale": 20.0,
            "angle_center": 0.5,
            "angle_sigmoid_scale": 5.0,
            "hand_change_time_threshold": 1.0,
            "force_time_scale": 0.75,
            "force_time_max": 2.25,
            "small_likelihood_value": 1e-8,
        }
    }


def run_trial(env, estimator_params, max_timesteps=None, render=False, sweep_label="", group_name="", param_name="", sweep_value=None, stop_on_done=True, reset_on_start=True):
    if reset_on_start:
        env.reset()

    # support optional connection-level overrides in estimator params dict under key "connection_params"
    connection_params = estimator_params.pop("connection_params", None) if isinstance(estimator_params, dict) else None

    component_building_functions, _, _ = get_building_functions_basic_drawer_motion(
        env, estimator_params=estimator_params, connection_params=connection_params
    )
    components = build_components(component_building_functions)
    gripper_component = components["GripperAction"]
    gripper_velo = components["EEVelocities"]
    kinematic_estimator = components["KinematicJointEstimator"]
    grasp_estimator = components["GraspLikelihoodEstimator"]

    curr_t = 0.0
    stop_joint_error = float("nan")
    stop_grasp_estimate = float("nan")
    stop_actual_grasped = False
    step_idx = 1
    # If max_timesteps is provided, limit the loop; otherwise run until env signals done.
    while True:
        run_component_sequence(components, torch.tensor(curr_t))
        curr_commanded_vel = gripper_velo.quantities["action_velo_ee"]
        curr_commanded_gripper = gripper_component.quantities["gripper_activation"]

        action = np.concatenate(
            [curr_commanded_vel.cpu().numpy(), [2 * curr_commanded_gripper.squeeze().cpu().numpy() - 1]]
        )

        _, rew, done, _ = env.step(action)
        base_env = env.env if hasattr(env, "env") else env
        true_joint_state = float(base_env.sim.data.qpos[base_env.cabinet_qpos_addrs])
        estimated_joint_state = float(kinematic_estimator.quantities["kinematic_joint"][2].item())
        grasp_estimate = float(grasp_estimator.quantities["likelihood_grasped_drawer"].item())
        obs = env._get_observations()
        actual_grasped = bool(np.asarray(obs.get("robot0_contact", base_env._has_gripper_contact)).item())
        stop_joint_error = estimated_joint_state - true_joint_state
        stop_grasp_estimate = grasp_estimate
        stop_actual_grasped = actual_grasped
        # display timestep info; show max if provided
        max_display = str(max_timesteps) if max_timesteps is not None else "inf"
        print(
            f"  [{group_name}.{param_name}={sweep_label}:{sweep_value}] "
            f"step {step_idx:04d}/{max_display}: "
            f"est={estimated_joint_state:.6f}, true={true_joint_state:.6f}, err={stop_joint_error:+.6f}, "
            f"grasp_est={grasp_estimate:.6f}, grasped={actual_grasped}",
            flush=True,
        )

        if (done or rew == 1.0) and stop_on_done:
            return True, step_idx, stop_joint_error, stop_grasp_estimate, stop_actual_grasped

        curr_t += env.control_timestep
        if render:
            env.render()

        # increment and check loop limit
        step_idx += 1
        if (max_timesteps is not None) and (step_idx > max_timesteps):
            return False, max_timesteps, stop_joint_error, stop_grasp_estimate, stop_actual_grasped


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


def main(env, max_timesteps_per_trial=500, render=False, rec_save_path=None):
    env.reset()

    # Print debug information about the drawer
    env.env.print_debug_info()

    # Get the observation using the original method
    env_obs = env._get_observations()

    # Access cabinet information in the same way as the original environment
    cabinet_position = env_obs["CabinetObject_pos"]
    cabinet_orientation = quat2mat(env_obs["CabinetObject_quat"])

    print(f"\nCabinet position from obs: {cabinet_position}")
    print(f"Cabinet orientation matrix from obs:\n{cabinet_orientation}")

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

    # sweep_jobs = sweep_jobs_est + sweep_jobs_conn
    sweep_jobs =  sweep_jobs_conn
    if RUN_ONLY_SINGLE_PARAMETER:
        sweep_jobs = filter_single_parameter_sweeps(sweep_jobs, SINGLE_SWEEP_GROUP, SINGLE_SWEEP_PARAM)
        if not sweep_jobs:
            raise ValueError(f"No sweep jobs matched {SINGLE_SWEEP_GROUP}.{SINGLE_SWEEP_PARAM}")
    results = []

    print(f"\nStarting sweep: {len(sweep_jobs)} trials, timeout={max_timesteps_per_trial} timesteps per trial")

    for trial_index, job in enumerate(sweep_jobs, start=1):
        if DEMO_SEED is not None and RESEED_EACH_TRIAL:
            set_global_seed(DEMO_SEED)
        group_name = job["group_name"]
        param_name = job["param_name"]
        sweep_label = job["sweep_label"]
        sweep_value = job["sweep_value"]
        print(
            f"Currently sweeping [{trial_index:03d}/{len(sweep_jobs):03d}]: "
            f"{group_name}.{param_name} -> {sweep_label} ({sweep_value})",
            flush=True,
        )

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
                render=render,
                sweep_label=sweep_label,
                group_name=group_name,
                param_name=param_name,
                sweep_value=sweep_value,
            )
        except Exception as exc:
            error_text = str(exc)
            success = False
            timesteps = max_timesteps_per_trial
            stop_joint_error = float("nan")
            stop_grasp_estimate = float("nan")
            stop_actual_grasped = False

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

    if rec_save_path is None:
        rec_save_path = str(Path(__file__).with_name("sweep_results.csv"))
    save_sweep_results(results, rec_save_path)

    total_success = sum(r["success"] for r in results)
    print(f"\nSweep complete. Successes: {total_success}/{len(results)}")
    print(f"Results saved to: {rec_save_path}")

def run_demo(render=DEFAULT_RENDER_SWEEP, max_timesteps_per_trial=500):
    set_global_seed(DEMO_SEED)
    # Define the desired initial joint positions for the Panda robot (7 joints)
    initial_panda_qpos = np.array([-0.2, 0.2, 0.1, -2.0, 0.0, 1.5, 0.7])
    # initial_panda_qpos = None

    # Pass the initial pose to the setup function
    env = setup_env(initial_qpos=initial_panda_qpos, render=render)
    main(env, max_timesteps_per_trial=max_timesteps_per_trial, render=render)


def run_default_loop(env, base_params, max_timesteps_per_trial=500, render=False, delay=DEFAULT_LOOP_DELAY):
    """Run trials repeatedly using the default estimator/connection params until interrupted."""
    trial = 0
    try:
        while True:
            if DEMO_SEED is not None and RESEED_EACH_TRIAL:
                set_global_seed(DEMO_SEED)
            trial += 1
            print(f"Default loop trial {trial}")
            try:
                success, timesteps, stop_joint_error, stop_grasp_estimate, stop_actual_grasped = run_trial(
                    env,
                    estimator_params=copy.deepcopy(base_params),
                    max_timesteps=None,
                    render=render,
                    sweep_label="default",
                    group_name="default",
                    param_name="default",
                    sweep_value=None,
                    stop_on_done=False,
                    reset_on_start=False,
                )
                print(f"Trial {trial}: success={success}, timesteps={timesteps}, err={stop_joint_error}")
            except Exception as exc:
                print(f"Trial {trial} failed with error: {exc}")

            # small pause between trials
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Default loop interrupted by user. Exiting.")


if __name__ == "__main__":
    set_global_seed(DEMO_SEED)
    # initial joint positions for Panda
    initial_panda_qpos = np.array([-0.2, 0.2, 0.1, -2.0, 0.0, 1.5, 0.7])
    env = setup_env(initial_qpos=initial_panda_qpos, render=DEFAULT_RENDER_SWEEP)

    if DEMO_MODE == "default_loop":
        base_params = get_default_estimator_params()
        run_default_loop(env, base_params, max_timesteps_per_trial=500, render=DEFAULT_RENDER_SWEEP)
    else:
        # default: run sweep
        main(env, max_timesteps_per_trial=500, render=DEFAULT_RENDER_SWEEP)
