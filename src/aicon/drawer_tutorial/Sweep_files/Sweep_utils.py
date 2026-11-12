import time
import csv
import copy
import json
import random
import inspect
import re
from pathlib import Path

import lovely_tensors as lt
import numpy as np
import torch
from robosuite.utils.transform_utils import quat2mat

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.drawer_tutorial.connections import DistGraspHandConnection,VisibleEEDrawerConnection
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

# The JSON sweep plan is the source of truth for which parameters are swept.
# We use the constructor defaults only to fill in the baseline/default values for
# the parameters listed in that JSON file.

_FALLBACK_ESTIMATOR_SWEEP_PARAMS = {
    "ee_pose": (
        "initial_uncertainty_scale", "action_process_noise", "proprio_update_noise",
    ),
    "visible": (
        "initial_likelihood_prior", "update_gain",
    ),
    "grasp_likelihood": (
        "baseline_measurement_likelihood",
        "initial_time_since_hand_change",
        "negative_innovation_threshold",
        "negative_innovation_scale",
    ),
    "drawer_position": (
        "initial_uncertainty_xy",
        "initial_uncertainty_depth",
        "meas_noise_factor",
        "visual_likelihood_steepness",
        "R_add_scale",
        "grasped_update_R_scale",
        "missed_absent_measurement_uncertainty_coeff",
        "forward_noise_base",
        "grasped_outlier_rejection_threshold",
        "measurement_existence_threshold",
        "grasped_uncertainty_threshold",
        "hand_change_recovery_time",
    ),
    "kinematic_joint": (
        "initial_axis_uncertainty_scale",
        "joint_initial_uncertainty_scale",
        "grasped_noise",
        "ungrasped_noise",
        "axis_process_noise",
        "joint_process_noise",
        "anchor_process_noise",
        "grasp_threshold",
        "outlier_rejection_treshold",
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


def _get_group_param_names_from_json(runtime_group_name, sweep_plan=None):
    sweep_plan = load_sweep_plan() if sweep_plan is None else sweep_plan
    if not isinstance(sweep_plan, dict):
        return []

    target = _normalize_group_name(runtime_group_name)
    for candidate in (sweep_plan.get("estimators", {}), sweep_plan.get("connections", {})):
        if not isinstance(candidate, dict):
            continue
        for group_key, params in candidate.items():
            if _normalize_group_name(group_key) == target and isinstance(params, dict):
                return list(params.keys())
    return []


def get_default_estimator_params():
    """Get estimator defaults from the constructors used by the experiment builder.

    The JSON sweep plan is the source of truth for the parameter names to include.
    """
    sweep_plan = load_sweep_plan()
    return {
        "ee_pose": _get_selected_constructor_defaults(
            EEPoseEstimator,
            _get_group_param_names_from_json("ee_pose", sweep_plan)
            or _FALLBACK_ESTIMATOR_SWEEP_PARAMS["ee_pose"],
        ),
        "visible": _get_selected_constructor_defaults(
            VisibleEstimator,
            _get_group_param_names_from_json("visible", sweep_plan)
            or _FALLBACK_ESTIMATOR_SWEEP_PARAMS["visible"],
        ),
        "grasp_likelihood": _get_selected_constructor_defaults(
            GraspedEstimator,
            _get_group_param_names_from_json("grasp_likelihood", sweep_plan)
            or _FALLBACK_ESTIMATOR_SWEEP_PARAMS["grasp_likelihood"],
        ),
        "drawer_position": _get_selected_constructor_defaults(
            DrawerPositionEstimator,
            _get_group_param_names_from_json("drawer_position", sweep_plan)
            or _FALLBACK_ESTIMATOR_SWEEP_PARAMS["drawer_position"],
        ),
        "kinematic_joint": _get_selected_constructor_defaults(
            KinematicJointEstimator,
            _get_group_param_names_from_json("kinematic_joint", sweep_plan)
            or _FALLBACK_ESTIMATOR_SWEEP_PARAMS["kinematic_joint"],
        ),
    }


def get_default_connection_params():
    """Get connection defaults from the constructors used by the experiment builder.

    Parameters are discovered from the JSON sweep plan when available.
    """
    sweep_plan = load_sweep_plan()
    return {
        "DistGraspHandConnection": _get_selected_constructor_defaults(
            DistGraspHandConnection,
            _get_group_param_names_from_json("DistGraspHandConnection", sweep_plan)
            or list(_get_constructor_defaults(DistGraspHandConnection).keys()),
        ),
        "VisibleEEDrawerConnection": _get_selected_constructor_defaults(
            VisibleEEDrawerConnection,
            _get_group_param_names_from_json("VisibleEEDrawerConnection", sweep_plan)
            or list(_get_constructor_defaults(VisibleEEDrawerConnection).keys()),
        ),
    }

DEFAULT_SWEEP_PLAN_PATH = Path(__file__).resolve().parent / "sweep_plan_actual_values.json"


def _normalize_group_name(name):
    normalized = re.sub(r'(?<!^)(?=[A-Z])', '_', str(name).strip())
    normalized = normalized.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    aliases = {
        "ee_pose_estimator": "ee_pose",
        "e_e_pose_estimator": "ee_pose",
        "eeposeestimator": "ee_pose",
        "drawer_position_estimator": "drawer_position",
        "drawer_position": "drawer_position",
        "grasped_estimator": "grasp_likelihood",
        "graspedestimator": "grasp_likelihood",
        "visible_estimator": "visible",
        "visibleestimator": "visible",
        "kinematic_joint_estimator": "kinematic_joint",
        "kinematic_joint": "kinematic_joint",
        "distgrasphandconnection": "distgrasp_hand_connection",
        "visibleeedrawerconnection": "visible_ee_drawer_connection",
    }
    return aliases.get(normalized, normalized)


def load_sweep_plan(path=None):
    sweep_path = Path(path) if path is not None else DEFAULT_SWEEP_PLAN_PATH
    if not sweep_path.exists():
        return {}
    with open(sweep_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_sweep_values(standard_value, group_name=None, param_name=None, sweep_plan_path=None):
    """Return actual sweep values from the JSON plan when available.

    The JSON stores the full sweep list with the baseline/default value at index 4. We skip that
    index here and leave the single baseline run to the standard job.
    """
    sweep_plan = load_sweep_plan(sweep_plan_path)
    if not sweep_plan:
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

    if group_name is None or param_name is None:
        return []

    norm_group = _normalize_group_name(group_name)
    plan_group = None
    for candidate in (sweep_plan.get("estimators", {}), sweep_plan.get("connections", {})):
        if not isinstance(candidate, dict):
            continue
        for group_key, params in candidate.items():
            if _normalize_group_name(group_key) == norm_group:
                plan_group = params
                break
        if plan_group is not None:
            break

    if plan_group is None:
        return []

    param_meta = plan_group.get(param_name)
    if not isinstance(param_meta, dict):
        return []

    values = list(param_meta.get("values", []))
    if not values:
        return []

    # Skip the baseline/default entry at index 4; the standard job runs that case once.
    return [
        (f"value_{index}", value)
        for index, value in enumerate(values)
        if index != 4
    ]


def get_standard_job(base_params):
    return {
        "group_name": "standard",
        "param_name": "standard",
        "standard_value": None,
        "sweep_label": "standard",
        "sweep_value": None,
        "trial_params": copy.deepcopy(base_params),
    }


def generate_single_parameter_sweeps(base_params, sweep_plan_path=None):
    for group_name in sorted(base_params.keys()):
        for param_name in sorted(base_params[group_name].keys()):
            print(f"Generating sweep for {group_name}.{param_name}")
            standard_value = base_params[group_name][param_name]
            sweep_values = get_sweep_values(standard_value, group_name=group_name, param_name=param_name, sweep_plan_path=sweep_plan_path)
            if not sweep_values:
                # Fallback to legacy scaling if the JSON plan is unavailable for this parameter.
                sweep_values = [
                    ("negative", -standard_value),
                    ("zero", 0.0),
                    ("x0.001", 0.001 * standard_value),
                    ("x0.2", 0.2 * standard_value),
                    ("x0.5", 0.5 * standard_value),
                    ("x2", 2.0 * standard_value),
                    ("x5", 5.0 * standard_value),
                    ("x1000", 1000.0 * standard_value),
                ]
            for sweep_label, sweep_value in sweep_values:
                # Skip the baseline/default entry exactly once by omitting index 4 in the JSON-derived values.
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
            disturbance=None, noise_scale=1.0,
              prior_noise_std=0.01, prior_noise_std_kinematic=0.0, return_gradient_traces=False,
              sensor_noise_overrides=None):
    """Sweep-compatible adapter around the shared demo trial runner."""
    status_label = f"{group_name}.{param_name}={sweep_label}:{sweep_value}"
    return run_demo_trial(
        env,
        device=None,
        estimator_params=estimator_params,
        max_timesteps=max_timesteps,
        stop_on_done=stop_on_done,
        reset_on_start=reset_on_start,
        random_disturbance=disturbance,
        noise_scale=noise_scale,
        prior_noise_std=prior_noise_std,
        prior_noise_std_kinematic=prior_noise_std_kinematic,
        sensor_noise_overrides=sensor_noise_overrides,
        # Sweep historically creates its diagnostic plots independently of MuJoCo rendering.
        status_label=status_label,
        render=False,
        return_gradient_traces=return_gradient_traces,
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


