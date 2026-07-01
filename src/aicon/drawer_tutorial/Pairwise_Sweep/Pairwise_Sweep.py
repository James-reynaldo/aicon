import sys
import copy
import itertools
import os
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.python_sequential import build_components, run_component_sequence
from aicon.drawer_tutorial.Experiment_store import ExperimentStore

from aicon.drawer_tutorial.Sweep import (
    setup_env,
    run_trial,
    set_global_seed,
    get_default_estimator_params as sweep_get_default_estimator_params,
    get_default_connection_params as sweep_get_default_connection_params,
)

RENDER = False
NUM_TRIALS_PER_JOB = 3
BOUNDARY_EXTENDED_TRIALS = 7
RANDOM_INIT_TIME = 0.5  # seconds of random movement at start
MAX_TIMESTEPS = 1000

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR_DISTURBANCE = DATA_DIR / "disturbance"
DATA_DIR_NOISE = DATA_DIR / "noise"
DATA_DIR_NORMAL = DATA_DIR / "normal"

# ============================================================================
# CONFIGURATION: Choose which parameter groups to include in pairwise sweep
# ============================================================================
# 
# Set SWEEP_ESTIMATOR_GROUPS and SWEEP_CONNECTION_GROUPS to control which
# parameters are included in the pairwise sweep. Non-swept parameters will
# still be included in trial configs with their standard values.
#
# AVAILABLE ESTIMATOR GROUPS:
#   - "ee_pose": end-effector pose estimation parameters (3 params)
#   - "visible": visibility estimation parameters (2 params)
#   - "grasp_likelihood": grasp likelihood parameters (11 params)
#   - "drawer_position": drawer position estimation parameters (12 params)
#   - "kinematic_joint": kinematic joint estimation parameters (9 params)
#
# AVAILABLE CONNECTION GROUPS:
#   - "DistGraspHandConnection": connection/grasping parameters (12 params)
#
# EXAMPLES:
#   # Sweep only drawer_position parameters in pairwise combinations:
#   SWEEP_ESTIMATOR_GROUPS = ["drawer_position"]
#   SWEEP_CONNECTION_GROUPS = []
#
#   # Sweep both drawer_position and grasp_likelihood:
#   SWEEP_ESTIMATOR_GROUPS = ["drawer_position", "grasp_likelihood"]
#   SWEEP_CONNECTION_GROUPS = []
#
#   # Include all available groups:
#   SWEEP_ESTIMATOR_GROUPS = ["ee_pose", "visible", "grasp_likelihood", "drawer_position", "kinematic_joint"]
#   SWEEP_CONNECTION_GROUPS = ["DistGraspHandConnection"]
#
#   # Or choose individual parameters within a group:
#   SWEEP_ESTIMATOR_GROUPS = {"drawer_position": ["meas_noise_factor", "R_add_scale"]}
#   SWEEP_CONNECTION_GROUPS = {"DistGraspHandConnection": ["ft_noise_offset", "dist_decay"]}
#
# ============================================================================

# SWEEP_ESTIMATOR_GROUPS = {"drawer_position": ["meas_noise_factor", "R_add_scale"]}
SWEEP_ESTIMATOR_GROUPS = {"kinematic_joint": [
    "initial_azimuth_default",
    "initial_elevation_default",
    "grasped_noise",
    "ungrasped_noise",
    "axis_azimuth_process_noise",
    "axis_elevation_process_noise",
    "joint_process_noise",
    "anchor_process_noise", # Does not seem to matter
    "grasp_threshold", # Does not matter much, just not zero
    "grasp_floor", # When value too high, estimation error quite big
    "outlier_rejection_treshold",
    "shift_clip_min",
]}
SWEEP_CONNECTION_GROUPS = {}  # Include connection parameters
# SWEEP_CONNECTION_GROUPS = {"DistGraspHandConnection": ["ft_noise_offset"]}  # Include connection parameters

def get_default_estimator_params():
    """Import complete estimator params from Sweep.py to ensure all parameters are included."""
    return sweep_get_default_estimator_params()


def get_default_connection_params():
    """Import complete connection params from Sweep.py to ensure all parameters are included."""
    return sweep_get_default_connection_params()


def get_all_jobs():
    base_params = get_default_estimator_params()
    base_conn_params = get_default_connection_params()

    sweep_specs = []

    def _collect_specs(params, prefix=None, connection=False, group_filter=None):
        """Collect sweep specs from params, optionally filtering by group names or parameter names."""
        for group_name in sorted(params.keys()):
            if group_filter is None:
                allowed_params = None
            elif isinstance(group_filter, dict):
                allowed_params = group_filter.get(group_name)
                if allowed_params is None:
                    continue
            elif group_name not in group_filter:
                continue
            for param_name in sorted(params[group_name].keys()):
                if allowed_params is not None and param_name not in allowed_params:
                    continue
                standard_value = params[group_name][param_name]
                sweep_values = list(get_sweep_values(standard_value, param_name=param_name))
                sweep_specs.append(
                    {
                        "group_name": group_name,
                        "param_name": param_name,
                        "standard_value": standard_value,
                        "sweep_values": sweep_values,
                        "connection": connection,
                    }
                )

    # Collect estimator parameter specs (filtered by SWEEP_ESTIMATOR_GROUPS)
    _collect_specs(base_params, connection=False, group_filter=SWEEP_ESTIMATOR_GROUPS)
    
    # Collect connection parameter specs (filtered by SWEEP_CONNECTION_GROUPS)
    _collect_specs(base_conn_params, connection=True, group_filter=SWEEP_CONNECTION_GROUPS)

    if not sweep_specs:
        return []

    jobs = []
    if len(sweep_specs) == 1:
        spec_combinations = [tuple(sweep_specs)]
    else:
        spec_combinations = list(itertools.combinations(sweep_specs, 2))

    for specs in spec_combinations:
        for combination in itertools.product(*(spec["sweep_values"] for spec in specs)):
            sweep_labels = tuple(sweep_label for sweep_label, _ in combination)

            # Skip the pairwise combination where every swept parameter is at its standard value
            if all(sweep_label == "standard" for sweep_label in sweep_labels):
                continue

            if sweep_labels in {
                ("x0.5", "standard"),
                ("x2", "standard"),
                ("standard", "x0.5"),
                ("standard", "x2"),
            }:
                continue

            trial_params = copy.deepcopy(base_params)
            connection_params = copy.deepcopy(base_conn_params)
            job_groups = []
            job_names = []
            for spec, (sweep_label, sweep_value) in zip(specs, combination):
                group_name = spec["group_name"]
                param_name = spec["param_name"]
                if spec["connection"]:
                    connection_params[group_name][param_name] = sweep_value
                    job_groups.append(f"connection.{group_name}")
                else:
                    trial_params[group_name][param_name] = sweep_value
                    job_groups.append(group_name)
                job_names.append(f"{group_name}.{param_name}={sweep_label}")

            trial_params["connection_params"] = connection_params

            jobs.append(
                {
                    "group_name": ",".join(job_groups),
                    "param_name": ",".join(spec["param_name"] for spec in specs),
                    "standard_value": None,
                    "sweep_label": ",".join(sweep_labels),
                    "sweep_value": None,
                    "trial_params": trial_params,
                    "job_name": "|".join(job_names),
                }
            )

    return jobs


def get_sweep_values(standard_value, param_name: str = None):
    """Return sweep values scaled by fixed multipliers.

    This always returns 0.2x, 0.5x, 1x, 2x, and 5x of the standard value.
    """
    return [
        ("x0.2", 0.2 * standard_value),
        ("x0.5", 0.5 * standard_value),
        ("standard", standard_value),
        ("x2", 2.0 * standard_value),
        ("x5", 5.0 * standard_value),
    ]


def make_compact_id(job: dict) -> str:
    """Return a compact deterministic id for a job: timestamp + short hash."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name = job.get("job_name") or f"{job.get('group_name','')}.{job.get('param_name','')}"
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{ts}-{h}"


def main(job_index:int, disturbance:float=None, noise_scale:float=None):
    jobs = get_all_jobs()
    print("total jobs:", len(jobs))
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Invalid job index {job_index}")

    job = jobs[job_index]

    print(
        f"[HPC] Running job {job_index}: "
        f"{job.get('job_name', job.get('group_name', ''))} "
        f"({job.get('sweep_label', '')})"
    )

    initial_panda_qpos = np.array([-0.56, 0.76, 0.1, -1.90, 1.11, 1.5, -0.32])

    env = setup_env(render=RENDER, initial_qpos=initial_panda_qpos)

    print("trial params:", job["trial_params"])

    results = []

    if disturbance != 0:
        directory = DATA_DIR_DISTURBANCE
    elif noise_scale != 0:
        directory = DATA_DIR_NOISE
    else:
        directory = DATA_DIR_NORMAL
    directory.mkdir(exist_ok=True, parents=True)
    db_path = directory / "experiment_store.db"
    store = ExperimentStore(str(db_path))

    job_metadata = {
        "experiment_type": "pairwise_sweep",
        "parameters": f"{job['group_name']}.{job['param_name']}",
        "sweep_labels": job["sweep_label"],
        "sweep_values": job["sweep_value"],
    }

    job_metadata.update(job.get("metadata", {}))

    try:
        for run in range(NUM_TRIALS_PER_JOB):
            set_global_seed(run)
            success, timesteps, err, grasp, grasped = run_trial(
                env,
                estimator_params=job["trial_params"],
                max_timesteps=MAX_TIMESTEPS,
                render=RENDER,
                sweep_label=job["sweep_label"],
                group_name=job["group_name"],
                param_name=job["param_name"],
                sweep_value=job["sweep_value"],
                random_init_time=RANDOM_INIT_TIME,
                random_std=1,
                disturbance=disturbance,
                noise_scale=noise_scale
            )

            store.add_trial(
                params=job["trial_params"],
                success=success,
                seed=run,
                timesteps=timesteps,
                error=err,
                metadata=job_metadata,
            )

            results.append((
                job["group_name"],
                job["param_name"],
                job["sweep_label"],
                job["sweep_value"],
                success,
                timesteps,
                err,
            ))

            print(f"run {run}: success={success}, steps={timesteps}, err={err}")

        initial_successes = sum(1 for result in results if result[4])
        if initial_successes in {1, 2}:
            print(
                f"Boundary success rate detected ({initial_successes}/{NUM_TRIALS_PER_JOB}), "
                f"running {BOUNDARY_EXTENDED_TRIALS} more trials to total {NUM_TRIALS_PER_JOB + BOUNDARY_EXTENDED_TRIALS} runs."
            )
            for run in range(NUM_TRIALS_PER_JOB, NUM_TRIALS_PER_JOB + BOUNDARY_EXTENDED_TRIALS):
                set_global_seed(run)
                success, timesteps, err, grasp, grasped = run_trial(
                    env,
                    estimator_params=job["trial_params"],
                    max_timesteps=MAX_TIMESTEPS,
                    render=RENDER,
                    sweep_label=job["sweep_label"],
                    group_name=job["group_name"],
                    param_name=job["param_name"],
                    sweep_value=job["sweep_value"],
                    random_init_time=RANDOM_INIT_TIME,
                    random_std=1,
                    disturbance=disturbance,
                    noise_scale=noise_scale
                )

                store.add_trial(
                    params=job["trial_params"],
                    success=success,
                    seed=run,
                    timesteps=timesteps,
                    error=err,
                    metadata=job_metadata,
                )

                results.append((
                    job["group_name"],
                    job["param_name"],
                    job["sweep_label"],
                    job["sweep_value"],
                    success,
                    timesteps,
                    err,
                ))
                print(f"run {run}: success={success}, steps={timesteps}, err={err}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        store.close()

    print(f"Experiment database saved to: {db_path}")
    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count-jobs":
        print(len(get_all_jobs()))
        sys.exit(0)

    idx = int(sys.argv[1])
    disturbance = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    noise_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(idx, disturbance=disturbance, noise_scale=noise_scale)
