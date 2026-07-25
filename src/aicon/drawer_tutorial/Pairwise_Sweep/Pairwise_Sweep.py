import sys
import copy
import itertools
import os
import hashlib
import sqlite3
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np

from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.python_sequential import build_components, run_component_sequence
from aicon.drawer_tutorial.Experiment_store import ExperimentStore, _hash_config

import time


Visible_Initial_qpos_list = [
np.array([-0.60657486,  0.55208371,  0.01406207, -1.88856343,  1.09944793,  1.29935419,  -0.16385514]), #visible
np.array([-0.60059587,  0.41057492,  0.01894018, -2.11816216,  1.080803,   1.35138319,  -0.24399737]), #visible
np.array([-0.60706375,  0.34607402,  0.09003703, -2.08550184,  1.2520165,   1.49328103,  -0.08097987]), #visible problem too close
np.array([-0.6024305,   0.56758879,  0.02335951, -1.86417613,  1.13117263,  1.33317896,  -0.14172676]), #visible
np.array([-0.62211521,  0.30425019,  -0.05303771, -2.14997687,  1.06525231,  1.27549496,  -0.19972245]), # invisible
]

Invisible_Initial_qpos_list = [
np.array([-0.48750612,  1.17415488,  0.19242183, -1.14334887,  1.13314345,  1.47119768,  0.04283408]),
np.array([-0.55455954,  0.57271839,  0.12380571, -2.1670548,   1.14339199,  1.55548018,  -0.3880041 ]), # invisible
np.array([-0.61155419,  0.60486737,  -0.03850908, -1.99964344,  1.0136531,   1.33582555,  -0.3462424 ]), # invisible at first
np.array([-0.63386333,  0.32862117,  -0.09482711, -2.24104107,  0.99977055,  1.30391341,  -0.32659168]), # invisible
np.array([-0.58865829,  0.70879424,  0.0393395,  -1.81579848,  1.08319597,  1.37122098,  -0.23145539]), #visible
]

from aicon.drawer_tutorial.Sweep import (
    setup_env,
    run_trial,
    set_global_seed,
    get_default_estimator_params as sweep_get_default_estimator_params,
    get_default_connection_params as sweep_get_default_connection_params,
)

NUM_TRIALS_PER_JOB = 10
BOUNDARY_EXTENDED_TRIALS = 7
RANDOM_INIT_TIME = 0  # seconds of random movement at start
MAX_TIMESTEPS = 1000

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR_DISTURBANCE = DATA_DIR / "disturbance"
DATA_DIR_NOISE = DATA_DIR / "noise"
DATA_DIR_NORMAL = DATA_DIR / "normal"
DB_SHARD_PREFIX = "experiment_store_job_"

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
SWEEP_ESTIMATOR_GROUPS = {
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
SWEEP_CONNECTION_GROUPS = {"DistGraspHandConnection": [
        "dist_decay",
        "close_dist_threshold",
        "ft_noise_offset",
        "low_likelihood_threshold",
        "gripper_activation_threshold",
        "uncertainty_dist_threshold",
        "small_likelihood_value",

        "uncertainty_bias",
        "uncertainty_scale_uncertainty",
        "uncertainty_scale_relevance",
        "dist_sigmoid_scale",
        "time_since_hand_change_threshold",
        "ft_tresh_multiplier",
        "ft_tresh_cap",
        ],}  # Include connection parameters
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
            
            # If job param_name is close_distance_threshold,joint_process_noise, print job number
            # if job_names == ["DistGraspHandConnection.close_dist_threshold=x2", "kinematic_joint.joint_process_noise=x0.2"]:
            #     print(f"Job index for {job_names}: {len(jobs)-1}")

    return jobs


def get_sweep_values(standard_value, param_name: str = None):
    """Return sweep values scaled by fixed multipliers.

    This always returns 0.2x, 0.5x, 1x, 2x, and 5x of the standard value.
    """
    return [
        ("x0.2", 0.2 * standard_value),
        ("x0.5", 0.5 * standard_value),
        ("x2", 2.0 * standard_value),
        ("x5", 5.0 * standard_value),
    ]


def make_compact_id(job: dict) -> str:
    """Return a compact deterministic id for a job: timestamp + short hash."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name = job.get("job_name") or f"{job.get('group_name','')}.{job.get('param_name','')}"
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{ts}-{h}"


def get_data_directory(disturbance: float = None, noise_scale: float = None) -> Path:
    if disturbance != 0:
        return DATA_DIR_DISTURBANCE
    if noise_scale != 0:
        return DATA_DIR_NOISE
    return DATA_DIR_NORMAL


def get_job_db_path(directory: Path, job_index: int, job: dict) -> Path:
    """Return a deterministic per-job database path to avoid write contention."""
    job_key = f"{job_index}:{job.get('job_name', '')}:{job.get('group_name', '')}:{job.get('param_name', '')}:{job.get('sweep_label', '')}"
    short_hash = hashlib.sha1(job_key.encode("utf-8")).hexdigest()[:10]
    return directory / f"{DB_SHARD_PREFIX}{job_index:05d}_{short_hash}.db"


def count_existing_trials(db_path: Path, params: dict) -> int:
    """Return how many trials already exist for this exact parameter config."""
    if not db_path.exists():
        return 0

    config_id = _hash_config(params)
    db_uri = f"file:{db_path}?mode=ro"

    with sqlite3.connect(db_uri, uri=True, timeout=60) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'trials'
            """
        )
        if cur.fetchone() is None:
            return 0

        cur.execute(
            """
            SELECT COUNT(*)
            FROM trials
            WHERE config_id = ?
            """,
            (config_id,),
        )
        return int(cur.fetchone()[0])


def merge_sharded_databases(source_dir: Path, target_db: Path) -> None:
    """Merge per-job shard databases into a single analysis database."""
    shard_paths = sorted(source_dir.glob(f"{DB_SHARD_PREFIX}*.db"))
    if not shard_paths:
        print(f"[HPC] No shard databases found in {source_dir}")
        return

    target_db.parent.mkdir(parents=True, exist_ok=True)
    target_store = ExperimentStore(str(target_db))

    try:
        for shard_path in shard_paths:
            if shard_path.resolve() == target_db.resolve():
                continue

            print(f"[HPC] Merging {shard_path.name} into {target_db.name}")
            try:
                target_store.conn.execute("ATTACH DATABASE ? AS srcdb", (str(shard_path),))
                target_store.conn.execute("BEGIN")
                target_store.conn.execute(
                    "INSERT OR IGNORE INTO configs (config_id, params) SELECT config_id, params FROM srcdb.configs"
                )
                target_store.conn.execute(
                    """
                    INSERT INTO trials (config_id, seed, success, timesteps, error, metadata)
                    SELECT config_id, seed, success, timesteps, error, metadata
                    FROM srcdb.trials
                    """
                )
                target_store.conn.commit()
            finally:
                try:
                    target_store.conn.execute("DETACH DATABASE srcdb")
                except Exception:
                    pass

        print(f"[HPC] Merge complete: {target_db}")
    finally:
        target_store.close()


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

    print("trial params:", job["trial_params"])

    directory = get_data_directory(disturbance=disturbance, noise_scale=noise_scale)
    directory.mkdir(exist_ok=True, parents=True)
    db_path = get_job_db_path(directory, job_index, job)

    start_timer = time.time()
    existing_trials = count_existing_trials(db_path, job["trial_params"])
    if existing_trials > 0:
        print(
            f"[HPC] Skipping job {job_index}: config already has "
            f"{existing_trials} trial(s) in {db_path}"
        )
        end_timer = time.time()
        print(f"[HPC] Counted existing trials in {end_timer - start_timer:.2f} seconds.")
        return

    initial_panda_qpos = np.array([-0.56, 0.76, 0.1, -1.90, 1.11, 1.5, -0.32])

    env = setup_env(initial_qpos=initial_panda_qpos)

    results = []

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
            if run < len(Visible_Initial_qpos_list):
                initial_panda_qpos = Visible_Initial_qpos_list[run]
            else:
                initial_panda_qpos = Invisible_Initial_qpos_list[run - len(Visible_Initial_qpos_list)]
            env = setup_env(initial_qpos=initial_panda_qpos)
            set_global_seed(run)
            success, timesteps, err, true_joint, grasp, kinematic_axis_error, anchor_error = run_trial(
                env,
                estimator_params=job["trial_params"],
                max_timesteps=MAX_TIMESTEPS,
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
                true_joint=true_joint,
                grasp=grasp,
                kinematic_axis_error=kinematic_axis_error,
                anchor_error=anchor_error,
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
                true_joint,
                grasp,
                kinematic_axis_error,
                anchor_error
            ))

            print(f"run {run}: success={success}, steps={timesteps}, err={err}")

        # initial_successes = sum(1 for result in results if result[4])
        # if initial_successes in {1, 2}:
        #     print(
        #         f"Boundary success rate detected ({initial_successes}/{NUM_TRIALS_PER_JOB}), "
        #         f"running {BOUNDARY_EXTENDED_TRIALS} more trials to total {NUM_TRIALS_PER_JOB + BOUNDARY_EXTENDED_TRIALS} runs."
        #     )
        #     for run in range(NUM_TRIALS_PER_JOB, NUM_TRIALS_PER_JOB + BOUNDARY_EXTENDED_TRIALS):
        #         set_global_seed(run)
        #         success, timesteps, err, true_joint, grasp, kinematic_axis_error, anchor_error = run_trial(
        #             env,
        #             estimator_params=job["trial_params"],
        #             max_timesteps=MAX_TIMESTEPS,
        #             sweep_label=job["sweep_label"],
        #             group_name=job["group_name"],
        #             param_name=job["param_name"],
        #             sweep_value=job["sweep_value"],
        #             random_init_time=RANDOM_INIT_TIME,
        #             random_std=1,
        #             disturbance=disturbance,
        #             noise_scale=noise_scale
        #         )

        #         store.add_trial(
        #             params=job["trial_params"],
        #             success=success,
        #             seed=run,
        #             timesteps=timesteps,
        #             error=err,
        #             true_joint=true_joint,
        #             grasp=grasp,
        #             kinematic_axis_error=kinematic_axis_error,
        #             anchor_error=anchor_error,
        #             metadata=job_metadata,
        #         )

        #         results.append((
        #             job["group_name"],
        #             job["param_name"],
        #             job["sweep_label"],
        #             job["sweep_value"],
        #             success,
        #             timesteps,
        #             err,
        #             true_joint,
        #             grasp,
        #             kinematic_axis_error,
        #             anchor_error
        #         ))
        #         print(f"run {run}: success={success}, steps={timesteps}, err={err}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        store.close()

    print(f"Experiment shard database saved to: {db_path}")


def _normalize_csv(value: str | None) -> str | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return ",".join(parts)


def find_jobs(
    parameters: str | None = None,
    sweep_labels: str | None = None,
    job_name_contains: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return matching jobs and print compact results with job indices."""
    parameters = _normalize_csv(parameters)
    sweep_labels = _normalize_csv(sweep_labels)

    jobs = get_all_jobs()
    matches = []

    for idx, job in enumerate(jobs):
        job_parameters = f"{job['group_name']}.{job['param_name']}"
        job_sweep_labels = _normalize_csv(job["sweep_label"])
        job_name = job.get("job_name", "")

        if parameters is not None and job_parameters != parameters:
            continue
        if sweep_labels is not None and job_sweep_labels != sweep_labels:
            continue
        if job_name_contains is not None and job_name_contains not in job_name:
            continue

        matches.append(
            {
                "job_index": idx,
                "job_name": job_name,
                "parameters": job_parameters,
                "sweep_labels": job["sweep_label"],
            }
        )

    print(f"total jobs: {len(jobs)}")
    print(
        "find filters:",
        {
            "experiment_type": "pairwise_sweep",
            "parameters": parameters,
            "sweep_labels": sweep_labels,
            "job_name_contains": job_name_contains,
        },
    )
    print(f"matches: {len(matches)}")

    for match in matches[: max(limit, 0)]:
        print(
            f"job_index={match['job_index']} | "
            f"parameters={match['parameters']} | "
            f"sweep_labels={match['sweep_labels']} | "
            f"job_name={match['job_name']}"
        )

    if len(matches) > max(limit, 0):
        remaining = len(matches) - max(limit, 0)
        print(f"... and {remaining} more. Increase --limit to show all.")

    return matches


def find_jobs_by_config_id(config_id: str, limit: int = 50) -> list[dict]:
    """Return matching jobs for a given config_id."""
    config_id = config_id.strip().lower()
    jobs = get_all_jobs()
    matches = []

    for idx, job in enumerate(jobs):
        job_config_id = _hash_config(job["trial_params"])
        if job_config_id != config_id:
            continue

        matches.append(
            {
                "job_index": idx,
                "job_name": job.get("job_name", ""),
                "parameters": f"{job['group_name']}.{job['param_name']}",
                "sweep_labels": job["sweep_label"],
                "config_id": job_config_id,
            }
        )

    print(f"total jobs: {len(jobs)}")
    print(f"find config_id: {config_id}")
    print(f"matches: {len(matches)}")

    for match in matches[: max(limit, 0)]:
        print(
            f"job_index={match['job_index']} | "
            f"config_id={match['config_id']} | "
            f"parameters={match['parameters']} | "
            f"sweep_labels={match['sweep_labels']} | "
            f"job_name={match['job_name']}"
        )

    if len(matches) > max(limit, 0):
        remaining = len(matches) - max(limit, 0)
        print(f"... and {remaining} more. Increase --limit to show all.")

    return matches


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or query pairwise sweep jobs.")
    parser.add_argument("job_index", nargs="?", type=int, help="Job index to run")
    parser.add_argument("disturbance", nargs="?", type=float, default=0)
    parser.add_argument("noise_scale", nargs="?", type=float, default=0)

    parser.add_argument("--count-jobs", action="store_true", help="Print number of jobs")
    parser.add_argument("--find-jobs", action="store_true", help="Find job indices by filters")
    parser.add_argument(
        "--find-json",
        type=str,
        help=(
            "JSON filter, e.g. "
            "'{\"experiment_type\":\"pairwise_sweep\",\"parameters\":\"kinematic_joint,kinematic_joint.initial_elevation_default,ungrasped_noise\",\"sweep_labels\":\"x0.2,x2\"}'"
        ),
    )
    parser.add_argument("--find-config-id", type=str, help="Find job indices by config_id")
    parser.add_argument("--parameters", type=str, help="Exact metadata parameters filter")
    parser.add_argument("--sweep-labels", type=str, help="Exact metadata sweep_labels filter")
    parser.add_argument("--job-name-contains", type=str, help="Substring filter on job_name")
    parser.add_argument("--limit", type=int, default=50, help="Max printed matches")
    parser.add_argument("--merge-shards", action="store_true", help="Merge shard databases into a single experiment_store.db")
    parser.add_argument("--merge-target", type=Path, help="Target database path for --merge-shards")

    return parser
    

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.count_jobs:
        print(len(get_all_jobs()))
        sys.exit(0)

    if args.merge_shards:
        directory = get_data_directory(disturbance=args.disturbance, noise_scale=args.noise_scale)
        target_db = args.merge_target or (directory / "experiment_store.db")
        merge_sharded_databases(directory, target_db)
        sys.exit(0)

    if args.find_jobs or args.find_json is not None or args.find_config_id is not None:
        find_parameters = args.parameters
        find_sweep_labels = args.sweep_labels
        find_job_name_contains = args.job_name_contains

        if args.find_json is not None:
            try:
                query = json.loads(args.find_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid --find-json payload: {exc}") from exc

            experiment_type = query.get("experiment_type")
            if experiment_type is not None and experiment_type != "pairwise_sweep":
                print(
                    "No matches: this script only handles experiment_type='pairwise_sweep'."
                )
                sys.exit(0)

            find_parameters = query.get("parameters", find_parameters)
            find_sweep_labels = query.get("sweep_labels", find_sweep_labels)

        if args.find_config_id is not None:
            find_jobs_by_config_id(args.find_config_id, limit=args.limit)
            sys.exit(0)

        find_jobs(
            parameters=find_parameters,
            sweep_labels=find_sweep_labels,
            job_name_contains=find_job_name_contains,
            limit=args.limit,
        )
        sys.exit(0)

    if args.job_index is None:
        parser.error("job_index is required unless --count-jobs or --find-jobs/--find-json is used")

    main(args.job_index, disturbance=args.disturbance, noise_scale=args.noise_scale)
