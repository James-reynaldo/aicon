"""
Pairwise sweep job generation and execution.

This script generates and runs pairwise parameter sweep jobs. All parameters
are defined in the sweep_plan_actual_values.json file - this script reads from
that JSON and generates all pairwise combinations of parameters defined there.

No hardcoded parameter groups: the JSON is the authoritative source of truth.
"""
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
from aicon.drawer_tutorial.Sweep_files.Experiment_store import ExperimentStore, _hash_config
from aicon.drawer_tutorial.Sweep_files.initial_qpos import initial_qpos_list

import time

from aicon.drawer_tutorial.Sweep_files.Sweep_utils import (
    setup_env,
    run_trial,
    set_global_seed,
    get_default_estimator_params as sweep_get_default_estimator_params,
    get_default_connection_params as sweep_get_default_connection_params,
    get_sweep_values as sweep_get_sweep_values,
    load_sweep_plan,
    _normalize_group_name,
)

NUM_TRIALS_PER_JOB = 10
RANDOM_INIT_TIME = 0  # seconds of random movement at start
MAX_TIMESTEPS = 1000

DATA_DIR = Path(__file__).resolve().parents[1] / "Ignored_files" / "Pairwise" / "data"
DATA_DIR_DISTURBANCE = DATA_DIR / "disturbance"
DATA_DIR_NOISE = DATA_DIR / "noise"
DATA_DIR_NORMAL = DATA_DIR / "normal"
DATA_DIR_BAD_PRIOR = DATA_DIR / "bad_prior"
DB_SHARD_PREFIX = "experiment_store_job_"

def get_default_estimator_params():
    """Import complete estimator params from Sweep.py to ensure all parameters are included."""
    return sweep_get_default_estimator_params()


def get_default_connection_params():
    """Import complete connection params from Sweep.py to ensure all parameters are included."""
    return sweep_get_default_connection_params()


def get_all_jobs():
    """Generate pairwise sweep jobs from JSON-defined parameters.
    
    This function reads the sweep plan from JSON and creates all pairwise
    combinations of parameters defined in that file. It does not depend on
    hardcoded parameter groups.
    """
    base_params = get_default_estimator_params()
    base_conn_params = get_default_connection_params()
    
    sweep_plan = load_sweep_plan()
    if not sweep_plan:
        print("[WARN] No sweep plan found; returning empty jobs list")
        return []

    sweep_specs = []

    # Collect estimator parameter specs from JSON
    for group_name, params_dict in sweep_plan.get("estimators", {}).items():
        for param_name in sorted(params_dict.keys()):
            # Match JSON group names to base_params keys using normalization
            def _find_matching_key(mapping, group_name):
                target = _normalize_group_name(group_name)
                for key in mapping.keys():
                    if _normalize_group_name(key) == target:
                        return key
                return None

            match_key = _find_matching_key(base_params, group_name)
            if match_key is None or param_name not in base_params.get(match_key, {}):
                continue
            standard_value = base_params[match_key][param_name]
            sweep_values = list(get_sweep_values(standard_value, group_name=group_name, param_name=param_name))
            if sweep_values:  # Only include if JSON provides sweep values
                sweep_specs.append(
                    {
                        "group_name": group_name,
                        "mapping_key": match_key,
                        "param_name": param_name,
                        "standard_value": standard_value,
                        "sweep_values": sweep_values,
                        "connection": False,
                    }
                )

    # Collect connection parameter specs from JSON
    for group_name, params_dict in sweep_plan.get("connections", {}).items():
        for param_name in sorted(params_dict.keys()):
            # Match JSON group names to base_conn_params keys using normalization
            def _find_matching_key_conn(mapping, group_name):
                target = _normalize_group_name(group_name)
                for key in mapping.keys():
                    if _normalize_group_name(key) == target:
                        return key
                return None

            match_key = _find_matching_key_conn(base_conn_params, group_name)
            if match_key is None or param_name not in base_conn_params.get(match_key, {}):
                continue
            standard_value = base_conn_params[match_key][param_name]
            sweep_values = list(get_sweep_values(standard_value, group_name=group_name, param_name=param_name))
            if sweep_values:  # Only include if JSON provides sweep values
                sweep_specs.append(
                    {
                        "group_name": group_name,
                        "mapping_key": match_key,
                        "param_name": param_name,
                        "standard_value": standard_value,
                        "sweep_values": sweep_values,
                        "connection": True,
                    }
                )

    if not sweep_specs:
        print("[WARN] No sweep specs found in JSON; returning empty jobs list")
        return []

    jobs = []
    if len(sweep_specs) == 1:
        spec_combinations = [tuple(sweep_specs)]
    else:
        spec_combinations = list(itertools.combinations(sweep_specs, 2))

    for specs in spec_combinations:
        for combo_indices, combination in enumerate(itertools.product(*(
            enumerate(spec["sweep_values"]) for spec in specs
        ))):
            # combination is now ((index0, (label0, value0)), (index1, (label1, value1)))
            indices = tuple(idx for idx, _ in combination)
            sweep_pairs = tuple(pair for _, pair in combination)
            sweep_labels = tuple(sweep_label for sweep_label, _ in sweep_pairs)
            
            # Skip if either chosen JSON index equals 4 (omitted row/column).
            # sweep_pairs contain labels like "value_2", "value_3", ... —
            # parse the numeric suffix and skip if any equals 4.
            json_indices = []
            for lbl, _ in sweep_pairs:
                try:
                    if isinstance(lbl, str) and lbl.startswith("value_"):
                        json_indices.append(int(lbl.split("_", 1)[1]))
                except Exception:
                    pass
            if 4 in json_indices:
                continue

            # Skip the pairwise combination where every swept parameter is at its standard value
            if all(sweep_label == "standard" for sweep_label in sweep_labels):
                continue

            trial_params = copy.deepcopy(base_params)
            connection_params = copy.deepcopy(base_conn_params)
            job_groups = []
            job_names = []
            for spec, (sweep_label, sweep_value) in zip(specs, sweep_pairs):
                group_name = spec["group_name"]
                param_name = spec["param_name"]
                mapping_key = spec.get("mapping_key", group_name)
                if spec["connection"]:
                    # mapping_key identifies the actual key in base_conn_params
                    connection_params[mapping_key][param_name] = sweep_value
                    job_groups.append(f"connection.{mapping_key}")
                else:
                    # mapping_key identifies the actual key in base_params
                    trial_params[mapping_key][param_name] = sweep_value
                    job_groups.append(mapping_key)
                job_names.append(f"{mapping_key}.{param_name}={sweep_label}")

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


def get_sweep_values(standard_value, group_name: str = None, param_name: str = None):
    """Get sweep values from JSON, reusing Sweep_utils logic.
    
    Returns only the nominal subset indices 2..6 (inclusive) from the JSON plan.
    Falls back to the shared logic in `sweep_get_sweep_values` when no JSON
    entry is available.
    """
    # Try to load the JSON plan and pick indices 2..6 if available.
    sweep_plan = load_sweep_plan()
    if not sweep_plan or group_name is None or param_name is None:
        return sweep_get_sweep_values(standard_value, group_name=group_name, param_name=param_name)

    # Find the plan group respecting normalization (estimators or connections)
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
        return sweep_get_sweep_values(standard_value, group_name=group_name, param_name=param_name)

    param_meta = plan_group.get(param_name)
    if not isinstance(param_meta, dict):
        return sweep_get_sweep_values(standard_value, group_name=group_name, param_name=param_name)

    values = list(param_meta.get("values", []))
    if not values:
        return sweep_get_sweep_values(standard_value, group_name=group_name, param_name=param_name)

    desired_indices = [2, 3, 5, 6]
    return [
        (f"value_{index}", values[index])
        for index in desired_indices
        if 0 <= index < len(values)
    ]


def make_compact_id(job: dict) -> str:
    """Return a compact deterministic id for a job: timestamp + short hash."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name = job.get("job_name") or f"{job.get('group_name','')}.{job.get('param_name','')}"
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{ts}-{h}"


def get_data_directory(disturbance: float = None, noise_scale: float = None, prior_noise_std: float = None, prior_noise_std_kinematic: float = None) -> Path:
    if disturbance is not None:
        return DATA_DIR_DISTURBANCE
    if noise_scale is not None:
        return DATA_DIR_NOISE
    if prior_noise_std is not None:
        return DATA_DIR_BAD_PRIOR

    return DATA_DIR_NORMAL


def get_job_db_path(directory: Path, job_index: int, job: dict) -> Path:
    """Return a deterministic per-job database path to avoid write contention."""
    job_key = f"{job_index}:{job.get('job_name', '')}:{job.get('group_name', '')}:{job.get('param_name', '')}:{job.get('sweep_label', '')}"
    short_hash = hashlib.sha1(job_key.encode("utf-8")).hexdigest()[:10]
    return directory / f"{DB_SHARD_PREFIX}{job_index:05d}.db"


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


def main(job_index:int, disturbance:float=None, noise_scale:float=None, prior_noise_std: float = None, prior_noise_std_kinematic: float = None):
    jobs = get_all_jobs()
    print(f"[HPC] Total jobs: {len(jobs)}")
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Invalid job index {job_index}")

    job = jobs[job_index]

    print(
        f"[HPC] Running job {job_index}: "
        f"{job.get('job_name', job.get('group_name', ''))} "
        f"({job.get('sweep_label', '')})"
    )

    print("trial params:", job["trial_params"])

    directory = get_data_directory(disturbance=disturbance, noise_scale=noise_scale, prior_noise_std=prior_noise_std, prior_noise_std_kinematic=prior_noise_std_kinematic)
    directory.mkdir(exist_ok=True, parents=True)
    db_path = get_job_db_path(directory, job_index, job)

    start_timer = time.time()
    # existing_trials = count_existing_trials(db_path, job["trial_params"])
    # if existing_trials > 0:
    #     print(
    #         f"[HPC] Skipping job {job_index}: config already has "
    #         f"{existing_trials} trial(s) in {db_path}"
    #     )
    #     end_timer = time.time()
    #     print(f"[HPC] Counted existing trials in {end_timer - start_timer:.2f} seconds.")
    #     return

    results = []

    store = ExperimentStore(str(db_path))

    job_metadata = {
        "experiment_type": "pairwise_sweep",
        "parameters": f"{job['group_name']}.{job['param_name']}",
        "sweep_labels": job["sweep_label"],
        "sweep_values": job["sweep_value"],
        "job_index": job_index,
    }
    if disturbance is not None:
        job_metadata["disturbance"] = disturbance
    if noise_scale is not None:
        job_metadata["noise_scale"] = noise_scale
    if prior_noise_std is not None:
        job_metadata["prior_noise_std"] = prior_noise_std
    if prior_noise_std_kinematic is not None:
        job_metadata["prior_noise_std_kinematic"] = prior_noise_std_kinematic

    job_metadata.update(job.get("metadata", {}))

    try:
        for run in range(NUM_TRIALS_PER_JOB):
            initial_panda_qpos = initial_qpos_list[run]
            env = setup_env(initial_qpos=initial_panda_qpos)
            set_global_seed(run)
            trial_kwargs = {
                "estimator_params": job["trial_params"],
                "max_timesteps": MAX_TIMESTEPS,
                "sweep_label": job["sweep_label"],
                "group_name": job["group_name"],
                "param_name": job["param_name"],
                "sweep_value": job["sweep_value"],
                "return_gradient_traces": True,
            }
            # Only pass these if they were actually specified
            if disturbance is not None:
                trial_kwargs["disturbance"] = disturbance

            if noise_scale is not None:
                trial_kwargs["noise_scale"] = noise_scale

            if prior_noise_std is not None:
                trial_kwargs["prior_noise_std"] = prior_noise_std

            if prior_noise_std_kinematic is not None:
                trial_kwargs["prior_noise_std_kinematic"] = prior_noise_std_kinematic
            success, timesteps, err, true_joint, grasp, actual_grasped, kinematic_axis_error, anchor_error, gradient_traces = run_trial(
                env,
                **trial_kwargs
            )

            # Register traces into the catalog and collect compact trace_ids
            trace_ids = []
            for trace in gradient_traces:
                if trace is None:
                    continue
                trace_ids.append(store.register_trace(trace))

            store.add_trial(
                params=job["trial_params"],
                success=success,
                seed=run,
                timesteps=timesteps,
                error=err,
                true_joint=true_joint,
                grasp=grasp,
                actual_grasped=actual_grasped,
                kinematic_axis_error=kinematic_axis_error,
                anchor_error=anchor_error,
                trace_ids=trace_ids,
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
                actual_grasped,
                kinematic_axis_error,
                anchor_error
            ))

            print(f"run {run}: success={success}, steps={timesteps}, err={err}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        store.close()

    print(f"Experiment shard database saved to: {db_path}")

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or query pairwise sweep jobs.")
    parser.add_argument(
        "job_index",
        type=int,
    )

    parser.add_argument(
        "--disturbance",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--prior-noise-std",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--prior-noise-std-kinematic",
        type=float,
        default=None,
    )

    parser.add_argument("--count-jobs", action="store_true", help="Print number of jobs")

    return parser
    

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.count_jobs:
        print(len(get_all_jobs()))
        sys.exit(0)

    if args.job_index is None:
        parser.error("job_index is required")

    main(
        args.job_index,
        disturbance=args.disturbance,
        noise_scale=args.noise_scale,
        prior_noise_std=args.prior_noise_std,
        prior_noise_std_kinematic=args.prior_noise_std_kinematic,
    )   
