import sys
import copy
import hashlib
import json
import numpy as np
from pathlib import Path

from aicon.drawer_tutorial.Sweep_files.Experiment_store import ExperimentStore
from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.python_sequential import build_components, run_component_sequence
from aicon.drawer_tutorial.Sweep_files.initial_qpos import initial_qpos_list

DATA_DIR = Path(__file__).resolve().parents[1] / "Ignored_files" / "data"
DATA_DIR_DISTURBANCE = DATA_DIR / "Single" / "disturbance"
DATA_DIR_NOISE = DATA_DIR / "Single" / "noise"
DATA_DIR_NORMAL = DATA_DIR / "Single" / "normal"
DATA_DIR_BAD_PRIOR = DATA_DIR / "Single" / "bad_prior"
DATA_DIR_BAD_PRIOR_KINEMATIC = DATA_DIR / "Single" / "bad_prior_kinematic"
DATA_DIR_TEMP = DATA_DIR / "Single" / "temp"
SWEEP_PLAN_PATH = Path(__file__).resolve().parent / "sweep_plan_actual_values.json"
NUM_TRIALS_PER_JOB = 10
BOUNDARY_EXTENDED_TRIALS = 7
RANDOM_INIT_TIME = 0  # seconds of random movement at start
MAX_TIMESTEPS = 1000
DB_SHARD_PREFIX = "experiment_store_job_"

# reuse your existing functions (copy them from current file)
from aicon.drawer_tutorial.Sweep_files.Sweep_utils import (
    get_default_estimator_params,
    get_default_connection_params,
    get_standard_job,
    setup_env,
    run_trial,
    set_global_seed,
)

def get_all_jobs():
    base_params = get_default_estimator_params()
    base_conn_params = get_default_connection_params()

    from aicon.drawer_tutorial.Sweep_files.Sweep_utils import generate_single_parameter_sweeps

    jobs_est = list(generate_single_parameter_sweeps(base_params, sweep_plan_path=str(SWEEP_PLAN_PATH)))
    for job in jobs_est:
        job["trial_params"]["connection_params"] = copy.deepcopy(base_conn_params)

    jobs_conn = list(generate_single_parameter_sweeps(base_conn_params, sweep_plan_path=str(SWEEP_PLAN_PATH)))
    for job in jobs_conn:
        job["group_name"] = f"connection.{job['group_name']}"
        conn_trial = job.pop("trial_params")
        trial_est_params = copy.deepcopy(base_params)
        trial_est_params["connection_params"] = conn_trial
        job["trial_params"] = trial_est_params

    standard_job = get_standard_job(base_params)
    standard_job["trial_params"]["connection_params"] = copy.deepcopy(base_conn_params)

    return jobs_est + jobs_conn + [standard_job]


def main(job_index: int, disturbance: float = None, noise_scale: float = None, prior_noise_std: float = None, prior_noise_std_kinematic: float = None, temp: bool = False,trial_index: int = None):
    jobs = get_all_jobs()
    print(total_jobs := len(jobs), "total jobs")
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Invalid job index {job_index}")

    job = jobs[job_index]

    print(f"[HPC] Running job {job_index}: {job['group_name']} {job['param_name']} {job['sweep_label']}")

    results = []

    if disturbance is not None:
        directory = DATA_DIR_DISTURBANCE
    elif noise_scale is not None:
        directory = DATA_DIR_NOISE
    elif prior_noise_std is not None:
        directory = DATA_DIR_BAD_PRIOR
    elif prior_noise_std_kinematic is not None:
        directory = DATA_DIR_BAD_PRIOR_KINEMATIC
    else:
        directory = DATA_DIR_NORMAL
    directory.mkdir(exist_ok=True, parents=True)

    # Use the same naming scheme as Pairwise_Sweep.py
    job_key = f"{job_index}:{job.get('job_name', '')}:{job.get('group_name', '')}:{job.get('param_name', '')}:{job.get('sweep_label', '')}"
    short_hash = hashlib.sha1(job_key.encode("utf-8")).hexdigest()[:10]
    db_path = directory / f"{DB_SHARD_PREFIX}{job_index:05d}.db"
    store = ExperimentStore(str(db_path))

    job_metadata = {
        "experiment_type": "1d_sweep",
        "parameter": f"{job['group_name']}.{job['param_name']}",
        "label": job["sweep_label"],
        "sweep_value": job["sweep_value"],
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
        for run in range(NUM_TRIALS_PER_JOB): #5,6,9
            #only run one trial if trial_index is specified
            if trial_index is not None:
                if trial_index < 0 or trial_index >= NUM_TRIALS_PER_JOB:
                    raise ValueError(f"Invalid trial index {trial_index}")
                run = trial_index
            initial_panda_qpos = initial_qpos_list[run]
            env = setup_env(initial_qpos=initial_panda_qpos)
            set_global_seed(run)  # set seed for reproducibility
            trial_kwargs = {
                "estimator_params": job["trial_params"],
                "max_timesteps": MAX_TIMESTEPS,
                "sweep_label": job["sweep_label"],
                "group_name": job["group_name"],
                "param_name": job["param_name"],
                "sweep_value": job["sweep_value"],
                "return_gradient_traces": True,
            }
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
            if trial_index is not None:
                break

    finally:
        try:
            env.close()
        except Exception:
            pass
        store.close()

    print(f"Experiment database saved to: {db_path}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "job_index",
        type=int,
    )

    parser.add_argument(
        "--trial_index",
        type=int,
        default=None,
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

    parser.add_argument(
        "--temp",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    main(
        args.job_index,
        disturbance=args.disturbance,
        noise_scale=args.noise_scale,
        prior_noise_std=args.prior_noise_std,
        prior_noise_std_kinematic=args.prior_noise_std_kinematic,
        temp=args.temp,
        trial_index=args.trial_index
    )
