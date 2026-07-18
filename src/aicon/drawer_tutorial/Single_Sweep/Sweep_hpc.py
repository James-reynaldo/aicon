import sys
import copy
import numpy as np
from pathlib import Path

from aicon.drawer_tutorial.Experiment_store import ExperimentStore
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.python_sequential import build_components, run_component_sequence

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR_DISTURBANCE = DATA_DIR / "disturbance"
DATA_DIR_NOISE = DATA_DIR / "noise"
DATA_DIR_NORMAL = DATA_DIR / "normal"
NUM_TRIALS_PER_JOB = 3
BOUNDARY_EXTENDED_TRIALS = 7
RANDOM_INIT_TIME = 0.5  # seconds of random movement at start
MAX_TIMESTEPS = 1000


# reuse your existing functions (copy them from current file)
from aicon.drawer_tutorial.Sweep import (
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

    # same logic as your sweep generator
    from aicon.drawer_tutorial.Sweep import generate_single_parameter_sweeps, filter_single_parameter_sweeps

    jobs_est = list(generate_single_parameter_sweeps(base_params))
    for job in jobs_est:
        job["trial_params"]["connection_params"] = copy.deepcopy(base_conn_params)

    jobs_conn = list(generate_single_parameter_sweeps(base_conn_params))
    for job in jobs_conn:
        job["group_name"] = f"connection.{job['group_name']}"
        conn_trial = job.pop("trial_params")
        trial_est_params = copy.deepcopy(base_params)
        trial_est_params["connection_params"] = conn_trial
        job["trial_params"] = trial_est_params

    standard_job = get_standard_job(base_params)
    standard_job["trial_params"]["connection_params"] = copy.deepcopy(base_conn_params)

    return jobs_est + jobs_conn


def main(job_index: int, disturbance: float = None, noise_scale: float = None):
    jobs = get_all_jobs()
    print(total_jobs := len(jobs), "total jobs")
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Invalid job index {job_index}")

    job = jobs[job_index]

    print(f"[HPC] Running job {job_index}: {job['group_name']} {job['param_name']} {job['sweep_label']}")

    initial_panda_qpos = np.array([-0.56, 0.76, 0.1, -1.90, 1.11, 1.5, -0.32])

    env = setup_env(initial_qpos=initial_panda_qpos)

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
        "experiment_type": "1d_sweep",
        "parameter": f"{job['group_name']}.{job['param_name']}",
        "label": job["sweep_label"],
        "sweep_value": job["sweep_value"],
    }
    job_metadata.update(job.get("metadata", {}))

    try:
        for run in range(NUM_TRIALS_PER_JOB):
            set_global_seed(run)  # set seed for reproducibility
            success, timesteps, err, grasp, grasped = run_trial(
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
                f"running {BOUNDARY_EXTENDED_TRIALS} more trials to total 10 runs."
            )
            for run in range(NUM_TRIALS_PER_JOB, NUM_TRIALS_PER_JOB + BOUNDARY_EXTENDED_TRIALS):
                set_global_seed(run)
                success, timesteps, err, grasp, grasped = run_trial(
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
    idx = int(sys.argv[1])
    disturbance = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    noise_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(idx, disturbance=disturbance, noise_scale=noise_scale)
