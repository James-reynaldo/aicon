import sys
import copy
import numpy as np

from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.python_sequential import build_components, run_component_sequence

NUM_TRIALS_PER_JOB = 3
RANDOM_INIT_TIME = 0.5  # seconds of random movement at start
RENDER = False
MAX_TIMESTEPS = 1000


# reuse your existing functions (copy them from current file)
from aicon.drawer_tutorial.Sweep import (
    get_default_estimator_params,
    get_default_connection_params,
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

    jobs_conn = list(generate_single_parameter_sweeps(base_conn_params))
    for job in jobs_conn:
        job["group_name"] = f"connection.{job['group_name']}"
        conn_trial = job.pop("trial_params")
        trial_est_params = copy.deepcopy(base_params)
        trial_est_params["connection_params"] = conn_trial
        job["trial_params"] = trial_est_params

    return jobs_est + jobs_conn


def main(job_index: int, disturbance: float = None, noise_scale: float = None):
    jobs = get_all_jobs()
    # print(total_jobs := len(jobs), "total jobs")
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Invalid job index {job_index}")

    job = jobs[job_index]

    print(f"[HPC] Running job {job_index}: {job['group_name']} {job['param_name']} {job['sweep_label']}")

    initial_panda_qpos = np.array([-0.56, 0.76, 0.1, -1.90, 1.11, 1.5, -0.32])

    env = setup_env(render=RENDER, initial_qpos=initial_panda_qpos)

    results = []

    try:
        for run in range(NUM_TRIALS_PER_JOB):
            set_global_seed(run)  # set seed for reproducibility
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

    # save per-job result
    import csv
    from pathlib import Path
    
    if disturbance is not None:
        out_file = Path("results_disturbance") / f"j_{job_index}_d_{disturbance}.csv"
    elif noise_scale is not None:
        out_file = Path("results_noise") / f"j_{job_index}_n_{noise_scale}.csv"
    else:
        out_file = Path("results") / f"j_{job_index}.csv"
    out_file.parent.mkdir(exist_ok=True)

    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "param_group",
            "param_name",
            "sweep_label",
            "sweep_value",
            "success",
            "timesteps",
            "error",
        ])
        writer.writerows(results)


if __name__ == "__main__":
    idx = int(sys.argv[1])
    disturbance = float(sys.argv[2]) if len(sys.argv) > 2 else None
    noise_scale = float(sys.argv[3]) if len(sys.argv) > 3 else None
    main(idx, disturbance=disturbance, noise_scale=noise_scale)