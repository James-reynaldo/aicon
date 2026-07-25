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
NUM_TRIALS_PER_JOB = 10
BOUNDARY_EXTENDED_TRIALS = 7
RANDOM_INIT_TIME = 0  # seconds of random movement at start
MAX_TIMESTEPS = 1000

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
    # standard_job["trial_params"]["connection_params"] = copy.deepcopy(base_conn_params)

    return jobs_est + jobs_conn + [standard_job]


def main(job_index: int, disturbance: float = None, noise_scale: float = None):
    jobs = get_all_jobs()
    print(total_jobs := len(jobs), "total jobs")
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"Invalid job index {job_index}")

    job = jobs[job_index]

    print(f"[HPC] Running job {job_index}: {job['group_name']} {job['param_name']} {job['sweep_label']}")

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
        for run in range(NUM_TRIALS_PER_JOB): #5,6,9
            if run < len(Visible_Initial_qpos_list):
                initial_panda_qpos = Visible_Initial_qpos_list[run]
            else:
                initial_panda_qpos = Invisible_Initial_qpos_list[run - len(Visible_Initial_qpos_list)]
            env = setup_env(initial_qpos=initial_panda_qpos)
            set_global_seed(run)  # set seed for reproducibility
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
        #         f"running {BOUNDARY_EXTENDED_TRIALS} more trials to total 10 runs."
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

    print(f"Experiment database saved to: {db_path}")


if __name__ == "__main__":
    idx = int(sys.argv[1])
    disturbance = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    noise_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(idx, disturbance=disturbance, noise_scale=noise_scale)
