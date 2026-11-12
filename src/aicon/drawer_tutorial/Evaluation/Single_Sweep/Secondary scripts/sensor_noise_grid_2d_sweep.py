import argparse
import copy
import itertools
from pathlib import Path

from aicon.drawer_tutorial.Sweep_files.Experiment_store import ExperimentStore
from aicon.drawer_tutorial.Sweep_files.Sweep_utils import (
    get_default_estimator_params,
    get_default_connection_params,
    run_trial,
    set_global_seed,
    setup_env,
)
from aicon.drawer_tutorial.initial_qpos import initial_qpos_list

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sensor_noise_grid_2d"
NUM_TRIALS_PER_JOB = 10
MAX_TIMESTEPS = 1000
RANDOM_INIT_TIME = 0

BEARING_SENSOR_NOISE_LEVELS = [1.0, 5.0, 10.0, 15.0, 25.0]
EE_SENSOR_NOISE_LEVELS = [1.0, 5.0, 10.0, 15.0, 25.0]
MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]


def get_all_jobs():
    base_params = get_default_estimator_params()
    base_conn_params = get_default_connection_params()
    jobs = []

    for bearing_noise in BEARING_SENSOR_NOISE_LEVELS:
        for ee_noise in EE_SENSOR_NOISE_LEVELS:
            sensor_noise_overrides = {
                "BearingSensor": bearing_noise,
                "EEPosSensor": ee_noise,
            }

            for meas_mult, radd_mult, proprio_mult in itertools.product(MULTIPLIERS, repeat=3):
                trial_params = copy.deepcopy(base_params)
                trial_params["connection_params"] = copy.deepcopy(base_conn_params)

                trial_params["drawer_position"]["meas_noise_factor"] = (
                    base_params["drawer_position"]["meas_noise_factor"] * meas_mult
                )
                trial_params["drawer_position"]["R_add_scale"] = (
                    base_params["drawer_position"]["R_add_scale"] * radd_mult
                )
                trial_params["ee_pose"]["proprio_update_noise"] = (
                    base_params["ee_pose"]["proprio_update_noise"] * proprio_mult
                )

                sweep_label = (
                    f"Bearing={bearing_noise}, EE={ee_noise}, "
                    f"meas={meas_mult}x, R_add={radd_mult}x, proprio={proprio_mult}x"
                )
                jobs.append(
                    {
                        "group_name": "sensor_noise.grid_2d",
                        "param_name": "drawer_position.meas_noise_factor,drawer_position.R_add_scale,ee_pose.proprio_update_noise",
                        "sweep_label": sweep_label,
                        "sweep_value": None,
                        "trial_params": trial_params,
                        "sensor_noise_overrides": sensor_noise_overrides,
                        "metadata": {
                            "experiment_type": "sensor_noise_grid_2d",
                            "sensor_name": "both",
                            "bearing_sensor_noise_level": bearing_noise,
                            "ee_sensor_noise_level": ee_noise,
                            "sensor_noise_level": bearing_noise,
                            "ee_sensor_noise_scale": ee_noise,
                            "meas_noise_factor_multiplier": meas_mult,
                            "R_add_scale_multiplier": radd_mult,
                            "proprio_update_noise_multiplier": proprio_mult,
                        },
                    }
                )

    return jobs


def main(job_index: int, trial_index: int | None = None, dry_run: bool = False):
    jobs = get_all_jobs()
    print(f"Total 2D sensor-noise-grid jobs: {len(jobs)}")
    if dry_run:
        for idx in range(min(5, len(jobs))):
            print(idx, jobs[idx]["sweep_label"])
        return

    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"job_index {job_index} out of range for {len(jobs)} jobs")

    job = jobs[job_index]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / "experiment_store.db"
    store = ExperimentStore(str(db_path))

    job_metadata = {
        "experiment_type": "sensor_noise_grid_2d",
        "parameter": job["param_name"],
        "label": job["sweep_label"],
        "sensor_name": job["metadata"]["sensor_name"],
        "bearing_sensor_noise_level": job["metadata"]["bearing_sensor_noise_level"],
        "ee_sensor_noise_level": job["metadata"]["ee_sensor_noise_level"],
        "job_index": job_index,
    }
    job_metadata.update(job.get("metadata", {}))

    try:
        for run in range(NUM_TRIALS_PER_JOB):
            if trial_index is not None:
                if trial_index < 0 or trial_index >= NUM_TRIALS_PER_JOB:
                    raise ValueError(f"Invalid trial index {trial_index}")
                run = trial_index

            initial_qpos = initial_qpos_list[run]
            env = setup_env(initial_qpos=initial_qpos)
            set_global_seed(run)

            trial_kwargs = {
                "estimator_params": job["trial_params"],
                "max_timesteps": MAX_TIMESTEPS,
                "sweep_label": job["sweep_label"],
                "group_name": job["group_name"],
                "param_name": job["param_name"],
                "sweep_value": job["sweep_value"],
                "random_init_time": RANDOM_INIT_TIME,
                "random_std": 1,
                "return_gradient_traces": True,
                "noise_scale": 1.0,
                "sensor_noise_overrides": job["sensor_noise_overrides"],
            }

            success, timesteps, err, true_joint, grasp, actual_grasped, kinematic_axis_error, anchor_error, gradient_traces = run_trial(
                env,
                **trial_kwargs,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full 2D sensor-noise grid sweep.")
    parser.add_argument("job_index", type=int)
    parser.add_argument("--trial-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    main(args.job_index, trial_index=args.trial_index, dry_run=args.dry_run)
