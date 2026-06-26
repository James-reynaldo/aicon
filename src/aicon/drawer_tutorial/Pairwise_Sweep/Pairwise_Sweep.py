import sys
import copy
import itertools
import os
import json
import hashlib
from datetime import datetime
import csv
import numpy as np

from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion
from aicon.middleware.python_sequential import build_components, run_component_sequence

from aicon.drawer_tutorial.Sweep import (
    setup_env,
    run_trial
)

RENDER = True
NUM_TRIALS_PER_JOB = 3
MAX_TIMESTEPS = 1000

def get_default_estimator_params():
    return {
        # "ee_pose": {
        #     "initial_uncertainty_scale": 0.001, # Seem to not matter at all
        #     "action_process_noise": 0.01, # Seem to not matter so much, just not negative (fail)
        #     "proprio_update_noise": 0.01, # Seem to not matter so much, just not negative (fail)
        # },
        # "visible": {
        #     "initial_likelihood_prior": 0.01,
        #     "initial_clip_min": 0.02,
        #     "initial_clip_max": 0.98,

        #     "update_gain": 0.5,
        #     # "clip_min": 1e-9,
        #     # "clip_max": 0.9999999,
        # },
        # "grasp_likelihood": {
        #     "initial_likelihood": 0.01, # Seem to not matter at all
        #     # "initially_grasped": False,
        #     "initially_grasped_likelihood": 0.99,
        #     # "clip_min": 1e-10,
        #     # "clip_max": 0.9999999,
        #     "baseline_measurement_likelihood": 0.05,
        #     "initial_clip_min": 0.02,
        #     "initial_clip_max": 0.98,
        #     "initial_time_since_hand_change": 2.0,

        #     "gripper_activation_threshold": 0.5,
        #     "close_distance_threshold": 0.03,
        #     "uncertainty_dist_threshold": 0.25,
        #     "hand_change_time_threshold": 1.0,
        #     "force_time_scale": 6.0,
        #     "force_time_max": 6.0,
        #     "low_likelihood_threshold": 0.1,
        #     "negative_innovation_threshold": -0.05,
        #     "negative_innovation_scale": 0.1,
        # },
        "drawer_position": {
            # "initial_depth": None,
            # "depth_prior": 0.8,# no need
            # "initial_uncertainty_scale": 200, # maybe no
            # "initial_uncertainty_xy": 0.2, # maybe no
            # "initial_uncertainty_depth": 1.0, # maybe no
            # "initial_uncertainty_xy_none": 0.1, # no need
            # "initial_uncertainty_depth_none": 0.3,# no need
            # # "sample_init_mean": False,
            # "sample_init_mean_likelihood_threshold": 0.6, # no need
            # "sample_init_mean_distance_threshold": 0.25, # no need
            # "sample_init_mean_uncertainty_multiplier": 2.0, # no need

            "meas_noise_factor": 0.025,
            "visual_likelihood_steepness": 5.0,
            "R_add_scale": 5.0,
            # "measurement_nan_reject_scale": 0.01, # no need
            # "forward_noise_grasped_coeff": 0.15, 
            # "forward_noise_base": 0.005, 
            # "grasped_update_R_scale": 0.03, 
            # "grasped_outlier_rejection_threshold": 0.05, 
            # "measurement_existence_threshold": 0.5,
            # "grasped_uncertainty_threshold": 0.1,
            # "missed_absent_measurement_uncertainty_coeff": 0.1, 
            # "hand_change_recovery_time": 0.5, # no need
            # "tf_lookup_timeout": 5.0, # no need
        },
        # "kinematic_joint": {
            # "initial_rotation_xy": None,
            # "initial_uncertainty_scale": None,
            # "sample_init_mean": False,
            # "initial_azimuth_default": -0.7853981633974483,
            # "initial_elevation_default": -1.5707963267948966,

            # "grasped_noise": 0.002, # When value too high, estimation error quite big
            # "ungrasped_noise": 0.001, # Seem to not matter at all
            # "axis_azimuth_process_noise": 0.001, # Does not seem to matter
            # "axis_elevation_process_noise": 0.001, # Does not seem to matter
            # "joint_process_noise": 0.2, # When negative, take a long time and high estimation error, when zero, high estimation error
            # "anchor_process_noise": 1e-6, # Does not seem to matter
            # "grasp_threshold": 0.2, # Does not matter much, just not zero
            # "grasp_floor": 0.2, # When value too high, estimation error quite big
            # "outlier_rejection_treshold": 4.0,
            # "shift_clip_min": 1e-10,
        # },
    }

def get_default_connection_params():
    # Defaults mirror the hard-coded values in DistGraspHandConnection
    return {
        # "DistGraspHandConnection": {
        #     "dist_decay": 4.0,
        #     "close_dist_threshold": 0.03,
        #     "ft_noise_offset": 5.0,
        #     "low_likelihood_threshold": 0.1,
        #     "gripper_activation_threshold": 0.5,
        #     "uncertainty_dist_threshold": 0.25,
        #     "small_likelihood_value": 1e-8,

        #     "uncertainty_bias": 0.2,
        #     "uncertainty_scale_uncertainty": 5.0,
        #     "uncertainty_scale_relevance": 20.0,
        #     "dist_sigmoid_scale": 5.0,
        #     "time_since_hand_change_threshold": 1.0,
        #     "ft_tresh_multiplier": 6.0,
        #     "ft_tresh_cap": 6.0,
        # }
    }


def get_all_jobs():
    base_params = get_default_estimator_params()
    base_conn_params = get_default_connection_params()

    sweep_specs = []

    def _collect_specs(params, prefix=None, connection=False):
        for group_name in sorted(params.keys()):
            for param_name in sorted(params[group_name].keys()):
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

    _collect_specs(base_params, connection=False)
    _collect_specs(base_conn_params, connection=True)

    if not sweep_specs:
        return []

    jobs = []
    if len(sweep_specs) == 1:
        spec_combinations = [tuple(sweep_specs)]
    else:
        spec_combinations = list(itertools.combinations(sweep_specs, 2))

    for specs in spec_combinations:
        for combination in itertools.product(*(spec["sweep_values"] for spec in specs)):
            trial_params = copy.deepcopy(base_params)
            connection_params = {}
            job_groups = []
            job_names = []
            for spec, (sweep_label, sweep_value) in zip(specs, combination):
                group_name = spec["group_name"]
                param_name = spec["param_name"]
                if spec["connection"]:
                    if group_name not in connection_params:
                        connection_params[group_name] = copy.deepcopy(base_conn_params[group_name])
                    connection_params[group_name][param_name] = sweep_value
                    job_groups.append(f"connection.{group_name}")
                else:
                    trial_params[group_name][param_name] = sweep_value
                    job_groups.append(group_name)
                job_names.append(f"{group_name}.{param_name}={sweep_label}")

            if connection_params:
                trial_params["connection_params"] = connection_params

            jobs.append(
                {
                    "group_name": ",".join(job_groups),
                    "param_name": ",".join(spec["param_name"] for spec in specs),
                    "standard_value": None,
                    "sweep_label": ",".join(label for label, _ in combination),
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


def save_job_metadata(job: dict, results: list, out_dir: str = "results_pairwise", job_index: int = None) -> tuple:
    """Save metadata JSON and per-run CSV for a completed job. Returns (json_path, csv_path).

    If `job_index` is provided, use it (as decimal) for filenames; otherwise fall back to compact hash id.
    """
    os.makedirs(out_dir, exist_ok=True)
    if job_index is not None:
        job_id = str(int(job_index))
    else:
        job_id = make_compact_id(job)
    base = os.path.join(out_dir, job_id)

    # build metadata
    successes = sum(1 for r in results if r[4])
    # For calculation of average timesteps, we can consider only successful runs
    avg_timesteps = sum(r[5] for r in results if r[4]) / successes if successes else None
    avg_err = sum(r[6] for r in results) / len(results) if results else None

    meta = {
        "id": job_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "job_name": job.get("job_name"),
        "group_name": job.get("group_name"),
        "param_name": job.get("param_name"),
        "sweep_label": job.get("sweep_label"),
        "sweep_value": job.get("sweep_value"),
        "trial_params": job.get("trial_params"),
        "runs": len(results),
        "summary": {
            "successes": successes,
            "avg_timesteps": avg_timesteps,
            "avg_err": avg_err,
        },
    }

    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    csv_path = base + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group_name", "param_name", "sweep_label", "sweep_value", "success", "timesteps", "err"])
        for rec in results:
            writer.writerow(rec)

    return json_path, csv_path

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

    try:
        for run in range(NUM_TRIALS_PER_JOB):
            success, timesteps, err, grasp, grasped = run_trial(
                env,
                estimator_params=job["trial_params"],
                max_timesteps=MAX_TIMESTEPS,
                render=RENDER,
                sweep_label=job["sweep_label"],
                group_name=job["group_name"],
                param_name=job["param_name"],
                sweep_value=job["sweep_value"],
                random_init_time=0.5,
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
        except Exception as e:
            print(f"Error closing environment: {e}")

    # save results and metadata
    try:
        json_path, csv_path = save_job_metadata(job, results, out_dir="results_pairwise", job_index=job_index)
        print(f"Saved results: {csv_path}")
        print(f"Saved metadata: {json_path}")
    except Exception as e:
        print(f"Error saving metadata/results: {e}")
    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--count-jobs":
        print(len(get_all_jobs()))
        sys.exit(0)

    idx = int(sys.argv[1])
    disturbance = float(sys.argv[2]) if len(sys.argv) > 2 else None
    noise_scale = float(sys.argv[3]) if len(sys.argv) > 3 else None
    main(idx, disturbance=disturbance, noise_scale=noise_scale)
