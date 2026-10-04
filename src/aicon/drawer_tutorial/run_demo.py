import ast
import copy
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
import torch
from PIL import Image
from robosuite.utils.transform_utils import quat2mat
from robosuite.wrappers import VisualizationWrapper

from aicon.drawer_tutorial.experiment_specifications import (
    OPEN_VALUE,
    get_building_functions_basic_drawer_motion,
)
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.drawer_tutorial.scene_visualizers import (
    CAMERA_FORWARD_INDICATOR,
    CAMERA_FORWARD_LENGTH,
    CAMERA_ORIGIN_INDICATOR,
    DRAWER_POSITION_INDICATOR,
    EE_POS_INDICATOR,
    KINEMATIC_AXIS_INDICATOR,
    KINEMATIC_AXIS_LENGTH,
    KINEMATIC_DRAWER_INDICATOR,
    KINEMATIC_DRAWER_INITIAL,
    create_demo_visualizers,
    update_demo_visualizers,
)
from aicon.middleware.python_sequential import build_components, run_component_sequence

RENDER = os.getenv("AICON_RENDER", "false").lower() == "true"
PLOTTER = os.getenv("AICON_PLOTTER", "false").lower() == "true"
MANUAL_CONTROL = os.getenv("AICON_MANUAL_CONTROL", "false").lower() == "true"
SAVE_PNG = os.getenv("AICON_SAVE_PNG", "true").lower() == "true"

# These toggles are intentionally kept in a single place to preserve the current
# runtime defaults and make the demo easy to configure in one location.
if RENDER:
    from robosuite.devices import Keyboard, SpaceMouse
    from robosuite.utils.input_utils import input2action


def render_frame(env, timestamp):
    """Render one frame when visualization is enabled for the drawer demo."""
    if RENDER:
        env.render()


def canonicalize_gradient_trace(trace):
    """Return a deterministic, canonical JSON string for the current gradient trace."""
    if trace is None:
        return None
    if isinstance(trace, torch.Tensor):
        payload = trace.detach().cpu().tolist()
    elif isinstance(trace, np.ndarray):
        payload = trace.tolist()
    elif isinstance(trace, (list, tuple)):
        payload = list(trace)
    elif isinstance(trace, str):
        payload = trace
    else:
        payload = str(trace)

    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        return json.dumps(str(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def trace_hash(trace):
    """Return a deterministic SHA-256 hash for a canonical trace."""
    canonical = canonicalize_gradient_trace(trace)
    if canonical is None:
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Disturbance configuration for periodic jerks.
DISTURBANCE_MAGNITUDE = 0.0

# Historical starting poses were explored during tuning. The final active default is
# intentionally the last assignment below so the demo keeps its current behavior.
# DEFAULT_INITIAL_PANDA_QPOS = np.array([...])
# ...
DEFAULT_INITIAL_PANDA_QPOS = np.array([-0.58865829,  0.70879424,  0.0393395,  -1.81579848,  1.08319597,  1.37122098,  -0.23145539])
def setup_env(device_type, initial_qpos=None):
    if RENDER:
        device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    # Create our custom environment
    env = DrawerOpenEnv(
        robots="Panda",
        has_renderer=RENDER,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="agentview", #'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
        horizon=100,
        control_freq=30,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        initial_qpos=initial_qpos,
    )

    env = VisualizationWrapper(
        env,
        indicator_configs=[
            {"name": KINEMATIC_AXIS_INDICATOR, "type": "capsule", "size": [0.005, KINEMATIC_AXIS_LENGTH / 2], "rgba": [0, 1, 1, 0.6]}, #cyan
            {"name": KINEMATIC_DRAWER_INDICATOR, "type": "sphere", "size": [0.01], "rgba": [1, 0, 1, 0.95]}, #magenta
            {"name": KINEMATIC_DRAWER_INITIAL, "type": "sphere", "size": [0.01], "rgba": [1, 1, 0, 0.95]}, #yellow
            {"name": EE_POS_INDICATOR, "type": "sphere", "size": [0.01], "rgba": [1, 0, 1, 0.95]}, #magenta
            {"name": DRAWER_POSITION_INDICATOR, "type": "sphere", "size": [0.01], "rgba": [0, 1, 0, 0.95]}, #green
            {"name": CAMERA_ORIGIN_INDICATOR, "type": "sphere", "size": [0.012], "rgba": [1, 0.2, 0.1, 1]}, #red
            {"name": CAMERA_FORWARD_INDICATOR, "type": "capsule", "size": [0.006, CAMERA_FORWARD_LENGTH / 2], "rgba": [1, 0.1, 0, 0.9]}, #red
        ],
    )
    env.reset()
    geom_id = env.sim.model.geom("block_geom").id

    print("BLOCK EXISTS")
    print("World position:", env.sim.data.geom_xpos[geom_id])
    print("Size:", env.sim.model.geom_size[geom_id])
    print("RGBA:", env.sim.model.geom_rgba[geom_id])
    if RENDER:
        device.start_control()
    # return env
    if RENDER:
        return env, device
    else:
        return env, None
    


def run_trial(env,device, estimator_params=None, max_timesteps=None, *, stop_on_done=False,
              reset_on_start=True,
              random_disturbance=None, 
               noise_scale=1.0, prior_noise_std =0.01,
              prior_noise_std_kinematic=0.0, sensor_noise_overrides=None,
              render=None, status_label=None, return_gradient_traces=False):
    """Shared drawer-control loop for the interactive demo and parameter sweeps."""
    image_dir = Path(__file__).resolve().parent / "demo"
    if SAVE_PNG:
        image_dir.mkdir(parents=True, exist_ok=True)
    if reset_on_start:
        env.reset()
    robot = env.robots[0]
    params = copy.deepcopy(estimator_params) if isinstance(estimator_params, dict) else estimator_params
    connection_params = params.pop("connection_params", None) if isinstance(params, dict) else None
    builders, _, _ = get_building_functions_basic_drawer_motion(
        env, estimator_params=params, connection_params=connection_params,
        noise_scale=noise_scale, prior_noise_std=prior_noise_std,
        prior_noise_std_kinematic=prior_noise_std_kinematic,
        sensor_noise_overrides=sensor_noise_overrides,
    )
    components = build_components(builders)
    gripper = components["GripperAction"]
    velocities = components["EEVelocities"]
    kinematic = components["KinematicJointEstimator"]
    grasp = components["GraspLikelihoodEstimator"]
    render = RENDER if render is None else render
    visualizers = create_demo_visualizers(
        env, components, visualize_kinematic_angles=True,
        visualize_kinematic_angles_plot=PLOTTER,
        visualize_ee=True, visualize_drawer_position=True,
        visualize_grasp_diagnostics=PLOTTER,
        visualize_gradient_trace=PLOTTER,
        visualize_camera_transform=True,
    ) if render or SAVE_PNG else None

    curr_t, step_idx = 0.0, 1
    joint_error = grasp_estimate = float("nan")
    actual_grasped = False
    done_wait_steps = 0
    base_env = env.env if hasattr(env, "env") else env
    initial_drawer_handle_pos = np.asarray(base_env.get_drawer_handle_pos(), dtype=np.float64).reshape(-1)
    job_gradient_traces = []
    print("initial_drawer_handle_pos:", initial_drawer_handle_pos)
    while True:
        run_component_sequence(components, torch.tensor(curr_t))
        current_trace = getattr(velocities, "gradient_trace", None)
        canonical_trace = canonicalize_gradient_trace(current_trace)
        if canonical_trace is not None:
            job_gradient_traces.append(canonical_trace)
        velocity = velocities.quantities["action_velo_ee"].cpu().numpy()
        gripper_action = gripper.quantities["gripper_activation"].squeeze().cpu().numpy()
        action = np.concatenate([velocity, np.zeros(3), [2 * gripper_action - 1]])

        if random_disturbance is not None and np.random.rand() < 0.1:
            if float(gripper_action) > 0.5:
                disturbance = np.zeros(3)
                disturbance[1] = abs(np.random.uniform(0.0, random_disturbance))
            else:
                disturbance = np.random.uniform(-random_disturbance, random_disturbance, 3)
            action[:3] += disturbance

        if MANUAL_CONTROL and render:
            action, _ = input2action(
                device=device, robot=robot, active_arm="right", env_configuration="single-arm-opposed"
            )
        obs, reward, done, _ = env.step(action)
        if SAVE_PNG:
            # Capture the resulting state
            img = env.sim.render(
                camera_name="agentview",
                width=512,
                height=512
            )
            img = np.flipud(img)
            filename = image_dir / f"demo_frame_{step_idx:04d}.png"
            Image.fromarray(img).save(
                filename
            )
        base_env = env.env if hasattr(env, "env") else env
        true_joint = float(base_env.sim.data.qpos[base_env.cabinet_qpos_addrs])
        estimated_joint = float(kinematic.quantities["kinematic_joint"][2].item())
        joint_error = true_joint - estimated_joint
        kinematic_joint = kinematic.quantities["kinematic_joint"].detach().cpu().numpy().reshape(-1)
        kinematic_axis_error = float(
            np.linalg.norm(kinematic_joint[:2] - np.array([np.pi / 2, np.pi / 2]))
        )
        kinematic_anchor_pos = kinematic_joint[3:6]
        anchor_error = float(np.linalg.norm(kinematic_anchor_pos - initial_drawer_handle_pos[:3]))
        grasp_estimate = float(grasp.quantities["likelihood_grasped_drawer"].item())
        grasp_bool = bool(grasp_estimate > 0.5)
        actual_grasped = bool(np.asarray(obs.get("robot0_contact", base_env._has_gripper_contact)).item())
        if visualizers is not None:
            update_demo_visualizers(visualizers, components, curr_t, obs.get("drawer_orientation"))
        if status_label is not None:
            limit = max_timesteps if max_timesteps is not None else "inf"
            print(f"  [{status_label}] step {step_idx:04d}/{limit}: true_joint={true_joint:.6f} estimated_joint={estimated_joint:.6f}"
                  f"kinematic_axis_error={kinematic_axis_error:.6f}, err={joint_error:+.6f}, "
                  f"anchor_error={anchor_error:.6f}, grasped={grasp_bool and actual_grasped}, gripper_activation={gripper_action}", flush=True)
        if stop_on_done and ((done or reward == 1.0 or done_wait_steps > 0)):
            done_wait_steps = min(done_wait_steps + 1, 50)
            if done_wait_steps >= 50:
                result = (True, step_idx, joint_error, true_joint, grasp_bool, actual_grasped, kinematic_axis_error, anchor_error)
                if return_gradient_traces:
                    return result + (job_gradient_traces,)
                return result
        else:
            done_wait_steps = 0
        curr_t += env.control_timestep

        #add time label to mujoco simulation

        if render:
            render_frame(env, curr_t)
        step_idx += 1
        if max_timesteps is not None and step_idx > max_timesteps:
            result = (False, max_timesteps, joint_error, true_joint, grasp_bool, actual_grasped, kinematic_axis_error, anchor_error)
            if return_gradient_traces:
                return result + (job_gradient_traces,)
            return result

        # Get current joint positions
        if MANUAL_CONTROL:
            panda_qpos = robot._joint_positions
            print(f"Current joint positions: {panda_qpos}")


def main(device, env, **kwargs):
    """Run the interactive demo through the shared trial runner."""
    if device is not None:
        device.start_control()
    return run_trial(
        env, device, stop_on_done=False, **kwargs
    )


def run_demo():
    env, device = setup_env("keyboard", initial_qpos=DEFAULT_INITIAL_PANDA_QPOS)
    main(device, env)


if __name__ == "__main__":
    run_demo()
