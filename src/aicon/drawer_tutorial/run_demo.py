import time
from pathlib import Path

import lovely_tensors as lt
import numpy as np
import robosuite as suite
import torch
# from robosuite.devices import Keyboard, SpaceMouse
# from robosuite.utils.input_utils import input2action
from robosuite.utils.transform_utils import quat2mat
from robosuite.wrappers import VisualizationWrapper

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion

# Import our custom environment
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.middleware.python_sequential import build_components, run_component_sequence


def setup_env(device_type, initial_qpos=None):
    # device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    # Create our custom environment
    env = DrawerOpenEnv(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        render_camera="agentview", #'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
        horizon=100,
        control_freq=30,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        initial_qpos=initial_qpos,
    )

    env = VisualizationWrapper(env)
    env.reset()
    # device.start_control()
    return env
    # return env, device


def main(env, rec_save_path=None):
    env.reset()
    robot = env.robots[0]
    # device.start_control()

    # Print debug information about the drawer
    env.env.print_debug_info()

    # Get the observation using the original method
    env_obs = env._get_observations()

    # Access cabinet information in the same way as the original environment
    cabinet_position = env_obs["CabinetObject_pos"]
    cabinet_orientation = quat2mat(env_obs["CabinetObject_quat"])

    # print(f"\nCabinet position from obs: {cabinet_position}")
    # print(f"Cabinet orientation matrix from obs:\n{cabinet_orientation}")
    # print(f"Initial robot joint state: {env_obs['robot0_joint_pos']}")

    # setup aicon
    component_building_functions, connection_building_functions, frame_rates = (
        get_building_functions_basic_drawer_motion(env)
    )
    components = build_components(component_building_functions)
    gripper_component = components["GripperAction"]
    gripper_velo = components["EEVelocities"]

    # Helper: map quantities to connection functions by inspecting connection function signatures
    def _connections_for_quantity(quantity_name):
        matches = []
        for comp_name, comp in components.items():
            conns = getattr(comp, "connections", {})
            for conn_key, conn in conns.items():
                try:
                    func = conn.define_implicit_connection_function()
                    code = getattr(func, "__code__", None)
                    if code is not None:
                        argcount = code.co_argcount
                        argnames = code.co_varnames[:argcount]
                    else:
                        argnames = ()
                except Exception:
                    argnames = ()
                if quantity_name in argnames:
                    matches.append((comp_name, conn_key, conn.__class__.__name__))
        return matches

    def _connections_linking_quantities(q1, q2):
        matches = []
        for comp_name, comp in components.items():
            conns = getattr(comp, "connections", {})
            for conn_key, conn in conns.items():
                try:
                    quantity_names = list(conn.connected_quantities.keys())
                except Exception:
                    quantity_names = []
                if q1 in quantity_names and q2 in quantity_names:
                    matches.append((comp_name, conn_key, conn.__class__.__name__, quantity_names))
        return matches

    def _print_connection_pairs_for_trace(trace_list):
        print("[conn-map] Mapping adjacent gradient-trace quantities to shared connections:")
        for q1, q2 in zip(trace_list, trace_list[1:]):
            found = _connections_linking_quantities(q1, q2)
            if not found:
                print(f"  {q1} -> {q2}: <no shared connection found>")
            else:
                print(f"  {q1} -> {q2}:")
                for comp_name, conn_key, cls, quantity_names in found:
                    print(f"    - component `{comp_name}` connection `{conn_key}` ({cls})")

    # Example trace (from your gradient trace). Adjust if needed.
    gradient_trace_example = [
        "ReduceJointStateDifference",
        "kinematic_joint",
        "position_drawer",
        "likelihood_grasped_drawer",
        "uncertainty_dist",
        "uncertainty_drawer",
        "pose_ee",
        "action_velo_ee",
    ]
    # _print_connection_pairs_for_trace(gradient_trace_example)

    # Start sim
    curr_t = 0
    counter = 0
    while True:
        
        run_component_sequence(components, torch.tensor(curr_t))
        curr_commanded_vel = gripper_velo.quantities["action_velo_ee"]
        curr_commanded_gripper = gripper_component.quantities["gripper_activation"]
        commanded_action = np.concatenate(
            [curr_commanded_vel.cpu().numpy(), np.zeros(3), [2 * curr_commanded_gripper.squeeze().cpu().numpy() - 1]]
        )

        # if curr_t < 1.5:
        #     action, _ = input2action(
        #         device=device, robot=robot, active_arm="right", env_configuration="single-arm-opposed"
        #     )
        # else:
        #     action = commanded_action

        action = commanded_action
        

        obs, rew, done, info = env.step(action)

        # Print current drawer position estimate from the estimator (if available)
        drawer_comp = components.get("DrawerPosEstimator")
        if drawer_comp is not None:
            drawer_pos_est = drawer_comp.quantities.get("position_drawer", None)
            drawer_uncertainty = drawer_comp.quantities.get("uncertainty_drawer", None)
            if drawer_pos_est is not None:
                try:
                    # Convert torch tensor to numpy for readable printing
                    if hasattr(drawer_pos_est, "cpu"):
                        dp = drawer_pos_est.cpu().numpy()
                    else:
                        dp = drawer_pos_est
                except Exception:
                    dp = drawer_pos_est
                # print(f"Drawer estimate at t={curr_t:.3f}: {dp}")
            if drawer_uncertainty is not None:
                try:
                    # Convert torch tensor to numpy for readable printing
                    if hasattr(drawer_uncertainty, "cpu"):
                        du = drawer_uncertainty.cpu().numpy()
                    else:
                        du = drawer_uncertainty
                    du_diag = np.diag(du)
                except Exception:
                    du_diag = drawer_uncertainty
                # print(f"Drawer uncertainty (diagonal) at t={curr_t:.3f}: {du_diag}")
        # Print camera bearing measurement and likelihoods for debugging
        bearing_comp = components.get("BearingSensor")
        if bearing_comp is not None:
            rel_pos_meas = bearing_comp.quantities.get("relative_position_in_CF_drawer", None)
            if rel_pos_meas is not None:
                try:
                    rp = rel_pos_meas.cpu().numpy()
                except Exception:
                    rp = rel_pos_meas
                # print(f"Processed bearing (sine-of-angles) at t={curr_t:.3f}: {rp}")

        vis_comp = components.get("VisibleEstimator")
        if vis_comp is not None:
            vis_like = vis_comp.quantities.get("likelihood_visible_drawer", None)
            if vis_like is not None:
                try:
                    vl = vis_like.cpu().numpy()
                except Exception:
                    vl = vis_like
                print(f"Visibility likelihood at t={curr_t:.3f}: {vl}")

        grasp_comp = components.get("GraspLikelihoodEstimator")
        if grasp_comp is not None:
            grasp_like = grasp_comp.quantities.get("likelihood_grasped_drawer", None)
            if grasp_like is not None:
                try:
                    gl = grasp_like.cpu().numpy()
                except Exception:
                    gl = grasp_like
                # print(f"Grasp likelihood: {gl}")

        # print(f"Robot joint state at t={curr_t:.3f}: {obs['robot0_joint_pos']}")

        # Print success state and joint info
        # if done:
        #     print("\n*** SUCCESS! Drawer opened. ***\n")

        curr_t += env.control_timestep
        env.render()


def run_demo():
    # Define the desired initial joint positions for the Panda robot (7 joints)
    initial_panda_qpos = np.array([-0.56, 0.76, 0.1, -1.90, 1.11, 1.5, -0.32])

    # initial_panda_qpos = None

    # Pass the initial pose to the setup function
    env = setup_env("keyboard", initial_qpos=initial_panda_qpos)
    main(env)


if __name__ == "__main__":
    run_demo()
