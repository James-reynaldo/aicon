import time
import copy
from collections import deque
from pathlib import Path

import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
import robosuite as suite
import torch

from robosuite.utils.transform_utils import quat2mat
from robosuite.wrappers import VisualizationWrapper
from experiment_specifications import OPEN_VALUE

from aicon.drawer_tutorial.experiment_specifications import get_building_functions_basic_drawer_motion

RENDER = True  # Set to True to visualize the environment

if RENDER:
    from robosuite.devices import Keyboard, SpaceMouse
    from robosuite.utils.input_utils import input2action

def render_frame(env, timestamp):
    """Render one frame when visualization is enabled for the drawer demo."""
    if RENDER:
        env.render()
# Disturbance configuration for periodic jerks
DISTURBANCE_INTERVAL = 1.0
DISTURBANCE_MAGNITUDE = 0

# Import our custom environment
from aicon.drawer_tutorial.robosuite_drawer_env import DrawerOpenEnv
from aicon.middleware.python_sequential import build_components, run_component_sequence


KINEMATIC_AXIS_LENGTH = 0.50
KINEMATIC_AXIS_INDICATOR = "kinematic_joint_axis"
KINEMATIC_DRAWER_INDICATOR = "kinematic_joint_drawer_position"
KINEMATIC_DRAWER_INITIAL = "kinematic_joint_drawer_initial_position"
EE_POS_INDICATOR = "ee_position"
DRAWER_POSITION_INDICATOR = "drawer_position"
# DEFAULT_INITIAL_PANDA_QPOS = np.array([-0.56, 0.76, 0.1, -1.90, 1.11, 1.5, -0.32])
# DEFAULT_INITIAL_PANDA_QPOS = np.array([-0.61320311,  0.85524774, -0.0179054,  -1.7072619,   1.06449885,  1.33379331, -0.28968686])
# DEFAULT_INITIAL_PANDA_QPOS = np.array([-0.59114603,  0.62355354, -0.00245896, -2.16374032,  1.05898897,  1.47348643, -0.47575451])
DEFAULT_INITIAL_PANDA_QPOS = np.array([-0.55801981,  0.83242474,  0.11796939, -1.66364752,  1.14582064,  1.44749675, -0.15419586])
# DEFAULT_INITIAL_PANDA_QPOS = np.array([-0.57798865,  0.72443188,  0.04076983, -1.99426233,  1.08883333,  1.4795042,  -0.39521093])
class GraspDiagnosticsPlotter:
    """Display the grasp-likelihood inputs and their decision thresholds live."""

    def __init__(self, grasp_connection, history_seconds=30.0):
        self.grasp_connection = grasp_connection
        self.history_seconds = history_seconds
        self.times = deque()
        self.force_magnitudes = deque()
        self.distances = deque()
        self.uncertainties = deque()
        self.gripper_activations = deque()
        self.force_thresholds = deque()

        plt.ion()
        self.figure, self.axes = plt.subplots(2, 2, num="Grasp diagnostics", figsize=(11, 7))
        self.figure.canvas.manager.set_window_title("Grasp diagnostics")
        self.figure.tight_layout(pad=3.0)

        self._series = [
            (self.axes[0, 0], "Force magnitude", "Force", "force_magnitudes", "force_thresholds", "FT_tresh"),
            (self.axes[0, 1], "EE–drawer distance", "Distance (m)", "distances", None, "close_dist_threshold"),
            (self.axes[1, 0], "Distance uncertainty", "Uncertainty", "uncertainties", None, "uncertainty_dist_threshold"),
            (self.axes[1, 1], "Gripper activation", "Activation", "gripper_activations", None, "gripper_activation_threshold"),
        ]
        self.lines = []
        for axis, title, ylabel, _, _, _ in self._series:
            axis.set_title(title)
            axis.set_xlabel("Simulation time (s)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            value_line, = axis.plot([], [], label="value")
            threshold_line, = axis.plot([], [], "--", color="tab:red", label="threshold")
            axis.legend(loc="best")
            self.lines.append((value_line, threshold_line))
        plt.show(block=False)

    @staticmethod
    def _scalar(value):
        return float(value.detach().cpu().reshape(-1)[0])

    def update(self, simulation_time, force_magnitude, distance_ee_drawer,
               uncertainty_dist, gripper_activation, time_since_hand_change):
        """Record one simulation sample and refresh the interactive plot."""
        if not plt.fignum_exists(self.figure.number):
            return

        self.times.append(float(simulation_time))
        self.force_magnitudes.append(self._scalar(force_magnitude))
        self.distances.append(self._scalar(distance_ee_drawer))
        self.uncertainties.append(self._scalar(uncertainty_dist))
        self.gripper_activations.append(self._scalar(gripper_activation))
        ft_tresh = min(
            self._scalar(time_since_hand_change[1]) * self.grasp_connection.ft_tresh_multiplier,
            self.grasp_connection.ft_tresh_cap,
        )
        self.force_thresholds.append(ft_tresh)

        while self.times[-1] - self.times[0] > self.history_seconds:
            for samples in (
                self.times, self.force_magnitudes, self.distances,
                self.uncertainties, self.gripper_activations, self.force_thresholds,
            ):
                samples.popleft()

        times = list(self.times)
        for (axis, _, _, values_name, thresholds_name, threshold_name), (value_line, threshold_line) in zip(self._series, self.lines):
            value_line.set_data(times, list(getattr(self, values_name)))
            if thresholds_name is None:
                threshold_values = [getattr(self.grasp_connection, threshold_name)] * len(times)
            else:
                threshold_values = list(getattr(self, thresholds_name))
            threshold_line.set_data(times, threshold_values)
            axis.relim()
            axis.autoscale_view()
            # Keep fixed-threshold panels focused on the decision region. The force
            # panel remains autoscaled because its FT_tresh changes over time.
            if thresholds_name is None:
                threshold = threshold_values[-1] if threshold_values else 0.0
                axis.set_ylim(0.0, max(5.0 * threshold, 1e-6))
            if len(times) > 1:
                axis.set_xlim(max(0.0, times[-1] - self.history_seconds), times[-1])

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


class KinematicJointAnglePlotter:
    """Display estimated joint parameters and their simulator references."""

    ANCHOR_Y_TARGET = 0.13032206

    def __init__(self, env, history_seconds=30.0):
        self.base_env = env.env if hasattr(env, "env") else env
        self.history_seconds = history_seconds
        self.times = deque()
        self.azimuths = deque()
        self.azimuth_refs = deque()
        self.elevations = deque()
        self.elevation_refs = deque()
        self.states = deque()
        self.state_targets = deque()
        self.anchor_ys = deque()
        self.anchor_y_targets = deque()
        self.joint_errors = deque()
        self.joint_error_targets = deque()
        self.drawer_position_errors = deque()
        self.drawer_position_error_targets = deque()

        plt.ion()
        self.figure, self.axes = plt.subplots(6, 1, num="Kinematic joint parameters", figsize=(9, 14))
        self.figure.canvas.manager.set_window_title("Kinematic joint angles")
        self.figure.tight_layout(pad=3.0)

        self._series = [
            (self.axes[0], "Azimuth", "Radians", "azimuths", "azimuth_refs"),
            (self.axes[1], "Elevation", "Radians", "elevations", "elevation_refs"),
            (self.axes[2], "Drawer state", "Position (m)", "states", "state_targets"),
            (self.axes[3], "Anchor Y", "Position (m)", "anchor_ys", "anchor_y_targets"),
            (self.axes[4], "Joint error", "Estimate − true (m)", "joint_errors", "joint_error_targets"),
            (self.axes[5], "Drawer position error", "Euclidean error (m)", "drawer_position_errors", "drawer_position_error_targets"),
        ]
        self.lines = []
        for axis, title, ylabel, _, _ in self._series:
            axis.set_title(title)
            axis.set_xlabel("Simulation time (s)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            value_line, = axis.plot([], [], label="estimate")
            ref_line, = axis.plot([], [], "--", color="tab:red", label="target")
            axis.legend(loc="best")
            self.lines.append((value_line, ref_line))

        plt.show(block=False)

    @staticmethod
    def _safe_scalar(value):
        return float(value.detach().cpu().reshape(-1)[0]) if hasattr(value, "detach") else float(value)

    def _orientation_to_az_el(self, orientation):
        axis = np.asarray(orientation, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return None, None
        axis = axis / norm
        azimuth = np.arctan2(axis[1], axis[0])
        elevation = np.arccos(np.clip(axis[2], -1.0, 1.0))
        return azimuth, elevation

    def update(self, simulation_time, estimated_joint, drawer_orientation,
               estimated_drawer_position=None):
        if not plt.fignum_exists(self.figure.number):
            return

        self.times.append(float(simulation_time))
        estimate = self._safe_scalar(estimated_joint[0])
        az_el = self._safe_scalar(estimated_joint[1]) if len(estimated_joint) > 1 else 0.0
        state = self._safe_scalar(estimated_joint[2]) if len(estimated_joint) > 2 else 0.0
        anchor_y = self._safe_scalar(estimated_joint[4]) if len(estimated_joint) > 4 else 0.0
        self.azimuths.append(estimate)
        self.elevations.append(az_el)
        self.states.append(state)
        target_state = OPEN_VALUE
        self.state_targets.append(target_state)
        self.anchor_ys.append(anchor_y)
        self.anchor_y_targets.append(self.ANCHOR_Y_TARGET)
        drawer_handle_pos = self.base_env.get_drawer_handle_pos()
        true_joint = float(self.base_env.sim.data.qpos[self.base_env.cabinet_qpos_addrs])
        self.joint_errors.append(state + true_joint)
        print(f"Estimated joint state: {state}, True joint state: {true_joint}, Joint error: {state - true_joint}")
        self.joint_error_targets.append(0.0)
        if estimated_drawer_position is None:
            drawer_position_error = float("nan")
        else:
            estimate_position = np.asarray(
                estimated_drawer_position.detach().cpu().numpy()
                if hasattr(estimated_drawer_position, "detach") else estimated_drawer_position,
                dtype=np.float64,
            ).reshape(-1)[:3]
            drawer_position_error = float(
                np.linalg.norm(estimate_position - np.asarray(drawer_handle_pos, dtype=np.float64).reshape(-1)[:3])
            )
        self.drawer_position_errors.append(drawer_position_error)
        self.drawer_position_error_targets.append(0.0)

        if drawer_orientation is not None:
            actual_azimuth, actual_elevation = self._orientation_to_az_el(drawer_orientation)
        else:
            actual_azimuth, actual_elevation = None, None

        self.azimuth_refs.append(actual_azimuth if actual_azimuth is not None else self.azimuths[-1])
        self.elevation_refs.append(actual_elevation if actual_elevation is not None else self.elevations[-1])

        while self.times[-1] - self.times[0] > self.history_seconds:
            for samples in (
                self.times, self.azimuths, self.azimuth_refs,
                self.elevations, self.elevation_refs, self.states, self.state_targets,
                self.anchor_ys, self.anchor_y_targets,
                self.joint_errors, self.joint_error_targets,
                self.drawer_position_errors, self.drawer_position_error_targets,
            ):
                samples.popleft()

        times = list(self.times)
        for (axis, _, _, values_name, refs_name), (value_line, ref_line) in zip(self._series, self.lines):
            values = list(getattr(self, values_name))
            refs = list(getattr(self, refs_name))
            value_line.set_data(times, values)
            ref_line.set_data(times, refs)
            if values_name == "anchor_ys":
                axis.set_ylim(0.0, 0.2)
            elif values_name == "drawer_position_errors":
                axis.set_ylim(0.0, 0.1)
            axis.relim()
            axis.autoscale_view()
            if len(times) > 1:
                axis.set_xlim(max(0.0, times[-1] - self.history_seconds), times[-1])

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


class KinematicJointVisualizer:
    """Draw the estimated joint axis as a pole extending from the drawer."""

    def __init__(self, env):
        self.env = env
        self.base_env = env.env if hasattr(env, "env") else env

    def update(self, kinematic_joint):
        """Update the in-scene arrow from the estimator's azimuth and elevation."""
        joint = kinematic_joint.detach().cpu().numpy()
        azimuth, elevation = joint[:2]
        state = joint[2]
        initial_point = joint[3:6]
        axis = np.array([
            np.cos(azimuth) * np.sin(elevation),
            np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation),
        ])

        # The kinematic model's predicted point is the drawer position. The capsule's
        # midpoint is offset so its near end starts at that point and sticks outward.
        drawer_position = initial_point + axis * state
        self.env.set_indicator_pos(KINEMATIC_DRAWER_INITIAL, initial_point)
        # self.env.set_indicator_pos(KINEMATIC_DRAWER_INDICATOR, drawer_position)
        self.env.set_indicator_pos(
            KINEMATIC_AXIS_INDICATOR,
            drawer_position + axis * (KINEMATIC_AXIS_LENGTH / 2),
        )
        axis_body_id = self.base_env.sim.model.body_name2id(KINEMATIC_AXIS_INDICATOR + "_body")
        self.base_env.sim.model.body_quat[axis_body_id] = self._z_axis_quaternion(axis)
        self.base_env.sim.forward()

    @staticmethod
    def _z_axis_quaternion(direction):
        """Return MuJoCo's wxyz quaternion rotating +z onto ``direction``."""
        z_axis = direction / np.linalg.norm(direction)
        helper = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(helper, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        trace = np.trace(rotation)
        if trace > 0:
            scale = 2 * np.sqrt(trace + 1.0)
            return np.array([0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                             (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale])
        index = np.argmax(np.diag(rotation))
        next_index, final_index = (index + 1) % 3, (index + 2) % 3
        scale = 2 * np.sqrt(1.0 + rotation[index, index] - rotation[next_index, next_index] - rotation[final_index, final_index])
        quaternion = np.zeros(4)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (rotation[final_index, next_index] - rotation[next_index, final_index]) / scale
        quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / scale
        quaternion[final_index + 1] = (rotation[final_index, index] + rotation[index, final_index]) / scale
        return quaternion


class EEPosVisualizer:
    """Draw the estimated ee pos as a sphere."""

    def __init__(self, env):
        self.env = env
        self.base_env = env.env if hasattr(env, "env") else env

    def update(self, ee):
        """Update the in-scene arrow from the estimator's azimuth and elevation."""
        # ``pose_ee`` is [x, y, z, rx, ry, rz]; the marker only has a position.
        ee_pos = ee.detach().cpu().numpy()[:3]

        # self.env.set_indicator_pos(EE_POS_INDICATOR, ee_pos)

        self.base_env.sim.forward()

class DrawerPositionVisualizer:
    """Draw the estimated drawer pos as a sphere."""

    def __init__(self, env):
        self.env = env
        self.base_env = env.env if hasattr(env, "env") else env

    def update(self, drawer_pos):
        """Update the in-scene arrow from the estimator's azimuth and elevation."""
        # ``pose_ee`` is [x, y, z, rx, ry, rz]; the marker only has a position.
        drawer_pos = drawer_pos.detach().cpu().numpy()[:3]

        self.env.set_indicator_pos(DRAWER_POSITION_INDICATOR, drawer_pos)

        self.base_env.sim.forward()

def create_demo_visualizers(env, components, visualize_kinematic_angles=True,
                            visualize_ee=True, visualize_grasp_diagnostics=True, visualize_drawer_position=True):
    """Create the optional visualizations shared by the demo and sweep runners."""
    grasp_estimator = components["GraspLikelihoodEstimator"]
    return {
        "joint": KinematicJointVisualizer(env) if visualize_kinematic_angles else None,
        "joint_angles": KinematicJointAnglePlotter(env) if visualize_kinematic_angles else None,
        "ee": EEPosVisualizer(env) if visualize_ee else None,
        "grasp": (
            GraspDiagnosticsPlotter(grasp_estimator.connections["GraspedLikelihood"])
            if visualize_grasp_diagnostics else None
        ),
        "drawer_position": DrawerPositionVisualizer(env) if visualize_drawer_position else None
    }


def update_demo_visualizers(visualizers, components, simulation_time, drawer_orientation=None):
    """Refresh all enabled demo visualizations for one estimator update."""
    joint_visualizer = visualizers["joint"]
    if joint_visualizer is not None:
        joint_visualizer.update(
            components["KinematicJointEstimator"].quantities["kinematic_joint"]
        )

    joint_angles_plotter = visualizers["joint_angles"]
    if joint_angles_plotter is not None:
        joint_angles_plotter.update(
            simulation_time,
            components["KinematicJointEstimator"].quantities["kinematic_joint"],
            drawer_orientation,
            components["DrawerPosEstimator"].quantities["position_drawer"],
        )

    drawer_position_visualizer = visualizers["drawer_position"]
    if drawer_position_visualizer is not None:
        drawer_position_visualizer.update(
            components["DrawerPosEstimator"].quantities["position_drawer"]
        )

    ee_visualizer = visualizers["ee"]
    if ee_visualizer is not None:
        ee_visualizer.update(components["EEPoseEstimator"].quantities["pose_ee"])

    grasp_plotter = visualizers["grasp"]
    if grasp_plotter is not None:
        grasp_plotter.update(
            simulation_time,
            components["EEForceSensor"].quantities["ee_force_mag_meas"],
            components["DistanceEstimator"].quantities["distance_ee_drawer"][0],
            components["DistanceEstimator"].quantities["uncertainty_dist"],
            components["GripperAction"].quantities["gripper_activation"],
            components["GraspLikelihoodEstimator"].quantities["time_since_hand_change"],
        )

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
        render_camera="sideview", #'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
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
            {"name": DRAWER_POSITION_INDICATOR, "type": "sphere", "size": [0.01], "rgba": [0, 1, 0, 0.95]} #green
        ],
    )
    env.reset()
    if RENDER:
        device.start_control()
    # return env
    if RENDER:
        return env, device
    else:
        return env, None


def run_trial(env,device, estimator_params=None, max_timesteps=None, *, stop_on_done=False,
              reset_on_start=True, random_init_time=0.0, random_std=0.1,
              random_disturbance=None, periodic_disturbance_interval=None,
              periodic_disturbance_magnitude=0.0, noise_scale=0.0,
              visualize_kinematic_angles=True, visualize_ee=True,
              visualize_drawer_position=True, visualize_grasp_diagnostics=True,
              render=None, status_label=None):
    """Shared drawer-control loop for the interactive demo and parameter sweeps."""
    if reset_on_start:
        env.reset()
    robot = env.robots[0]
    params = copy.deepcopy(estimator_params) if isinstance(estimator_params, dict) else estimator_params
    connection_params = params.pop("connection_params", None) if isinstance(params, dict) else None
    builders, _, _ = get_building_functions_basic_drawer_motion(
        env, estimator_params=params, connection_params=connection_params, noise_scale=noise_scale,
    )
    components = build_components(builders)
    gripper = components["GripperAction"]
    velocities = components["EEVelocities"]
    kinematic = components["KinematicJointEstimator"]
    grasp = components["GraspLikelihoodEstimator"]
    render = RENDER if render is None else render
    visualizers = create_demo_visualizers(
        env, components, visualize_kinematic_angles=visualize_kinematic_angles,
        visualize_ee=visualize_ee, visualize_drawer_position=visualize_drawer_position,
        visualize_grasp_diagnostics=visualize_grasp_diagnostics,
    ) if render else None

    curr_t, step_idx = 0.0, 1
    joint_error = grasp_estimate = float("nan")
    actual_grasped = False
    while True:
        run_component_sequence(components, torch.tensor(curr_t))
        velocity = velocities.quantities["action_velo_ee"].cpu().numpy()
        gripper_action = gripper.quantities["gripper_activation"].squeeze().cpu().numpy()
        action = np.concatenate([velocity, np.zeros(3), [2 * gripper_action - 1]])
        if curr_t < random_init_time:
            action = np.concatenate([np.random.uniform(-random_std, random_std, 3), np.zeros(3), [0]])
        if random_disturbance is not None and np.random.rand() < 0.1:
            action[:3] += np.random.uniform(-random_disturbance, random_disturbance, 3)
        if periodic_disturbance_interval is not None and (
            int(curr_t / periodic_disturbance_interval)
            != int((curr_t - env.control_timestep) / periodic_disturbance_interval)
        ):
            action[:3] += np.random.uniform(
                -periodic_disturbance_magnitude, periodic_disturbance_magnitude, 3
            )

        # action, _ = input2action(
        #     device=device, robot=robot, active_arm="right", env_configuration="single-arm-opposed"
        # )
        obs, reward, done, _ = env.step(action)
        base_env = env.env if hasattr(env, "env") else env
        true_joint = float(base_env.sim.data.qpos[base_env.cabinet_qpos_addrs])
        estimated_joint = float(kinematic.quantities["kinematic_joint"][2].item())
        joint_error = estimated_joint + true_joint
        grasp_estimate = float(grasp.quantities["likelihood_grasped_drawer"].item())
        actual_grasped = bool(np.asarray(obs.get("robot0_contact", base_env._has_gripper_contact)).item())
        if visualizers is not None:
            update_demo_visualizers(visualizers, components, curr_t, obs.get("drawer_orientation"))
        if status_label is not None:
            limit = max_timesteps if max_timesteps is not None else "inf"
            print(f"  [{status_label}] step {step_idx:04d}/{limit}: est={estimated_joint:.6f}, "
                  f"true={true_joint:.6f}, err={joint_error:+.6f}, "
                  f"grasp_est={grasp_estimate:.6f}, grasped={actual_grasped}", flush=True)
        if stop_on_done and (done or reward == 1.0):
            return True, step_idx, joint_error, grasp_estimate, actual_grasped
        curr_t += env.control_timestep

        #add time label to mujoco simulation

        if render:
            render_frame(env, curr_t)
        step_idx += 1
        if max_timesteps is not None and step_idx > max_timesteps:
            return False, max_timesteps, joint_error, grasp_estimate, actual_grasped

        # Get current joint positions
        panda_qpos = robot._joint_positions
        print(f"Current joint positions: {panda_qpos}")


def main(device, env, **kwargs):
    """Run the interactive demo through the shared trial runner."""
    if device is not None:
        device.start_control()
    return run_trial(
        env, device, stop_on_done=False, periodic_disturbance_interval=DISTURBANCE_INTERVAL,
        periodic_disturbance_magnitude=DISTURBANCE_MAGNITUDE, **kwargs,
    )


def run_demo():
    env, device = setup_env("keyboard", initial_qpos=DEFAULT_INITIAL_PANDA_QPOS)
    main(device, env)


if __name__ == "__main__":
    run_demo()
