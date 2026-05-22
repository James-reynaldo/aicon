import torch

from aicon.drawer_tutorial.actions import GripperAction, VeloEEAction
from aicon.drawer_tutorial.connections import build_connections
from aicon.drawer_tutorial.estimators import (
    DrawerPositionEstimator,
    EEPoseEstimator,
    GraspedEstimator,
    VisibleEstimator,
    KinematicJointEstimator,
)
from aicon.drawer_tutorial.goals import DrawerOpenViaJointGoal
from aicon.drawer_tutorial.sensors import BearingSensor, DrawerPoseSenser, EEForceSensor, EEPoseSensor


def get_building_functions_basic_drawer_motion(sim_env_pointer, estimator_params=None, connection_params=None):
    """
    Builds the connections and components for the basic drawer motion experiment.
    """
    estimator_params = estimator_params or {}
    connection_builders = build_connections(connection_params)
    component_builders = {
        "GripperAction": lambda mockbuild: GripperAction("GripperAction",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("GraspedLikelihood",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        ),
        "EEForceSensor": lambda mockbuild: EEForceSensor("EEForceSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("GraspedLikelihood",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "EEPosSensor": lambda mockbuild: EEPoseSensor("EEPosSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("DirectMeasurement",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "DrawerPosSensor": lambda mockbuild: DrawerPoseSenser("DrawerPosSensor",
                                                        connections={k: connection_builders[k] for k in
                                                                     ("DrawerDirectMeasurement",)},
                                                        device=torch.device("cpu"),
                                                        dtype=torch.double, mockbuild=mockbuild,
                                                        sim_env_pointer=sim_env_pointer),
        "BearingSensor": lambda mockbuild: BearingSensor("BearingSensor",
                                connections={k: connection_builders[k] for k in
                                         ("ProjectiveGeometry",)},
                                device=torch.device("cpu"),
                                dtype=torch.double, mockbuild=mockbuild,
                                sim_env_pointer=sim_env_pointer),
        "EEVelocities": lambda mockbuild: VeloEEAction("EEVelocities",
                                                       connections={k: connection_builders[k] for k in
                                                                    ("ForwardKinematics",)},
                                                       device=torch.device("cpu"),
                                                       dtype=torch.double, mockbuild=mockbuild,
                                                       ),
        "EEPoseEstimator": lambda mockbuild: EEPoseEstimator("EEPoseEstimator",
                                     connections={k: connection_builders[k] for k in
                                          ("ForwardKinematics",
                                           "DirectMeasurement",
                                           "ProjectiveGeometry",
                                           "GraspedDrawerKinematics",
                                           "GraspedLikelihood",
                                           "VisibleLikelihood",
                                           )},
                                                             device=torch.device("cpu"),
                                                             dtype=torch.double, mockbuild=mockbuild,
                                                             **estimator_params.get("ee_pose", {})),
                "DrawerPosEstimator": lambda mockbuild: DrawerPositionEstimator("DrawerPosEstimator",
                                                                                connections={k: connection_builders[k] for k in
                                                                                                 (
                                                                                                        #  "DrawerDirectMeasurement",
                                                                                                    "ProjectiveGeometry",
                                                                                                    "GraspedDrawerKinematics",
                                                                                                    "DrawerKinematics",
                                                                                                    "VisibleLikelihood",
                                                                                                 )},
                                                                        device=torch.device("cpu"),
                                                                        dtype=torch.double, mockbuild=mockbuild,
                                                                        **estimator_params.get("drawer_position", {}),
                                                                        ),
        "GraspLikelihoodEstimator": lambda mockbuild: GraspedEstimator("GraspLikelihoodEstimator",
                                                                       connections={k: connection_builders[k] for k in
                                                                                    ("GraspedDrawerKinematics",
                                                                                     "GraspedLikelihood",
                                                                                     "DrawerKinematics",
                                                                                     )},
                                                                       dtype=torch.double, mockbuild=mockbuild,
                                                                       **estimator_params.get("grasp_likelihood", {}),
                                                                       ),
        "VisibleEstimator": lambda mockbuild: VisibleEstimator("VisibleEstimator",
                                       connections={k: connection_builders[k] for k in
                                            ("VisibleLikelihood",)},
                                       device=torch.device("cpu"),
                                       dtype=torch.double, mockbuild=mockbuild,
                                       **estimator_params.get("visible", {})),
        "KinematicJointEstimator": lambda mockbuild: KinematicJointEstimator("KinematicJointEstimator",
                                                                         connections={k: connection_builders[k] for k in
                                                                                      ("DrawerKinematics",
                                                                                       )},
                                                                         dtype=torch.double, mockbuild=mockbuild,
                                                                         goals={"ReduceJointStateDifference": lambda d, t,
                                                                                                            mockbuild=False: DrawerOpenViaJointGoal(
                                                                             is_active=True, dtype=t, device=d,
                                                                             mockbuild=mockbuild, open_value=-0.03)},
                                                                         **estimator_params.get("kinematic_joint", {}),
                                                                         )
    }
    frame_rates = {
        "GripperAction": 10,
        "EEForceSensor": 10,
        "EEPosSensor": 10,
        "DrawerPosSensor": 10,
        "BearingSensor": 10,
        "EEVelocities": 10,
        "EEPoseEstimator": 10,
        "DrawerPosEstimator": 10,
        "GraspLikelihoodEstimator": 10,
        "VisibleEstimator": 10,
        "KinematicJointEstimator": 10,
    }
    return component_builders, connection_builders, frame_rates