"""
Utility functions for the drawer experiment.
This module provides helper functions for processing sensor data and calculating likelihoods
in the drawer manipulation environment.
"""

import torch

from aicon.inference.util import gradient_preserving_clipping
from aicon.math.util_3d import homogeneous_transform_inverse


def pose_vec_to_homogeneous(pose_ee: torch.Tensor) -> torch.Tensor:
    """Convert a 6D pose [x, y, z, ax, ay, az] to a 4x4 homogeneous matrix.

    The first 3 entries are translation in world frame and the last 3 entries
    are a rotation vector (axis-angle, i.e. axis * angle in radians).
    """
    t = pose_ee[:3]
    rotvec = pose_ee[3:]
    theta = torch.norm(rotvec)
    eps = torch.tensor(1e-10, dtype=pose_ee.dtype, device=pose_ee.device)

    I = torch.eye(3, dtype=pose_ee.dtype, device=pose_ee.device)
    theta_safe = torch.clamp(theta, min=eps)
    k = rotvec / theta_safe
    kx, ky, kz = k[0], k[1], k[2]
    zero = torch.zeros((), dtype=pose_ee.dtype, device=pose_ee.device)
    K = torch.stack(
        [
            torch.stack([zero, -kz, ky]),
            torch.stack([kz, zero, -kx]),
            torch.stack([-ky, kx, zero]),
        ],
        dim=0,
    )
    R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * torch.mm(K, K)

    upper = torch.cat([R, t.unsqueeze(1)], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=pose_ee.dtype, device=pose_ee.device)
    return torch.cat([upper, bottom], dim=0)


def get_sine_of_angles(relative_pos: torch.Tensor) -> torch.Tensor:
    """
    Calculate the sine of angles from a relative position vector.
    
    Args:
        relative_pos: 3D position vector [x, y, z]
    
    Returns:
        torch.Tensor: Vector of sine values for each angle [sin(ax), sin(ay), sin(az)]
    """
    sin_ax = relative_pos[0] / torch.norm(relative_pos)
    sin_ay = relative_pos[1] / torch.norm(relative_pos)
    sin_az = relative_pos[2] / torch.norm(relative_pos)
    only_angles_sin = torch.concatenate([sin_ax.unsqueeze(0), sin_ay.unsqueeze(0), sin_az.unsqueeze(0)],
                                        dim=0)
    return only_angles_sin


def likelihood_func_visible(pose_ee: torch.Tensor, position_drawer: torch.Tensor,
                          H_ee_to_cam: torch.Tensor, steepness: float = 15.0) -> torch.Tensor:
    """
    Calculate the likelihood that the drawer is visible in the camera frame.
    This function considers the field of view constraints of the camera.
    
    Args:
        pose_ee: End-effector pose [x, y, z, ax, ay, az] (axis-angle rotation vector)
        position_drawer: Drawer position in world coordinates
        H_ee_to_cam: Homogeneous transform from end-effector to camera frame
        steepness: Steepness parameter for the error function (default: 15.0)
    
    Returns:
        torch.Tensor: Likelihood value between 0 and 1
    """
    H_world_to_ee = homogeneous_transform_inverse(pose_vec_to_homogeneous(pose_ee))
    # print(f"util pose_ee: {pose_ee}")
    # print(f"util position_drawer: {position_drawer}")
    # print(f"util world subtraction (drawer - ee_xyz): {relative_world}")
    relative_pos = torch.einsum("ki,ij,j->k",
                                homogeneous_transform_inverse(H_ee_to_cam),
                                H_world_to_ee,
                                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype,
                                                                       device=position_drawer.device)]))[:3]
    # print(f"util relative_pos: {relative_pos}")
    # determine angle from principal camera axis to current relative vector
    if relative_pos[2] <= 0:
        angle_x = torch.pi - torch.abs(torch.arcsin(relative_pos[0] / torch.norm(relative_pos[[0, 2]])))
        angle_y = torch.pi - torch.abs(torch.arcsin(relative_pos[1] / torch.norm(relative_pos[[1, 2]])))
    else:
        angle_x = torch.abs(torch.arcsin(relative_pos[0] / torch.norm(relative_pos[[0, 2]])))
        angle_y = torch.abs(torch.arcsin(relative_pos[1] / torch.norm(relative_pos[[1, 2]])))
    # now apply error function depending on how far in/out of FOV it is
    # 0.6 = 69.4 / 180 * PI / 2.0 with 69.4° FOV in x
    # 0.37 = 42.5 / 180 * PI / 2.0 with 42.5° FOV in y
    # clipping 0.1 above to not go into 0 area of the error function
    angle_x = gradient_preserving_clipping(angle_x, 0.0, 0.57)
    likelihood_x = 0.5 - 0.5 * torch.erf((angle_x - 0.37) * steepness)
    angle_y = gradient_preserving_clipping(angle_y, 0.0, 0.8)
    likelihood_y = 0.5 - 0.5 * torch.erf((angle_y - 0.6) * steepness)
    likelihood = likelihood_x * likelihood_y
    # print(f"util angle_x: {angle_x}, likelihood_x: {likelihood_x}")
    # print(f"util angle_y: {angle_y}, likelihood_y: {likelihood_y}")
    return likelihood


def likelihood_dist_func(pose_ee: torch.Tensor, position_drawer: torch.Tensor,
                        H_ee_to_cam: torch.Tensor) -> torch.Tensor:
    """
    Calculate the likelihood based on the distance between the end-effector and drawer.
    
    Args:
        pose_ee: End-effector pose [x, y, z, ax, ay, az] (axis-angle rotation vector)
        position_drawer: Drawer position in world coordinates
        H_ee_to_cam: Homogeneous transform from end-effector to camera frame
    
    Returns:
        torch.Tensor: Likelihood value between 0 and 1
    """
    H_world_to_ee = homogeneous_transform_inverse(pose_vec_to_homogeneous(pose_ee))
    relative_pos = torch.einsum("ki,ij,j->k",
                                homogeneous_transform_inverse(H_ee_to_cam),
                                H_world_to_ee,
                                torch.cat([position_drawer, torch.ones(1, dtype=position_drawer.dtype,
                                                                       device=position_drawer.device)]))[:3]
    likelihood = torch.exp(-torch.pow(torch.norm(relative_pos) - 0.5, 2) / 2 / 0.1)
    return likelihood