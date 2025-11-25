#!/usr/bin/env python3
"""Loss functions for mask-based planning."""
import torch
import torch.nn.functional as F


def project_to_image(points, camera_info):
    """Project 3D points in camera frame to pixel coordinates.

    Args:
        points: (B, N, 3) points in camera frame.
        camera_info: dict with fx, fy, cx, cy
    Returns:
        normalized grid coords for grid_sample: (B, N, 2)
    """
    fx, fy = camera_info["fx"], camera_info["fy"]
    cx, cy = camera_info["cx"], camera_info["cy"]
    u = points[..., 0] * fx / points[..., 2] + cx
    v = points[..., 1] * fy / points[..., 2] + cy
    # normalize to [-1,1]
    w = camera_info["width"]
    h = camera_info["height"]
    u_norm = (u / (w - 1)) * 2 - 1
    v_norm = (v / (h - 1)) * 2 - 1
    return torch.stack([u_norm, v_norm], dim=-1)


def compute_J_c_mask(mask_safe_pred, projected_points, dt):
    """Collision cost using predicted mask."""
    danger = 1.0 - mask_safe_pred
    grid = projected_points.view(projected_points.shape[0], -1, 1, 2)
    samples = F.grid_sample(danger, grid, align_corners=True)
    samples = samples.view(projected_points.shape[0], -1)
    return (samples * dt).sum(dim=1).mean()


def compute_J_s(jerk, dt):
    return (jerk.pow(2).sum(dim=-1) * dt).mean()


def compute_J_g(p_T, goal):
    return ((p_T - goal) ** 2).sum(dim=-1).mean()
