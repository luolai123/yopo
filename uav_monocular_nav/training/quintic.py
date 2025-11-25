#!/usr/bin/env python3
"""
Quintic polynomial trajectory generator using torch.
"""
import torch


def solve_quintic(p0, v0, a0, pT, vT, aT, T, num_samples=50):
    """Solve 1D quintic coefficients and sample trajectory.

    Args:
        p0, v0, a0: (...,) tensors
        pT, vT, aT: (...,) tensors
        T: scalar or tensor of shape (...,)
    Returns:
        dict with pos, vel, acc, jerk, t
    """
    T = T.unsqueeze(-1) if T.dim() == p0.dim() else T
    t_vec = torch.linspace(0.0, 1.0, steps=num_samples, device=p0.device)
    t_samples = t_vec * T

    M = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 2, 6, 12, 20],
    ], dtype=torch.float32, device=p0.device)

    boundary = torch.stack([p0, v0, a0, pT, vT, aT], dim=-1)
    coeffs = torch.linalg.solve(M, boundary)
    a0_, a1, a2, a3, a4, a5 = [c.unsqueeze(-1) for c in coeffs.unbind(-1)]

    powers = torch.stack([t_samples ** i for i in range(6)], dim=-1)
    pos = (coeffs.unsqueeze(-2) * powers).sum(-1)

    vel_powers = torch.stack([
        torch.zeros_like(t_samples),
        torch.ones_like(t_samples),
        2 * t_samples,
        3 * t_samples ** 2,
        4 * t_samples ** 3,
        5 * t_samples ** 4,
    ], dim=-1)
    vel = (coeffs.unsqueeze(-2) * vel_powers).sum(-1)

    acc_powers = torch.stack([
        torch.zeros_like(t_samples),
        torch.zeros_like(t_samples),
        2 * torch.ones_like(t_samples),
        6 * t_samples,
        12 * t_samples ** 2,
        20 * t_samples ** 3,
    ], dim=-1)
    acc = (coeffs.unsqueeze(-2) * acc_powers).sum(-1)

    jerk = torch.diff(acc, dim=-1, prepend=acc[..., :1]) / (t_samples[1] - t_samples[0])
    return {
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "jerk": jerk,
        "t": t_samples,
    }
