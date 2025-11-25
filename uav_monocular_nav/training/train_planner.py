#!/usr/bin/env python3
"""Planner network training script."""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models import PlannerNet, terminal_state_from_net
from dataset_tools import PlannerDataset
from quintic import solve_quintic
from losses import compute_J_c_mask, compute_J_s, compute_J_g, project_to_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="planner.pth")
    parser.add_argument("--lambda_s", type=float, default=0.1)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = PlannerDataset(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = PlannerNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dummy camera model parameters
    camera_info = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}

    for epoch in range(args.epochs):
        total_loss = 0.0
        for image, p0, v0, a0, goal in loader:
            image = image.to(device)
            p0 = p0.to(device)
            v0 = v0.to(device)
            a0 = a0.to(device)
            goal = goal.to(device)

            delta_p_raw, y_v, y_a = model(image)
            p_T, v_T, a_T, T_pred = terminal_state_from_net(delta_p_raw, y_v, y_a, p0)

            traj = solve_quintic(p0, v0, a0, p_T, v_T, a_T, T_pred)
            dt = T_pred.unsqueeze(-1) / traj["t"].shape[-1]

            # project positions
            points_cam = traj["pos"].permute(0, 2, 1)  # (B, 3, N)
            points_cam = points_cam.permute(0, 2, 1)
            grid = project_to_image(points_cam, camera_info)
            mask_safe_pred = torch.ones((image.shape[0], 1, 480, 640), device=device)

            J_c = compute_J_c_mask(mask_safe_pred, grid, dt)
            J_s = compute_J_s(traj["jerk"], dt)
            J_g = compute_J_g(p_T, goal)
            loss = args.lambda_s * J_s + args.lambda_c * J_c + args.lambda_g * J_g

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"Saved planner model to {args.save}")


if __name__ == "__main__":
    main()
