#!/usr/bin/env python3
"""
Model definitions for segmentation and planner networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, 1, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SegmentationNet(nn.Module):
    """Lightweight U-Net style encoder-decoder."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, s=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, s=2)

        self.dec2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.out_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        y = self.dec2(x3)
        y = F.relu(y + x2)
        y = self.dec1(y)
        y = F.relu(y + x1)
        logits = self.out_head(y)
        return torch.sigmoid(logits)


class PlannerNet(nn.Module):
    """Predict terminal state from image features."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 9),
        )

    def forward(self, image):
        features = self.backbone(image)
        params = self.head(features)
        delta_p_raw = params[:, 0:3]
        y_v = params[:, 3:6]
        y_a = params[:, 6:9]
        return delta_p_raw, y_v, y_a


def terminal_state_from_net(delta_p_raw, y_v, y_a, p0, r_max=3.0, v_scale=2.0, a_scale=2.0):
    delta_p = torch.tanh(delta_p_raw) * r_max
    p_T = p0 + delta_p
    v_T = torch.tanh(y_v) * v_scale
    a_T = torch.tanh(y_a) * a_scale
    r = torch.norm(delta_p, dim=-1, keepdim=True) + 1e-6
    T = 2.0 * r / v_scale
    return p_T, v_T, a_T, T.squeeze(-1)
