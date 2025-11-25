#!/usr/bin/env python3
"""Dataset helpers for training segmentation and planner networks."""
import json
import os
from dataclasses import dataclass
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    image_path: str
    mask_path: str
    state_path: str
    goal_path: str


class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(root_dir, "images")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".png", ".png")
        image = Image.open(os.path.join(self.root_dir, "images", img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.root_dir, "masks", mask_name)).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = (mask > 127).float()
        return image, mask


class PlannerDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(root_dir, "images")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, "images", img_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        state = json.load(open(os.path.join(self.root_dir, "states", img_name.replace(".png", ".json"))))
        goal = json.load(open(os.path.join(self.root_dir, "goals", img_name.replace(".png", ".json"))))
        p0 = torch.tensor(state["position"], dtype=torch.float32)
        v0 = torch.tensor(state["velocity"], dtype=torch.float32)
        a0 = torch.tensor(state.get("acceleration", [0, 0, 0]), dtype=torch.float32)
        goal_vec = torch.tensor(goal["goal"], dtype=torch.float32)
        return image, p0, v0, a0, goal_vec
