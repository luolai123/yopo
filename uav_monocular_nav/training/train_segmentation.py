#!/usr/bin/env python3
"""Training script for segmentation network."""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models import SegmentationNet
from dataset_tools import SegmentationDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Dataset root containing images/masks")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=str, default="segmentation.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToTensor(),
    ])
    dataset = SegmentationDataset(args.data, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = SegmentationNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"Saved segmentation model to {args.save}")


if __name__ == "__main__":
    main()
