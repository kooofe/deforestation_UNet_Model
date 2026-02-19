from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from deforest.models.unet import UNet


def load_model(checkpoint: str | Path, in_channels: int, base_channels: int, device: str) -> UNet:
    model = UNet(in_channels=in_channels, base_channels=base_channels)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def predict_chip(model: UNet, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        logits = model(t)
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    return probs
