from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3)
    num = 2 * (probs * targets).sum(dims)
    den = (probs + targets).sum(dims) + eps
    return 1 - (num / den).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.bce(logits, targets) + dice_loss(logits, targets)
