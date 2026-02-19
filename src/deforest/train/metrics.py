from __future__ import annotations

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / (den + 1e-6)


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "dice": dice.item(),
        "iou": iou.item(),
    }
