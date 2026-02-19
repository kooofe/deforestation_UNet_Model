from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _scale_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, None)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - p2) / (p98 - p2), 0, 1)


def save_panel(pre_rgb: np.ndarray, post_rgb: np.ndarray, mask: np.ndarray, out_path: str | Path) -> None:
    pre = _scale_rgb(pre_rgb)
    post = _scale_rgb(post_rgb)

    overlay = post.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask * 0.7, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pre)
    axes[0].set_title("PRE RGB")
    axes[1].imshow(post)
    axes[1].set_title("POST RGB")
    axes[2].imshow(overlay)
    axes[2].set_title("DETECTION")
    for ax in axes:
        ax.axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
