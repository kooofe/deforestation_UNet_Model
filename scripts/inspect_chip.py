from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deforest.data.chips_io import load_npz


def _sanitize_chip(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    y = y.astype(np.float32, copy=False)
    y_max = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
    if y_max > 1.5:
        y = (y > 127).astype(np.float32)
    else:
        y = (y > 0.5).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y, valid


def _save_rgb_images(x: np.ndarray, out_dir: Path, chip_stem: str) -> None:
    """Saves pre and post RGB images."""
    # Channel indices for pre- and post-disaster RGB
    # Sentinel-2: B4(red), B3(green), B2(blue)
    pre_rgb_indices = [2, 1, 0]   # B4, B3, B2
    post_rgb_indices = [10, 9, 8] # B4, B3, B2

    def _create_rgb(indices: list[int]) -> np.ndarray:
        rgb = x[indices, :, :]
        rgb = np.moveaxis(rgb, 0, -1)  # (C, H, W) -> (H, W, C)

        # Contrast stretching
        low, high = np.percentile(rgb, [2, 98])
        rgb = np.clip(rgb, low, high)
        rgb = (rgb - low) / (high - low)
        return rgb

    pre_rgb = _create_rgb(pre_rgb_indices)
    post_rgb = _create_rgb(post_rgb_indices)

    plt.imsave(out_dir / f"{chip_stem}_pre_rgb.png", pre_rgb)
    plt.imsave(out_dir / f"{chip_stem}_post_rgb.png", post_rgb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", required=True, help="Path to a .npz chip file")
    args = parser.parse_args()

    chip_path = Path(args.chip)
    if not chip_path.exists():
        raise SystemExit(f"Chip not found: {chip_path}")

    x, y, meta = load_npz(chip_path)
    x, y, valid = _sanitize_chip(x, y)

    unique_vals = np.unique(y)[:10]
    print(f"chip={chip_path}")
    print(f"y.unique(first10)={unique_vals.tolist()}")
    print(f"y.mean={float(y.mean()):.6f}")

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_dir / f"{chip_path.stem}_y.png", y, cmap="gray", vmin=0, vmax=1)
    plt.imsave(out_dir / f"{chip_path.stem}_valid.png", valid, cmap="gray", vmin=0, vmax=1)

    _save_rgb_images(x, out_dir, chip_path.stem)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
