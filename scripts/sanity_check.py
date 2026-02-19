from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deforest.data.chips_io import list_npz, load_npz


def _parse_idx(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("RGB index must have 3 comma-separated values.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _scale_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, None)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - p2) / (p98 - p2), 0, 1)


def _save_preview(x: np.ndarray, y: np.ndarray, pre_idx: tuple[int, int, int], post_idx: tuple[int, int, int], out_dir: Path) -> None:
    pre = np.stack([x[pre_idx[0]], x[pre_idx[1]], x[pre_idx[2]]], axis=-1)
    post = np.stack([x[post_idx[0]], x[post_idx[1]], x[post_idx[2]]], axis=-1)
    pre = _scale_rgb(pre)
    post = _scale_rgb(post)

    overlay = post.copy()
    mask = y.astype(np.float32)
    overlay[..., 0] = np.clip(overlay[..., 0] + mask * 0.7, 0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_dir / "preview_pre.png", pre)
    plt.imsave(out_dir / "preview_post.png", post)
    plt.imsave(out_dir / "preview_overlay.png", overlay)


def _first_nonfinite(x: np.ndarray) -> tuple[int, int, int, str] | None:
    mask = ~np.isfinite(x)
    if not mask.any():
        return None
    idx = np.argwhere(mask)[0]
    if x.ndim == 2:
        ch = 0
        row, col = int(idx[0]), int(idx[1])
        value = x[row, col]
    else:
        ch = int(idx[0])
        row = int(idx[1]) if idx.shape[0] > 1 else 0
        col = int(idx[2]) if idx.shape[0] > 2 else 0
        value = x[tuple(idx)]
    if np.isnan(value):
        kind = "nan"
    elif np.isposinf(value):
        kind = "posinf"
    elif np.isneginf(value):
        kind = "neginf"
    else:
        kind = "nonfinite"
    return ch, row, col, kind


def _nan_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    if not np.isfinite(arr).any():
        nan = float("nan")
        return nan, nan, nan, nan
    return (
        float(np.nanmin(arr)),
        float(np.nanmax(arr)),
        float(np.nanmean(arr)),
        float(np.nanstd(arr)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips_dir", required=True)
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--stats_channels", type=int, default=8)
    parser.add_argument("--save_preview", action="store_true")
    parser.add_argument("--preview_index", type=int, default=0)
    parser.add_argument("--preview_dir", default="outputs/preview")
    parser.add_argument("--pre_rgb_idx", default="2,1,0")
    parser.add_argument("--post_rgb_idx", default="10,9,8")
    parser.add_argument("--strict_sanity", action="store_true", default=True)
    parser.add_argument("--no_strict_sanity", dest="strict_sanity", action="store_false")
    args = parser.parse_args()

    paths = list_npz(args.chips_dir)
    if not paths:
        raise SystemExit("No chips found")

    channels = None
    pos_count = 0
    checked = 0
    pos_ratios = []
    warned = set()

    pre_idx = _parse_idx(args.pre_rgb_idx)
    post_idx = _parse_idx(args.post_rgb_idx)

    for i, p in enumerate(paths[: args.max]):
        x, y, meta = load_npz(p)
        x = x.astype(np.float32, copy=False)
        nonfinite = _first_nonfinite(x)
        if nonfinite is not None:
            ch, row, col, kind = nonfinite
            msg = f"Non-finite in x before sanitize: file={p} ch={ch} row={row} col={col} type={kind}"
            if args.strict_sanity:
                raise ValueError(msg)
            if p not in warned:
                print(f"WARNING: {msg}")
                warned.add(p)
        x_sanitized = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if channels is None:
            channels = x.shape[0]
        if x.shape[0] != channels:
            print(f"Channel mismatch: {p}")
        if x.shape[1:] != y.shape:
            print(f"Shape mismatch: {p}")

        y = y.astype(np.float32, copy=False)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y_bin = (y > 0.5).astype(np.float32)
        ratio = float(y_bin.mean())
        pos_ratios.append(ratio)
        if y_bin.sum() > 0:
            pos_count += 1
        checked += 1

        print(f"Chip: {p.name}")
        print(f"  x.shape={x.shape} y.shape={y.shape}")
        unique_vals = np.unique(y)[:10]
        print(f"  y.unique(first10)={unique_vals.tolist()}")
        print(f"  y.mean={ratio:.6f}")
        if ratio > 0.4:
            print("  WARNING: Label positive ratio unusually high; check that y is deforestation mask, not valid/SAFE mask or inverted.")
        print(f"  x.finite_after_sanitize={bool(np.isfinite(x_sanitized).all())}")
        nan_ratio_all = (~np.isfinite(x)).reshape(x.shape[0], -1).mean(axis=1) * 100.0
        nan_ratio_list = [round(float(r), 2) for r in nan_ratio_all]
        print(f"  nan_ratio_per_channel%={nan_ratio_list}")

        stats_n = min(int(args.stats_channels), x.shape[0])
        for c in range(stats_n):
            band = x[c]
            band_s = x_sanitized[c]
            nan_ratio = float((~np.isfinite(band)).mean() * 100.0)
            raw_min, raw_max, raw_mean, raw_std = _nan_stats(band)
            san_min = float(band_s.min())
            san_max = float(band_s.max())
            san_mean = float(band_s.mean())
            san_std = float(band_s.std())
            print(
                f"  ch{c:02d} nan_ratio={nan_ratio:.2f}% "
                f"raw(min={raw_min:.4f} max={raw_max:.4f} mean={raw_mean:.4f} std={raw_std:.4f}) "
                f"san(min={san_min:.4f} max={san_max:.4f} mean={san_mean:.4f} std={san_std:.4f})"
            )

        if args.save_preview and i == args.preview_index:
            _save_preview(x_sanitized, y_bin, pre_idx, post_idx, Path(args.preview_dir))
            print(f"  Preview saved to {Path(args.preview_dir)}")

    ratios = np.array(pos_ratios) if pos_ratios else np.array([0.0])
    avg_ratio = float(ratios.mean())
    med_ratio = float(np.median(ratios))

    top = sorted(zip(pos_ratios, paths[: args.max]), key=lambda t: t[0], reverse=True)[:5]

    print(f"Checked {checked} chips")
    print(f"Channels: {channels}")
    print(f"Chips with positives: {pos_count}")
    print(f"Avg positive ratio: {avg_ratio:.6f}")
    print(f"Median positive ratio: {med_ratio:.6f}")
    print("Top-5 positive ratio chips:")
    for ratio, path in top:
        print(f"  {path.name}: {ratio:.6f}")


if __name__ == "__main__":
    main()
