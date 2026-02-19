from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np


def binarize_label(y: np.ndarray) -> np.ndarray:
    if np.isfinite(y).any() and float(np.nanmax(y)) > 1.5:
        return (y > 127).astype(np.uint8)
    return (y > 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val splits from existing chips after filtering no-data chips."
    )
    parser.add_argument("--chips_dir", default="data/chips")
    parser.add_argument("--train_out", default="data/chips/train_valid.txt")
    parser.add_argument("--val_out", default="data/chips/val_valid.txt")
    parser.add_argument("--min_valid_frac", type=float, default=0.10)
    parser.add_argument("--min_pos_frac", type=float, default=0.0)
    parser.add_argument(
        "--neg_to_pos_ratio",
        type=float,
        default=0.0,
        help="If >0, cap negatives to at most ratio * positives.",
    )
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chips_dir = Path(args.chips_dir)
    paths = sorted(chips_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"No chips found in {chips_dir}")

    positives: list[Path] = []
    negatives: list[Path] = []
    dropped_low_valid = 0
    dropped_tiny_pos = 0

    for p in paths:
        with np.load(p) as d:
            x = d["x"]
            y = d["y"]
        valid_frac = float(np.isfinite(x).all(axis=0).mean())
        if valid_frac < args.min_valid_frac:
            dropped_low_valid += 1
            continue
        yb = binarize_label(y)
        pos_frac = float(yb.mean())
        if pos_frac > 0.0:
            if pos_frac < args.min_pos_frac:
                dropped_tiny_pos += 1
                continue
            positives.append(p)
        else:
            negatives.append(p)

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    if args.neg_to_pos_ratio > 0:
        max_neg = int(round(len(positives) * args.neg_to_pos_ratio))
        negatives = negatives[:max_neg]

    kept = positives + negatives
    rng.shuffle(kept)

    n_val = int(len(kept) * args.val_frac)
    val_paths = kept[:n_val]
    train_paths = kept[n_val:]

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text("\n".join(str(p) for p in train_paths), encoding="utf-8")
    val_out.write_text("\n".join(str(p) for p in val_paths), encoding="utf-8")

    print(f"total_chips={len(paths)}")
    print(f"dropped_low_valid={dropped_low_valid}")
    print(f"dropped_tiny_pos={dropped_tiny_pos}")
    print(f"kept={len(kept)} positives={len(positives)} negatives={len(negatives)}")
    if kept:
        print(f"kept_pos_fraction={len(positives)/len(kept):.3f}")
    print(f"train={len(train_paths)} val={len(val_paths)}")
    print(f"train_list={train_out}")
    print(f"val_list={val_out}")


if __name__ == "__main__":
    main()

