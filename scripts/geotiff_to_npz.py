from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random
import numpy as np
import rasterio
from tqdm import tqdm
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deforest.data.chips_io import save_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Chip GeoTIFFs into smaller samples for training.")
    parser.add_argument("--tif_dir", required=True, help="Directory containing GeoTIFF files to process.")
    parser.add_argument("--out_dir", required=True, help="Directory to save the output .npz chip files.")
    parser.add_argument("--chip_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max_chips_per_tif", type=int, default=0, help="Max chips per TIF. 0 means all possible. Windows are shuffled if this is set.")
    parser.add_argument("--label_band", type=int, default=-1, help="The band index of the label mask (1-based). Negative values count from the end.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_valid_frac",
        type=float,
        default=0.10,
        help="Minimum fraction of finite pixels across all feature bands. Chips below this are skipped.",
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--require_positive", action="store_true", help="Only save chips that contain at least one positive (non-zero) label pixel.")
    group.add_argument("--require_negative", action="store_true", help="Only save chips that contain only negative (zero) label pixels.")

    args = parser.parse_args()

    tif_dir = Path(args.tif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(list(tif_dir.glob("*.tif*")))
    if not tif_files:
        print(f"Error: No .tif or .tiff files found in {tif_dir}", file=sys.stderr)
        return
    
    print(f"Found {len(tif_files)} TIFF files to process in {tif_dir}")
    print(f"Saving chips to {out_dir}")

    rng = random.Random(args.seed)
    total_saved = 0

    for tif_path in tqdm(tif_files, desc="Overall Progress"):
        try:
            with rasterio.open(tif_path) as src:
                h, w = src.height, src.width
                if h < args.chip_size or w < args.chip_size:
                    continue
                
                band_count = src.count
                label_band_idx = args.label_band
                if label_band_idx < 0:
                    label_band_idx += band_count + 1

                windows = []
                for row in range(0, h - args.chip_size + 1, args.stride):
                    for col in range(0, w - args.chip_size + 1, args.stride):
                        windows.append(Window(col, row, args.chip_size, args.chip_size))

                # Shuffle windows for random sampling if max_chips is set, which is useful for negative sampling
                if args.max_chips_per_tif and args.max_chips_per_tif < len(windows):
                    rng.shuffle(windows)
                    windows = windows[: args.max_chips_per_tif]

                saved_this_file = 0
                skipped_low_valid = 0
                skipped_by_class_filter = 0
                chip_iterator = tqdm(windows, desc=f"Chipping {tif_path.name}", leave=False)
                for window in chip_iterator:
                    data = src.read(window=window)
                    if data.shape[0] < band_count:
                        continue

                    y = data[label_band_idx - 1]
                    y_for_check = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    is_positive = bool(y_for_check.sum() > 0)

                    if args.require_positive and not is_positive:
                        skipped_by_class_filter += 1
                        continue
                    if args.require_negative and is_positive:
                        skipped_by_class_filter += 1
                        continue
                    
                    x = np.delete(data, label_band_idx - 1, axis=0)
                    valid_mask = np.isfinite(x).all(axis=0)
                    valid_frac = float(valid_mask.mean())
                    if valid_frac < args.min_valid_frac:
                        skipped_low_valid += 1
                        continue

                    transform = rasterio.windows.transform(window, src.transform)
                    meta = {
                        "crs": src.crs.to_string() if src.crs else None,
                        "transform": list(transform)[:6],
                        "height": args.chip_size,
                        "width": args.chip_size,
                        "source": str(tif_path.name),
                        "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
                        "valid_frac": valid_frac,
                    }

                    chip_id = f"{tif_path.stem}_chip_{window.row_off}_{window.col_off}"
                    out_path = out_dir / f"{chip_id}.npz"
                    save_npz(out_path, x, y, meta)
                    saved_this_file += 1
                
                if saved_this_file > 0:
                    tqdm.write(
                        f"  - Saved {saved_this_file} chips from {tif_path.name} "
                        f"(skipped_low_valid={skipped_low_valid}, skipped_class_filter={skipped_by_class_filter})."
                    )
                total_saved += saved_this_file

        except Exception as e:
            tqdm.write(f"  - ERROR processing {tif_path.name}: {e}", file=sys.stderr)
            continue
    
    print(f"\nFinished: Saved a total of {total_saved} chips to {out_dir}")




if __name__ == "__main__":
    main()
