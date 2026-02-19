from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage as ndi


DEFAULT_BAND_ORDER = [
    "pre_B2",
    "pre_B3",
    "pre_B4",
    "pre_B8",
    "pre_B11",
    "pre_B12",
    "pre_NDVI",
    "pre_NBR",
    "post_B2",
    "post_B3",
    "post_B4",
    "post_B8",
    "post_B11",
    "post_B12",
    "post_NDVI",
    "post_NBR",
    "dNDVI",
    "dNBR",
    "label",
]


def _band_index(src: rasterio.io.DatasetReader, name: str) -> int | None:
    desc = list(src.descriptions or [])
    if desc:
        for idx, d in enumerate(desc, start=1):
            if d == name:
                return idx
    if name in DEFAULT_BAND_ORDER:
        return DEFAULT_BAND_ORDER.index(name) + 1
    return None


def _get_band(src: rasterio.io.DatasetReader, name: str) -> np.ndarray:
    idx = _band_index(src, name)
    if idx is None:
        raise KeyError(f"Band '{name}' not found.")
    return src.read(idx).astype(np.float32, copy=False)


def _compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-6)


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= (radius * radius)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["ndvi", "hansen"], default="ndvi")
    parser.add_argument("--ndvi_band_name", default="pre_NDVI")
    parser.add_argument("--ndvi_thr", type=float, default=0.45)
    parser.add_argument("--smooth_px", type=int, default=1)
    args = parser.parse_args()

    raster_path = Path(args.raster)
    out_path = Path(args.out)
    if not raster_path.exists():
        raise SystemExit(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        if args.method == "ndvi":
            try:
                forest_prob = _get_band(src, args.ndvi_band_name)
            except KeyError:
                nir = _get_band(src, "pre_B8")
                red = _get_band(src, "pre_B4")
                forest_prob = _compute_ndvi(nir, red)
        else:
            try:
                forest_prob = _get_band(src, "treecover2000")
            except KeyError as exc:
                raise SystemExit("Hansen mode requires a 'treecover2000' band in the raster.") from exc

        forest_prob = np.nan_to_num(forest_prob, nan=0.0, posinf=0.0, neginf=0.0)
        forest_mask = (forest_prob > float(args.ndvi_thr))

        if args.smooth_px > 0:
            struct = _disk(int(args.smooth_px))
            forest_mask = ndi.binary_closing(forest_mask, structure=struct)
            forest_mask = ndi.binary_opening(forest_mask, structure=struct)

        forest_mask_u8 = forest_mask.astype(np.uint8)

        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(forest_mask_u8, 1)

    unique_count = int(np.unique(forest_prob).size)
    print(
        "forest_prob stats: min={:.3f} max={:.3f} mean={:.3f} unique_count={}".format(
            float(forest_prob.min()), float(forest_prob.max()), float(forest_prob.mean()), unique_count
        )
    )
    print(f"forest_mask sum: {int(forest_mask_u8.sum())} ratio: {float(forest_mask_u8.mean()):.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
