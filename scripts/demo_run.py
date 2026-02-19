from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deforest.config import load_config, ensure_dir
from deforest.data.chips_io import list_npz, load_npz
from deforest.infer.infer import load_model, predict_chip
from deforest.infer.postprocess import threshold_mask, clean_mask
from deforest.infer.vectorize import mask_to_geojson, save_geojson
from deforest.infer.viz import save_panel


def get_channel_index(channels: list[str], name: str) -> int:
    if name not in channels:
        raise ValueError(f"Channel {name} not found")
    return channels.index(name)


def extract_rgb(x: np.ndarray, channels: list[str], prefix: str) -> np.ndarray:
    r = x[get_channel_index(channels, f"{prefix}_B4")]
    g = x[get_channel_index(channels, f"{prefix}_B3")]
    b = x[get_channel_index(channels, f"{prefix}_B2")]
    return np.stack([r, g, b], axis=-1)


def area_per_pixel_m2(meta: dict) -> float:
    transform = meta.get("transform")
    if transform and len(transform) >= 6:
        px_w = abs(transform[0])
        px_h = abs(transform[4])
        if px_w > 0 and px_h > 0:
            return px_w * px_h
    return 100.0


def baseline_mask(x: np.ndarray, channels: list[str], dndvi_thr: float, dnbr_thr: float) -> np.ndarray:
    dndvi = x[get_channel_index(channels, "dNDVI")]
    dnbr = x[get_channel_index(channels, "dNBR")]
    return ((dndvi < dndvi_thr) & (dnbr < dnbr_thr)).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    chips_dir = Path(cfg["chips_dir"])
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/demo/run_001"))

    channels = cfg.get("channels", [])
    chip_paths = list_npz(chips_dir)
    if not chip_paths:
        raise SystemExit(f"No chips found in {chips_dir}")

    checkpoint = Path(cfg.get("checkpoint", ""))
    use_model = checkpoint.exists()

    device = "cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"

    model = None
    if use_model:
        model_cfg = cfg.get("model", {})
        in_channels = int(model_cfg.get("in_channels", len(channels)))
        base_channels = int(model_cfg.get("base_channels", 32))
        model = load_model(checkpoint, in_channels=in_channels, base_channels=base_channels, device=device)

    thresh = float(cfg.get("threshold", 0.6))
    min_area_ha = float(cfg.get("min_area_ha", 1.0))
    post_cfg = cfg.get("postprocess", {})
    morph_open = int(post_cfg.get("morph_open", 0))
    morph_close = int(post_cfg.get("morph_close", 0))
    base_cfg = cfg.get("baseline", {})

    all_features = []
    total_area_m2 = 0.0

    panel_index = int(cfg.get("panel_chip_index", 0))
    panel_saved = False

    for i, chip_path in enumerate(chip_paths):
        x, _, meta = load_npz(chip_path)
        if use_model:
            prob = predict_chip(model, x, device)
            raw_mask = threshold_mask(prob, thresh)
        else:
            raw_mask = baseline_mask(x, channels, float(base_cfg.get("dndvi_thr", -0.2)), float(base_cfg.get("dnbr_thr", -0.2)))

        px_area = area_per_pixel_m2(meta)
        min_area_px = int(np.ceil((min_area_ha * 10000.0) / px_area))
        mask = clean_mask(raw_mask, min_area_px=min_area_px, morph_open=morph_open, morph_close=morph_close)

        geojson = mask_to_geojson(mask, transform=meta.get("transform"), crs=meta.get("crs"))
        all_features.extend(geojson.get("features", []))

        total_area_m2 += mask.sum() * px_area

        if i == panel_index and not panel_saved:
            pre_rgb = extract_rgb(x, channels, "pre")
            post_rgb = extract_rgb(x, channels, "post")
            save_panel(pre_rgb, post_rgb, mask, output_dir / "panel.png")
            panel_saved = True

    out_geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }
    save_geojson(out_geojson, output_dir / "detections.geojson")

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["num_polygons", "total_area_ha"])
        writer.writeheader()
        writer.writerow(
            {
                "num_polygons": len(all_features),
                "total_area_ha": round(total_area_m2 / 10000.0, 4),
            }
        )


if __name__ == "__main__":
    main()
