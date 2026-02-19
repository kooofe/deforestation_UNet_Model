from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio import warp
from rasterio import features
from rasterio.windows import Window
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry import shape
from pyproj import Geod
import matplotlib.pyplot as plt
import json
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deforest.config import load_config, resolve_path
from deforest.models.unet import UNet
from deforest.infer.infer import predict_chip
from deforest.infer.postprocess import threshold_mask
from deforest.infer.viz import save_panel
from deforest.features.indices import ndvi, nbr

DEFAULT_CHANNELS = [
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
]


def _resolve(base: Path | None, path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    if base is None:
        return Path(path_value)
    return resolve_path(base, path_value)


def _get_arg_or_cfg(args, cfg: dict, key: str, default=None):
    val = getattr(args, key)
    if val is not None:
        return val
    return cfg.get(key, default)


def _infer_model_params(state: dict) -> tuple[int, int]:
    weight = state["model"]["down1.block.0.weight"]
    base_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    return in_channels, base_channels


def _prepare_x(x: np.ndarray, expected_channels: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    valid = np.isfinite(x).all(axis=0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if expected_channels == x.shape[0] + 1:
        x = np.concatenate([x, valid[None, ...]], axis=0)
    elif expected_channels != x.shape[0]:
        raise ValueError(f"Input channels mismatch: expected {expected_channels}, got {x.shape[0]}")
    return x


def _stack_pre_post(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    if pre.shape[0] != 6 or post.shape[0] != 6:
        raise ValueError("Expected 6 bands for pre and post (B2,B3,B4,B8,B11,B12).")
    pre_bands = {
        "B2": pre[0],
        "B3": pre[1],
        "B4": pre[2],
        "B8": pre[3],
        "B11": pre[4],
        "B12": pre[5],
    }
    post_bands = {
        "B2": post[0],
        "B3": post[1],
        "B4": post[2],
        "B8": post[3],
        "B11": post[4],
        "B12": post[5],
    }
    pre_ndvi = ndvi(pre_bands["B8"], pre_bands["B4"])
    pre_nbr = nbr(pre_bands["B8"], pre_bands["B12"])
    post_ndvi = ndvi(post_bands["B8"], post_bands["B4"])
    post_nbr = nbr(post_bands["B8"], post_bands["B12"])
    dndvi = post_ndvi - pre_ndvi
    dnbr = post_nbr - pre_nbr

    stack = np.stack(
        [
            pre_bands["B2"],
            pre_bands["B3"],
            pre_bands["B4"],
            pre_bands["B8"],
            pre_bands["B11"],
            pre_bands["B12"],
            pre_ndvi,
            pre_nbr,
            post_bands["B2"],
            post_bands["B3"],
            post_bands["B4"],
            post_bands["B8"],
            post_bands["B11"],
            post_bands["B12"],
            post_ndvi,
            post_nbr,
            dndvi,
            dnbr,
        ],
        axis=0,
    )
    return stack.astype(np.float32, copy=False)


def _scale_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, None)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 - p2 < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - p2) / (p98 - p2), 0, 1)


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= (radius * radius)


# VISUALIZATION ONLY — does not affect masks/polygons/summary.
def _compute_viz_mask(
    mask: np.ndarray,
    *,
    auto: bool,
    target_coverage: float,
    min_px: int,
    max_px: int,
) -> tuple[np.ndarray, int, float, float]:
    mask_bool = mask.astype(bool)
    total = float(mask_bool.size)
    coverage_raw = float(mask_bool.sum()) / total if total else 0.0
    chosen = max(0, int(min_px))
    mask_viz = mask_bool
    if chosen > 0:
        mask_viz = ndi.binary_dilation(mask_bool, structure=_disk(chosen))
    coverage_viz = float(mask_viz.sum()) / total if total else 0.0
    if auto and coverage_raw < target_coverage:
        for r in range(chosen, int(max_px) + 1):
            mask_viz = ndi.binary_dilation(mask_bool, structure=_disk(r)) if r > 0 else mask_bool
            coverage_viz = float(mask_viz.sum()) / total if total else 0.0
            chosen = r
            if coverage_viz >= target_coverage:
                break
    return mask_viz, chosen, coverage_raw, coverage_viz


# VISUALIZATION ONLY — does not affect masks/polygons/summary.
def _build_overlay(
    rgb: np.ndarray,
    mask_viz: np.ndarray,
    *,
    alpha: float,
    color: np.ndarray,
    outline: bool,
    outline_px: int,
) -> np.ndarray:
    rgb_scaled = _scale_rgb(rgb)
    overlay = rgb_scaled.copy()
    mask_bool = mask_viz.astype(bool)
    if mask_bool.any():
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color
    if outline and mask_bool.any():
        boundary = mask_bool ^ ndi.binary_erosion(mask_bool, structure=_disk(outline_px))
        overlay[boundary] = color
    return overlay


def _save_overlay(rgb: np.ndarray, overlay: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, overlay)


def _save_panel_images(
    pre: np.ndarray,
    post: np.ndarray,
    overlay: np.ndarray,
    out_path: Path,
    *,
    show_legend: bool,
    legend_loc: str,
    title_prefix: str,
    footer_text: str | None,
    color: np.ndarray,
) -> None:
    import matplotlib.patches as mpatches

    pre_scaled = _scale_rgb(pre)
    post_scaled = _scale_rgb(post)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pre_scaled)
    axes[0].set_title("PRE RGB")
    axes[1].imshow(post_scaled)
    axes[1].set_title("POST RGB")
    axes[2].imshow(overlay)
    axes[2].set_title("DETECTION")
    for ax in axes:
        ax.axis("off")
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
    if show_legend:
        patch = mpatches.Patch(color=color, label="Detections (visualized)")
        axes[2].legend(handles=[patch], loc=legend_loc, frameon=True, fontsize=8)
    if footer_text:
        fig.text(0.5, 0.01, footer_text, ha="center", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_forest_diag(forest_ok: np.ndarray, out_path: Path) -> None:
    if forest_ok.size == 0:
        return
    boundary = forest_ok.astype(bool) ^ ndi.binary_erosion(forest_ok.astype(bool))
    base = np.zeros_like(forest_ok, dtype=np.float32)
    overlay = np.stack([base, base, base], axis=-1)
    overlay[..., 1] = np.clip(boundary.astype(np.float32) * 0.9, 0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, overlay)


def _get_rgb_indices(channels: list[str], prefix: str) -> tuple[int, int, int] | None:
    try:
        r = channels.index(f"{prefix}_B4") + 1
        g = channels.index(f"{prefix}_B3") + 1
        b = channels.index(f"{prefix}_B2") + 1
        return r, g, b
    except ValueError:
        return None


def _band_index(src: rasterio.io.DatasetReader, name: str) -> int | None:
    desc = list(src.descriptions or [])
    if desc:
        for idx, d in enumerate(desc, start=1):
            if d == name:
                return idx
    if name in DEFAULT_CHANNELS:
        return DEFAULT_CHANNELS.index(name) + 1
    return None


def _read_band(
    src: rasterio.io.DatasetReader,
    name: str,
    *,
    fallback_idx: int | None = None,
) -> np.ndarray | None:
    idx = _band_index(src, name)
    if idx is None:
        idx = fallback_idx
    if idx is None:
        return None
    return src.read(idx).astype(np.float32, copy=False)


def _pixel_area_m2(transform: Affine, crs, width: int, height: int) -> float | None:
    if crs and crs.is_projected:
        return float(abs(transform.a * transform.e))
    if crs and crs.is_geographic:
        cx = transform.c + transform.a * (width / 2.0)
        cy = transform.f + transform.e * (height / 2.0)
        geod = Geod(ellps="WGS84")
        _, _, dx = geod.inv(cx, cy, cx + transform.a, cy)
        _, _, dy = geod.inv(cx, cy, cx, cy + transform.e)
        return float(abs(dx * dy))
    return None


def _vectorize(mask: np.ndarray, transform: Affine, crs) -> gpd.GeoDataFrame:
    shapes = features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
    geoms = [shape(geom) for geom, val in shapes if val == 1]
    return gpd.GeoDataFrame({"geometry": geoms}, crs=crs)


def _write_empty_geojson(path: Path, crs) -> None:
    geojson = {"type": "FeatureCollection", "features": []}
    if crs:
        geojson["crs"] = {"type": "name", "properties": {"name": str(crs)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)


def _pick_device(value: str | None) -> str:
    if value and value.lower() != "auto":
        if value.lower() == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return value.lower()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _align_forest_mask(path: Path, ref_src, out_shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(path) as src:
        forest = src.read(1)
        print(f"forest_mask src: shape={forest.shape} crs={src.crs} transform={src.transform}")
        print(f"prob raster: shape={out_shape} crs={ref_src.crs} transform={ref_src.transform}")
        print(f"forest_mask bounds: {src.bounds}")
        print(f"prob raster bounds: {ref_src.bounds}")
        if forest.shape != out_shape or src.transform != ref_src.transform or src.crs != ref_src.crs:
            dst = np.zeros(out_shape, dtype=forest.dtype)
            warp.reproject(
                source=forest,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_src.transform,
                dst_crs=ref_src.crs,
                resampling=warp.Resampling.nearest,
            )
            forest = dst
        forest = np.nan_to_num(forest, nan=0.0, posinf=0.0, neginf=0.0)
        unique_count = int(np.unique(forest).size)
        print(
            "forest_mask stats: min={:.3f} max={:.3f} mean={:.3f} unique_count={}".format(
                float(forest.min()), float(forest.max()), float(forest.mean()), unique_count
            )
        )
    return forest


def _tile_candidates(mask: np.ndarray, tile: int, stride: int) -> list[tuple[int, int, int]]:
    height, width = mask.shape
    candidates = []
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            row_end = min(row + tile, height)
            col_end = min(col + tile, width)
            area = int(mask[row:row_end, col:col_end].sum())
            if area > 0:
                candidates.append((row, col, area))
    return candidates


def _save_tile_panel(
    pre_rgb: np.ndarray,
    post_rgb: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    *,
    alpha: float,
    dilate_px: int,
    outline: bool,
    outline_px: int,
    color: np.ndarray,
) -> None:
    mask_viz = ndi.binary_dilation(mask.astype(bool), structure=_disk(dilate_px)) if dilate_px > 0 else mask.astype(bool)
    overlay = _build_overlay(
        post_rgb,
        mask_viz,
        alpha=alpha,
        color=color,
        outline=outline,
        outline_px=outline_px,
    )
    _save_panel_images(
        pre_rgb,
        post_rgb,
        overlay,
        out_path,
        show_legend=False,
        legend_loc="lower right",
        title_prefix="",
        footer_text=None,
        color=color,
    )


def _save_gallery(images: list[np.ndarray], cols: int, out_path: Path) -> None:
    if not images:
        return
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes[:, None]
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx < len(images):
                ax.imshow(images[idx])
            idx += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)


def _mask_sum_ratio(mask: np.ndarray) -> tuple[int, float]:
    total = int(mask.sum())
    ratio = float(mask.mean())
    return total, ratio


def _cc_stats(mask_bool: np.ndarray) -> tuple[int, np.ndarray]:
    labeled, num = ndi.label(mask_bool)
    if num == 0:
        return 0, np.array([], dtype=np.int64)
    sizes = ndi.sum(mask_bool, labeled, index=np.arange(1, num + 1)).astype(np.int64)
    return num, sizes


def _print_cc_stats(label: str, mask_bool: np.ndarray) -> None:
    num, sizes = _cc_stats(mask_bool)
    if num == 0:
        print(f"{label}: components=0")
        return
    print(
        "{}: components={} min={} median={} max={}".format(
            label,
            num,
            int(sizes.min()),
            int(np.median(sizes)),
            int(sizes.max()),
        )
    )


def _apply_morphology(mask_bool: np.ndarray, open_px: int, close_px: int, safe_postproc: bool) -> np.ndarray:
    if open_px <= 0 and close_px <= 0:
        return mask_bool
    morphed = mask_bool
    if open_px > 0:
        morphed = ndi.binary_opening(morphed, iterations=open_px)
    if close_px > 0:
        morphed = ndi.binary_closing(morphed, iterations=close_px)
    if safe_postproc and morphed.sum() == 0 and mask_bool.sum() > 0:
        print("WARNING: Morphology removed all detections; skipping morphology.")
        return mask_bool
    return morphed


def _filter_by_area(mask_bool: np.ndarray, min_area_px: int, safe_postproc: bool) -> tuple[np.ndarray, int]:
    if min_area_px <= 1:
        return mask_bool, min_area_px
    labeled, num = ndi.label(mask_bool)
    if num == 0:
        return mask_bool, min_area_px
    sizes = ndi.sum(mask_bool, labeled, index=np.arange(1, num + 1))
    keep = sizes >= min_area_px
    mask_cc = np.isin(labeled, np.where(keep)[0] + 1)
    if safe_postproc and mask_cc.sum() == 0 and mask_bool.sum() > 0:
        for factor in (4, 0):
            if factor == 0:
                fallback_px = 1
            else:
                fallback_px = max(1, int(np.ceil(min_area_px / factor)))
            keep = sizes >= fallback_px
            mask_cc = np.isin(labeled, np.where(keep)[0] + 1)
            if mask_cc.sum() > 0:
                print(f"WARNING: Area filter removed all detections; fallback min_area_px={fallback_px}")
                return mask_cc, fallback_px
        print("WARNING: Area filter removed all detections; keeping original mask.")
        return mask_bool, min_area_px
    return mask_cc, min_area_px


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--raster")
    parser.add_argument("--aoi_pre")
    parser.add_argument("--aoi_post")
    parser.add_argument("--out_dir")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min_area_m2", type=float)
    parser.add_argument("--tile", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--tile_gallery", type=int)
    parser.add_argument("--forest_mask_path")
    parser.add_argument("--forest_source")
    parser.add_argument("--forest_thr", type=float)
    parser.add_argument("--device")
    parser.add_argument("--open_px", type=int)
    parser.add_argument("--close_px", type=int)
    parser.add_argument("--safe_postproc", dest="safe_postproc", action="store_true")
    parser.add_argument("--no_safe_postproc", dest="safe_postproc", action="store_false")
    parser.add_argument("--morph_close", type=int)
    parser.add_argument("--viz_alpha", type=float)
    parser.add_argument("--viz_color")
    parser.add_argument("--viz_dilate_px", type=int)
    parser.add_argument("--viz_outline", dest="viz_outline", action="store_true")
    parser.add_argument("--no_viz_outline", dest="viz_outline", action="store_false")
    parser.add_argument("--viz_outline_px", type=int)
    parser.add_argument("--viz_dilate_auto", dest="viz_dilate_auto", action="store_true")
    parser.add_argument("--no_viz_dilate_auto", dest="viz_dilate_auto", action="store_false")
    parser.add_argument("--viz_target_coverage", type=float)
    parser.add_argument("--viz_dilate_max_px", type=int)
    parser.add_argument("--viz_dilate_min_px", type=int)
    parser.add_argument("--viz_boundary_alpha", type=float)
    parser.add_argument("--viz_show_legend", dest="viz_show_legend", action="store_true")
    parser.add_argument("--no_viz_show_legend", dest="viz_show_legend", action="store_false")
    parser.add_argument("--viz_legend_loc")
    parser.add_argument("--viz_title_prefix")
    parser.add_argument("--viz_footer", dest="viz_footer", action="store_true")
    parser.add_argument("--no_viz_footer", dest="viz_footer", action="store_false")
    parser.add_argument("--viz_mode")
    parser.add_argument("--viz_tile", type=int)
    parser.add_argument("--viz_stride", type=int)
    parser.add_argument("--viz_zoom_largest_component", dest="viz_zoom_largest_component", action="store_true")
    parser.add_argument("--no_viz_zoom_largest_component", dest="viz_zoom_largest_component", action="store_false")
    parser.add_argument("--zoom_margin_px", type=int)
    parser.add_argument("--gallery_topk", type=int)
    parser.add_argument("--gallery_tile_px", type=int)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--no_debug", dest="debug", action="store_false")
    parser.add_argument("--drop_extra_band", dest="drop_extra_band", action="store_true")
    parser.add_argument("--keep_extra_band", dest="drop_extra_band", action="store_false")
    parser.add_argument("--save_probs", dest="save_probs", action="store_true")
    parser.add_argument("--no_save_probs", dest="save_probs", action="store_false")
    parser.set_defaults(save_probs=None)
    parser.set_defaults(drop_extra_band=True)
    parser.set_defaults(safe_postproc=None)
    parser.set_defaults(viz_outline=None)
    parser.set_defaults(viz_dilate_auto=None)
    parser.set_defaults(viz_show_legend=None)
    parser.set_defaults(viz_footer=None)
    parser.set_defaults(viz_zoom_largest_component=None)
    parser.set_defaults(debug=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    cfg_base = Path(args.config).resolve().parent if args.config else None

    checkpoint = _resolve(cfg_base, _get_arg_or_cfg(args, cfg, "checkpoint"))
    raster = _resolve(cfg_base, _get_arg_or_cfg(args, cfg, "raster"))
    aoi_pre = _resolve(cfg_base, _get_arg_or_cfg(args, cfg, "aoi_pre"))
    aoi_post = _resolve(cfg_base, _get_arg_or_cfg(args, cfg, "aoi_post"))
    out_dir = _resolve(cfg_base, _get_arg_or_cfg(args, cfg, "out_dir")) or Path("outputs/demo/run_001")
    threshold = float(_get_arg_or_cfg(args, cfg, "threshold", 0.85))
    min_area_m2 = float(_get_arg_or_cfg(args, cfg, "min_area_m2", 5000))
    tile = int(_get_arg_or_cfg(args, cfg, "tile", 256))
    stride = int(_get_arg_or_cfg(args, cfg, "stride", tile))
    tile_gallery = int(_get_arg_or_cfg(args, cfg, "tile_gallery", 12))
    forest_mask_path = _resolve(cfg_base, _get_arg_or_cfg(args, cfg, "forest_mask_path"))
    forest_source = _get_arg_or_cfg(args, cfg, "forest_source")
    forest_thr_default = 30.0 if forest_source else 0.6
    forest_thr = float(_get_arg_or_cfg(args, cfg, "forest_thr", forest_thr_default))
    device = _pick_device(_get_arg_or_cfg(args, cfg, "device", "auto"))
    open_px = int(_get_arg_or_cfg(args, cfg, "open_px", 0))
    close_px_cfg = _get_arg_or_cfg(args, cfg, "close_px", None)
    if close_px_cfg is None:
        close_px = int(_get_arg_or_cfg(args, cfg, "morph_close", 2))
    else:
        close_px = int(close_px_cfg)
    safe_postproc = bool(_get_arg_or_cfg(args, cfg, "safe_postproc", True))
    viz_alpha = float(_get_arg_or_cfg(args, cfg, "viz_alpha", 0.75))
    viz_color = _get_arg_or_cfg(args, cfg, "viz_color", [1.0, 0.0, 0.0])
    if isinstance(viz_color, str):
        viz_color = [float(v) for v in viz_color.split(",")]
    viz_color = np.array(viz_color, dtype=np.float32)
    viz_outline = bool(_get_arg_or_cfg(args, cfg, "viz_outline", True))
    viz_outline_px = int(_get_arg_or_cfg(args, cfg, "viz_outline_px", 2))
    viz_dilate_auto = bool(_get_arg_or_cfg(args, cfg, "viz_dilate_auto", True))
    viz_target_coverage = float(_get_arg_or_cfg(args, cfg, "viz_target_coverage", 0.005))
    viz_dilate_min_px = int(_get_arg_or_cfg(args, cfg, "viz_dilate_min_px", 0))
    viz_dilate_max_px = int(_get_arg_or_cfg(args, cfg, "viz_dilate_max_px", 12))
    viz_boundary_alpha = float(_get_arg_or_cfg(args, cfg, "viz_boundary_alpha", 1.0))
    viz_show_legend = bool(_get_arg_or_cfg(args, cfg, "viz_show_legend", True))
    viz_legend_loc = str(_get_arg_or_cfg(args, cfg, "viz_legend_loc", "lower right"))
    viz_title_prefix = str(_get_arg_or_cfg(args, cfg, "viz_title_prefix", ""))
    viz_footer = bool(_get_arg_or_cfg(args, cfg, "viz_footer", True))
    viz_mode = str(_get_arg_or_cfg(args, cfg, "viz_mode", "final"))
    viz_tile = int(_get_arg_or_cfg(args, cfg, "viz_tile", tile))
    viz_stride = int(_get_arg_or_cfg(args, cfg, "viz_stride", stride))
    viz_zoom_largest_component = bool(_get_arg_or_cfg(args, cfg, "viz_zoom_largest_component", True))
    zoom_margin_px = int(_get_arg_or_cfg(args, cfg, "zoom_margin_px", 64))
    gallery_topk = int(_get_arg_or_cfg(args, cfg, "gallery_topk", 12))
    gallery_tile_px = int(_get_arg_or_cfg(args, cfg, "gallery_tile_px", 192))
    debug = bool(_get_arg_or_cfg(args, cfg, "debug", False))
    save_probs = bool(_get_arg_or_cfg(args, cfg, "save_probs", True))
    drop_extra_band = bool(_get_arg_or_cfg(args, cfg, "drop_extra_band", True))

    if checkpoint is None:
        raise SystemExit("Missing --checkpoint")
    if raster is None and (aoi_pre is None or aoi_post is None):
        raise SystemExit("Provide --raster or both --aoi_pre and --aoi_post")
    if out_dir is None:
        raise SystemExit("Missing --out_dir")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(checkpoint, map_location=device)
    in_channels, base_channels = _infer_model_params(state)
    model = UNet(in_channels=in_channels, base_channels=base_channels)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    channels = cfg.get("channels") or state.get("config", {}).get("channels") or DEFAULT_CHANNELS

    if raster is not None:
        src = rasterio.open(raster)
        ref_src = src
        band_count = src.count
        if band_count == in_channels + 1:
            if not drop_extra_band:
                src.close()
                raise SystemExit(
                    f"Raster band count {band_count} has 1 extra band for model input {in_channels}. "
                    "Use --drop_extra_band to drop the last band."
                )
            read_band_count = in_channels
        elif band_count == in_channels:
            read_band_count = band_count
        else:
            src.close()
            raise SystemExit(f"Raster band count {band_count} incompatible with model input {in_channels}")
    else:
        pre_src = rasterio.open(aoi_pre)
        post_src = rasterio.open(aoi_post)
        ref_src = pre_src
        if pre_src.count != post_src.count:
            raise SystemExit("Pre/post band counts differ.")
        if pre_src.width != post_src.width or pre_src.height != post_src.height:
            raise SystemExit("Pre/post dimensions differ.")
        if pre_src.transform != post_src.transform:
            raise SystemExit("Pre/post transforms differ.")
        if pre_src.crs != post_src.crs:
            raise SystemExit("Pre/post CRS differ.")

    height = ref_src.height
    width = ref_src.width
    transform = ref_src.transform
    crs = ref_src.crs

    prob_accum = np.zeros((height, width), dtype=np.float32)
    count_accum = np.zeros((height, width), dtype=np.float32)

    for row in range(0, height, stride):
        for col in range(0, width, stride):
            window = Window(col, row, tile, tile)
            if raster is not None:
                x = src.read(indexes=list(range(1, read_band_count + 1)), window=window, boundless=True, fill_value=0)
                x = _prepare_x(x, in_channels)
            else:
                pre = pre_src.read(window=window, boundless=True, fill_value=0)
                post = post_src.read(window=window, boundless=True, fill_value=0)
                x = _stack_pre_post(pre, post)
                x = _prepare_x(x, in_channels)

            probs = predict_chip(model, x, device)
            row_end = min(row + tile, height)
            col_end = min(col + tile, width)
            probs_crop = probs[: row_end - row, : col_end - col]
            prob_accum[row:row_end, col:col_end] += probs_crop
            count_accum[row:row_end, col:col_end] += 1.0

    prob = prob_accum / np.maximum(count_accum, 1e-6)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    mask_raw = threshold_mask(prob, threshold)
    print(f"prob: min={float(prob.min()):.6f} max={float(prob.max()):.6f} mean={float(prob.mean()):.6f}")
    raw_sum, raw_ratio = _mask_sum_ratio(mask_raw)
    print(f"mask raw sum: {raw_sum} ratio: {raw_ratio:.6f}")

    px_area_m2 = _pixel_area_m2(transform, crs, width, height)
    if px_area_m2 and min_area_m2 > 0:
        min_area_px = int(np.ceil(min_area_m2 / px_area_m2))
    else:
        min_area_px = 1
    if crs and crs.is_projected:
        print(
            "area conversion: pixel_size_m=({:.3f},{:.3f}) min_area_m2={} min_area_px={}".format(
                abs(transform.a), abs(transform.e), min_area_m2, min_area_px
            )
        )
    else:
        print(
            "area conversion: pixel_area_m2={:.3f} min_area_m2={} min_area_px={}".format(
                px_area_m2 or 0.0, min_area_m2, min_area_px
            )
        )

    mask_morph = _apply_morphology(mask_raw.astype(bool), open_px, close_px, safe_postproc)
    morph_sum, morph_ratio = _mask_sum_ratio(mask_morph.astype(np.uint8))
    print(f"mask morph sum: {morph_sum} ratio: {morph_ratio:.6f}")

    _print_cc_stats("components after morph", mask_morph)

    mask_cc, min_area_px_used = _filter_by_area(mask_morph, min_area_px, safe_postproc)
    cc_sum, cc_ratio = _mask_sum_ratio(mask_cc.astype(np.uint8))
    print(f"mask cc sum: {cc_sum} ratio: {cc_ratio:.6f} (min_area_px={min_area_px_used})")
    _print_cc_stats("components after area filter", mask_cc.astype(bool))

    det_mask_full = mask_cc.astype(np.uint8)
    det_mask = det_mask_full
    forest_filter_enabled = forest_mask_path is not None or forest_source is not None
    forest_mask = None
    forest_ok = None
    if forest_mask_path is not None:
        forest_mask = _align_forest_mask(forest_mask_path, ref_src, (height, width))
    if forest_mask is not None:
        forest_mask_f = forest_mask.astype(np.float32)
        forest_ok = forest_mask_f >= forest_thr
        if forest_mask_f.max() <= 1.0 and forest_thr > 1.0:
            print("WARNING: forest mask appears binary/probabilistic; ignoring forest_thr > 1.0")
            forest_ok = forest_mask_f >= 0.5
        forest_ok_sum = int(forest_ok.sum())
        survive = int((det_mask.astype(bool) & forest_ok).sum())
        print(f"forest_ok sum: {forest_ok_sum} | survive after forest filter: {survive}")
        det_mask = det_mask & forest_ok
        if safe_postproc and det_mask.sum() == 0 and det_mask_full.sum() > 0:
            print("WARNING: Forest filter removed all detections; skipping forest filter.")
            diag_path = out_dir / "forest_mask_diag.png"
            _save_forest_diag(forest_ok.astype(np.uint8), diag_path)
            det_mask = det_mask_full
    if forest_ok is not None:
        inside_forest = int((det_mask_full.astype(bool) & forest_ok).sum())
        removed_by_forest = int(det_mask_full.sum()) - inside_forest
        print(f"forest overlap: inside={inside_forest} removed={removed_by_forest}")
    post_sum, post_ratio = _mask_sum_ratio(det_mask.astype(np.uint8))
    print(f"mask postproc sum: {post_sum} ratio: {post_ratio:.6f}")

    # Forest change gate (optional) - applied after morphology and before polygonization.
    enable_forest_change_gate = bool(cfg.get("enable_forest_change_gate", True))
    if enable_forest_change_gate:
        gate_mode = str(cfg.get("gate_mode", "either")).lower()
        pre_ndvi_min = float(cfg.get("pre_ndvi_min", 0.5))
        dndvi_max = float(cfg.get("dndvi_max", -0.15))
        dnbr_max = float(cfg.get("dnbr_max", -0.10))

        pre_ndvi = None
        dndvi = None
        dnbr = None
        if raster is not None:
            pre_ndvi = _read_band(src, "pre_NDVI", fallback_idx=7)
            dndvi = _read_band(src, "dNDVI", fallback_idx=17)
            dnbr = _read_band(src, "dNBR", fallback_idx=18)
            if pre_ndvi is None or dndvi is None or dnbr is None:
                pre_b8 = _read_band(src, "pre_B8", fallback_idx=4)
                pre_b4 = _read_band(src, "pre_B4", fallback_idx=3)
                pre_b12 = _read_band(src, "pre_B12", fallback_idx=6)
                post_b8 = _read_band(src, "post_B8", fallback_idx=12)
                post_b4 = _read_band(src, "post_B4", fallback_idx=11)
                post_b12 = _read_band(src, "post_B12", fallback_idx=14)
                if pre_b8 is not None and pre_b4 is not None:
                    pre_ndvi = (pre_b8 - pre_b4) / (pre_b8 + pre_b4 + 1e-6)
                if post_b8 is not None and post_b4 is not None and pre_ndvi is not None:
                    post_ndvi = (post_b8 - post_b4) / (post_b8 + post_b4 + 1e-6)
                    dndvi = post_ndvi - pre_ndvi
                if pre_b8 is not None and pre_b12 is not None:
                    pre_nbr = (pre_b8 - pre_b12) / (pre_b8 + pre_b12 + 1e-6)
                else:
                    pre_nbr = None
                if post_b8 is not None and post_b12 is not None and pre_nbr is not None:
                    post_nbr = (post_b8 - post_b12) / (post_b8 + post_b12 + 1e-6)
                    dnbr = post_nbr - pre_nbr
        else:
            pre_b8 = pre_src.read(4).astype(np.float32, copy=False)
            pre_b4 = pre_src.read(3).astype(np.float32, copy=False)
            pre_b12 = pre_src.read(6).astype(np.float32, copy=False)
            post_b8 = post_src.read(4).astype(np.float32, copy=False)
            post_b4 = post_src.read(3).astype(np.float32, copy=False)
            post_b12 = post_src.read(6).astype(np.float32, copy=False)
            pre_ndvi = (pre_b8 - pre_b4) / (pre_b8 + pre_b4 + 1e-6)
            post_ndvi = (post_b8 - post_b4) / (post_b8 + post_b4 + 1e-6)
            dndvi = post_ndvi - pre_ndvi
            pre_nbr = (pre_b8 - pre_b12) / (pre_b8 + pre_b12 + 1e-6)
            post_nbr = (post_b8 - post_b12) / (post_b8 + post_b12 + 1e-6)
            dnbr = post_nbr - pre_nbr

        if pre_ndvi is None or dndvi is None or dnbr is None:
            print("WARNING: forest_change_gate missing required bands; skipping gate.")
        else:
            pre_ndvi = np.nan_to_num(pre_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
            dndvi = np.nan_to_num(dndvi, nan=0.0, posinf=0.0, neginf=0.0)
            dnbr = np.nan_to_num(dnbr, nan=0.0, posinf=0.0, neginf=0.0)

            gate = pre_ndvi >= pre_ndvi_min
            if gate_mode == "ndvi":
                gate &= dndvi <= dndvi_max
            elif gate_mode == "nbr":
                gate &= dnbr <= dnbr_max
            else:
                gate &= (dndvi <= dndvi_max) | (dnbr <= dnbr_max)

            before_gate = int(det_mask.sum())
            det_mask = det_mask & gate
            after_gate = int(det_mask.sum())
            removed_pct = 0.0 if before_gate == 0 else (before_gate - after_gate) / before_gate * 100.0
            print(f"forest_change_gate: before={before_gate} after={after_gate} removed={removed_pct:.2f}%")

            if after_gate > 0:
                idx = det_mask.astype(bool)
                print(
                    "gate stats (survivors): pre_NDVI min={:.3f} mean={:.3f} max={:.3f}".format(
                        float(pre_ndvi[idx].min()),
                        float(pre_ndvi[idx].mean()),
                        float(pre_ndvi[idx].max()),
                    )
                )
                print(
                    "gate stats (survivors): dNDVI min={:.3f} mean={:.3f} max={:.3f}".format(
                        float(dndvi[idx].min()),
                        float(dndvi[idx].mean()),
                        float(dndvi[idx].max()),
                    )
                )
                print(
                    "gate stats (survivors): dNBR min={:.3f} mean={:.3f} max={:.3f}".format(
                        float(dnbr[idx].min()),
                        float(dnbr[idx].mean()),
                        float(dnbr[idx].max()),
                    )
                )

    change_gate_cfg = cfg.get("change_gate", {}) if isinstance(cfg.get("change_gate", {}), dict) else {}
    change_gate_enabled = bool(change_gate_cfg.get("enabled", False))
    change_gate_mask = None
    if change_gate_enabled:
        if raster is None:
            print("WARNING: change_gate enabled but raster not provided; skipping change gate.")
        else:
            det_mask_before_gate = det_mask.copy()
            gate = np.ones_like(det_mask, dtype=bool)
            gate_any = False
            if change_gate_cfg.get("use_dndvi", True):
                name = str(change_gate_cfg.get("dndvi_band_name", "dNDVI"))
                idx = _band_index(src, name)
                if idx is None:
                    print(f"WARNING: change_gate dNDVI band '{name}' not found; skipping dNDVI gate.")
                else:
                    dndvi = src.read(idx).astype(np.float32, copy=False)
                    dndvi = np.nan_to_num(dndvi, nan=0.0, posinf=0.0, neginf=0.0)
                    thr = float(change_gate_cfg.get("dndvi_thr", -0.10))
                    gate &= dndvi < thr
                    gate_any = True
            if change_gate_cfg.get("use_dnbr", False):
                name = str(change_gate_cfg.get("dnbr_band_name", "dNBR"))
                idx = _band_index(src, name)
                if idx is None:
                    print(f"WARNING: change_gate dNBR band '{name}' not found; skipping dNBR gate.")
                else:
                    dnbr = src.read(idx).astype(np.float32, copy=False)
                    dnbr = np.nan_to_num(dnbr, nan=0.0, posinf=0.0, neginf=0.0)
                    thr = float(change_gate_cfg.get("dnbr_thr", -0.08))
                    gate &= dnbr < thr
                    gate_any = True
            if gate_any:
                change_gate_mask = gate
                before_sum = int(det_mask.sum())
                det_mask = det_mask & gate
                after_sum = int(det_mask.sum())
                print(f"change gate: before={before_sum} after={after_sum}")
                if safe_postproc and after_sum == 0 and before_sum > 0:
                    print("WARNING: change gate removed all detections; skipping change gate.")
                    det_mask = det_mask_before_gate
            else:
                print("WARNING: change_gate enabled but no valid bands found; skipping.")

    prob_path = out_dir / "probs.tif"
    mask_path = out_dir / "mask.tif"
    mask_full_path = out_dir / "mask_full.tif"
    overlay_path = out_dir / "overlay.png"
    panel_path = out_dir / "panel.png"
    polygons_path = out_dir / "polygons.geojson"
    summary_path = out_dir / "summary.csv"
    tiles_dir = out_dir / "tiles"
    gallery_path = out_dir / "gallery.png"

    gdf = _vectorize(det_mask, transform, crs)
    mask_final = det_mask
    total_area_m2 = 0.0
    if not gdf.empty:
        gdf_area = gdf
        if gdf.crs and gdf.crs.is_geographic:
            gdf_area = gdf.to_crs("EPSG:3857")
        areas = gdf_area.area
        if min_area_m2 > 0:
            keep = areas >= min_area_m2
            if keep.sum() == 0 and det_mask.sum() > 0 and safe_postproc:
                print("WARNING: Polygon filter removed all detections; skipping polygon filter.")
            else:
                gdf = gdf.loc[keep].copy()
                areas = areas[keep]
        total_area_m2 = float(areas.sum()) if len(areas) else 0.0
        if not gdf.empty and gdf is not gdf_area:
            mask_final = features.rasterize(
                ((geom, 1) for geom in gdf.geometry),
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
            )
    pre_area_m2 = float(det_mask_full.sum() * px_area_m2) if px_area_m2 else float(det_mask_full.sum())

    final_sum, final_ratio = _mask_sum_ratio(mask_final.astype(np.uint8))
    print(f"mask final sum: {final_sum} ratio: {final_ratio:.6f}")

    if not gdf.empty:
        gdf.to_file(polygons_path, driver="GeoJSON")
    else:
        _write_empty_geojson(polygons_path, crs)

    profile = ref_src.profile.copy()
    profile.update(count=1, compress="lzw")
    if save_probs:
        profile.update(dtype="float32")
        with rasterio.open(prob_path, "w", **profile) as dst:
            dst.write(prob.astype(np.float32), 1)

    profile.update(dtype="uint8", nodata=0)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask_final.astype(np.uint8), 1)
    with rasterio.open(mask_full_path, "w", **profile) as dst:
        dst.write(det_mask_full.astype(np.uint8), 1)

    if raster is not None:
        post_idx = _get_rgb_indices(channels, "post") or (11, 10, 9)
        pre_idx = _get_rgb_indices(channels, "pre") or (3, 2, 1)
        post_rgb = src.read(list(post_idx)).transpose(1, 2, 0)
        pre_rgb = src.read(list(pre_idx)).transpose(1, 2, 0)
    else:
        post_rgb = post_src.read([3, 2, 1]).transpose(1, 2, 0)
        pre_rgb = pre_src.read([3, 2, 1]).transpose(1, 2, 0)
    post_rgb = np.nan_to_num(post_rgb, nan=0.0, posinf=0.0, neginf=0.0)
    pre_rgb = np.nan_to_num(pre_rgb, nan=0.0, posinf=0.0, neginf=0.0)
    # VISUALIZATION ONLY — does not affect masks/polygons/summary.
    mask_viz_source = mask_final.astype(np.uint8)
    if viz_mode.lower() == "forest_change":
        if forest_ok is None or change_gate_mask is None:
            print("WARNING: viz_mode=forest_change but forest_ok/change_gate missing; using mask_final.")
        else:
            mask_viz_source = (mask_final.astype(bool) & forest_ok & change_gate_mask).astype(np.uint8)
            if mask_viz_source.sum() == 0 and mask_final.sum() > 0:
                print("WARNING: viz_mode=forest_change produced empty mask; using mask_final.")
                mask_viz_source = mask_final.astype(np.uint8)

    mask_viz, chosen_dilate_px, coverage_raw, coverage_viz = _compute_viz_mask(
        mask_viz_source,
        auto=viz_dilate_auto,
        target_coverage=viz_target_coverage,
        min_px=viz_dilate_min_px,
        max_px=viz_dilate_max_px,
    )
    print(
        "viz_auto: coverage_raw={:.6f} coverage_viz={:.6f} chosen_dilate_px={} outline_px={}".format(
            coverage_raw, coverage_viz, chosen_dilate_px, viz_outline_px
        )
    )
    overlay_full = _build_overlay(
        post_rgb,
        mask_viz,
        alpha=viz_alpha,
        color=viz_color,
        outline=viz_outline,
        outline_px=viz_outline_px,
    )
    boundary = mask_final.astype(bool) ^ ndi.binary_erosion(mask_final.astype(bool))
    if viz_outline_px > 1:
        boundary = ndi.binary_dilation(boundary, structure=_disk(viz_outline_px))
    overlay_boundary = _build_overlay(
        post_rgb,
        boundary,
        alpha=viz_boundary_alpha,
        color=viz_color,
        outline=False,
        outline_px=1,
    )
    _save_overlay(post_rgb, overlay_full, overlay_path)
    _save_overlay(post_rgb, overlay_boundary, out_dir / "overlay_boundary.png")
    footer_text = None
    if viz_footer:
        parts = [f"thr={threshold}", f"min_area_m2={min_area_m2}"]
        parts.append(f"forest_filter={bool(forest_filter_enabled)}")
        if forest_filter_enabled:
            parts.append(f"forest_thr={forest_thr}")
        if isinstance(cfg.get("change_gate", {}), dict) and cfg.get("change_gate", {}).get("enabled", False):
            parts.append("change_gate=on")
        footer_text = " | ".join(parts)
    _save_panel_images(
        pre_rgb,
        post_rgb,
        overlay_full,
        panel_path,
        show_legend=viz_show_legend,
        legend_loc=viz_legend_loc,
        title_prefix=viz_title_prefix,
        footer_text=footer_text,
        color=viz_color,
    )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_polygons",
                "total_area_m2",
                "total_area_ha",
                "threshold",
                "min_area_m2",
                "forest_filter_enabled",
                "forest_thr",
                "pre_forest_area_m2",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_polygons": int(len(gdf)),
                "total_area_m2": round(total_area_m2, 3),
                "total_area_ha": round(total_area_m2 / 10000.0, 3),
                "threshold": threshold,
                "min_area_m2": min_area_m2,
                "forest_filter_enabled": bool(forest_filter_enabled),
                "forest_thr": forest_thr,
                "pre_forest_area_m2": round(pre_area_m2, 3),
            }
        )

    if raster is not None:
        src.close()
    else:
        pre_src.close()
        post_src.close()

    # Zoom panel (visualization only)
    mask_for_zoom = mask_viz_source if mask_viz_source.sum() > 0 else mask_final.astype(np.uint8)
    panel_zoom_path = out_dir / "panel_zoom.png"
    zoom_pixels = 0
    if mask_for_zoom.sum() > 0:
        if viz_zoom_largest_component:
            labeled, num = ndi.label(mask_for_zoom.astype(bool))
            if num > 0:
                sizes = ndi.sum(mask_for_zoom.astype(bool), labeled, index=np.arange(1, num + 1))
                idx = int(np.argmax(sizes))
                slices = ndi.find_objects(labeled)
                sl = slices[idx] if slices and idx < len(slices) else None
                if sl is not None:
                    min_r, max_r = sl[0].start, sl[0].stop - 1
                    min_c, max_c = sl[1].start, sl[1].stop - 1
                else:
                    rows, cols = np.where(mask_for_zoom > 0)
                    min_r, max_r = int(rows.min()), int(rows.max())
                    min_c, max_c = int(cols.min()), int(cols.max())
            else:
                rows, cols = np.where(mask_for_zoom > 0)
                min_r, max_r = int(rows.min()), int(rows.max())
                min_c, max_c = int(cols.min()), int(cols.max())
        else:
            rows, cols = np.where(mask_for_zoom > 0)
            min_r, max_r = int(rows.min()), int(rows.max())
            min_c, max_c = int(cols.min()), int(cols.max())
        print(f"det bbox px: r[{min_r},{max_r}] c[{min_c},{max_c}]")
        min_r = max(0, min_r - zoom_margin_px)
        min_c = max(0, min_c - zoom_margin_px)
        max_r = min(height - 1, max_r + zoom_margin_px)
        max_c = min(width - 1, max_c + zoom_margin_px)
        zoom_pixels = int(mask_for_zoom[min_r : max_r + 1, min_c : max_c + 1].sum())
        print(f"zoom crop det pixels: {zoom_pixels}")
        pre_zoom = pre_rgb[min_r : max_r + 1, min_c : max_c + 1]
        post_zoom = post_rgb[min_r : max_r + 1, min_c : max_c + 1]
        overlay_zoom = overlay_full[min_r : max_r + 1, min_c : max_c + 1]
        boundary_zoom = overlay_boundary[min_r : max_r + 1, min_c : max_c + 1]
        _save_panel_images(
            pre_zoom,
            post_zoom,
            overlay_zoom,
            panel_zoom_path,
            show_legend=viz_show_legend,
            legend_loc=viz_legend_loc,
            title_prefix=viz_title_prefix,
            footer_text=footer_text,
            color=viz_color,
        )
        _save_overlay(post_zoom, boundary_zoom, out_dir / "overlay_boundary_zoom.png")
    else:
        print("WARNING: mask_final is empty; skipping panel_zoom.png")

    if mask_for_zoom.sum() > 0 and (max_r - min_r < 32 or max_c - min_c < 32):
        print("NOTE: bbox is very small; consider lowering threshold or increasing viz_dilate_max_px.")

    if debug:
        debug_dir = out_dir / "debug"
        _save_mask_png(mask_raw, debug_dir / "mask_raw.png")
        _save_mask_png(mask_morph.astype(np.uint8), debug_dir / "mask_morph.png")
        _save_mask_png(mask_cc.astype(np.uint8), debug_dir / "mask_cc.png")
        if forest_ok is not None:
            _save_mask_png(forest_ok.astype(np.uint8), debug_dir / "forest_ok.png")
        if change_gate_mask is not None:
            _save_mask_png(change_gate_mask.astype(np.uint8), debug_dir / "change_gate.png")
        _save_mask_png(det_mask.astype(np.uint8), debug_dir / "mask_postproc.png")
        _save_mask_png(mask_final.astype(np.uint8), debug_dir / "mask_final.png")

    gallery_images = []
    if tile_gallery > 0:
        tiles_dir.mkdir(parents=True, exist_ok=True)
        candidates = _tile_candidates(mask_for_zoom, viz_tile, viz_stride)
        candidates.sort(key=lambda t: t[2], reverse=True)
        for idx, (row, col, _) in enumerate(candidates[:tile_gallery], start=1):
            row_end = min(row + viz_tile, height)
            col_end = min(col + viz_tile, width)
            pre_tile = pre_rgb[row:row_end, col:col_end]
            post_tile = post_rgb[row:row_end, col:col_end]
            mask_tile = mask_for_zoom[row:row_end, col:col_end]
            tile_path = tiles_dir / f"tile_{idx:02d}_r{row}_c{col}.png"
            _save_tile_panel(
                pre_tile,
                post_tile,
                mask_tile,
                tile_path,
                alpha=viz_alpha,
                dilate_px=chosen_dilate_px,
                outline=viz_outline,
                outline_px=viz_outline_px,
                color=viz_color,
            )
            gallery_images.append(plt.imread(tile_path))
        if gallery_images:
            cols = min(4, len(gallery_images))
            _save_gallery(gallery_images, cols, gallery_path)

    # Gallery zoom by top-K components
    gallery_zoom_path = out_dir / "gallery_zoom.png"
    gallery_zoom_images = []
    if gallery_topk > 0 and mask_for_zoom.sum() > 0:
        labeled, num = ndi.label(mask_for_zoom.astype(bool))
        if num > 0:
            sizes = ndi.sum(mask_for_zoom.astype(bool), labeled, index=np.arange(1, num + 1))
            order = np.argsort(sizes)[::-1]
            slices = ndi.find_objects(labeled)
            for idx in order[:gallery_topk]:
                sl = slices[idx]
                if sl is None:
                    continue
                r0, r1 = sl[0].start, sl[0].stop
                c0, c1 = sl[1].start, sl[1].stop
                center_r = (r0 + r1) // 2
                center_c = (c0 + c1) // 2
                half = gallery_tile_px // 2
                rs = max(0, center_r - half)
                cs = max(0, center_c - half)
                re = min(height, rs + gallery_tile_px)
                ce = min(width, cs + gallery_tile_px)
                rs = max(0, re - gallery_tile_px)
                cs = max(0, ce - gallery_tile_px)
                pre_crop = pre_rgb[rs:re, cs:ce]
                post_crop = post_rgb[rs:re, cs:ce]
                overlay_crop = overlay_full[rs:re, cs:ce]
                panel = np.concatenate(
                    [
                        _scale_rgb(pre_crop),
                        _scale_rgb(post_crop),
                        overlay_crop,
                    ],
                    axis=1,
                )
                gallery_zoom_images.append(panel)
        if gallery_zoom_images:
            cols = min(4, len(gallery_zoom_images))
            _save_gallery(gallery_zoom_images, cols, gallery_zoom_path)
    if gallery_topk > 0:
        print(f"gallery_zoom tiles: {len(gallery_zoom_images)}")

    outputs = [mask_path, mask_full_path, overlay_path, panel_path, polygons_path, summary_path]
    if save_probs:
        outputs.insert(0, prob_path)
    if tile_gallery > 0 and gallery_path.exists():
        outputs.append(gallery_path)
    if panel_zoom_path.exists():
        outputs.append(panel_zoom_path)
    if gallery_zoom_path.exists():
        outputs.append(gallery_zoom_path)
    boundary_path = out_dir / "overlay_boundary.png"
    if boundary_path.exists():
        outputs.append(boundary_path)
    boundary_zoom_path = out_dir / "overlay_boundary_zoom.png"
    if boundary_zoom_path.exists():
        outputs.append(boundary_zoom_path)
    print("Outputs:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
