from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy as pix2xy
from scipy import ndimage as ndi


def _band_map(ds: rasterio.io.DatasetReader) -> Dict[str, int]:
    """
    Map band description -> 1-based band index.
    """
    m = {}
    if ds.descriptions:
        for i, d in enumerate(ds.descriptions, start=1):
            if d:
                m[str(d).strip()] = i
    return m


def _get_band(ds, name: str, fallback_idx: Optional[int] = None) -> np.ndarray:
    m = _band_map(ds)
    if name in m:
        return ds.read(m[name]).astype(np.float32)
    if fallback_idx is not None:
        return ds.read(fallback_idx).astype(np.float32)
    raise KeyError(f"Band '{name}' not found in descriptions, and no fallback_idx provided.")


def _safe_nd(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (a - b) / (a + b + eps)


def _component_centroid_rc(mask: np.ndarray) -> Tuple[float, float]:
    # returns (row, col) centroid
    ys, xs = np.nonzero(mask)
    return float(np.mean(ys)), float(np.mean(xs))


def _ring(mask: np.ndarray, r: int) -> np.ndarray:
    # ring around mask: dilate - mask
    if r <= 0:
        return np.zeros_like(mask, dtype=bool)
    dil = ndi.binary_dilation(mask, iterations=r)
    ring = np.logical_and(dil, ~mask)
    return ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacked", required=True, help="Path to stacked.tif (pre/post bands + optional dNDVI/dNBR).")
    ap.add_argument("--probs", required=True, help="Path to probs.tif from infer.")
    ap.add_argument("--mask", required=True, help="Path to final mask GeoTIFF (mask.tif or mask_full.tif).")
    ap.add_argument("--out_csv", default="outputs/inspect_components.csv")
    ap.add_argument("--out_geojson", default="outputs/inspect_points.geojson")
    ap.add_argument("--topk", type=int, default=200, help="How many components to keep (sorted by area desc).")
    ap.add_argument("--ring_px", type=int, default=6, help="Ring thickness (pixels) to compare inside vs around.")
    ap.add_argument("--min_area_m2", type=float, default=200.0)
    ap.add_argument("--water_ndwi_thr", type=float, default=0.15, help="If NDWI is high -> likely water.")
    ap.add_argument("--forest_pre_ndvi_thr", type=float, default=0.45, help="If pre NDVI low -> not forest.")
    ap.add_argument("--defor_dndvi_thr", type=float, default=-0.12, help="Deforestation hint: dNDVI below this.")
    ap.add_argument("--defor_dnbr_thr", type=float, default=-0.08, help="Deforestation hint: dNBR below this.")
    args = ap.parse_args()

    stacked_path = Path(args.stacked)
    probs_path = Path(args.probs)
    mask_path = Path(args.mask)
    out_csv = Path(args.out_csv)
    out_geojson = Path(args.out_geojson)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)

    # --- read probs + mask (must be same grid ideally) ---
    with rasterio.open(probs_path) as pds, rasterio.open(mask_path) as mds:
        probs = pds.read(1).astype(np.float32)
        m = mds.read(1)

        # mask could be 0/255 or 0/1
        mask = (m > 0)

        if probs.shape != mask.shape:
            raise ValueError(f"Shape mismatch probs{probs.shape} vs mask{mask.shape}. Use aligned outputs.")

        transform = pds.transform
        crs = pds.crs
        # pixel area (approx) in m² for projected CRS
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        px_area_m2 = float(px_w * px_h)

    # --- connected components on mask ---
    lab, n = ndi.label(mask)
    if n == 0:
        print("No detections (mask has 0 components).")
        return

    # component areas in pixels
    areas_px = ndi.sum(np.ones_like(mask, dtype=np.int32), labels=lab, index=np.arange(1, n + 1))
    areas_px = np.asarray(areas_px, dtype=np.float64)
    areas_m2 = areas_px * px_area_m2

    # filter by min area
    keep_ids = np.where(areas_m2 >= float(args.min_area_m2))[0] + 1
    if keep_ids.size == 0:
        print(f"All components filtered by min_area_m2={args.min_area_m2}.")
        return

    # sort by area desc
    keep_ids = keep_ids[np.argsort(areas_m2[keep_ids - 1])[::-1]]
    keep_ids = keep_ids[: int(args.topk)]

    # --- read stacked bands for diagnostics ---
    with rasterio.open(stacked_path) as sds:
        # Try to use stored dNDVI/dNBR first (best).
        bm = _band_map(sds)

        has_dndvi = "dNDVI" in bm
        has_dnbr = "dNBR" in bm

        # Needed base bands (Sentinel-2 style from your exporter)
        # pre: B3 (green), B8 (nir), B4 (red), B11 (swir1), B12 (swir2)
        pre_B3 = _get_band(sds, "pre_B3")
        pre_B8 = _get_band(sds, "pre_B8")
        pre_B4 = _get_band(sds, "pre_B4")
        pre_B11 = _get_band(sds, "pre_B11")
        # post
        post_B3 = _get_band(sds, "post_B3")
        post_B8 = _get_band(sds, "post_B8")
        post_B4 = _get_band(sds, "post_B4")
        post_B11 = _get_band(sds, "post_B11")

        # NDVI/NBR (compute even if present, for consistency)
        pre_ndvi = _safe_nd(pre_B8, pre_B4)
        post_ndvi = _safe_nd(post_B8, post_B4)
        pre_nbr = _safe_nd(pre_B8, pre_B11)
        post_nbr = _safe_nd(post_B8, post_B11)

        dndvi = _get_band(sds, "dNDVI") if has_dndvi else (post_ndvi - pre_ndvi)
        dnbr = _get_band(sds, "dNBR") if has_dnbr else (post_nbr - pre_nbr)

        # NDWI (water-ish): (G - NIR) / (G + NIR)
        pre_ndwi = _safe_nd(pre_B3, pre_B8)
        post_ndwi = _safe_nd(post_B3, post_B8)

        # sanity check grid match (assume same grid as probs/mask)
        if pre_ndvi.shape != mask.shape:
            raise ValueError(
                f"stacked shape {pre_ndvi.shape} != probs/mask shape {mask.shape}. "
                "If stacked differs, use aligned stacked for this run."
            )

    rows: List[dict] = []
    features: List[dict] = []

    for cid in keep_ids:
        cmask = (lab == cid)
        area_px = int(np.sum(cmask))
        area_m2 = float(area_px * px_area_m2)

        # probs stats inside
        p_inside = probs[cmask]
        p_mean = float(np.mean(p_inside)) if p_inside.size else 0.0
        p_max = float(np.max(p_inside)) if p_inside.size else 0.0

        # ring comparison (optional)
        ring = _ring(cmask, int(args.ring_px))
        # avoid going outside image (ring already boolean)
        ring_any = np.any(ring)

        # compute stats inside (and ring)
        def _stat(a: np.ndarray, msk: np.ndarray) -> Tuple[float, float]:
            v = a[msk]
            if v.size == 0:
                return 0.0, 0.0
            return float(np.mean(v)), float(np.median(v))

        pre_ndvi_mean, pre_ndvi_med = _stat(pre_ndvi, cmask)
        post_ndvi_mean, post_ndvi_med = _stat(post_ndvi, cmask)
        dndvi_mean, dndvi_med = _stat(dndvi, cmask)

        pre_nbr_mean, pre_nbr_med = _stat(pre_nbr, cmask)
        post_nbr_mean, post_nbr_med = _stat(post_nbr, cmask)
        dnbr_mean, dnbr_med = _stat(dnbr, cmask)

        pre_ndwi_mean, pre_ndwi_med = _stat(pre_ndwi, cmask)
        post_ndwi_mean, post_ndwi_med = _stat(post_ndwi, cmask)

        if ring_any:
            ring_dndvi_mean, _ = _stat(dndvi, ring)
            ring_dnbr_mean, _ = _stat(dnbr, ring)
            ring_pre_ndvi_mean, _ = _stat(pre_ndvi, ring)
        else:
            ring_dndvi_mean = ring_dnbr_mean = ring_pre_ndvi_mean = 0.0

        # heuristics
        likely_water = (pre_ndwi_mean > args.water_ndwi_thr) or (post_ndwi_mean > args.water_ndwi_thr)
        likely_forest_before = (pre_ndvi_mean >= args.forest_pre_ndvi_thr)

        defor_hint = (dndvi_mean <= args.defor_dndvi_thr) and (dnbr_mean <= args.defor_dnbr_thr)
        # stronger if inside drops more than ring
        defor_contrast = (dndvi_mean - ring_dndvi_mean) < -0.05 or (dnbr_mean - ring_dnbr_mean) < -0.05

        likely_deforestation = (likely_forest_before and defor_hint and defor_contrast and (not likely_water))

        # centroid -> map coords
        r0, c0 = _component_centroid_rc(cmask)
        x0, y0 = pix2xy(transform, r0, c0, offset="center")

        rows.append(
            dict(
                id=int(cid),
                area_px=area_px,
                area_m2=area_m2,
                prob_mean=p_mean,
                prob_max=p_max,
                pre_ndvi_mean=pre_ndvi_mean,
                post_ndvi_mean=post_ndvi_mean,
                dndvi_mean=dndvi_mean,
                pre_nbr_mean=pre_nbr_mean,
                post_nbr_mean=post_nbr_mean,
                dnbr_mean=dnbr_mean,
                pre_ndwi_mean=pre_ndwi_mean,
                post_ndwi_mean=post_ndwi_mean,
                ring_pre_ndvi_mean=ring_pre_ndvi_mean,
                ring_dndvi_mean=ring_dndvi_mean,
                ring_dnbr_mean=ring_dnbr_mean,
                likely_water=int(likely_water),
                likely_forest_before=int(likely_forest_before),
                defor_hint=int(defor_hint),
                defor_contrast=int(defor_contrast),
                likely_deforestation=int(likely_deforestation),
                x=float(x0),
                y=float(y0),
            )
        )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": int(cid),
                    "area_m2": area_m2,
                    "prob_mean": p_mean,
                    "prob_max": p_max,
                    "dndvi_mean": dndvi_mean,
                    "dnbr_mean": dnbr_mean,
                    "pre_ndvi_mean": pre_ndvi_mean,
                    "pre_ndwi_mean": pre_ndwi_mean,
                    "likely_water": int(likely_water),
                    "likely_deforestation": int(likely_deforestation),
                },
                "geometry": {"type": "Point", "coordinates": [float(x0), float(y0)]},
            }
        )

    # write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # write GeoJSON points
    gj = {"type": "FeatureCollection", "features": features, "crs": {"type": "name", "properties": {"name": str(crs)}}}
    out_geojson.write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")

    # quick console summary
    n_water = sum(r["likely_water"] for r in rows)
    n_defor = sum(r["likely_deforestation"] for r in rows)
    print(f"Components kept: {len(rows)} (min_area_m2={args.min_area_m2})")
    print(f"Likely water/artefact: {n_water}")
    print(f"Likely deforestation (heuristic): {n_defor}")
    print(f"Wrote:\n  {out_csv}\n  {out_geojson}")


if __name__ == "__main__":
    main()
