from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import ee
import yaml
from tqdm import tqdm


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_geojson(path: str | Path) -> dict:
    """Reads GeoJSON safely even if it has UTF-8 BOM."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON not found: {p.resolve()}")
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def ee_init(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize(project=project)


def geojson_to_ee_geometry(geojson: dict) -> ee.Geometry:
    gtype = geojson.get("type")
    if gtype == "FeatureCollection":
        return ee.FeatureCollection(geojson).geometry()
    if gtype == "Feature":
        return ee.Feature(geojson).geometry()
    return ee.Geometry(geojson)


def _extract_coords(geom: dict) -> list[tuple[float, float]]:
    geom_type = geom.get("type")
    coords: list[tuple[float, float]] = []
    if geom_type == "Feature":
        return _extract_coords(geom.get("geometry", {}))
    if geom_type == "FeatureCollection":
        for feat in geom.get("features", []):
            coords.extend(_extract_coords(feat))
        return coords
    if geom_type == "Polygon":
        for ring in geom.get("coordinates", []):
            for xy in ring:
                coords.append((xy[0], xy[1]))
        return coords
    if geom_type == "MultiPolygon":
        for poly in geom.get("coordinates", []):
            for ring in poly:
                for xy in ring:
                    coords.append((xy[0], xy[1]))
        return coords
    if geom_type == "Point":
        xy = geom.get("coordinates", [])
        if len(xy) >= 2:
            coords.append((xy[0], xy[1]))
        return coords
    if geom_type == "MultiPoint":
        for xy in geom.get("coordinates", []):
            coords.append((xy[0], xy[1]))
        return coords
    return coords


def _infer_utm_epsg_from_lon_lat(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) / 6.0) + 1
    epsg_base = 32600 if lat >= 0 else 32700
    return f"EPSG:{epsg_base + zone}"


def infer_utm_epsg_from_geojson(geojson: dict) -> str:
    coords = _extract_coords(geojson)
    if not coords:
        raise ValueError("AOI GeoJSON has no coordinates.")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lon = (min(lons) + max(lons)) / 2.0
    lat = (min(lats) + max(lats)) / 2.0
    return _infer_utm_epsg_from_lon_lat(lon, lat)


def infer_utm_epsg_from_bbox(bbox: list[float]) -> str:
    if len(bbox) != 4:
        raise ValueError("searchBox must be [minLon, minLat, maxLon, maxLat]")
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    lon = (min_lon + max_lon) / 2.0
    lat = (min_lat + max_lat) / 2.0
    return _infer_utm_epsg_from_lon_lat(lon, lat)


# ---------- EE image helpers (float32 discipline) ----------
def f32(img: ee.Image) -> ee.Image:
    return ee.Image(img).toFloat()


def add_indices(img: ee.Image) -> ee.Image:
    img = ee.Image(img)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    return img.addBands([f32(ndvi), f32(nbr)])


def scl_mask(img: ee.Image, exclude_water: bool) -> ee.Image:
    scl = img.select("SCL")
    # Strict clear mask: vegetation + bare soil only (exclude unclassified/cloudy artifacts).
    clear = scl.eq(4).Or(scl.eq(5))
    if not exclude_water:
        clear = clear.Or(scl.eq(6))  # water
    return clear


def qa60_mask(img: ee.Image) -> ee.Image:
    qa60 = img.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    cloud_free = qa60.bitwiseAnd(cloud_bit).eq(0)
    cirrus_free = qa60.bitwiseAnd(cirrus_bit).eq(0)
    return cloud_free.And(cirrus_free)


def apply_mask(img: ee.Image, use_scl_mask: bool, exclude_water: bool, use_qa60_mask: bool) -> ee.Image:
    img = ee.Image(img)
    if use_scl_mask:
        img = img.updateMask(scl_mask(img, exclude_water))
    if use_qa60_mask:
        img = img.updateMask(qa60_mask(img))
    return img


def get_s2_collection(
    aoi: ee.Geometry,
    start: str,
    end: str,
    use_scl_mask: bool,
    use_qa60_mask: bool,
    exclude_water: bool,
    max_cloud_pct: float,
) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(lambda im: apply_mask(im, use_scl_mask, exclude_water, use_qa60_mask))
        .map(add_indices)
    )


def make_composite(
    aoi: ee.Geometry,
    start: str,
    end: str,
    use_scl_mask: bool,
    use_qa60_mask: bool,
    exclude_water: bool,
    max_cloud_pct: float,
) -> ee.Image:
    s2 = get_s2_collection(aoi, start, end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
    comp = s2.qualityMosaic("NDVI").select(["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NBR"])
    return f32(comp)


# ---------- Label / density helpers ----------
def get_gfc_asset(cfg: dict) -> str:
    # Ты можешь менять в yaml: gfcAsset: "UMD/hansen/global_forest_change_2023_v1_11"
    return str(cfg.get("gfcAsset", "UMD/hansen/global_forest_change_2023_v1_11"))


def build_loss_mask_only(gfc: ee.Image, years: list[int]) -> ee.Image:
    """Hansen loss only (no forest, no DW)."""
    lossyear = gfc.select("lossyear")
    yrs = [int(y) for y in years]
    loss = lossyear.remap(yrs, [1] * len(yrs), 0).eq(1)
    return loss


def build_weak_label_mask(
    gfc: ee.Image,
    region: ee.Geometry,
    pre_start: str,
    post_end: str,
    treecover_thresh: float,
    target_loss_years: list[int],
    use_dynamic_world: bool,
    dw_trees_prob_thresh: float,
) -> ee.Image:
    forest = gfc.select("treecover2000").gte(treecover_thresh)
    loss = build_loss_mask_only(gfc, target_loss_years)
    weak = forest.And(loss)

    if use_dynamic_world:
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(region)
            .filterDate(pre_start, post_end)
            .select("trees")
            .mean()
        )
        weak = weak.And(dw.gte(dw_trees_prob_thresh))

    return weak  # 0/1 image


def build_forest_mask(
    gfc: ee.Image,
    region: ee.Geometry,
    pre_start: str,
    post_end: str,
    treecover_thresh: float,
    use_dynamic_world: bool,
    dw_trees_prob_thresh: float,
) -> ee.Image:
    forest = gfc.select("treecover2000").gte(treecover_thresh)
    if use_dynamic_world:
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(region)
            .filterDate(pre_start, post_end)
            .select("trees")
            .mean()
        )
        forest = forest.And(dw.gte(dw_trees_prob_thresh))
    return forest


def count_positive_pixels(mask01: ee.Image, region: ee.Geometry, scale: float) -> ee.Number:
    """Counts positive pixels (mask==1) inside region."""
    # mask01 expected 0/1 (not selfMasked)
    stats = mask01.rename("m").reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1e13,
    )
    return ee.Number(stats.get("m"))


def auto_select_aois(
    *,
    cfg: dict,
    search_box: list[float],
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    scale_for_check: float,
    treecover_thresh: float,
    target_loss_years: list[int],
    use_dynamic_world: bool,
    dw_trees_prob_thresh: float,
    use_scl_mask: bool,
    use_qa60_mask: bool,
    exclude_water: bool,
    max_cloud_pct: float,
    dndvi_thr: float,
    dnbr_thr: float,
    label_dilate: int,
    debug: bool,
    density_radius_m: float,
    pick_top_fraction: float,
    chip_km: float,
    num_to_find: int,
    seed: int = 42,
) -> list[ee.Geometry]:
    """
    Picks multiple AOIs around dense Hansen loss pixels, verifying each has positives.
    Returns a list of valid AOI geometries.
    """
    if len(search_box) != 4:
        raise ValueError("searchBox must be [minLon, minLat, maxLon, maxLat]")

    region = ee.Geometry.Rectangle(search_box)
    gfc = ee.Image(get_gfc_asset(cfg))

    density_years = cfg.get("densityLossYears", None) or list(target_loss_years)
    density_years = [int(y) for y in density_years]
    widen_years_fallback = [int(y) for y in cfg.get("densityWidenYears", [22, 23])]

    max_candidates = int(cfg.get("autoMaxCandidates", 200))
    max_tries = int(cfg.get("autoMaxTries", 60))
    min_pos_pixels = int(cfg.get("minPositivePixels", 1))
    chip_m = (chip_km * 1000.0) / 2.0

    def make_candidates(years_for_density: list[int]) -> ee.FeatureCollection:
        """
        Generates candidate points for AOIs by sampling from forest loss areas.
        """
        loss_mask = build_loss_mask_only(gfc, years_for_density)
        forest_mask = gfc.select("treecover2000").gte(treecover_thresh)
        candidate_source = forest_mask.And(loss_mask).selfMask().rename("class")
        return candidate_source.stratifiedSample(
            numPoints=max_candidates,
            classBand='class',
            region=region,
            scale=250,
            seed=seed,
            geometries=True
        )

    def aoi_from_point_geom(pt_geom: ee.Geometry) -> ee.Geometry:
        return pt_geom.buffer(chip_m).bounds()

    gfc_forest = gfc.select("treecover2000").gte(treecover_thresh)
    gfc_loss = build_loss_mask_only(gfc, target_loss_years)
    simple_gfc_mask = gfc_forest.And(gfc_loss)

    def check_aoi_cheap(aoi: ee.Geometry) -> int:
        pos = count_positive_pixels(simple_gfc_mask, aoi, scale=scale_for_check).getInfo()
        return int(pos or 0)

    def check_aoi_full(aoi: ee.Geometry) -> int:
        label = build_weak_label_mask(
            gfc=gfc, region=aoi, pre_start=pre_start, post_end=post_end,
            treecover_thresh=treecover_thresh, target_loss_years=target_loss_years,
            use_dynamic_world=use_dynamic_world, dw_trees_prob_thresh=dw_trees_prob_thresh,
        )
        pre = make_composite(aoi, pre_start, pre_end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
        post = make_composite(aoi, post_start, post_end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
        dndvi = f32(post.select("NDVI").subtract(pre.select("NDVI")).rename("dNDVI"))
        dnbr = f32(post.select("NBR").subtract(pre.select("NBR")).rename("dNBR"))
        change = dndvi.lt(dndvi_thr).Or(dnbr.lt(dnbr_thr))
        label = label.And(change)
        if label_dilate > 0:
            label = label.focal_max(label_dilate)
        pos = count_positive_pixels(label, aoi, scale=scale_for_check).getInfo()
        return int(pos or 0)

    def try_pick_from_candidates(
        cands: ee.FeatureCollection, *, existing_aois: Optional[list[ee.Geometry]] = None, needed: Optional[int] = None
    ) -> list[ee.Geometry]:
        found_aois: list[ee.Geometry] = list(existing_aois or [])
        target_total = int(needed if needed is not None else num_to_find)
        if target_total <= len(found_aois):
            return found_aois
        lst = cands.toList(max_candidates)
        tries = max_tries
        to_find = target_total - len(found_aois)
        print(f"[AUTO] Checking up to {tries} candidates to find {to_find} AOI(s)...")

        for i in tqdm(range(tries), desc="Finding suitable AOIs"):
            try:
                feat = ee.Feature(lst.get(i))
                pt = feat.geometry()
                aoi = aoi_from_point_geom(pt)

                if any(aoi.intersects(existing_aoi, maxError=1).getInfo() for existing_aoi in found_aois):
                    tqdm.write(f"\n[AUTO] Skipping candidate {i} as it overlaps with an already found AOI.")
                    continue

                gfc_pos = check_aoi_cheap(aoi)
                if gfc_pos < min_pos_pixels:
                    continue

                tqdm.write(f"\n[AUTO] Candidate {i} has potential ({gfc_pos} GFC pixels). Performing full check...")
                full_pos = check_aoi_full(aoi)

                if full_pos >= min_pos_pixels:
                    tqdm.write(f"\n[AUTO] Found valid AOI at candidate {i} (final_positive_pixels={full_pos}).")
                    found_aois.append(aoi)
                    if len(found_aois) >= target_total:
                        tqdm.write(f"\n[AUTO] Found target of {target_total} AOI(s).")
                        break
            except ee.EEException as e:
                if "List.get: index" in str(e):
                    tqdm.write(f"\n[AUTO] All candidates checked ({i} total).")
                    break
                tqdm.write(f"[AUTO] WARN: Skipping candidate {i} due to GEE error: {e}")
                continue
        return found_aois

    print(f"[AUTO] densityLossYears={density_years} targetLossYears={target_loss_years}")
    cands = make_candidates(density_years)
    found = try_pick_from_candidates(cands, existing_aois=[], needed=num_to_find)
    if len(found) < num_to_find:
        print(f"[AUTO] Found only {len(found)}/{num_to_find} AOIs. Widening density years -> {widen_years_fallback}")
        cands2 = make_candidates(widen_years_fallback)
        found = try_pick_from_candidates(cands2, existing_aois=found, needed=num_to_find)
    
    if not found:
        print("[AUTO] No positives found. Falling back to center of searchBox for one AOI.")
        center = ee.Geometry.Point([(search_box[0] + search_box[2]) / 2.0, (search_box[1] + search_box[3]) / 2.0])
        return [center.buffer((chip_km * 1000.0) / 2.0).bounds()]

    return found[:num_to_find]


def build_label_image_for_aoi(
    aoi: ee.Geometry,
    *,
    cfg: dict,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    use_scl_mask: bool,
    use_qa60_mask: bool,
    exclude_water: bool,
    max_cloud_pct: float,
    treecover_thresh: float,
    target_loss_years: list[int],
    use_dynamic_world: bool,
    dw_trees_prob_thresh: float,
    dndvi_thr: float,
    dnbr_thr: float,
    label_dilate: int,
    pre: Optional[ee.Image] = None,
    post: Optional[ee.Image] = None,
    dndvi: Optional[ee.Image] = None,
    dnbr: Optional[ee.Image] = None,
) -> ee.Image:
    gfc = ee.Image(get_gfc_asset(cfg))
    if pre is None:
        pre = make_composite(aoi, pre_start, pre_end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
    if post is None:
        post = make_composite(aoi, post_start, post_end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
    if dndvi is None:
        dndvi = f32(post.select("NDVI").subtract(pre.select("NDVI")).rename("dNDVI"))
    if dnbr is None:
        dnbr = f32(post.select("NBR").subtract(pre.select("NBR")).rename("dNBR"))

    weak = build_weak_label_mask(
        gfc=gfc,
        region=aoi,
        pre_start=pre_start,
        post_end=post_end,
        treecover_thresh=treecover_thresh,
        target_loss_years=target_loss_years,
        use_dynamic_world=use_dynamic_world,
        dw_trees_prob_thresh=dw_trees_prob_thresh,
    )
    change = dndvi.lt(dndvi_thr).Or(dnbr.lt(dnbr_thr))
    label = weak.And(change)
    if label_dilate > 0:
        label = label.focal_max(label_dilate)
    return label.unmask(0).rename("label")


def preview_aoi_positive_counts(
    aois: list[ee.Geometry],
    *,
    cfg: dict,
    pre_start: str,
    pre_end: str,
    post_start: str,
    post_end: str,
    use_scl_mask: bool,
    use_qa60_mask: bool,
    exclude_water: bool,
    max_cloud_pct: float,
    treecover_thresh: float,
    target_loss_years: list[int],
    use_dynamic_world: bool,
    dw_trees_prob_thresh: float,
    dndvi_thr: float,
    dnbr_thr: float,
    label_dilate: int,
    scale_for_check: float,
) -> None:
    print(f"[PREVIEW] Estimating positives at scale={scale_for_check} m")
    total_pos = 0
    total_area_m2 = 0.0
    ok = 0

    for idx, aoi in enumerate(aois, start=1):
        try:
            label = build_label_image_for_aoi(
                aoi,
                cfg=cfg,
                pre_start=pre_start,
                pre_end=pre_end,
                post_start=post_start,
                post_end=post_end,
                use_scl_mask=use_scl_mask,
                use_qa60_mask=use_qa60_mask,
                exclude_water=exclude_water,
                max_cloud_pct=max_cloud_pct,
                treecover_thresh=treecover_thresh,
                target_loss_years=target_loss_years,
                use_dynamic_world=use_dynamic_world,
                dw_trees_prob_thresh=dw_trees_prob_thresh,
                dndvi_thr=dndvi_thr,
                dnbr_thr=dnbr_thr,
                label_dilate=label_dilate,
            )
            pos = int(count_positive_pixels(label, aoi, scale=scale_for_check).getInfo() or 0)
            area_m2 = float(aoi.area(maxError=1).getInfo() or 0.0)
            px_total = int(round(area_m2 / (scale_for_check * scale_for_check))) if scale_for_check > 0 else 0
            pos_ratio = (pos / px_total) if px_total > 0 else 0.0
            print(
                "[PREVIEW] AOI {}: positives={} area_km2={:.2f} pos_ratio={:.4%}".format(
                    idx, pos, area_m2 / 1e6, pos_ratio
                )
            )
            total_pos += pos
            total_area_m2 += area_m2
            ok += 1
        except ee.EEException as e:
            print(f"[PREVIEW] AOI {idx}: failed to evaluate positives ({e})")

    if ok == 0:
        print("[PREVIEW] No AOIs were successfully evaluated.")
        return

    total_px = int(round(total_area_m2 / (scale_for_check * scale_for_check))) if scale_for_check > 0 else 0
    total_ratio = (total_pos / total_px) if total_px > 0 else 0.0
    print(
        "[PREVIEW] TOTAL: aois_ok={}/{} positives={} area_km2={:.2f} pos_ratio={:.4%}".format(
            ok, len(aois), total_pos, total_area_m2 / 1e6, total_ratio
        )
    )


def generate_stack_for_aoi(
    aoi: ee.Geometry, *, cfg: dict, pre_start: str, pre_end: str, post_start: str, post_end: str,
    use_scl_mask: bool, use_qa60_mask: bool, exclude_water: bool, max_cloud_pct: float,
    treecover_thresh: float, target_loss_years: list[int], use_dynamic_world: bool,
    dw_trees_prob_thresh: float, dndvi_thr: float, dnbr_thr: float, label_dilate: int
) -> ee.Image:
    """Generates the full image stack for a single AOI."""
    pre = make_composite(aoi, pre_start, pre_end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
    post = make_composite(aoi, post_start, post_end, use_scl_mask, use_qa60_mask, exclude_water, max_cloud_pct)
    dndvi = f32(post.select("NDVI").subtract(pre.select("NDVI")).rename("dNDVI"))
    dnbr = f32(post.select("NBR").subtract(pre.select("NBR")).rename("dNBR"))

    label = f32(
        build_label_image_for_aoi(
            aoi,
            cfg=cfg,
            pre_start=pre_start,
            pre_end=pre_end,
            post_start=post_start,
            post_end=post_end,
            use_scl_mask=use_scl_mask,
            use_qa60_mask=use_qa60_mask,
            exclude_water=exclude_water,
            max_cloud_pct=max_cloud_pct,
            treecover_thresh=treecover_thresh,
            target_loss_years=target_loss_years,
            use_dynamic_world=use_dynamic_world,
            dw_trees_prob_thresh=dw_trees_prob_thresh,
            dndvi_thr=dndvi_thr,
            dnbr_thr=dnbr_thr,
            label_dilate=label_dilate,
            pre=pre,
            post=post,
            dndvi=dndvi,
            dnbr=dnbr,
        )
    )

    stacked = (
        pre.rename(["pre_B2", "pre_B3", "pre_B4", "pre_B8", "pre_B11", "pre_B12", "pre_NDVI", "pre_NBR"])
        .addBands(post.rename(["post_B2", "post_B3", "post_B4", "post_B8", "post_B11", "post_B12", "post_NDVI", "post_NBR"]))
        .addBands(dndvi)
        .addBands(dnbr)
        .addBands(label)
    )
    return f32(stacked)


def _require(cfg: dict, key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing config key: {key}")
    return cfg[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project = cfg.get("eeProject") or cfg.get("project") or None
    ee_init(project)

    aoi_mode = str(cfg.get("aoiMode", "geojson")).lower()

    pre_start, pre_end = str(_require(cfg, "preStart")), str(_require(cfg, "preEnd"))
    post_start, post_end = str(_require(cfg, "postStart")), str(_require(cfg, "postEnd"))

    scale = float(cfg.get("scale", 10))
    max_cloud_pct = float(cfg.get("maxCloudPct", 60))
    use_scl_mask = bool(cfg.get("useSCLMask", True))
    use_qa60_mask = bool(cfg.get("useQA60Mask", True))
    exclude_water = bool(cfg.get("excludeWater", True))
    treecover_thresh = float(cfg.get("treecoverThresh", 30))
    target_loss_years = [int(y) for y in cfg.get("targetLossYears", [22, 23])]
    use_dynamic_world = bool(cfg.get("useDynamicWorld", True))
    dw_trees_prob_thresh = float(cfg.get("dwTreesProbThresh", 0.6))
    dndvi_thr = float(cfg.get("dndviThr", -0.2))
    dnbr_thr = float(cfg.get("dnbrThr", -0.15))
    export_forest_mask = bool(cfg.get("exportForestMask", True))
    debug = bool(cfg.get("debug", False))
    label_dilate = int(cfg.get("labelDilatePixels", 0))
    preview_only = bool(cfg.get("previewOnly", False) or args.preview_only)
    positive_check_scale = float(cfg.get("positiveCheckScale", 30))

    aois = []
    geojson_for_crs = None
    search_box_for_crs = None

    if aoi_mode == "auto":
        search_box = cfg.get("searchBox", [78.0, 47.0, 87.5, 52.5])
        search_box_for_crs = search_box
        aois = auto_select_aois(
            cfg=cfg, search_box=search_box,
            pre_start=pre_start, pre_end=pre_end, post_start=post_start, post_end=post_end,
            scale_for_check=30, treecover_thresh=treecover_thresh, target_loss_years=target_loss_years,
            use_dynamic_world=use_dynamic_world, dw_trees_prob_thresh=dw_trees_prob_thresh,
            use_scl_mask=use_scl_mask, use_qa60_mask=use_qa60_mask, exclude_water=exclude_water,
            max_cloud_pct=max_cloud_pct, dndvi_thr=dndvi_thr, dnbr_thr=dnbr_thr,
            label_dilate=label_dilate, debug=debug,
            density_radius_m=float(cfg.get("densityRadiusM", 800)),
            pick_top_fraction=float(cfg.get("pickTopFraction", 0.90)),
            chip_km=float(cfg.get("chipKm", 10)),
            num_to_find=int(cfg.get("autoAoiCount", 1)),
            seed=int(cfg.get("seed", 42)),
        )
    else:
        aoi_path = Path(_require(cfg, "aoi_geojson_path"))
        geojson = read_geojson(aoi_path)
        geojson_for_crs = geojson
        aois = [geojson_to_ee_geometry(geojson)]

    if not aois:
        print("No AOIs found or provided. Exiting.")
        return
    print(f"Found {len(aois)} AOI(s) to process.")

    export_crs = str(cfg.get("exportCRS", "")).strip()
    if not export_crs:
        if search_box_for_crs:
            export_crs = infer_utm_epsg_from_bbox(search_box_for_crs)
        elif geojson_for_crs:
            export_crs = infer_utm_epsg_from_geojson(geojson_for_crs)
    print(f"AOI mode: {aoi_mode}, Resolved CRS: {export_crs}, Scale: {scale} m")

    full_region = aois[0]
    for i in range(1, len(aois)):
        full_region = full_region.union(aois[i], maxError=1)

    print_diagnostics = bool(cfg.get("printDiagnostics", True))
    if print_diagnostics:
        aoi_area_km2 = full_region.area(maxError=1).divide(1e6).getInfo()
        print(f"Total AOI area: {aoi_area_km2:.2f} km2")

    if preview_only:
        preview_aoi_positive_counts(
            aois,
            cfg=cfg,
            pre_start=pre_start,
            pre_end=pre_end,
            post_start=post_start,
            post_end=post_end,
            use_scl_mask=use_scl_mask,
            use_qa60_mask=use_qa60_mask,
            exclude_water=exclude_water,
            max_cloud_pct=max_cloud_pct,
            treecover_thresh=treecover_thresh,
            target_loss_years=target_loss_years,
            use_dynamic_world=use_dynamic_world,
            dw_trees_prob_thresh=dw_trees_prob_thresh,
            dndvi_thr=dndvi_thr,
            dnbr_thr=dnbr_thr,
            label_dilate=label_dilate,
            scale_for_check=positive_check_scale,
        )
        print("Preview mode enabled: skipping EE export task creation.")
        return

    image_stacks = []
    for aoi in aois:
        stack = generate_stack_for_aoi(
            aoi, cfg=cfg, pre_start=pre_start, pre_end=pre_end, post_start=post_start, post_end=post_end,
            use_scl_mask=use_scl_mask, use_qa60_mask=use_qa60_mask, exclude_water=exclude_water,
            max_cloud_pct=max_cloud_pct, treecover_thresh=treecover_thresh,
            target_loss_years=target_loss_years, use_dynamic_world=use_dynamic_world,
            dw_trees_prob_thresh=dw_trees_prob_thresh, dndvi_thr=dndvi_thr,
            dnbr_thr=dnbr_thr, label_dilate=label_dilate
        )
        image_stacks.append(stack)

    mosaic = ee.ImageCollection.fromImages(image_stacks).mosaic()
    region_mode = str(cfg.get("regionMode", "aoi")).lower()
    export_region = full_region.bounds() if region_mode == "bounds" else full_region
    print(f"Export region mode: {region_mode}")

    task = ee.batch.Export.image.toDrive(
        image=mosaic,
        description=str(cfg.get("taskDescription", "deforest_stacked_image")),
        folder=str(cfg.get("exportFolder", "deforest_demo")),
        fileNamePrefix=str(cfg.get("fileNamePrefix", "stacked")),
        region=export_region,
        scale=scale, crs=export_crs, maxPixels=float(cfg.get("maxPixels", 1e13)),
        fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True},
    )
    task.start()
    print(f"Task started: {task.id}")

    if export_forest_mask:
        gfc = ee.Image(get_gfc_asset(cfg))
        forest_masks = []
        for aoi in aois:
            mask = build_forest_mask(
                gfc=gfc, region=aoi, pre_start=pre_start, post_end=post_end,
                treecover_thresh=treecover_thresh, use_dynamic_world=use_dynamic_world,
                dw_trees_prob_thresh=dw_trees_prob_thresh
            ).rename("forest_mask")
            forest_masks.append(mask)
        
        forest_mosaic = ee.ImageCollection.fromImages(forest_masks).mosaic()
        forest_task = ee.batch.Export.image.toDrive(
            image=f32(forest_mosaic),
            description=str(cfg.get("forestTaskDescription", "deforest_forest_mask")),
            folder=str(cfg.get("exportFolder", "deforest_demo")),
            fileNamePrefix=str(cfg.get("forestFileNamePrefix", "forest_mask")),
            region=export_region,
            scale=scale, crs=export_crs, maxPixels=float(cfg.get("maxPixels", 1e13)),
            fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True},
        )
        forest_task.start()
        print(f"Forest mask task started: {forest_task.id}")

    poll_seconds = int(cfg.get("pollSeconds", 30))
    while True:
        status = task.status()
        state = status.get("state")
        print(f"Main task state: {state}")
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            if state != "COMPLETED":
                print(f"Error: {status.get('error_message')}")
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
