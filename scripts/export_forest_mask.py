from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import ee
import yaml


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
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
        except Exception as exc:
            if not project:
                raise SystemExit(
                    "Earth Engine requires a project. Set eeProject in the config or "
                    "export EE_PROJECT/GOOGLE_CLOUD_PROJECT."
                ) from exc
            raise


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


def _require(cfg: dict, key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing config key: {key}")
    return cfg[key]


def _default_dw_dates(cfg: dict) -> tuple[str, str]:
    pre_start = cfg.get("preStart")
    post_end = cfg.get("postEnd")
    if pre_start and post_end:
        return str(pre_start), str(post_end)
    today = dt.date.today()
    start = today.replace(year=today.year - 2)
    return start.isoformat(), today.isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project = cfg.get("eeProject") or cfg.get("project") or os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    ee_init(project)

    aoi_mode = str(cfg.get("aoiMode", "geojson")).lower()
    if aoi_mode == "auto":
        search_box = cfg.get("searchBox")
        if not search_box:
            raise ValueError("searchBox missing for aoiMode=auto")
        aoi = ee.Geometry.Rectangle(search_box)
        geojson = None
    else:
        aoi_path = Path(_require(cfg, "aoi_geojson_path"))
        geojson = read_geojson(aoi_path)
        aoi = geojson_to_ee_geometry(geojson)

    export_crs = str(cfg.get("exportCRS", "")).strip()
    if not export_crs:
        if aoi_mode == "auto":
            export_crs = infer_utm_epsg_from_bbox(cfg.get("searchBox", []))
        else:
            export_crs = infer_utm_epsg_from_geojson(geojson or {})

    scale = float(cfg.get("scale", 30))
    forest_source = str(cfg.get("forestSource", "hansen")).lower()
    treecover_thresh = float(cfg.get("treecoverThresh", 30))
    dw_trees_prob_thresh = float(cfg.get("dwTreesProbThresh", 0.6))
    dw_start = cfg.get("dwStart")
    dw_end = cfg.get("dwEnd")
    if not dw_start or not dw_end:
        dw_start, dw_end = _default_dw_dates(cfg)

    gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    forest_h = gfc.select("treecover2000").gte(treecover_thresh)
    treecover2000 = gfc.select("treecover2000").toUint8().rename("treecover2000")

    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(aoi)
        .filterDate(str(dw_start), str(dw_end))
        .select("trees")
        .mean()
    )
    forest_dw = dw.gte(dw_trees_prob_thresh)

    if forest_source == "hansen":
        forest = forest_h
    elif forest_source == "dynamic_world":
        forest = forest_dw
    elif forest_source == "hansen_and_dw":
        forest = forest_h.And(forest_dw)
    else:
        raise ValueError("forestSource must be hansen, dynamic_world, or hansen_and_dw")

    forest01 = forest.rename("forest01").toUint8()
    export_img = forest01
    if forest_source in ("hansen", "hansen_and_dw"):
        export_img = export_img.addBands(treecover2000)

    region_mode = str(cfg.get("regionMode", "aoi")).lower()
    region = aoi.bounds() if region_mode == "bounds" else aoi

    task = ee.batch.Export.image.toDrive(
        image=export_img,
        description=str(cfg.get("taskDescription", "forest_mask_export")),
        folder=str(cfg.get("exportFolder", "deforest_demo")),
        fileNamePrefix=str(cfg.get("fileNamePrefix", "forest_mask")),
        region=region,
        scale=scale,
        crs=export_crs,
        maxPixels=float(cfg.get("maxPixels", 1e13)),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )

    task.start()
    print(f"Task started: {task.id}")
    print("Run example: python scripts/export_forest_mask.py --config configs/forest_mask.yaml")


if __name__ == "__main__":
    main()
