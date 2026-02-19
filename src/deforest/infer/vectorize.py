from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import Affine


def mask_to_geojson(mask: np.ndarray, transform, crs: str | None) -> dict:
    if not isinstance(transform, Affine):
        transform = Affine(*transform)

    shapes = features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
    features_list = []
    for geom, val in shapes:
        if val == 0:
            continue
        features_list.append({"type": "Feature", "geometry": geom, "properties": {}})

    geojson = {
        "type": "FeatureCollection",
        "features": features_list,
    }
    if crs:
        geojson["crs"] = {"type": "name", "properties": {"name": crs}}
    return geojson


def save_geojson(geojson: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)
