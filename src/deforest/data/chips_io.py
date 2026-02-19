from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import rasterio


def save_npz(path: str | Path, x: np.ndarray, y: np.ndarray, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(path, x=x.astype(np.float32), y=y.astype(np.uint8), meta=meta_json)


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.uint8)
        meta_raw = data["meta"]
        meta_str = meta_raw.item() if hasattr(meta_raw, "item") else str(meta_raw)
        meta = json.loads(meta_str)
    return x, y, meta


def list_npz(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return sorted(data_dir.rglob("*.npz"))


def geotiff_to_npz(tif_path: str | Path, out_dir: str | Path, label_band: int = -1) -> list[Path]:
    """
    Convert a multiband GeoTIFF into per-tile npz files.
    Assumes the last band is the label mask by default.
    """
    tif_path = Path(tif_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    with rasterio.open(tif_path) as src:
        data = src.read()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None

    if label_band < 0:
        label_band = data.shape[0] - 1

    x = data[:label_band]
    y = data[label_band]

    meta = {
        "crs": crs,
        "transform": list(transform)[:6],
        "height": height,
        "width": width,
        "source": str(tif_path),
    }

    out_path = out_dir / (tif_path.stem + ".npz")
    save_npz(out_path, x, y, meta)
    outputs.append(out_path)

    return outputs
