# Deforestation Change Detection Demo

This repo provides an end-to-end pipeline to build weak labels, train a lightweight U-Net, and run a demo that outputs GeoTIFF masks, PNG overlays/panels, GeoJSON polygons, and summary CSVs.

## Quick start

1) Create environment and install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Export stacked AOI (GEE)

```powershell
python scripts\export_stack.py --config configs\export.yaml
```

3) Optional forest mask (for forest-only filtering)

```powershell
python scripts\export_forest_mask.py --config configs\forest_mask.yaml
```

Or build a mask from the stacked raster:

```powershell
python scripts\make_forest_mask.py --raster data\stacked.tif --out outputs\forest_mask.tif --ndvi_thr 0.45
```

4) Convert chips + sanity check

```powershell
python scripts\geotiff_to_npz.py --tif_dir data\stacked_tifs --out_dir data\chips --chip_size 256 --stride 256 --min_valid_frac 0.10
python scripts\sanity_check.py --chips_dir data\chips --no_strict_sanity
```


5) Train

```powershell
python scripts\train.py --config configs\train.yaml
```

6) Run demo inference

```powershell
python scripts\infer_demo.py --config configs\infer_demo.yaml
```

## Outputs (demo)

- `outputs/checkpoints/best.pt` (from training)
- `outputs/demo/run_001/probs.tif` (optional)
- `outputs/demo/run_001/mask.tif` (final mask)
- `outputs/demo/run_001/mask_full.tif` (pre-forest/postproc mask)
- `outputs/demo/run_001/overlay.png`, `panel.png`, `panel_zoom.png`
- `outputs/demo/run_001/overlay_boundary.png` (optional)
- `outputs/demo/run_001/gallery.png`, `gallery_zoom.png` (optional)
- `outputs/demo/run_001/polygons.geojson`
- `outputs/demo/run_001/summary.csv`

## Notes

- Earth Engine exports require authentication and a project (`eeProject`).
- The pipeline assumes Sentinel-2 L2A in a metric CRS (UTM) for area estimates.
- `scripts/infer_demo.py` supports forest masks and change-gating via `configs/infer_demo.yaml`.
- The older chip-based demo (`scripts/demo_run.py`) is still available.
