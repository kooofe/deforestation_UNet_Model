from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TILE_RE = re.compile(r"stacked-(\d{10})-(\d{10})$")


def parse_offsets(tile_stem: str) -> tuple[int, int]:
    m = TILE_RE.match(tile_stem)
    if not m:
        raise ValueError(f"Unexpected tile name: {tile_stem}")
    return int(m.group(1)), int(m.group(2))


def forest_mask_for_tile(tile_stem: str, forest_masks_dir: Path) -> Path:
    x_off, y_off = parse_offsets(tile_stem)
    fx = (x_off // 32768) * 32768
    fy = (y_off // 32768) * 32768
    path = forest_masks_dir / f"forest_mask-{fx:010d}-{fy:010d}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing forest mask for {tile_stem}: {path}")
    return path


def read_summary_row(summary_csv: Path) -> dict[str, str] | None:
    if not summary_csv.exists():
        return None
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, None)


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def aggregate_outputs(out_root: Path) -> None:
    tile_dirs = sorted([p for p in out_root.iterdir() if p.is_dir()])
    merged_geojson = out_root / "polygons_merged.geojson"
    summary_rows: list[dict[str, str]] = []

    with merged_geojson.open("w", encoding="utf-8") as out_f:
        out_f.write('{"type":"FeatureCollection","features":[\n')
        first = True
        for tile_dir in tile_dirs:
            tile_id = tile_dir.name
            poly_path = tile_dir / "polygons.geojson"
            if poly_path.exists():
                data = json.loads(poly_path.read_text(encoding="utf-8"))
                for feat in data.get("features", []):
                    props = feat.get("properties") or {}
                    props["tile_id"] = tile_id
                    feat["properties"] = props
                    if not first:
                        out_f.write(",\n")
                    json.dump(feat, out_f, separators=(",", ":"))
                    first = False

            row = read_summary_row(tile_dir / "summary.csv")
            if row:
                row = dict(row)
                row["tile_id"] = tile_id
                summary_rows.append(row)
        out_f.write("\n]}\n")

    per_tile_summary = out_root / "tile_summaries.csv"
    if summary_rows:
        fieldnames = ["tile_id"] + [k for k in summary_rows[0].keys() if k != "tile_id"]
        with per_tile_summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    aggregate_summary = out_root / "aggregate_summary.csv"
    total_tiles = len(summary_rows)
    total_polygons = int(sum(to_float(r.get("num_polygons")) for r in summary_rows))
    total_area_m2 = float(sum(to_float(r.get("total_area_m2")) for r in summary_rows))
    total_area_ha = total_area_m2 / 10000.0
    total_prefilter_area_m2 = float(sum(to_float(r.get("pre_forest_area_m2")) for r in summary_rows))
    with aggregate_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tiles_processed",
                "total_polygons",
                "total_area_m2",
                "total_area_ha",
                "total_prefilter_area_m2",
                "generated_utc",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tiles_processed": total_tiles,
                "total_polygons": total_polygons,
                "total_area_m2": round(total_area_m2, 3),
                "total_area_ha": round(total_area_ha, 3),
                "total_prefilter_area_m2": round(total_prefilter_area_m2, 3),
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )


def cleanup_tile_artifacts(tile_out: Path) -> None:
    keep = {"summary.csv", "polygons.geojson", "run.log"}
    for p in tile_out.iterdir():
        if p.name in keep:
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def run_tile(
    *,
    python_exe: Path,
    repo_root: Path,
    infer_config: Path,
    checkpoint: Path,
    raster: Path,
    forest_mask: Path,
    out_dir: Path,
    threshold: float,
    tile: int,
    stride: int,
    device: str,
    min_area_m2: float,
) -> tuple[int, str]:
    cmd = [
        str(python_exe),
        "scripts/infer_demo.py",
        "--config",
        str(infer_config),
        "--checkpoint",
        str(checkpoint),
        "--raster",
        str(raster),
        "--forest_mask_path",
        str(forest_mask),
        "--out_dir",
        str(out_dir),
        "--threshold",
        str(threshold),
        "--tile",
        str(tile),
        "--stride",
        str(stride),
        "--tile_gallery",
        "0",
        "--no_save_probs",
        "--min_area_m2",
        str(min_area_m2),
        "--device",
        device,
    ]
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run infer_demo over all stacked tiles and aggregate outputs.")
    parser.add_argument("--infer_config", default="configs/infer_demo_usable_vf90_ratio3_thr055.yaml")
    parser.add_argument("--checkpoint", default="outputs/train/run_usable_20260210_vf90_ratio3_pw12/best.pt")
    parser.add_argument("--stacked_dir", default="data/stacked_tifs")
    parser.add_argument("--forest_masks_dir", default="data/forest_masks")
    parser.add_argument("--out_root", default="outputs/demo/batch_usable_vf90_ratio3_thr055")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--min_area_m2", type=float, default=1000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_tiles", type=int, default=0, help="0 means all.")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--keep_tile_artifacts", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_exe = Path(sys.executable).resolve()
    infer_config = (repo_root / args.infer_config).resolve()
    checkpoint = (repo_root / args.checkpoint).resolve()
    stacked_dir = (repo_root / args.stacked_dir).resolve()
    forest_masks_dir = (repo_root / args.forest_masks_dir).resolve()
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not infer_config.exists():
        raise SystemExit(f"Missing infer config: {infer_config}")
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    if not stacked_dir.exists():
        raise SystemExit(f"Missing stacked dir: {stacked_dir}")
    if not forest_masks_dir.exists():
        raise SystemExit(f"Missing forest masks dir: {forest_masks_dir}")

    tiles = sorted(stacked_dir.glob("stacked-*.tif"))
    if args.max_tiles > 0:
        tiles = tiles[: args.max_tiles]
    if not tiles:
        raise SystemExit(f"No tiles found in {stacked_dir}")

    resume = not args.no_resume
    failures: list[dict[str, str]] = []
    start_all = time.time()
    completed = 0
    skipped = 0

    print(f"Tiles to process: {len(tiles)}")
    print(f"Out root: {out_root}")
    print(f"Resume mode: {resume}")
    for idx, tile_path in enumerate(tiles, start=1):
        tile_stem = tile_path.stem
        tile_out = out_root / tile_stem
        summary_csv = tile_out / "summary.csv"
        polygons_geojson = tile_out / "polygons.geojson"
        if resume and summary_csv.exists() and polygons_geojson.exists():
            skipped += 1
            print(f"[{idx}/{len(tiles)}] SKIP {tile_stem}")
            continue

        tile_out.mkdir(parents=True, exist_ok=True)
        try:
            forest_mask = forest_mask_for_tile(tile_stem, forest_masks_dir)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append({"tile": tile_stem, "error": msg})
            print(f"[{idx}/{len(tiles)}] FAIL {tile_stem} {msg}")
            continue

        print(f"[{idx}/{len(tiles)}] RUN  {tile_stem}")
        t0 = time.time()
        code, log_text = run_tile(
            python_exe=python_exe,
            repo_root=repo_root,
            infer_config=infer_config,
            checkpoint=checkpoint,
            raster=tile_path.resolve(),
            forest_mask=forest_mask.resolve(),
            out_dir=tile_out.resolve(),
            threshold=args.threshold,
            tile=args.tile,
            stride=args.stride,
            device=args.device,
            min_area_m2=args.min_area_m2,
        )
        (tile_out / "run.log").write_text(log_text, encoding="utf-8")
        dt = time.time() - t0
        if code != 0:
            failures.append({"tile": tile_stem, "error": f"infer_demo exit code {code}"})
            print(f"[{idx}/{len(tiles)}] FAIL {tile_stem} ({dt:.1f}s)")
            continue

        completed += 1
        print(f"[{idx}/{len(tiles)}] DONE {tile_stem} ({dt:.1f}s)")
        if not args.keep_tile_artifacts:
            cleanup_tile_artifacts(tile_out)

    failures_csv = out_root / "failures.csv"
    if failures:
        with failures_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tile", "error"])
            writer.writeheader()
            writer.writerows(failures)
    elif failures_csv.exists():
        failures_csv.unlink()

    aggregate_outputs(out_root)
    total_dt = time.time() - start_all
    print(
        "Finished: completed={} skipped={} failed={} elapsed_sec={:.1f}".format(
            completed, skipped, len(failures), total_dt
        )
    )
    print(f"Merged polygons: {out_root / 'polygons_merged.geojson'}")
    print(f"Aggregate summary: {out_root / 'aggregate_summary.csv'}")
    print(f"Per-tile summary: {out_root / 'tile_summaries.csv'}")
    if failures:
        print(f"Failures: {failures_csv}")


if __name__ == "__main__":
    main()
