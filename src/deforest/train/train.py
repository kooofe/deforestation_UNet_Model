from __future__ import annotations

from pathlib import Path
import random
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from deforest.config import load_config, ensure_dir
from deforest.data.dataset import ChipsDataset
from deforest.data.splits import make_splits, read_list
from deforest.models.unet import UNet
from deforest.models.losses import BCEDiceLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_sample_weights(paths: list[Path]) -> list[float]:
    weights = []
    for p in paths:
        with np.load(p) as data:
            y = data["y"]
        w = 2.0 if y.sum() > 0 else 1.0
        weights.append(w)
    return weights


def _tensor_stats(t: torch.Tensor) -> dict[str, float | bool | tuple[int, ...]]:
    t_det = t.detach()
    if not t_det.is_floating_point():
        t_det = t_det.float()
    if t_det.numel() == 0:
        return {
            "shape": tuple(t_det.shape),
            "finite": True,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "shape": tuple(t_det.shape),
        "finite": bool(torch.isfinite(t_det).all().item()),
        "min": float(t_det.min().item()),
        "max": float(t_det.max().item()),
        "mean": float(t_det.mean().item()),
        "std": float(t_det.std().item()),
    }


def _extract_paths(meta) -> list[str]:
    if meta is None:
        return []
    if isinstance(meta, dict):
        paths = meta.get("path")
        if isinstance(paths, list):
            return [str(p) for p in paths]
        if isinstance(paths, str):
            return [paths]
    if isinstance(meta, (list, tuple)):
        out = []
        for item in meta:
            if isinstance(item, dict) and "path" in item:
                out.append(str(item["path"]))
        return out
    return []


def _binary_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> tuple[float, float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()
    tp = float((preds * targets).sum().item())
    fp = float((preds * (1 - targets)).sum().item())
    fn = float(((1 - preds) * targets).sum().item())
    return tp, fp, fn


def _counts_to_metrics(tp: float, fp: float, fn: float, eps: float = 1e-6) -> dict[str, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision": precision, "recall": recall, "dice": dice, "iou": iou}


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_dir = Path(cfg["data_dir"])
    train_list = Path(cfg["train_list"])
    val_list = Path(cfg["val_list"])
    val_frac = float(cfg.get("val_frac", 0.2))

    train_paths = read_list(train_list)
    val_paths = read_list(val_list)
    if not train_paths or not val_paths:
        train_paths, val_paths = make_splits(data_dir, train_list, val_list, val_frac=val_frac, seed=seed)

    strict_sanity = bool(cfg.get("strict_sanity", False))
    append_valid_channel = bool(cfg.get("append_valid_channel", False))
    debug_return_meta = bool(cfg.get("debug_return_meta", False))
    nan_log_path = cfg.get("nan_log_path", "outputs/warnings/nan_files.log")
    warn_once_global = bool(cfg.get("warn_once_global", True))
    train_ds = ChipsDataset(
        train_paths,
        augment=True,
        strict_sanity=strict_sanity,
        return_meta=debug_return_meta,
        append_valid_channel=append_valid_channel,
        warn_log_path=nan_log_path,
        warn_once_global=warn_once_global,
    )
    val_ds = ChipsDataset(
        val_paths,
        augment=False,
        strict_sanity=strict_sanity,
        return_meta=debug_return_meta,
        append_valid_channel=append_valid_channel,
        warn_log_path=nan_log_path,
        warn_once_global=warn_once_global,
    )

    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    pin_memory = device == "cuda"

    sampler = None
    if cfg.get("oversample_pos", True):
        weights = compute_sample_weights(train_paths)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model_cfg = cfg.get("model", {})
    channels = cfg.get("channels", [])
    requested_in_channels = model_cfg.get("in_channels")
    if requested_in_channels is None:
        base_in_channels = len(channels) if channels else 18
    else:
        base_in_channels = int(requested_in_channels)
    if append_valid_channel and (requested_in_channels is None or base_in_channels == len(channels)):
        in_channels = base_in_channels + 1
    else:
        in_channels = base_in_channels
    base_channels = int(model_cfg.get("base_channels", 32))

    model = UNet(in_channels=in_channels, base_channels=base_channels).to(device)

    pos_weight = cfg.get("pos_weight")
    loss_fn = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))

    use_amp = bool(cfg.get("amp", False) and device == "cuda")
    scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
    val_threshold = float(cfg.get("val_threshold", 0.5))

    epochs = int(cfg.get("epochs", 10))
    output_dir = ensure_dir(cfg.get("output_dir", "outputs/train/run_001"))
    checkpoints_dir = ensure_dir("outputs/checkpoints")
    best_path = checkpoints_dir / "best.pt"
    best_run_path = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.csv"
    metrics_has_header = metrics_path.exists()

    best_dice = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}"), start=1):
            meta = None
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss).all():
                x_stats = _tensor_stats(x)
                y_stats = _tensor_stats(y)
                logits_stats = _tensor_stats(logits)
                zero_pct = float((x == 0).float().mean().item() * 100.0)
                print("Non-finite loss detected; aborting.")
                print(f"  epoch={epoch} step={step} loss={loss.item()}")
                print(
                    "  x: shape={shape} finite={finite} min={min:.6f} max={max:.6f} mean={mean:.6f} std={std:.6f}".format(
                        **x_stats
                    )
                )
                print(f"  x.zero_pct={zero_pct:.2f}%")
                print(
                    "  y: shape={shape} finite={finite} min={min:.6f} max={max:.6f} mean={mean:.6f} std={std:.6f}".format(
                        **y_stats
                    )
                )
                print(
                    "  logits: shape={shape} finite={finite} min={min:.6f} max={max:.6f} mean={mean:.6f} std={std:.6f}".format(
                        **logits_stats
                    )
                )
                paths = _extract_paths(meta)
                if paths:
                    print("  batch_paths:")
                    for p in paths[:10]:
                        print(f"    {p}")
                raise RuntimeError("Non-finite loss encountered during training.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss = train_loss_sum / max(1, train_steps)

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_tp = 0.0
        val_fp = 0.0
        val_fn = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val {epoch}"):
                x = x.to(device)
                y = y.unsqueeze(1).to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss_sum += float(loss.item())
                val_steps += 1
                tp, fp, fn = _binary_counts(logits, y, threshold=val_threshold)
                val_tp += tp
                val_fp += fp
                val_fn += fn

        val_loss = val_loss_sum / max(1, val_steps)
        all_metrics = _counts_to_metrics(val_tp, val_fp, val_fn)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"dice={all_metrics['dice']:.4f} iou={all_metrics['iou']:.4f} "
            f"precision={all_metrics['precision']:.4f} recall={all_metrics['recall']:.4f} "
            f"val_threshold={val_threshold:.2f}"
        )
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall"],
            )
            if not metrics_has_header:
                writer.writeheader()
                metrics_has_header = True
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "dice": round(all_metrics["dice"], 6),
                    "iou": round(all_metrics["iou"], 6),
                    "precision": round(all_metrics["precision"], 6),
                    "recall": round(all_metrics["recall"], 6),
                }
            )

        if all_metrics["dice"] > best_dice:
            best_dice = all_metrics["dice"]
            torch.save({"model": model.state_dict(), "config": cfg}, best_path)
            torch.save({"model": model.state_dict(), "config": cfg}, best_run_path)

        torch.save({"model": model.state_dict(), "config": cfg}, output_dir / "last.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    train(args.config)
