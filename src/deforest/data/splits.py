from __future__ import annotations

from pathlib import Path
import random

from .chips_io import list_npz


def make_splits(data_dir: str | Path, train_list: str | Path, val_list: str | Path, val_frac: float = 0.2, seed: int = 42) -> tuple[list[Path], list[Path]]:
    data_dir = Path(data_dir)
    train_list = Path(train_list)
    val_list = Path(val_list)

    paths = list_npz(data_dir)
    random.Random(seed).shuffle(paths)

    n_val = int(len(paths) * val_frac)
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]

    train_list.parent.mkdir(parents=True, exist_ok=True)
    val_list.parent.mkdir(parents=True, exist_ok=True)

    train_list.write_text("\n".join(str(p) for p in train_paths), encoding="utf-8")
    val_list.write_text("\n".join(str(p) for p in val_paths), encoding="utf-8")

    return train_paths, val_paths


def read_list(list_path: str | Path) -> list[Path]:
    list_path = Path(list_path)
    if not list_path.exists():
        return []
    lines = [p.strip() for p in list_path.read_text(encoding="utf-8").splitlines() if p.strip()]
    return [Path(p) for p in lines]
