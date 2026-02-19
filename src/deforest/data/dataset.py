from __future__ import annotations

from pathlib import Path
import hashlib
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

from .chips_io import load_npz


class ChipsDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        augment: bool = False,
        return_meta: bool = False,
        strict_sanity: bool = False,
        append_valid_channel: bool = False,
        warn_log_path: str | Path | None = None,
        warn_once_global: bool = True,
    ):
        self.paths = paths
        self.augment = augment
        self.return_meta = return_meta
        self.strict_sanity = strict_sanity
        self.append_valid_channel = append_valid_channel
        self.warn_log_path = Path(warn_log_path) if warn_log_path else None
        self.warn_once_global = warn_once_global
        self._warned_files: set[str] = set()

    def __len__(self) -> int:
        return len(self.paths)

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=2)
            y = np.flip(y, axis=1)
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=0)
        return x.copy(), y.copy()

    @staticmethod
    def _first_nonfinite(x: np.ndarray) -> tuple[int, int, int, str] | None:
        mask = ~np.isfinite(x)
        if not mask.any():
            return None
        idx = np.argwhere(mask)[0]
        if x.ndim == 2:
            ch = 0
            row, col = int(idx[0]), int(idx[1])
            value = x[row, col]
        else:
            ch = int(idx[0])
            row = int(idx[1]) if idx.shape[0] > 1 else 0
            col = int(idx[2]) if idx.shape[0] > 2 else 0
            value = x[tuple(idx)]
        if np.isnan(value):
            kind = "nan"
        elif np.isposinf(value):
            kind = "posinf"
        elif np.isneginf(value):
            kind = "neginf"
        else:
            kind = "nonfinite"
        return ch, row, col, kind

    def _mark_warned_global(self, path_str: str) -> bool:
        if not self.warn_log_path:
            return True
        warn_dir = self.warn_log_path.parent / "nan_warned"
        warn_dir.mkdir(parents=True, exist_ok=True)
        sentinel = warn_dir / ".global_warning_lock"
        try:
            fd = os.open(str(sentinel), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _append_warn_log(self, msg: str) -> None:
        if not self.warn_log_path:
            return
        self.warn_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.warn_log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        x, y, meta = load_npz(path)
        x = x.astype("float32", copy=False)
        valid = np.isfinite(x).all(axis=0).astype(np.float32)
        nonfinite = self._first_nonfinite(x)
        if nonfinite is not None:
            ch, row, col, kind = nonfinite
            msg = f"Non-finite in x before sanitize: file={path} ch={ch} row={row} col={col} type={kind}"
            if self.strict_sanity:
                raise ValueError(msg)
            if self.warn_once_global:
                if self._mark_warned_global(str(path)):
                    warnings.warn(msg)
                    self._append_warn_log(msg)
            else:
                if str(path) not in self._warned_files:
                    warnings.warn(msg)
                    self._append_warn_log(msg)
                    self._warned_files.add(str(path))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        y = y.astype("float32", copy=False)
        y_max = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
        if y_max > 1.5:
            y = (y > 127).astype(np.float32)
        else:
            y = (y > 0.5).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if self.append_valid_channel:
            x = np.concatenate([x, valid[None, ...]], axis=0)
        if self.augment:
            x, y = self._augment(x, y)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if self.return_meta:
            meta = dict(meta)
            meta["path"] = str(path)
            return x_t, y_t, meta
        return x_t, y_t
