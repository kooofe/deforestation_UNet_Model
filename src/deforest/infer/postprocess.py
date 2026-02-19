from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def threshold_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(np.uint8)


def clean_mask(mask: np.ndarray, min_area_px: int, morph_open: int = 0, morph_close: int = 0) -> np.ndarray:
    cleaned = mask.astype(bool)

    if morph_open > 0:
        cleaned = ndi.binary_opening(cleaned, iterations=morph_open)
    if morph_close > 0:
        cleaned = ndi.binary_closing(cleaned, iterations=morph_close)

    labeled, num = ndi.label(cleaned)
    if num == 0:
        return cleaned.astype(np.uint8)

    sizes = ndi.sum(cleaned, labeled, index=np.arange(1, num + 1))
    keep = sizes >= min_area_px
    cleaned = np.isin(labeled, np.where(keep)[0] + 1)
    return cleaned.astype(np.uint8)
