from __future__ import annotations

import numpy as np


def apply_scl_mask(scl: np.ndarray) -> np.ndarray:
    """
    Sentinel-2 Scene Classification Layer valid classes for clear land.
    Keeps: 4 vegetation, 5 bare, 6 water, 7 unclassified
    """
    valid = {4, 5, 6, 7}
    mask = np.isin(scl, list(valid))
    return mask


def apply_qa60_mask(qa60: np.ndarray) -> np.ndarray:
    """
    QA60 bits 10/11 are clouds/cirrus.
    Returns a boolean mask where True means clear.
    """
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = ((qa60 & cloud_bit) == 0) & ((qa60 & cirrus_bit) == 0)
    return mask
