from __future__ import annotations

import numpy as np


def ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (nir - red) / (nir + red + eps)


def nbr(nir: np.ndarray, swir2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (nir - swir2) / (nir + swir2 + eps)


def compute_indices(pre_bands: dict, post_bands: dict) -> dict:
    pre_ndvi = ndvi(pre_bands["B8"], pre_bands["B4"])
    pre_nbr = nbr(pre_bands["B8"], pre_bands["B12"])
    post_ndvi = ndvi(post_bands["B8"], post_bands["B4"])
    post_nbr = nbr(post_bands["B8"], post_bands["B12"])
    return {
        "pre_NDVI": pre_ndvi,
        "pre_NBR": pre_nbr,
        "post_NDVI": post_ndvi,
        "post_NBR": post_nbr,
        "dNDVI": post_ndvi - pre_ndvi,
        "dNBR": post_nbr - pre_nbr,
    }
