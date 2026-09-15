# -*- coding: utf-8 -*-
"""Image-type classification: natural / synthetic / hybrid.

Ported from the old project's ``core/image_classifier.py`` (pure NumPy, GUI-free).
Used only for the Compare tab's "Detected image type: <type>" pill — it does not
affect any compression maths.

Criteria: unique-colour count, Shannon entropy, flat-area ratio, edge sharpness.
"""
from __future__ import annotations

import numpy as np


def _to_gray_255(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float64)
    if img.max() <= 1.01:
        img = img * 255.0
    return img


def _entropy(image: np.ndarray) -> float:
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-8)
    nz = hist[hist > 0]
    ent = -np.sum(nz * np.log2(nz))
    return float(np.clip(ent / 8.0, 0.0, 1.0))


def _edge_sharpness(image: np.ndarray, threshold: int = 2) -> float:
    dx = np.abs(np.diff(image, axis=1))
    dy = np.abs(np.diff(image, axis=0))
    edges = np.concatenate([dx.ravel(), dy.ravel()])
    strong = edges[edges > threshold]
    if strong.size == 0:
        return 0.0
    return float(np.clip((strong.mean() / 255.0) * 3.0, 0.0, 1.0))


def _flat_areas(image: np.ndarray) -> float:
    dx = np.diff(image, axis=1) == 0
    dy = np.diff(image, axis=0) == 0
    ratio = (np.sum(dx) + np.sum(dy)) / (dx.size + dy.size)
    return float(np.clip(ratio, 0.0, 1.0))


def classify(image: np.ndarray) -> tuple[str, dict]:
    """Return (image_type, features) where image_type is one of
    'natural' / 'synthetic' / 'hybrid'."""
    g = _to_gray_255(image)
    u8 = np.clip(g, 0, 255).astype(np.uint8)
    unique_colors = int(np.unique(u8).size)
    entropy = _entropy(g)
    flatness = _flat_areas(g)
    sharpness = _edge_sharpness(g)

    features = {
        "unique_colors": unique_colors,
        "entropy": round(entropy, 3),
        "flat_areas": round(flatness, 3),
        "edge_sharpness": round(sharpness, 3),
    }

    is_low_color = unique_colors < 80
    is_high_flat = flatness > 0.45

    if is_low_color or is_high_flat:
        if entropy < 0.40:
            return "synthetic", features
        if entropy < 0.60:
            return "hybrid", features
        if not is_low_color:
            return "natural", features
    if 0.25 < flatness <= 0.45 and unique_colors > 120:
        return "hybrid", features
    if unique_colors > 180 and flatness < 0.25:
        return "natural", features
    if sharpness > 0.6 and flatness > 0.3:
        return "synthetic", features
    return ("natural" if unique_colors > 150 else "hybrid"), features


def classify_type(image: np.ndarray) -> str:
    return classify(image)[0]
