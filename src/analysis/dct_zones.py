# -*- coding: utf-8 -*-
"""DCT frequency-zone breakdown of the whole image.

Ported from the old project's ``backend/services/dct_analysis.py``. For every
8x8 block: 2D DCT, keep only one of four frequency zones (DC+low-freq,
horizontal AC, vertical AC, diagonal AC), inverse-DCT, and accumulate the
spatial contribution. Unquantized -> independent of target BPP. This is the
DCT-side counterpart of the DWT Lab's LL/HL/LH/HH single-level bands.
"""
from __future__ import annotations

import cv2
import numpy as np

ZONE_LABELS = ["DC + low-freq", "Horizontal AC", "Vertical AC", "Diagonal AC"]


def zone_masks_8x8() -> list[np.ndarray]:
    """Four 8x8 masks over the DCT coefficient plane: top-left 4x4 quadrant is
    DC+low-freq, the other three quadrants are the horizontal / vertical /
    diagonal AC zones."""
    bs = 8
    dc_lf = np.zeros((bs, bs), np.float32)
    horiz = np.zeros((bs, bs), np.float32)
    vert = np.zeros((bs, bs), np.float32)
    diag = np.zeros((bs, bs), np.float32)
    for u in range(bs):
        for v in range(bs):
            if u < 4 and v < 4:
                dc_lf[u, v] = 1.0
            elif u >= 4 and v < 4:
                horiz[u, v] = 1.0
            elif u < 4 and v >= 4:
                vert[u, v] = 1.0
            else:
                diag[u, v] = 1.0
    return [dc_lf, horiz, vert, diag]


def zone_layer_images(gray_u8: np.ndarray, block_size: int = 8
                      ) -> tuple[list[np.ndarray], list[float]]:
    """Returns (four float [0,1] spatial layer images, four energy percentages
    summing to ~100)."""
    g = np.asarray(gray_u8, dtype=np.float32)
    if g.ndim == 3:
        g = g[:, :, 0]
    if g.max() > 1.5:
        g = g / 255.0
    h0, w0 = g.shape
    ph = ((h0 + block_size - 1) // block_size) * block_size
    pw = ((w0 + block_size - 1) // block_size) * block_size
    padded = np.zeros((ph, pw), np.float32)
    padded[:h0, :w0] = g

    masks = zone_masks_8x8()
    acc = [np.zeros((ph, pw), np.float32) for _ in range(4)]
    energy = [0.0, 0.0, 0.0, 0.0]

    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            blk = padded[i:i + block_size, j:j + block_size]
            d = cv2.dct(blk)
            for zi, m in enumerate(masks):
                zm = d * m
                energy[zi] += float(np.sum(zm ** 2))
                acc[zi][i:i + block_size, j:j + block_size] += cv2.idct(zm)

    total = sum(energy)
    pct = [25.0] * 4 if total < 1e-18 else [100.0 * e / total for e in energy]

    out = []
    for arr in acc:
        a = np.abs(arr[:h0, :w0])
        mx = float(a.max())
        vis = (a / mx if mx > 1e-9 else np.zeros_like(a))
        out.append(vis)
    return out, pct
