# -*- coding: utf-8 -*-
"""PR-QMF (Perfect Reconstruction Quadrature Mirror Filter) — 2-channel,
separable 2D, plus a genuine uniform M=4 band split.

Ported from the old project's ``core/pr_qmf.py``. 2/4/6-tap PR-QMF prototype
filters that PyWavelets does not ship. Output layout matches pywt.dwt2:
(LL, (cH, cV, cD)) = (LL, (LH, HL, HH)). Pure NumPy + scipy.ndimage.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve1d

FILTERS: dict[str, dict[str, list[float]]] = {
    "prqmf2": {
        "h0": [0.7071, 0.7071],
        "h1": [-0.7071, 0.7071],
        "g0": [0.7071, 0.7071],
        "g1": [0.7071, -0.7071],
    },
    "prqmf4": {
        "h0": [-0.1294, 0.2241, 0.8365, 0.4830],
        "h1": [-0.4830, 0.8365, -0.2241, -0.1294],
        "g0": [0.4830, 0.8365, 0.2241, -0.1294],
        "g1": [-0.1294, -0.2241, 0.8365, -0.4830],
    },
    "prqmf6": {
        "h0": [0.0352, -0.0854, -0.1350, 0.4599, 0.8069, 0.3327],
        "h1": [-0.3327, 0.8069, -0.4599, -0.1350, 0.0854, 0.0352],
        "g0": [0.3327, 0.8069, 0.4599, -0.1350, -0.0854, 0.0352],
        "g1": [0.0352, 0.0854, -0.1350, -0.4599, 0.8069, -0.3327],
    },
}

PRQMF_UI_OPTIONS = [
    ("prqmf2", "prqmf2 (2-tap prototype)"),
    ("prqmf4", "prqmf4 (4-tap prototype)"),
    ("prqmf6", "prqmf6 (6-tap prototype)"),
]


def is_pr_qmf(name: str) -> bool:
    return name in FILTERS


def _taps(coeffs) -> np.ndarray:
    return np.asarray(coeffs, dtype=np.float64)


def _conv_circ(x: np.ndarray, h: np.ndarray, axis: int = -1) -> np.ndarray:
    """Circular (periodization) 1D convolution — convolve1d, mode='wrap'."""
    return convolve1d(np.asarray(x, dtype=np.float64), np.asarray(h, dtype=np.float64),
                      axis=axis, mode="wrap")


def analyze_1d(x, h0, h1):
    """1D analysis: circular convolution + downsample by 2 (even indices)."""
    x = np.asarray(x, dtype=np.float64)
    return _conv_circ(x, _taps(h0))[::2], _conv_circ(x, _taps(h1))[::2]


def synthesize_1d(low, high, g0, g1, out_len):
    """1D synthesis: upsample + circular filter + sum. convolve1d centres the
    kernel, so analysis+synthesis produces a net 1-sample circular shift that
    np.roll(..., 1) compensates."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    n_low = (out_len + 1) // 2
    up_low = np.zeros(out_len, dtype=np.float64)
    up_high = np.zeros(out_len, dtype=np.float64)
    up_low[0::2] = low[:n_low]
    up_high[0::2] = high[:n_low]
    rec = _conv_circ(up_low, _taps(g0)) + _conv_circ(up_high, _taps(g1))
    return np.roll(rec, 1)


def _analyze_along_axis(data, h0, h1, axis):
    lo_full = _conv_circ(data, h0, axis=axis)
    hi_full = _conv_circ(data, h1, axis=axis)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(None, None, 2)
    return lo_full[tuple(sl)], hi_full[tuple(sl)]


def _synthesize_along_axis(low, high, g0, g1, out_len, axis):
    shape = list(low.shape)
    shape[axis] = out_len
    up_low = np.zeros(shape, dtype=np.float64)
    up_high = np.zeros(shape, dtype=np.float64)
    sl = [slice(None)] * len(shape)
    sl[axis] = slice(None, None, 2)
    n_keep = (out_len + 1) // 2
    sl_src = [slice(None)] * low.ndim
    sl_src[axis] = slice(0, n_keep)
    up_low[tuple(sl)] = low[tuple(sl_src)]
    up_high[tuple(sl)] = high[tuple(sl_src)]
    rec = _conv_circ(up_low, g0, axis=axis) + _conv_circ(up_high, g1, axis=axis)
    return np.roll(rec, 1, axis=axis)


def dwt2_mband4(data: np.ndarray, name: str = "prqmf4"):
    """Genuine 2D separable M=4 band split: a uniform, critically-sampled
    4-band split along rows, then columns -> 4x4 = 16 subbands total (one
    level, all the same size — the 2D counterpart of the course's M-Band
    Subband Coding diagram, k0=k1=k2=k3=4).

    Returns (bands, orig_shape) where bands[r][c] (r,c in 0..3) is one
    16th-size 2D subband, row-major from low to high frequency on both axes.
    """
    if name not in FILTERS:
        raise ValueError(f"unknown PR-QMF: {name}")
    filt = FILTERS[name]
    h0, h1 = _taps(filt["h0"]), _taps(filt["h1"])
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("dwt2_mband4 expects a 2D array")
    orig_shape = (x.shape[0], x.shape[1])

    def _split4(arr, axis):
        lo_full = _conv_circ(arr, h0, axis=axis)
        hi_full = _conv_circ(arr, h1, axis=axis)
        sl2 = [slice(None)] * arr.ndim
        sl2[axis] = slice(None, None, 2)
        lo, hi = lo_full[tuple(sl2)], hi_full[tuple(sl2)]
        ll = _conv_circ(lo, h0, axis=axis)[tuple(sl2)]
        lh = _conv_circ(lo, h1, axis=axis)[tuple(sl2)]
        hl = _conv_circ(hi, h0, axis=axis)[tuple(sl2)]
        hh = _conv_circ(hi, h1, axis=axis)[tuple(sl2)]
        return [ll, lh, hl, hh]

    col_split = _split4(x, axis=1)
    bands = [[None] * 4 for _ in range(4)]
    for c, band in enumerate(col_split):
        row_split = _split4(band, axis=0)
        for r in range(4):
            bands[r][c] = row_split[r]
    return bands, orig_shape


def idwt2_mband4(bands, orig_shape, name: str = "prqmf4") -> np.ndarray:
    """Inverse of :func:`dwt2_mband4`."""
    if name not in FILTERS:
        raise ValueError(f"unknown PR-QMF: {name}")
    filt = FILTERS[name]
    g0, g1 = _taps(filt["g0"]), _taps(filt["g1"])
    h, w = orig_shape

    def _merge4(quads, axis, out_len):
        half = (out_len + 1) // 2
        lo = _synthesize_along_axis(quads[0], quads[1], g0, g1, half, axis=axis)
        hi = _synthesize_along_axis(quads[2], quads[3], g0, g1, out_len - half, axis=axis)
        return _synthesize_along_axis(lo, hi, g0, g1, out_len, axis=axis)

    col_bands = [_merge4([bands[r][c] for r in range(4)], axis=0, out_len=h) for c in range(4)]
    return _merge4(col_bands, axis=1, out_len=w)


def band_energy_pct(bands) -> dict[str, float]:
    e = {}
    total = 0.0
    for r in range(4):
        for c in range(4):
            v = float(np.sum(np.asarray(bands[r][c], dtype=np.float64) ** 2))
            e[f"B{r}{c}"] = v
            total += v
    if total < 1e-18:
        return {k: 6.25 for k in e}
    return {k: 100.0 * v / total for k, v in e.items()}


def max_level_prqmf(shape, name: str) -> int:
    filt_len = len(FILTERS[name]["h0"])
    return max(1, int(math.log2(min(shape) / (filt_len - 1))))
