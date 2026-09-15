# -*- coding: utf-8 -*-
"""Lossless coding — zlib DEFLATE on the raw samples.

Ported from the old project's ``stages/stage_lossless.py``. Pixel-exact: the
reconstruction is bit-for-bit identical to the input. Used by the Lossless tab.
"""
from __future__ import annotations

import zlib

import numpy as np


def compress(gray_u8: np.ndarray, level: int = 9) -> tuple[bytes, int]:
    """DEFLATE the raw uint8 samples. Returns (stream, original_byte_count)."""
    level = int(np.clip(level, 1, 9))
    g = np.ascontiguousarray(np.asarray(gray_u8, dtype=np.uint8))
    raw = g.tobytes()
    return zlib.compress(raw, level=level), len(raw)


def decompress(stream: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`compress` — exact recovery of the uint8 array."""
    raw = zlib.decompress(stream)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def lossless_roundtrip(gray_u8: np.ndarray, level: int = 9) -> tuple[np.ndarray, int, int]:
    """Compress then decompress. Returns (recon_u8, compressed_bytes, original_bytes)."""
    g = np.ascontiguousarray(np.asarray(gray_u8, dtype=np.uint8))
    stream, orig_bytes = compress(g, level)
    recon = decompress(stream, g.shape)
    return recon, len(stream), orig_bytes
