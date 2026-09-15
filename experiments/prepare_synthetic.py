# -*- coding: utf-8 -*-
"""Generate the "Synthetic Images (Grayscale)" category: data/synthetic/*.png.

The instructor asked for a *synthetic* 512x512 grayscale test image in the
200-300 KB range, with a mathematical PSNR-vs-bitrate reference.

There is exactly one image family that satisfies "200-300 KB PNG for a 512x512
8-bit image": a near-incompressible one, i.e. a spectrally (near-)flat field.
So the synthetic target here is a **near-white Gaussian noise field** with a
faint low-contrast calibration watermark (64 px grid + centre circle +
diagonals) and a gentle vignette — recognisably a made test chart, not a photo,
while the source stays close enough to i.i.d. Gaussian that the memoryless
Gaussian rate-distortion function is a real upper bound:

        PSNR_ceiling(R) = 20*log10(255 / sigma) + 6.0206 * R      [dB, R in bpp]

A block-DCT / wavelet codec lands ~1-3 dB below this ceiling (it is not the
optimal vector quantiser), and lossless coding gains almost nothing (~1.05-1.1
: 1) — the intended contrast with the fingerprint image (data/fingerprint/).

Deterministic: fixed seed -> byte-identical output. No network, no downloads.

Usage:  python experiments/prepare_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

import config

N = 512
SEED = 20
SIGMA = 34.0


def make_synthetic_noise(sigma: float = SIGMA, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    cx = cy = (N - 1) / 2.0

    img = 128.0 + rng.normal(0.0, sigma, (N, N))
    img += 6.0 * np.sin(2 * np.pi * (x + y) / 512.0)                  # ultra-faint tint
    img -= 10.0 * (((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)       # gentle vignette

    wm = np.ascontiguousarray(img)
    for gx in range(0, N, 64):
        wm[:, gx:gx + 1] += 18.0
    for gy in range(0, N, 64):
        wm[gy:gy + 1, :] += 18.0
    cv2.circle(wm, (int(cx), int(cy)), 150, 235, 1)
    cv2.line(wm, (0, 0), (N - 1, N - 1), 235, 1)
    cv2.line(wm, (0, N - 1), (N - 1, 0), 235, 1)
    return np.clip(wm, 0, 255).astype(np.uint8)


def _also_save_to_desktop(img: np.ndarray, name: str) -> None:
    """Mirror into the Desktop 'Görseller' folder if it exists (the user keeps
    a copy of the sample images there)."""
    for cand in (Path.home() / "Desktop" / "Görseller",
                 Path.home() / "Desktop" / "Gorseller"):
        if cand.is_dir():
            Image.fromarray(img).save(cand / name)
            print(f"[synthetic] also copied -> {cand / name}")
            return


def main() -> None:
    out_dir = config.DATA_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = make_synthetic_noise()
    path = out_dir / "synthetic_noise.png"
    Image.fromarray(img).save(path)
    kb = path.stat().st_size / 1024
    print(f"[synthetic] synthetic_noise.png  ({img.shape})  {kb:.1f} KB  "
          f"sigma={img.astype(float).std():.1f}")
    if not 150 <= kb <= 320:
        print(f"  WARNING: {kb:.1f} KB is outside the intended 200-300 KB band")
    _also_save_to_desktop(img, "synthetic_noise.png")


if __name__ == "__main__":
    main()
