# -*- coding: utf-8 -*-
"""Generate the "Fingerprint (Lossless)" category: data/fingerprint/*.png.

A synthetic 512x512 grayscale fingerprint built with **Gabor noise**: white
noise is band-passed on an annulus in the Fourier domain (ridge frequency
~1/8.5 px) for a set of orientations, and the oriented bands are blended by a
smooth orientation field that has a loop core and a delta singularity. Sparse
pores, an elliptical fade mask and a light paper background finish it.
Seamless (no cumulative-phase artefacts), procedural and seeded — byte-identical
output, no network.

Why it lives in its own category: fingerprints are the textbook case for
*lossless* coding (forensic/biometric — no detail may be discarded). This image
is used to show:
  * lossless (zlib DEFLATE on the raw samples) reconstructs it bit-exactly,
    MSE == 0, at a modest ratio (~1.2-1.35 : 1 — ridge texture is high-entropy,
    which is exactly why operational systems use lossy WSQ / JPEG2000 at ~15:1);
  * on the lossy side, wavelet/JPEG2000 beats block-DCT/JPEG by ~0.7-2.6 dB
    across bitrates (the historical reason the FBI moved fingerprint coding to
    JPEG2000).

Usage:  python experiments/prepare_fingerprint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import config

N = 512
SEED = 3


def _orientation_field() -> np.ndarray:
    """Smooth ridge-orientation field with a loop core (above centre) and a
    delta singularity (below-left) — the standard two-singularity model."""
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    cx = cy = (N - 1) / 2.0
    core = (cx + 8.0, cy - 48.0)
    delta = (cx - 30.0, cy + 130.0)
    th = (0.5 * np.arctan2(y - core[1], x - core[0])
          - 0.5 * np.arctan2(y - delta[1], x - delta[0]))
    return th + 0.10 * np.sin(y / 110.0) + 0.06 * np.cos(x / 130.0)


def make_fingerprint(seed: int = SEED, f0: float = 0.118, n_ang: int = 16,
                     grain: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, N))
    Fn = np.fft.fft2(noise)
    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.fftfreq(N)[None, :]
    fr = np.sqrt(fy ** 2 + fx ** 2)
    fang = np.arctan2(fy, fx)
    radial = np.exp(-((fr - f0) ** 2) / (2 * (0.020 ** 2)))          # ridge-frequency band

    th = _orientation_field()
    acc = np.zeros((N, N))
    wsum = np.zeros((N, N))
    for k in range(n_ang):
        a = np.pi * k / n_ang
        wedge = np.exp(np.cos(2 * (fang - (a + np.pi / 2))) * 6.0)   # spectrum energy _|_ ridge
        wedge /= wedge.max()
        band = np.real(np.fft.ifft2(Fn * radial * wedge))
        w = np.exp(np.cos(2 * (th - a)) * 5.0)                       # local orientation ~ a
        acc += w * band
        wsum += w
    ridges = acc / (wsum + 1e-9)
    ridges = (ridges - ridges.mean()) / ridges.std()

    img = 128.0 + 95.0 * np.tanh(1.6 * ridges)                       # crisp black/white ridges
    img = img + rng.normal(0.0, grain, (N, N))
    img[rng.random((N, N)) < 0.0016] = 244.0                         # sweat pores

    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    cx = cy = (N - 1) / 2.0
    r = ((x - cx) / 205.0) ** 2 + ((y - cy) / 245.0) ** 2
    mask = np.clip(1.32 - r, 0.0, 1.0) ** 1.6
    out = mask * img + (1.0 - mask) * (239.0 + rng.normal(0.0, 0.7, (N, N)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _also_save_to_desktop(img: np.ndarray, name: str) -> None:
    for cand in (Path.home() / "Desktop" / "Görseller",
                 Path.home() / "Desktop" / "Gorseller"):
        if cand.is_dir():
            Image.fromarray(img).save(cand / name)
            print(f"[fingerprint] also copied -> {cand / name}")
            return


def main() -> None:
    out_dir = config.DATA_DIR / "fingerprint"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = make_fingerprint()
    path = out_dir / "fingerprint.png"
    Image.fromarray(img).save(path)
    print(f"[fingerprint] fingerprint.png  ({img.shape})  "
          f"{path.stat().st_size / 1024:.1f} KB")
    _also_save_to_desktop(img, "fingerprint.png")


if __name__ == "__main__":
    main()
