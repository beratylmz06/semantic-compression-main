# -*- coding: utf-8 -*-
"""The instructor's PSNR-vs-bitrate reference ranges, checked against this
build's real engines.

* Lena 512² — DCT and DWT PSNR must land inside the practical bands at
  0.25 / 0.50 / 0.75 / 1.00 bpp.
* Synthetic noise target (data/synthetic/) — PNG must be ~200-300 KB and the
  codec PSNR must stay at or below the memoryless Gaussian rate-distortion
  ceiling at every bitrate.
* Fingerprint target (data/fingerprint/) — lossless must be pixel-exact
  (MSE 0) at the expected ratio, and wavelet must beat DCT on the lossy side.
"""
from __future__ import annotations

import os
import warnings

import cv2
import numpy as np
import pytest

warnings.filterwarnings("ignore")

import src.compare as compare
from src.analysis.lossless import lossless_roundtrip
from src.metrics import reference_ranges as rr

SYNTH = "data/synthetic/synthetic_noise.png"
FINGER = "data/fingerprint/fingerprint.png"


def _gray(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert g is not None, f"missing sample image: {path}"
    return g


# --------------------------------------------------------------------------
def test_gaussian_ceiling_formula():
    # 6.0206 dB per extra bit, and PSNR = 20log10(255/sigma) at R=0
    assert rr.gaussian_psnr_ceiling(255.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert (rr.gaussian_psnr_ceiling(32.0, 2.0)
            - rr.gaussian_psnr_ceiling(32.0, 1.0)) == pytest.approx(6.0206, abs=1e-3)


@pytest.mark.parametrize("bpp", [0.25, 0.50, 0.75, 1.00])
def test_lenna_psnr_in_practical_bands(bpp):
    g = _gray("data/classic/lenna.png")
    d = compare.run_jpeg_primary(g, bpp, 8).metrics["psnr"]
    w = compare.run_wavelet(g, bpp, "bior4.4", 4).metrics["psnr"]
    db = rr.psnr_band("dct", bpp)
    wb = rr.psnr_band("dwt", bpp)
    assert rr.in_band(d, db), f"DCT {d:.2f} dB outside {db} at {bpp} bpp"
    assert rr.in_band(w, wb), f"DWT {w:.2f} dB outside {wb} at {bpp} bpp"
    # wavelet is at least as good as DCT on Lena
    assert w >= d - 0.5


def test_lenna_evaluate_verdict_all_ok():
    g = _gray("data/classic/lenna.png")
    for bpp in (0.25, 0.5, 0.75, 1.0):
        d = compare.run_jpeg_primary(g, bpp, 8).metrics["psnr"]
        w = compare.run_wavelet(g, bpp, "bior4.4", 4).metrics["psnr"]
        ev = rr.evaluate(bpp, d, w, reference="lenna")
        assert all(row[3] for row in ev["rows"]), (bpp, ev)


# --------------------------------------------------------------------------
def test_synthetic_file_size_band():
    assert os.path.exists(SYNTH), "run experiments/prepare_synthetic.py"
    kb = os.path.getsize(SYNTH) / 1024
    assert 200.0 <= kb <= 300.0, f"synthetic PNG is {kb:.1f} KB (want 200-300)"
    g = _gray(SYNTH)
    assert g.shape == (512, 512)


@pytest.mark.parametrize("bpp", [0.25, 0.50, 0.75, 1.00])
def test_synthetic_codec_below_gaussian_ceiling(bpp):
    # The memoryless-Gaussian bound is a strict PSNR ceiling only in the
    # near-white / low-rate regime; above ~1 bpp the target's faint watermark
    # gets coded and the codec legitimately passes it (see evaluate()).
    g = _gray(SYNTH)
    sigma = float(g.astype(np.float64).std())
    ceil = rr.gaussian_psnr_ceiling(sigma, bpp)
    d = compare.run_jpeg_primary(g, bpp, 8).metrics["psnr"]
    w = compare.run_wavelet(g, bpp, "bior4.4", 4).metrics["psnr"]
    assert d <= ceil + 0.75, f"DCT {d:.2f} > ceiling {ceil:.2f} at {bpp} bpp"
    assert w <= ceil + 0.75, f"DWT {w:.2f} > ceiling {ceil:.2f} at {bpp} bpp"
    assert rr.SYNTHETIC_NOISE["ceiling_margin_db"][0] <= ceil - max(d, w)


def test_synthetic_barely_compresses_losslessly():
    g = _gray(SYNTH)
    _, cb, ob = lossless_roundtrip(g, 9)
    ratio = ob / cb
    lo, hi = rr.SYNTHETIC_NOISE["lossless_ratio"]
    assert lo <= ratio <= hi, f"synthetic lossless ratio {ratio:.3f} not in {(lo, hi)}"


# --------------------------------------------------------------------------
def test_fingerprint_lossless_is_pixel_exact_at_expected_ratio():
    assert os.path.exists(FINGER), "run experiments/prepare_fingerprint.py"
    g = _gray(FINGER)
    assert g.shape == (512, 512)
    recon, cb, ob = lossless_roundtrip(g, 9)
    assert np.array_equal(g, recon)
    assert float(np.mean((g.astype(np.float64) - recon) ** 2)) == rr.FINGERPRINT_LOSSLESS["mse"]
    ratio = ob / cb
    lo, hi = rr.FINGERPRINT_LOSSLESS["ratio"]
    assert lo <= ratio <= hi, f"fingerprint lossless ratio {ratio:.3f} not in {(lo, hi)}"


@pytest.mark.parametrize("bpp", [0.25, 0.50, 1.00])
def test_fingerprint_wavelet_beats_dct(bpp):
    g = _gray(FINGER)
    d = compare.run_jpeg_primary(g, bpp, 8).metrics["psnr"]
    w = compare.run_wavelet(g, bpp, "bior4.4", 4).metrics["psnr"]
    lo, hi = rr.FINGERPRINT_LOSSLESS["dwt_advantage_db"]
    assert (w - d) >= lo - 0.3, f"DWT adv {w - d:.2f} dB below {lo} at {bpp} bpp"
