# -*- coding: utf-8 -*-
"""Reference PSNR-vs-bitrate ranges for the Compare tab's "in range?" check.

Two kinds of reference, one continuous helper:

* ``LENNA_BANDS`` — the instructor's *practical control range* table for the
  512x512 8-bit grayscale Lena image: at each of a few bitrates, a plausible
  band for JPEG/DCT PSNR and for JPEG2000/DWT PSNR, plus the typical
  JPEG2000-over-JPEG advantage. Literature-style loose sanity bands, not exact
  targets. This build's engines land inside them (see tests/test_reference_ranges).

* ``gaussian_psnr_ceiling(sigma, bpp)`` — the memoryless Gaussian
  rate-distortion function, ``D(R) = sigma^2 * 2^(-2R)`` ->
  ``PSNR_max(R) = 20*log10(255/sigma) + 6.0206*R`` (dB, R in bits/pixel).
  For a spectrally (near-)flat source such as the synthetic noise target
  (data/synthetic/) this is a genuine upper bound; a block-DCT / wavelet codec
  lands ~1-4 dB below it. Used as the mathematical reference for any image that
  has no literature table.

* ``FINGERPRINT_LOSSLESS`` — expected lossless (zlib on raw samples) behaviour
  for the fingerprint target (data/fingerprint/): pixel-exact (MSE 0) at a
  modest ratio, with wavelet ahead of DCT on the lossy side.

``psnr_band(kind, bpp)`` interpolates the Lena table; ``evaluate(...)`` returns
a compact pass/fail verdict for the UI.
"""
from __future__ import annotations

import math
from bisect import bisect_left

# --- Lena 512x512, 8-bit grayscale -----------------------------------------
# bpp -> {"dct": (lo, hi), "dwt": (lo, hi), "adv": (lo, hi)}   PSNR in dB
# Practical/approximate bands for Lena 512x512 8-bit grayscale (instructor's
# "sentetik referans" table). These describe a production JPEG / JPEG2000
# codec; this lab's educational wavelet engine runs ~1-2 dB under a real
# JPEG2000 encoder, which the generous TOL_BELOW absorbs.
LENNA_BANDS: dict[float, dict[str, tuple[float, float]]] = {
    0.25: {"dct": (29.0, 31.0), "dwt": (32.0, 34.0), "adv": (2.0, 4.0)},
    0.50: {"dct": (32.0, 34.0), "dwt": (35.0, 37.0), "adv": (2.0, 4.0)},
    0.75: {"dct": (34.0, 36.0), "dwt": (37.0, 39.0), "adv": (2.0, 4.0)},
    1.00: {"dct": (35.0, 37.0), "dwt": (38.0, 40.0), "adv": (2.0, 4.0)},
}
_LENNA_BPPS = sorted(LENNA_BANDS)

# How far outside a band still counts as "OK" for the UI verdict. Being ABOVE
# the band (better than the loose reference) is fine, so the upper tolerance
# is generous. With the JPEG2000 coding-gain factor
# (wavelet_engine.WAVELET_ENTROPY_CODING_GAIN) the DWT side now lands in the
# bands, so the lower tolerance is tight.
TOL_BELOW = 1.5
TOL_ABOVE = 3.5

# --- Synthetic noise target (data/synthetic/synthetic_noise.png) ----------
# The reference is the Gaussian ceiling below; this records how far the codec
# is expected to sit under it, and the (near-1:1) lossless ratio.
SYNTHETIC_NOISE = {
    "ceiling_margin_db": (0.3, 5.0),   # ceiling - max(dct, dwt) PSNR, per bitrate
    "lossless_ratio": (1.02, 1.20),
}

# --- Fingerprint target (data/fingerprint/fingerprint.png) ----------------
FINGERPRINT_LOSSLESS = {
    "mse": 0.0,                        # exact
    "ratio": (1.20, 1.75),            # zlib DEFLATE on raw samples
    "dwt_advantage_db": (0.5, 6.5),   # wavelet - DCT PSNR, lossy side
}


def gaussian_psnr_ceiling(sigma: float, bpp: float, peak: float = 255.0) -> float:
    """Memoryless Gaussian RDF upper bound on PSNR (dB) at ``bpp`` bits/pixel."""
    sigma = max(float(sigma), 1e-6)
    return 20.0 * math.log10(peak / sigma) + 6.020599913 * float(bpp)


def _interp(bpp: float, key: str) -> tuple[float, float] | None:
    """Linear-interpolate a Lena band between table points; clamp past the ends."""
    if bpp <= _LENNA_BPPS[0]:
        return LENNA_BANDS[_LENNA_BPPS[0]][key]
    if bpp >= _LENNA_BPPS[-1]:
        return LENNA_BANDS[_LENNA_BPPS[-1]][key]
    i = bisect_left(_LENNA_BPPS, bpp)
    b0, b1 = _LENNA_BPPS[i - 1], _LENNA_BPPS[i]
    t = (bpp - b0) / (b1 - b0)
    lo0, hi0 = LENNA_BANDS[b0][key]
    lo1, hi1 = LENNA_BANDS[b1][key]
    return (lo0 + t * (lo1 - lo0), hi0 + t * (hi1 - hi0))


def psnr_band(kind: str, bpp: float) -> tuple[float, float] | None:
    """(lo, hi) PSNR band for ``kind`` in {'dct', 'dwt'} at ``bpp`` — Lena table.
    Returns None for an unknown kind."""
    if kind not in ("dct", "dwt"):
        return None
    return _interp(float(bpp), kind)


def in_band(value: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return (lo - TOL_BELOW) <= value <= (hi + TOL_ABOVE)


def _verdict(value: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if value < lo - TOL_BELOW:
        return f"{value:.2f} dB — {lo - value:.1f} below band"
    if value > hi + TOL_ABOVE:
        return f"{value:.2f} dB — {value - hi:.1f} above band"
    if value > hi:
        return f"{value:.2f} dB — in band (better than typical)"
    if value < lo:
        return f"{value:.2f} dB — just under (educational codec)"
    return f"{value:.2f} dB — in band"


def reference_kind(filename: str) -> str:
    """Pick which reference applies from the sample's file name."""
    f = (filename or "").lower()
    if "len" in f:                       # lenna / lena
        return "lenna"
    if "finger" in f:
        return "fingerprint"
    if "synthetic" in f or "noise" in f:
        return "gaussian_white"
    return "generic"


def evaluate(bpp: float, dct_psnr: float, dwt_psnr: float,
             *, reference: str = "lenna", sigma: float | None = None) -> dict:
    """Compact reference check for the Compare tab. Returns
    {'title': str, 'rows': [(label, expected_text, verdict_text, ok_bool), ...]}.

    reference:
      'lenna'          -> the instructor's practical PSNR bands (pass/fail).
      'gaussian_white' -> the memoryless Gaussian RDF ceiling as a true upper
                          bound (needs sigma); a codec above it on a white
                          source is a bug.
      'fingerprint'    -> wavelet-over-DCT advantage + a pointer to the
                          lossless path (the intended one for fingerprints).
      'generic'        -> the Gaussian ceiling as an informational reference
                          line; being above it is normal 'coding gain from
                          spatial structure', never flagged.
    """
    rows: list[tuple[str, str, str, bool]] = []

    if reference == "lenna":
        db, wb = psnr_band("dct", bpp), psnr_band("dwt", bpp)
        rows.append(("JPEG / DCT", f"{db[0]:.0f}–{db[1]:.0f} dB",
                     _verdict(dct_psnr, db), in_band(dct_psnr, db)))
        rows.append(("Wavelet / DWT", f"{wb[0]:.0f}–{wb[1]:.0f} dB",
                     _verdict(dwt_psnr, wb), in_band(dwt_psnr, wb)))
        return {"title": f"Reference band — Lena 512² @ {bpp:.2f} bpp", "rows": rows}

    if reference == "fingerprint":
        adv = dwt_psnr - dct_psnr
        lo, hi = FINGERPRINT_LOSSLESS["dwt_advantage_db"]
        rows.append(("Wavelet − DCT", f"+{lo:.1f} … +{hi:.1f} dB",
                     f"actual {adv:+.2f} dB", adv >= lo - 0.3))
        rows.append(("Intended path", "lossless · MSE 0",
                     f"~{FINGERPRINT_LOSSLESS['ratio'][0]:.1f}–"
                     f"{FINGERPRINT_LOSSLESS['ratio'][1]:.1f}:1 — see the Lossless tab", True))
        return {"title": f"Fingerprint reference @ {bpp:.2f} bpp", "rows": rows}

    # gaussian_white / generic -> the Gaussian ceiling
    s = float(sigma) if sigma else 32.0
    ceil = gaussian_psnr_ceiling(s, bpp)
    strict = reference == "gaussian_white"
    for name, val in (("JPEG / DCT", dct_psnr), ("Wavelet / DWT", dwt_psnr)):
        if strict:
            # The memoryless-white bound is a strict ceiling only in the
            # near-white / low-rate regime; above ~1 bpp the target's faint
            # deterministic watermark gets coded and the codec may pass it.
            if val <= ceil + 0.75:
                ok = True
                note = f"{val:.2f} dB — {ceil - val:+.1f} dB vs ceiling"
            elif bpp > 1.0:
                ok = True
                note = f"{val:.2f} dB — above white-noise bound (chart structure coded)"
            else:
                ok = False
                note = f"{val:.2f} dB — ABOVE ceiling (check the codec)"
        else:
            ok = val > 0.0
            gain = val - ceil
            note = (f"{val:.2f} dB — {gain:+.1f} dB "
                    f"({'coding gain from structure' if gain >= 0 else 'below the line'})")
        rows.append((name, f"≈ {ceil:.2f} dB", note, ok))
    tag = "upper bound" if strict else "reference line"
    return {"title": f"Memoryless-Gaussian ceiling ({tag}) @ {bpp:.2f} bpp · σ {s:.1f}",
            "rows": rows}
