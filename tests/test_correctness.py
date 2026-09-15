# -*- coding: utf-8 -*-
"""Focused correctness tests for Signal Compression Lab.

Covers the engines (unchanged), the metrics, the ported analysis modules, and
that the Gradio UI builds and its two live compute paths (Compare, Lossless)
run. The old UI-wiring regression tests were dropped along with the tabs they
guarded (Semantic ROI, the Plotly RD chart, the before/after slider).
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest

warnings.filterwarnings("ignore")

import config
import src.compare as compare
from src.analysis.image_classifier import classify, classify_type
from src.analysis.lossless import compress, decompress, lossless_roundtrip
from src.engines import dct_engine, wavelet_engine
from src.engines.jpeg2000_engine import JPEG2000_AVAILABLE
from src.engines.real_jpeg_engine import REAL_JPEG_AVAILABLE
from src.metrics.quality import (bpp_from_bits, calculate_metrics,
                                 compression_ratio, mse, psnr, raw_bits,
                                 ratio_to_bpp, ssim)
from src.pgm_io import parse_pgm
from src.roi.bit_allocation import match_bpp
from src.viz import cards, dct_block

RNG = np.random.default_rng(20260902)
LENNA = "data/classic/lenna.png"
CLASSIC = ["data/classic/lenna.png", "data/classic/boat.png",
           "data/classic/couple.png", "data/classic/man.png"]


@pytest.fixture
def gray512() -> np.ndarray:
    img = cv2.imread(LENNA, cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (512, 512)
    return img


@pytest.fixture
def random_gray() -> np.ndarray:
    return RNG.integers(0, 256, (128, 128), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_bpp_and_ratio_formulas():
    shape = (100, 200)
    total_bits = 100 * 200 * 0.5
    assert bpp_from_bits(total_bits, shape) == pytest.approx(0.5)
    assert raw_bits(shape) == 100 * 200 * 8
    assert compression_ratio(shape, total_bits) == pytest.approx((100 * 200 * 8) / total_bits)


def test_ratio_to_bpp_roundtrip():
    assert ratio_to_bpp(8.0, channels=1) == pytest.approx(1.0)
    assert ratio_to_bpp(24.0, channels=3) == pytest.approx(1.0)


def test_psnr_mse_ssim_identity_and_noise(gray512):
    assert mse(gray512, gray512) == 0.0
    assert np.isinf(psnr(gray512, gray512))
    assert ssim(gray512, gray512) == pytest.approx(1.0, abs=1e-9)
    noisy = np.clip(gray512.astype(np.int16) + RNG.integers(-25, 25, gray512.shape), 0, 255).astype(np.uint8)
    m = mse(gray512, noisy)
    assert m > 0
    assert psnr(gray512, noisy) == pytest.approx(10.0 * np.log10(255.0 ** 2 / m), rel=1e-9)
    assert 0.0 < ssim(gray512, noisy) < 1.0


def test_psnr_nan_on_undefined_region(gray512):
    empty = np.zeros(gray512.shape, dtype=bool)
    assert np.isnan(psnr(gray512, gray512, empty))


def test_calculate_metrics_grayscale_baseline_not_3x(gray512):
    m = calculate_metrics(gray512, gray512.copy(), 0.5)
    assert m["original_size_bytes"] == pytest.approx(512 * 512 * 8 / 8.0)  # 262144, not 3x
    assert m["bpp"] == 0.5
    assert np.isinf(m["psnr"])


# ---------------------------------------------------------------------------
# Engines — DCT
# ---------------------------------------------------------------------------
def test_dct_block_roundtrip_no_quant():
    for _ in range(5):
        blk = RNG.integers(0, 256, (8, 8)).astype(np.float64)
        from scipy.fft import dctn, idctn
        rec = idctn(dctn(blk - 128, norm="ortho"), norm="ortho") + 128
        assert np.max(np.abs(rec - blk)) < 1e-9


def test_dct_engine_compress_lower_quality_more_zeros(gray512):
    rgb = np.stack([gray512] * 3, -1)
    r_hi, bpp_hi = dct_engine.compress_image(rgb, 90)
    r_lo, bpp_lo = dct_engine.compress_image(rgb, 15)
    assert bpp_lo < bpp_hi
    assert r_hi.shape == rgb.shape and r_lo.shape == rgb.shape


def test_dct_block_mosaic_quantized_is_rgb(gray512):
    m = dct_block.dct_block_mosaic_quantized(gray512, 40)
    assert m.ndim == 3 and m.shape[2] == 3 and m.dtype == np.uint8
    assert m.shape[:2] == gray512.shape


# ---------------------------------------------------------------------------
# Engines — Wavelet / DWT
# ---------------------------------------------------------------------------
def test_wavelet_lossless_roundtrip_pywt(gray512):
    import pywt
    coeffs = pywt.wavedec2(gray512.astype(np.float64), "bior4.4", level=3, mode="periodization")
    rec = pywt.waverec2(coeffs, "bior4.4", mode="periodization")[:512, :512]
    assert np.max(np.abs(rec - gray512)) < 1e-6


def test_wavelet_engine_compress_bpp_decreases_with_step(gray512):
    rgb = np.stack([gray512] * 3, -1)
    _, bpp_fine = wavelet_engine.compress_image(rgb, 4.0, "bior4.4")
    _, bpp_coarse = wavelet_engine.compress_image(rgb, 40.0, "bior4.4")
    assert bpp_coarse < bpp_fine


def test_max_decomposition_level_shrinks_with_filter_length():
    lvls = [wavelet_engine.max_decomposition_level((128, 128), w) for w in ("db2", "db4", "db8", "db12")]
    assert lvls == sorted(lvls, reverse=True) and lvls[-1] >= 1


# ---------------------------------------------------------------------------
# Engines — real codecs
# ---------------------------------------------------------------------------
def test_match_bpp_converges(gray512):
    rgb = np.stack([gray512] * 3, -1)
    recon, bpp, q = match_bpp(lambda x: dct_engine.compress_image(rgb, x),
                              0.6, 1, 100, True)
    assert abs(bpp - 0.6) / 0.6 < 0.15
    assert 1 <= q <= 100 and recon.shape == rgb.shape


def test_real_jpeg_actual_bpp_from_real_bytes(gray512):
    if not REAL_JPEG_AVAILABLE:
        pytest.skip("no libjpeg")
    from src.engines import real_jpeg_engine as rj
    recon, bpp, size_bytes, q = rj.compress_at_target_bpp(gray512, 0.5)
    assert bpp == pytest.approx(size_bytes * 8 / (512 * 512), rel=1e-9)
    assert recon.shape == gray512.shape and 1 <= q <= 100


def test_real_jpeg_psnr_in_reasonable_band(gray512):
    """Loose sanity band on lenna — no hardcoded exact number."""
    if not REAL_JPEG_AVAILABLE:
        pytest.skip("no libjpeg")
    from src.engines import real_jpeg_engine as rj
    prev = -1.0
    for target, (lo, hi) in {0.25: (24, 33), 0.50: (27, 36), 1.00: (31, 40)}.items():
        recon, bpp, *_ = rj.compress_at_target_bpp(gray512, target)
        p = psnr(gray512, recon)
        assert lo <= p <= hi, f"target={target} psnr={p:.2f} outside [{lo},{hi}]"
        assert p >= prev - 0.5
        prev = p


def test_jpeg2000_roundtrips(gray512):
    if not JPEG2000_AVAILABLE:
        pytest.skip("no OpenJPEG")
    from src.engines import jpeg2000_engine as jp2
    recon, bpp, size_bytes = jp2.compress_at_target_bpp(gray512, 0.5)
    assert recon.shape == gray512.shape and bpp > 0 and size_bytes > 0


# ---------------------------------------------------------------------------
# compare.py — the Compare tab's two engines
# ---------------------------------------------------------------------------
def test_compare_run_jpeg_primary(gray512):
    res = compare.run_jpeg_primary(gray512, 0.5, 8)
    assert res.recon.shape == gray512.shape
    for k in ("psnr", "ssim", "mse", "bpp", "compression_ratio", "compressed_size_bytes"):
        assert k in res.metrics
    assert "quality" in res.extra


def test_compare_run_wavelet_exposes_step_and_filter(gray512):
    res = compare.run_wavelet(gray512, 0.5, "bior4.4", config.WAVELET_LEVELS)
    assert res.recon.shape == gray512.shape
    assert res.is_real_codec is False
    assert "base_step" in res.extra and res.extra["wavelet"] == "bior4.4"


def test_compare_psnr_increases_with_bpp(gray512):
    lo = compare.run_jpeg_primary(gray512, 0.25, 8).metrics["psnr"]
    hi = compare.run_jpeg_primary(gray512, 1.0, 8).metrics["psnr"]
    assert hi > lo


# ---------------------------------------------------------------------------
# Ported analysis modules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", CLASSIC)
def test_image_classifier_returns_valid_type(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    t, feats = classify(img)
    assert t in ("natural", "synthetic", "hybrid")
    assert {"unique_colors", "entropy", "flat_areas", "edge_sharpness"} <= set(feats)


def test_image_classifier_flat_image_is_synthetic():
    flat = np.full((128, 128), 120, dtype=np.uint8)
    assert classify_type(flat) == "synthetic"


@pytest.mark.parametrize("path", CLASSIC)
def test_lossless_is_pixel_exact(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    recon, comp_bytes, orig_bytes = lossless_roundtrip(g, 6)
    assert np.array_equal(g, recon)
    assert comp_bytes < orig_bytes
    assert orig_bytes == g.size


def test_lossless_compress_decompress_symmetry(random_gray):
    stream, orig = compress(random_gray, 9)
    back = decompress(stream, random_gray.shape)
    assert np.array_equal(random_gray, back) and orig == random_gray.size


# ---------------------------------------------------------------------------
# PGM I/O
# ---------------------------------------------------------------------------
def _write_pgm(path, arr, binary=True):
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"{'P5' if binary else 'P2'}\n{w} {h}\n255\n".encode())
        if binary:
            f.write(arr.astype(np.uint8).tobytes())
        else:
            f.write(b" ".join(str(int(v)).encode() for v in arr.ravel()))


def test_pgm_p5_and_p2_roundtrip(tmp_path):
    arr = RNG.integers(0, 256, (17, 23), dtype=np.uint8)
    for binary in (True, False):
        p = tmp_path / f"t_{binary}.pgm"
        _write_pgm(p, arr, binary)
        out = parse_pgm(str(p))
        assert out.dtype == np.uint8 and np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# cards.py primitives
# ---------------------------------------------------------------------------
def test_cards_primitives():
    assert 'class="card"' in cards.card("Title", "<p>x</p>")
    assert 'card-title' in cards.card_title("HELLO")
    g = cards.metric_grid([("PSNR", "30 dB"), ("SSIM", "0.9", "", "accent")])
    assert 'metric-grid' in g and "30 dB" in g and "accent" in g
    assert "<h2>" in cards.page_header("H", "p")
    assert "Detected image type" in cards.image_type_pill("natural")
    assert cards.fmt_psnr(float("inf")) == "∞ (lossless)"
    assert cards.fmt_psnr(float("nan")) == "N/A"


# ---------------------------------------------------------------------------
# App — builds + the two live compute paths
# ---------------------------------------------------------------------------
class _Req:
    session_hash = "pytest"


def test_build_ui_constructs():
    import app
    demo = app.build_ui()
    assert type(demo).__name__ == "Blocks"


def test_compute_working_point_output(gray512):
    import app
    app._mark_latest_image("pytest", 7)
    rgb = np.stack([gray512] * 3, -1)
    out = app.compute_working_point(rgb, 7, 0.5, "bior4.4", {"filename": "lenna"}, _Req())
    assert len(out) == 7
    dct_grid, wav_grid, dct_img, wav_img, type_pill, dwt_layer, ref_html = out
    assert "Quality (from bpp)" in dct_grid and "PSNR" in dct_grid
    assert "Quant. step" in wav_grid
    assert dct_img.shape == gray512.shape and wav_img.shape == gray512.shape
    assert "Detected image type" in type_pill
    assert dwt_layer is not None and dwt_layer.ndim >= 2
    assert "Reference band" in ref_html and "JPEG / DCT" in ref_html
    # a non-Lena image falls back to the Gaussian RD ceiling
    out2 = app.compute_working_point(rgb, 7, 0.5, "bior4.4", {"filename": "noise"}, _Req())
    assert "ceiling" in out2[6]


def test_compute_rd_scatter_and_render(gray512):
    import app
    app._mark_latest_image("pytest", 8)
    rgb = np.stack([gray512] * 3, -1)
    state, fig = app.compute_rd(rgb, 8, 0.5, "bior4.4", {}, _Req())
    assert len(state["dct"]) >= 5 and len(state["dwt"]) >= 5 and fig is not None
    # points are (bpp, psnr) and psnr rises with bpp
    ds = sorted(state["dct"])
    assert ds[0][1] < ds[-1][1]
    assert app.render_rd(state, 1.2) is not None


def test_compute_working_point_stale_is_noop(gray512):
    import app
    app._mark_latest_image("pytest", 100)
    out = app.compute_working_point(np.stack([gray512] * 3, -1), 99, 0.5, "bior4.4", {}, _Req())
    assert out == ("", "", None, None, "", None, "")


def test_dct_zones_energy_sums_to_100(gray512):
    from src.analysis.dct_zones import ZONE_LABELS, zone_layer_images
    layers, pct = zone_layer_images(gray512)
    assert len(layers) == 4 and len(pct) == 4
    assert sum(pct) == pytest.approx(100.0, abs=0.5)
    assert pct[0] > sum(pct[1:])  # DC + low-freq dominates a natural photo
    assert all(0.0 <= float(l.min()) and float(l.max()) <= 1.0 for l in layers)


def test_dct_core_panels_and_metrics(gray512):
    import app
    app._mark_latest_image("pytest", 3)
    rgb = np.stack([gray512] * 3, -1)
    out = app._dct_core(rgb, 3, 0.5, None, None, _Req())
    assert len(out) == 12
    preview, pipe, rc, p1, p2, p3, p4, p5, p6, p6c, zz, zzc = out
    assert preview.ndim == 3
    assert rc == (256, 256)  # centre block of a 512x512
    for p in (p1, p2, p3, p4, p5, p6):
        assert p.ndim == 3 and p.shape[2] == 3
    assert "JPEG quality" in pipe and "Block PSNR" in pipe
    assert "dB vs. original block" in p6c
    assert zzc.startswith('<p class="caption">First 16: [') or "First 16" in zzc
    assert zz.ndim == 3


def test_dct_click_snaps_to_8x8_grid(gray512):
    import app
    app._mark_latest_image("pytest", 3)

    class _Evt:
        index = (123, 201)  # (x, y)

    out = app.dct_from_click(np.stack([gray512] * 3, -1), 3, 0.5, _Evt(), _Req())
    assert out[2] == (201 // 8 * 8, 123 // 8 * 8)  # (row, col)


def test_dct_block_panel_grid_helpers(gray512):
    from src.viz import dct_block as db
    m = db.matrix_grid_rgb(np.arange(-32, 32).reshape(8, 8).astype(float), value_fmt="int")
    assert m.shape == (8 * 40, 8 * 40, 3)
    assert db.zigzag_order_grid_rgb().shape[2] == 3
    assert db.zigzag_bar_rgb(np.zeros((8, 8))).shape[2] == 3
    p = db.jpeg_block_panels(gray512, 40, 40, 80)
    assert p["block_row"] == 40 and p["block_col"] == 80
    assert len(p["panels"]) == 6 and len(p["zigzag_first16"]) == 16


# ---------------------------------------------------------------------------
# DWT Lab + PR-QMF
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("proto", ["prqmf2", "prqmf4", "prqmf6"])
def test_pr_qmf_mband4_roundtrips(proto, gray512):
    from src.engines import pr_qmf
    bands, shape = pr_qmf.dwt2_mband4(gray512.astype(np.float64), proto)
    assert len(bands) == 4 and all(len(row) == 4 for row in bands)
    assert bands[0][0].shape == (128, 128)  # 512 / 4
    rec = pr_qmf.idwt2_mband4(bands, shape, proto)
    # rounded 4-dp prototype coefficients -> near-perfect reconstruction
    assert np.mean((gray512 - rec) ** 2) < 1e-2
    epct = pr_qmf.band_energy_pct(bands)
    assert len(epct) == 16 and sum(epct.values()) == pytest.approx(100.0, abs=0.5)
    assert epct["B00"] == max(epct.values())      # lowest-freq quarter has most energy


def test_single_level_bands_energy(gray512):
    from src.viz import subbands
    b = subbands.single_level_bands(gray512.astype(np.float64), "bior4.4")
    assert set(b) == {"LL", "HL", "LH", "HH"}
    pcts = {k: v[1] for k, v in b.items()}
    assert sum(pcts.values()) == pytest.approx(100.0, abs=0.5)
    assert pcts["LL"] > sum(v for k, v in pcts.items() if k != "LL")
    assert all(b[k][0].dtype == np.uint8 for k in b)


def test_pyramid_and_mband_mosaics_are_labeled_rgb(gray512):
    import pywt

    from src.viz import subbands
    from src.engines import pr_qmf
    coeffs = pywt.wavedec2(gray512.astype(np.float64) - 128, "bior4.4", level=3)
    pm = subbands.pyramid_labeled(coeffs)
    assert pm.ndim == 3 and pm.shape[2] == 3
    bands, _ = pr_qmf.dwt2_mband4(gray512.astype(np.float64), "prqmf4")
    mm = subbands.mband_mosaic_labeled(bands)
    assert mm.ndim == 3 and mm.shape == (512, 512, 3)


def test_dwt_recompute_scalar_and_uniform(gray512):
    import app
    app._mark_latest_image("pytest", 4)
    rgb = np.stack([gray512] * 3, -1)
    o = app.dwt_recompute(rgb, 4, "bior4.4", app._QMODE_SCALAR, app._DWT_AUTO,
                          0.5, 20.0, _Req())
    assert len(o) == 11
    quant, llt, lli, hlt, hli, lht, lhi, hht, hhi, mosaic, cap = o
    assert "Step" in quant and "Subbands" in quant
    for img in (lli, hli, lhi, hhi):
        assert img.dtype == np.uint8
    assert "energy" in llt and mosaic.ndim == 3
    assert cap.strip().startswith('<p class="caption">LL3 · LH3')
    o2 = app.dwt_recompute(rgb, 4, "db4", app._QMODE_UNIFORM, app._DWT_MANUAL,
                           0.5, 15.0, _Req())
    assert "Manual override" in o2[0]


def test_pr_qmf_recompute_output(gray512):
    import app
    app._mark_latest_image("pytest", 4)
    check, energy, mosaic = app.pr_qmf_recompute(np.stack([gray512] * 3, -1), 4,
                                                 "prqmf4", _Req())
    assert "Analysis→Synthesis MSE" in check and ">16<" in check
    assert energy.count('class="metric"') == 4
    assert mosaic.ndim == 3


def test_run_lossless_pixel_exact(gray512):
    import app
    app._mark_latest_image("pytest", 5)
    out = app.run_lossless(np.stack([gray512] * 3, -1), 5, _Req())
    assert len(out) == 6
    bpp_grid, check, pill, size, orig, recon = out
    assert "Bits per pixel" in bpp_grid and "8.000 bpp" in bpp_grid
    assert "MSE" in check and "0.000000" in check
    assert "Pixel-exact" in pill
    assert np.array_equal(orig, recon)


# ---------------------------------------------------------------------------
# Guard: no Semantic/YOLO leftovers in the shipped app
# (the reference-range check is a wanted feature again — see
# src/metrics/reference_ranges.py — so it is deliberately not on this list.)
# ---------------------------------------------------------------------------
def test_no_yolo_or_semantic_imports_in_app():
    src = open("app.py", encoding="utf-8").read()
    for bad in ("ultralytics", "src.semantic", "importance_map", "yolo",
                "interactive_plots", "run_semantic_pipeline", "rd_interactive_figure"):
        assert bad not in src, f"app.py still references {bad!r}"
