# -*- coding: utf-8 -*-
"""Signal Compression Lab — Gradio web UI.

Run:
    python app.py
then open http://127.0.0.1:7860 in a browser.

Tabs: Compare / DCT Lab / DWT Lab / Lossless / Theory. The layout mirrors the
old React app (frontend/src/pages/*.tsx of "signalprojelast 2"); the maths
comes from this project's own engines (src/engines/*, src/compare.py,
src/metrics/quality.py). Warm "ivory / teal / copper" palette from
src/viz/style.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- Compatibility shim for a known gradio 4.44 / gradio_client 1.3 bug ---
# gradio_client.utils.get_type() / _json_schema_to_python_type() crash with
# "TypeError: argument of type 'bool' is not iterable" when a component's JSON
# schema contains a bare boolean (e.g. additionalProperties: false, which
# gr.State(dict) produces). It only affects the auto-generated API docs, never
# the UI — so we degrade it to "Any" instead of taking the server down.
try:  # pragma: no cover - environment-dependent
    import gradio_client.utils as _gcu

    _orig_json_schema = _gcu._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        try:
            return _orig_json_schema(schema, defs)
        except TypeError:
            return "Any"

    _gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
    _orig_get_type = _gcu.get_type
    _gcu.get_type = lambda schema: (
        "Any" if not isinstance(schema, dict) else _orig_get_type(schema)
    )
except Exception:  # noqa: BLE001 - best effort only
    pass

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import config
import src.compare as compare
from src.analysis.image_classifier import classify_type
from src.analysis.lossless import lossless_roundtrip
from src.engines import dct_engine, pr_qmf, real_jpeg_engine, wavelet_engine
from src.engines.wavelet_engine import decompose_for_viz, max_decomposition_level
from src.metrics import reference_ranges
from src.metrics.quality import psnr, ssim
from src.pgm_io import parse_pgm
from src.ui_theme import FORCE_LIGHT_JS, LAB_CSS, LAB_THEME
from src.viz import cards, dct_block, subbands
from src.viz.plots import rd_line_figure

MAX_SIDE = 640  # large uploads are downscaled for speed
_BPP_MIN, _BPP_MAX = 0.1, 5.0
_LEVELS = config.WAVELET_LEVELS
# Rate-distortion sweep: fixed encode parameters, no per-point bisection
# (a scatter of measured (bpp, psnr) points is all the chart needs).
# Rate-distortion curve: BOTH methods are encoded at the SAME target bitrates
# (each point is bisected to that bpp via compare.run_*), so the two curves
# share an x-grid and start / end together — the way an RD comparison must be
# read. Runs only on image / filter change; the bpp slider only redraws.
_RD_BPP_POINTS = [0.15, 0.25, 0.40, 0.60, 0.85, 1.20, 1.70, 2.40, 3.20]



# =============================================================================
# Image freshness tracking: an auto-analysis chain can take seconds; if the
# user picks another image meanwhile, the stale queued job is skipped before
# it starts. Gradio's per-session queue is sequential, so a dict keyed by
# request.session_hash is correct for a single-user local tool.
# =============================================================================
_LATEST_IMAGE_ID: dict[str, int] = {}


def _mark_latest_image(session_hash: str, image_id) -> None:
    _LATEST_IMAGE_ID[session_hash] = int(image_id)


def _is_stale(session_hash: str, image_id) -> bool:
    return _LATEST_IMAGE_ID.get(session_hash, -1) != int(image_id)


def _prepare(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]).astype(np.float64)


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(_to_gray(image), 0, 255).astype(np.uint8)


def _reduce_if_true_grayscale(image: np.ndarray) -> np.ndarray:
    """active_img is always RGB; for real grayscale sources (the classic PNGs,
    grayscale JPEG / PGM uploads) that is just an R=G=B triple. Passing that
    triple to the engines as 3 channels sends them down the YCbCr/colour path
    (non-zero chroma bytes stolen from luma -> lower PSNR; a 3x-too-large raw
    baseline). Real colour images (R,G,B genuinely differ) are left untouched.
    """
    if image.ndim != 3:
        return image
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if np.array_equal(r, g) and np.array_equal(g, b):
        return r
    return image


# =============================================================================
# Categorised sample-image source (used only by the single global picker)
# =============================================================================
_MAX_GALLERY_ITEMS = 6


def _category_dir(label: str) -> str:
    return next(k for k, l in config.UI_IMAGE_CATEGORIES if l == label)


def _category_images(label: str) -> list[str]:
    folder = config.DATA_DIR / _category_dir(label)
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.glob("*.png"))[:_MAX_GALLERY_ITEMS]


def _on_category_change(label: str):
    return gr.update(value=_category_images(label))


def build_active_image_bar():
    """The single canonical image picker + active-image bar for the whole app.
    Placed above gr.Tabs(); Compare / DCT Lab / DWT Lab / Lossless all share:

      active_img  - gr.Image(visible=False): canonical RGB numpy array
      active_id   - gr.Number(visible=False): counter, bumped on every pick /
                    upload; the single trigger for all "image changed" chains
      active_meta - gr.State(dict): filename / category / width / height / ...

    Returns (active_img, active_id, active_meta, home_btn).
    """
    default_label = config.UI_IMAGE_CATEGORIES[0][1]

    # format="png" (not the default lossy "webp"): Gradio re-encodes the
    # component value on every postprocess/preprocess hop, and webp would
    # corrupt the "original" pixels the engines see (measured: up to 20/255
    # per 8-bit channel). PNG is lossless.
    active_img = gr.Image(visible=False, type="numpy", format="png")
    active_id = gr.Number(visible=False, value=0, precision=0)
    active_meta = gr.State({})

    with gr.Column(visible=True) as empty_group:
        gr.HTML(cards.hero_empty_html(
            "Select an Image",
            "The image you pick is used at the same time by ALL of the "
            "Compare, DCT Lab, DWT Lab and Lossless tabs.",
        ))
        with gr.Row(elem_classes=["hero-buttons"]):
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=0, min_width=420):
                with gr.Row():
                    select_btn = gr.Button("Pick Sample", size="sm", variant="secondary")
                    upload_btn = gr.Button("Upload Image", size="sm", variant="secondary")
            with gr.Column(scale=1):
                pass

    with gr.Row(visible=False, elem_classes=["active-image-bar"]) as bar_row:
        bar_html = gr.HTML()
        change_btn = gr.Button("Change image", size="sm", variant="secondary")
        home_btn = gr.Button("⌂ Home", size="sm", variant="secondary")

    with gr.Accordion("Categories and samples", open=False, visible=False) as picker_acc:
        cat_radio = gr.Radio([l for _, l in config.UI_IMAGE_CATEGORIES], value=default_label,
                             label="Category")
        gallery = gr.Gallery(value=_category_images(default_label), columns=_MAX_GALLERY_ITEMS, rows=1,
                             height=180, show_label=False, object_fit="contain",
                             allow_preview=False)
    with gr.Group(visible=False) as upload_group:
        # gr.File (not gr.Image): gr.Image(type="filepath") re-encodes every
        # upload via PIL in this Gradio version, so Path.stat would report the
        # re-encoded copy's size, not the real file. gr.File passes the file
        # through untouched, so the KB figure is real.
        upload_img = gr.File(
            file_types=["image"], type="filepath", label="Upload your image")
        # gr.Image's browser-side upload filter is a fixed "image/*"; .pgm is
        # usually reported as application/octet-stream and rejected there.
        # gr.File matches by extension, so PGM gets its own small uploader.
        upload_pgm = gr.File(
            file_types=[".pgm", ".pnm", "image/x-portable-graymap", "application/octet-stream"],
            type="filepath",
            label="or upload a .pgm (Netpbm grayscale, P2/P5)",
        )

    select_btn.click(
        lambda: (gr.update(visible=True, open=True), gr.update(visible=False)),
        None, [picker_acc, upload_group])
    upload_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False, open=False)),
        None, [upload_group, picker_acc])
    change_btn.click(
        lambda: (gr.update(visible=True, open=True), gr.update(visible=False)),
        None, [picker_acc, upload_group])
    home_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False),
                 gr.update(visible=False, open=False), gr.update(visible=False)),
        None, [empty_group, bar_row, picker_acc, upload_group])
    cat_radio.change(_on_category_change, cat_radio, gallery)

    def _select(cat_label, current_id, evt: gr.SelectData, request: gr.Request):
        path = evt.value["image"]["path"]
        file_size_bytes = Path(path).stat().st_size
        raw = np.array(Image.open(path).convert("RGB"))
        orig_h, orig_w = raw.shape[:2]  # before _prepare — never hide a resize
        img = _prepare(raw)
        name = Path(path).stem
        h, w = img.shape[:2]
        meta = {"filename": name, "category": cat_label, "width": w, "height": h,
                "source_type": "sample", "file_size_bytes": file_size_bytes}
        new_id = int(current_id) + 1
        _mark_latest_image(request.session_hash, new_id)
        return (
            img, new_id, meta,
            cards.active_image_bar_html(name, cat_label, w, h, orig_w, orig_h, file_size_bytes),
            gr.update(visible=True), gr.update(visible=False, open=False),
            gr.update(visible=False),
        )

    def _upload(path, current_id, request: gr.Request):
        if path is None:
            return (gr.update(),) * 7
        file_size_bytes = Path(path).stat().st_size
        raw = np.array(Image.open(path).convert("RGB"))
        orig_h, orig_w = raw.shape[:2]
        img = _prepare(raw)
        h, w = img.shape[:2]
        meta = {"filename": "Uploaded image", "category": None, "width": w, "height": h,
                "source_type": "upload", "file_size_bytes": file_size_bytes}
        new_id = int(current_id) + 1
        _mark_latest_image(request.session_hash, new_id)
        return (
            img, new_id, meta,
            cards.active_image_bar_html("Uploaded image", None, w, h, orig_w, orig_h, file_size_bytes),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False),
        )

    def _upload_pgm(path, current_id, request: gr.Request):
        if path is None:
            return (gr.update(),) * 7
        file_size_bytes = Path(path).stat().st_size
        try:
            gray = parse_pgm(path)  # P2/P5, dims from the header — see src/pgm_io
        except ValueError as exc:
            raise gr.Error(f"PGM parse error: {exc}") from exc
        orig_h, orig_w = gray.shape
        rgb = np.stack([gray, gray, gray], axis=-1)
        img = _prepare(rgb)
        h, w = img.shape[:2]
        meta = {"filename": "Uploaded image (PGM)", "category": None, "width": w, "height": h,
                "source_type": "upload", "file_size_bytes": file_size_bytes}
        new_id = int(current_id) + 1
        _mark_latest_image(request.session_hash, new_id)
        return (
            img, new_id, meta,
            cards.active_image_bar_html("Uploaded image (PGM)", None, w, h, orig_w, orig_h, file_size_bytes),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False),
        )

    gallery.select(
        _select, [cat_radio, active_id],
        [active_img, active_id, active_meta, bar_html, bar_row, picker_acc, empty_group])
    upload_img.change(
        _upload, [upload_img, active_id],
        [active_img, active_id, active_meta, bar_html, bar_row, upload_group, empty_group])
    upload_pgm.change(
        _upload_pgm, [upload_pgm, active_id],
        [active_img, active_id, active_meta, bar_html, bar_row, upload_group, empty_group])

    return active_img, active_id, active_meta, home_btn


# =============================================================================
# TAB 1 — COMPARE  (mirrors frontend/src/pages/CompareLab.tsx)
# =============================================================================
def _metric_grid_for(engine_metrics: dict, lead_label: str, lead_value: str,
                     lead_tone: str) -> str:
    m = engine_metrics
    return cards.metric_grid([
        (lead_label, lead_value, "", lead_tone),
        ("PSNR", f"{m['psnr']:.2f} dB"),
        ("MSE", f"{m['mse']:.3f}"),
        ("SSIM", f"{m['ssim']:.4f}"),
        ("Actual BPP", f"{m['bpp']:.3f}"),
        ("Compression", f"{m['compression_ratio']:.1f}:1"),
        ("Size", cards.fmt_size_kb(m["compressed_size_bytes"])),
    ])


def _clamp_bpp(v) -> float:
    try:
        return float(np.clip(float(v), _BPP_MIN, _BPP_MAX))
    except (TypeError, ValueError):
        return 0.5


def _sync_bpp_from_num(v):
    """Manual bpp entry -> one normalised value written to BOTH the slider and
    the number field, so the two lock together in a single hop (no feedback
    loop, no double recompute)."""
    x = round(_clamp_bpp(v), 2)
    return x, x


def _wire_manual_bpp(num, slider, *steps):
    """Wire a manual-bpp number field so that TYPING a value (no Enter needed)
    or pressing Enter moves the slider and re-runs `steps` — each a
    (fn, inputs, outputs) triple, chained in order.

    `.input` (not `.change`) is deliberate: it fires only on genuine user
    input, so the slider->number mirror writing this field programmatically
    can't loop back here. `trigger_mode="always_last"` coalesces per-keystroke
    fires into a single trailing recompute."""
    for trigger in (num.input, num.submit):
        dep = trigger(_sync_bpp_from_num, num, [slider, num],
                      trigger_mode="always_last")
        for fn, ins, outs in steps:
            dep = dep.then(fn, ins, outs)


def _reference_html(image_meta, tb, gray, dct_psnr, dwt_psnr) -> str:
    """The 'is the PSNR in the reference range?' card. The reference depends on
    the sample: Lena -> the instructor's practical bands; the synthetic noise
    target -> the Gaussian rate-distortion ceiling (a true upper bound);
    the fingerprint -> the wavelet-over-DCT advantage + the lossless pointer;
    anything else -> the Gaussian ceiling as an informational reference line."""
    fname = image_meta.get("filename", "") if isinstance(image_meta, dict) else ""
    kind = reference_ranges.reference_kind(fname)
    ev = reference_ranges.evaluate(tb, dct_psnr, dwt_psnr, reference=kind,
                                   sigma=float(np.std(gray)))
    return cards.reference_check(ev["title"], ev["rows"])


def compute_working_point(image, image_id, target_bpp, wavelet, image_meta,
                          request: gr.Request):
    """Fast path (~0.5 s): the JPEG/DCT and Wavelet/DWT result at the current
    target bpp — metrics, reconstructions, the DWT subband mosaic, image-type
    pill and the reference-range check. Runs on every bpp / filter / image
    change.

    Returns 7 items: (dct_grid, wav_grid, dct_recon, wav_recon, type_pill,
    dwt_layer, reference_html)."""
    empty = ("", "", None, None, "", None, "")
    if image is None or _is_stale(request.session_hash, image_id):
        return empty
    image = _reduce_if_true_grayscale(_prepare(image))
    tb = _clamp_bpp(target_bpp)

    dct = compare.run_jpeg_primary(image, tb, 8)
    wav = compare.run_wavelet(image, tb, wavelet, _LEVELS)

    q_val = dct.extra.get("quality")
    q_disp = f"{int(round(q_val))}" if isinstance(q_val, (int, float)) else "-"
    wav_step = float(wav.extra.get("base_step", 0.0))

    dct_grid = _metric_grid_for(dct.metrics, "Quality (from bpp)", q_disp, "accent")
    wav_grid = _metric_grid_for(wav.metrics, "Quant. step (Δ)", f"{wav_step:.2f}", "violet")
    type_pill = cards.image_type_pill(classify_type(image))

    gray = _to_gray(image)
    lvl = min(_LEVELS, max_decomposition_level(gray.shape, wavelet))
    # periodization -> tiles halve exactly (no seams/overlap); quantize the
    # detail bands at the current step so the layer tracks the target bpp.
    vcoef = decompose_for_viz(gray, wavelet, lvl, mode="periodization")
    dwt_layer = subbands.quantized_pyramid(vcoef, wav_step or 8.0)
    ref_html = _reference_html(image_meta, tb, np.clip(gray, 0, 255),
                               dct.metrics["psnr"], wav.metrics["psnr"])

    return (dct_grid, wav_grid, dct.recon, wav.recon, type_pill, dwt_layer, ref_html)


def compute_rd(image, image_id, target_bpp, wavelet, image_meta,
               request: gr.Request):
    """Slow path (~1-2 s): the rate-distortion curves. BOTH JPEG/DCT and
    Wavelet/DWT are encoded at every bitrate in _RD_BPP_POINTS (the same
    bisection the working point uses), so the two curves share an x-grid and
    begin / end at the same bitrate. Runs only on image / filter change.
    (`image_meta` is unused; it just keeps the Compare input list shared.)
    Returns (rd_state, figure)."""
    if image is None or _is_stale(request.session_hash, image_id):
        return {"dct": [], "dwt": []}, None
    image = _reduce_if_true_grayscale(_prepare(image))
    gray = np.clip(_to_gray(image), 0, 255).astype(np.uint8)

    dct_pts, dwt_pts = [], []
    for tb in _RD_BPP_POINTS:
        d = compare.run_jpeg_primary(gray, tb, 8)
        w_ = compare.run_wavelet(gray, tb, wavelet, _LEVELS)
        dct_pts.append((d.metrics["bpp"], d.metrics["psnr"]))
        dwt_pts.append((w_.metrics["bpp"], w_.metrics["psnr"]))

    dct_pts.sort()
    dwt_pts.sort()
    state = {"dct": dct_pts, "dwt": dwt_pts}
    return state, rd_line_figure(dct_pts, dwt_pts, _clamp_bpp(target_bpp))


def render_rd(rd_state, target_bpp):
    """Re-draw the RD chart from the cached scatter with a new target line —
    cheap, used on bpp change so the marker moves without a re-sweep."""
    rd_state = rd_state or {"dct": [], "dwt": []}
    return rd_line_figure(rd_state.get("dct", []), rd_state.get("dwt", []),
                          _clamp_bpp(target_bpp))


# =============================================================================
# TAB 5 — THEORY  (text verbatim from frontend/src/pages/Theory.tsx)
# =============================================================================
_THEORY = [
    ("DCT (JPEG)",
     "JPEG splits the image into 8&times;8 pixel blocks and applies a 2D "
     "Discrete Cosine Transform to each one independently. Each block produces "
     "one DC coefficient (average brightness) and 63 AC coefficients "
     "(frequency content). DCT does <em>not</em> produce true LL/LH/HL/HH "
     "subbands &mdash; those names belong to DWT. Coefficients are quantized "
     "against a quality-scaled matrix (more aggressive at low quality), then "
     "read out in a zigzag order &mdash; low frequency first &mdash; so runs "
     "of quantized-to-zero high-frequency coefficients compress well with "
     "entropy coding."),
    ("DWT (JPEG2000)",
     "JPEG2000-style coding applies a Discrete Wavelet Transform to the "
     "<em>whole</em> image at once, recursively splitting it into an "
     "approximation band (LL) and three detail bands (LH horizontal, HL "
     "vertical, HH diagonal) per level. Unlike block-based DCT, DWT has no "
     "block artifacts and degrades gracefully at low bit-rates. This lab "
     "supports Haar, Daubechies (db2&ndash;db32), Biorthogonal 9/7 (the real "
     "JPEG2000 wavelet), and custom Perfect-Reconstruction QMF filter banks."),
    ("Bits Per Pixel (BPP)",
     "BPP is the compressed size in bits divided by the pixel count &mdash; "
     "the standard way to compare compression efficiency independent of image "
     "resolution. This project drives both DCT and DWT from the <em>same</em> "
     "target BPP: for DCT, a full quality sweep (1&ndash;100) finds the JPEG "
     "quality whose encoded size is closest to the target; for DWT, a binary "
     "search finds the quantization step whose coded size matches it."),
    ("PSNR / SSIM / MSE",
     "<strong>MSE</strong> is the mean squared pixel difference between "
     "original and reconstructed images. <strong>PSNR</strong> (dB) is a "
     "log-scaled inverse of MSE &mdash; higher is better, and it is the most "
     "common lossy-compression quality metric. <strong>SSIM</strong> compares "
     "local luminance, contrast, and structure instead of raw pixel "
     "differences, which tracks perceived quality more closely than PSNR, "
     "especially at low bit-rates."),
]


def _theory_html() -> str:
    body = "".join(
        cards.card(title, f'<p style="font-size:12.5px;line-height:1.6;'
                          f'color:var(--lab-text-secondary);margin:0">{text}</p>')
        for title, text in _THEORY
    )
    return f'<div class="sc-grid" style="flex-wrap:wrap">{body}</div>'


# =============================================================================
# TAB 4 — LOSSLESS  (mirrors frontend/src/pages/LosslessLab.tsx)
# =============================================================================
def run_lossless(image, image_id, request: gr.Request):
    """Lossless has no target-bitrate knob (it can't trade quality for rate),
    so instead of an "effort" slider this reports the bit-rate the entropy
    coder actually achieves. zlib is run at its maximum level — the
    reconstruction is identical at any level, so there is nothing to tune.

    Returns 6 items: (bpp_grid, check_grid, check_pill, size_grid, orig, recon)."""
    empty = ("", "", "", "", None, None)
    if image is None or _is_stale(request.session_hash, image_id):
        return empty
    image = _reduce_if_true_grayscale(_prepare(image))
    gray = _to_gray_u8(image)
    h, w = gray.shape
    recon, comp_bytes, orig_bytes = lossless_roundtrip(gray, 9)

    m = float(np.mean((gray.astype(np.float64) - recon.astype(np.float64)) ** 2))
    is_exact = m == 0.0
    s = ssim(gray, recon)
    ratio = orig_bytes / max(1, comp_bytes)
    bpp = comp_bytes * 8.0 / (h * w)
    saving = (1.0 - comp_bytes / max(1, orig_bytes)) * 100.0

    bpp_grid = cards.metric_grid([
        ("Bits per pixel", f"{bpp:.3f}", "achieved — no target-rate control", "accent"),
        ("Uncompressed", "8.000 bpp"),
        ("Space saving", f"{saving:.1f}%"),
    ])
    check_grid = cards.metric_grid([
        ("MSE", f"{m:.6f}", "", "accent" if is_exact else ""),
        ("PSNR", "exact match" if is_exact else f"{psnr(gray, recon):.2f} dB"),
        ("SSIM", f"{s:.6f}"),
    ])
    check_pill = cards.pill(
        "✓ Pixel-exact — 0 bits of information lost" if is_exact else "computing…",
        good=is_exact)
    size_grid = cards.metric_grid([
        ("Compressed", cards.fmt_size_kb(comp_bytes)),
        ("Original", cards.fmt_size_kb(orig_bytes)),
        ("Ratio", f"{ratio:.2f}:1", "", "accent"),
    ])
    return bpp_grid, check_grid, check_pill, size_grid, gray, recon


# =============================================================================
# TAB 2 — DCT LAB  (mirrors frontend/src/pages/DctLab.tsx)
# =============================================================================
def _dct_core(image, image_id, target_bpp, row, col, request: gr.Request):
    empty = ("", "", (None, None), None, None, None, None, None, None, "", None, "")
    if image is None or _is_stale(request.session_hash, image_id):
        return empty
    image = _reduce_if_true_grayscale(_prepare(image))
    gray = _to_gray_u8(image)
    tb = _clamp_bpp(target_bpp)

    res = compare.run_jpeg_primary(gray, tb, 8)
    q_val = res.extra.get("quality")
    q_int = int(round(q_val)) if isinstance(q_val, (int, float)) else 50
    degraded = res.recon if res.recon.ndim == 2 else _to_gray_u8(res.recon)

    p = dct_block.jpeg_block_panels(gray, q_int, row, col)
    bi, bj = p["block_row"], p["block_col"]

    preview = dct_block.image_with_block_box(np.clip(degraded, 0, 255).astype(np.uint8), bi, bj)
    pipeline = cards.metric_grid([
        ("Block size", "8×8"),
        ("JPEG quality", str(q_int), "derived from target bpp", "accent"),
        ("Row", str(bi)),
        ("Col", str(bj)),
        ("Block PSNR", cards.fmt_psnr(p["block_psnr"]), "", "accent"),
        ("Block MSE", f"{p['block_mse']:.3f}"),
    ])
    panels = [img for _, _, img in p["panels"]]
    p6_caption = cards.caption(p["panels"][5][1])
    zz_caption = cards.caption("First 16: [" + ", ".join(str(v) for v in p["zigzag_first16"]) + "]")

    return (preview, pipeline, (bi, bj),
            panels[0], panels[1], panels[2], panels[3], panels[4], panels[5],
            p6_caption, p["zigzag_bar"], zz_caption)


def dct_from_bpp(image, image_id, target_bpp, rc_state, request: gr.Request):
    r, c = rc_state if rc_state else (None, None)
    return _dct_core(image, image_id, target_bpp, r, c, request)


def dct_from_click(image, image_id, target_bpp, evt: gr.SelectData, request: gr.Request):
    x, y = evt.index
    return _dct_core(image, image_id, target_bpp, int(y), int(x), request)


def dct_zones(image, image_id, request: gr.Request):
    empty = ("", None, "", None, "", None, "", None)
    if image is None or _is_stale(request.session_hash, image_id):
        return empty
    from src.analysis.dct_zones import ZONE_LABELS, zone_layer_images
    image = _reduce_if_true_grayscale(_prepare(image))
    layers, pct = zone_layer_images(_to_gray_u8(image))
    out = []
    for label, layer, p in zip(ZONE_LABELS, layers, pct):
        out.append(cards.card_title(f"{label} — {p:.1f}% energy"))
        out.append((np.clip(layer, 0, 1) * 255).astype(np.uint8))
    return tuple(out)


# =============================================================================
# TAB 3 — DWT LAB  (mirrors frontend/src/pages/DwtLab.tsx)
# =============================================================================
_DWT_LEVEL = 3
_DWT_STEP_MIN, _DWT_STEP_MAX = 0.5, 150.0
_DWT_AUTO = "Auto (derived from target BPP)"
_DWT_MANUAL = "Manual quantization step (Δ)"
_QMODE_SCALAR = "Scalar (band-weighted)"
_QMODE_UNIFORM = "Uniform"


def _dwt_uniform_channel(gray_c: np.ndarray, wavelet: str, levels: int, step: float):
    """One flat quantization step for every subband (vs the engine's
    band-weighted default). Returns (recon 0-255, total_bits)."""
    import pywt

    from src.engines.wavelet_engine import _quantize_band, decompose_for_viz
    coeffs = decompose_for_viz(gray_c, wavelet, levels)
    q0, s0, b0 = _quantize_band(coeffs[0], step, None, 1.0)
    total = b0
    new = [q0.astype(np.float64) * s0]
    for li in range(1, len(coeffs)):
        bands = []
        for bi in range(3):
            q, s, b = _quantize_band(coeffs[li][bi], step, None, 1.0)
            total += b
            bands.append(q.astype(np.float64) * s)
        new.append(tuple(bands))
    rec = pywt.waverec2(new, wavelet, mode="symmetric")
    rec = rec[:gray_c.shape[0], :gray_c.shape[1]] + 128.0
    return np.clip(rec, 0, 255), total


def _dwt_encode(gray: np.ndarray, wavelet: str, levels: int, qmode: str, step: float):
    h, w = gray.shape
    if qmode == _QMODE_UNIFORM:
        return _dwt_uniform_channel(gray, wavelet, levels, step)
    recon, total_bits = wavelet_engine.compress_channel(gray, step, wavelet, levels)
    return np.clip(recon, 0, 255), total_bits


def dwt_recompute(image, image_id, wavelet, qmode, step_ctrl, target_bpp,
                  manual_step, request: gr.Request):
    empty = ("", "", None, "", None, "", None, "", None, None, "")
    if image is None or _is_stale(request.session_hash, image_id):
        return empty
    image = _reduce_if_true_grayscale(_prepare(image))
    gray = _to_gray(image)
    h, w = gray.shape
    lvl = min(_DWT_LEVEL, max_decomposition_level(gray.shape, wavelet))

    if step_ctrl == _DWT_MANUAL:
        step = float(np.clip(manual_step, _DWT_STEP_MIN, _DWT_STEP_MAX))
    else:
        from src.roi.bit_allocation import match_bpp
        tb = _clamp_bpp(target_bpp)
        _, _, step = match_bpp(
            lambda s: (lambda r, tb_: (r, tb_ / (h * w)))(*_dwt_encode(gray, wavelet, lvl, qmode, s)),
            tb, 0.5, 512.0, False)

    n_sub = 3 * lvl + 1
    q_rows = [("Step (Δ)", f"{step:.2f}", "", "violet"), ("Subbands", str(n_sub))]
    if step_ctrl == _DWT_MANUAL:
        q_rows.append(("Mode", "Manual override", "", "accent"))
    quant_grid = cards.metric_grid(q_rows)

    bands = subbands.single_level_bands(gray, wavelet)
    out = []
    for key in ("LL", "HL", "LH", "HH"):
        img_u8, pct = bands[key]
        out.append(cards.card_title(f"{key} — {pct:.1f}% energy"))
        out.append(img_u8)

    coeffs = decompose_for_viz(gray, wavelet, lvl, mode="periodization")
    mosaic = subbands.pyramid_labeled(coeffs)
    labels = " · ".join(
        f"{b}{k}" for k in range(lvl, 0, -1) for b in (("LL",) if k == lvl else ()) + ("LH", "HL", "HH")
    )
    cap = cards.caption(f"{labels} — labeled directly on the image above.")

    return (quant_grid, out[0], out[1], out[2], out[3], out[4], out[5],
            out[6], out[7], mosaic, cap)


def pr_qmf_recompute(image, image_id, prototype, request: gr.Request):
    empty = ("", "", None)
    if image is None or _is_stale(request.session_hash, image_id):
        return empty
    from src.engines import pr_qmf
    image = _reduce_if_true_grayscale(_prepare(image))
    gray = _to_gray(image)

    bands, shape = pr_qmf.dwt2_mband4(gray, prototype)
    rec = pr_qmf.idwt2_mband4(bands, shape, prototype)
    recon_mse = float(np.mean((gray - rec) ** 2))

    check = cards.metric_grid([
        ("Analysis→Synthesis MSE", f"{recon_mse:.2e}", "~1e-8..1e-3 = floating-point noise, i.e. lossless", "accent"),
        ("Subbands", "16"),
    ])
    epct = pr_qmf.band_energy_pct(bands)
    top4 = sorted(epct.items(), key=lambda kv: -kv[1])[:4]
    energy = cards.metric_grid([(k, f"{v:.1f}%") for k, v in top4])
    mosaic = subbands.mband_mosaic_labeled(bands)
    return check, energy, mosaic


# =============================================================================
# UI
# =============================================================================
_HEADER_HTML = """
<div class="lab-header">
  <p class="lab-title">Signal Compression Lab</p>
  <p class="lab-subtitle">DCT · Wavelet · Rate–Distortion Analysis</p>
</div>"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Signal Compression Lab", theme=LAB_THEME, css=LAB_CSS,
                   js=FORCE_LIGHT_JS) as demo:
        with gr.Row():
            gr.HTML(_HEADER_HTML)
            badge = gr.HTML('<span class="badge">Waiting for image</span>')

        active_img, active_id, active_meta, home_btn = build_active_image_bar()

        with gr.Tabs():
            # ---------------- Compare ----------------
            with gr.Tab("Compare"):
                gr.HTML(cards.page_header(
                    "JPEG (DCT) vs JPEG2000 (DWT)",
                    "Same target bit-rate, both methods -- the JPEG side tracks "
                    "BPP just like the wavelet side."))

                with gr.Row(elem_classes=["sc-grid"]):
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Target Bit-Rate"))
                        with gr.Row(elem_classes=["sc-bpp"]):
                            cmp_bpp = gr.Slider(_BPP_MIN, _BPP_MAX, value=0.5, step=0.01,
                                                label="Target BPP")
                            cmp_bpp_num = gr.Number(value=0.5, precision=2, label="Manual bpp",
                                                    elem_classes=["sc-num"], minimum=_BPP_MIN,
                                                    maximum=_BPP_MAX)
                        gr.HTML(cards.caption("Drag the slider or type a value in the box."))
                        cmp_type = gr.HTML()
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Wavelet (DWT side)"))
                        cmp_wav = gr.Dropdown(config.WAVELET_UI_OPTIONS,
                                              value=config.WAVELET_DEFAULT_FILTER,
                                              label="Filter family")

                with gr.Row(elem_classes=["sc-grid"]):
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("JPEG / DCT"))
                        cmp_dct_grid = gr.HTML()
                        cmp_dct_img = gr.Image(show_label=False, elem_classes=["image-frame"])
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Wavelet / DWT"))
                        cmp_wav_grid = gr.HTML()
                        cmp_wav_img = gr.Image(show_label=False, elem_classes=["image-frame"])

                cmp_ref = gr.HTML()

                with gr.Column(elem_classes=["card"]):
                    gr.HTML(cards.card_title("JPEG2000 layer: multiresolution subbands"))
                    cmp_dwt_layer = gr.Image(show_label=False, elem_classes=["image-frame"])
                    gr.HTML(cards.caption(
                        "Contrast-stretched wavelet coefficient pyramid — "
                        "tiled, non-overlapping."))

                with gr.Column(elem_classes=["card"]):
                    gr.HTML(cards.card_title("Rate–Distortion Analysis"))
                    cmp_rd = gr.Plot(show_label=False)

                cmp_rd_state = gr.State({"dct": [], "dwt": []})
                _point_out = [cmp_dct_grid, cmp_wav_grid, cmp_dct_img, cmp_wav_img,
                              cmp_type, cmp_dwt_layer, cmp_ref]
                _point_in = [active_img, active_id, cmp_bpp, cmp_wav, active_meta]
                _rd_out = [cmp_rd_state, cmp_rd]

                # Number field tracks the slider while dragging. It's a
                # programmatic set, so the .input handler below won't refire.
                cmp_bpp.change(lambda v: round(float(v), 2), cmp_bpp, cmp_bpp_num).then(
                    render_rd, [cmp_rd_state, cmp_bpp], cmp_rd)  # RD star follows the slider live
                # Slider released -> fast working point, then re-draw RD from cache.
                cmp_bpp.release(compute_working_point, _point_in, _point_out).then(
                    render_rd, [cmp_rd_state, cmp_bpp], cmp_rd)
                # Manual bpp entry (typing — no Enter needed — or Enter) moves the
                # slider and recomputes the working point + RD marker.
                _wire_manual_bpp(cmp_bpp_num, cmp_bpp,
                                 (compute_working_point, _point_in, _point_out),
                                 (render_rd, [cmp_rd_state, cmp_bpp], cmp_rd))
                # Filter changed -> working point + full RD sweep.
                cmp_wav.change(compute_working_point, _point_in, _point_out).then(
                    compute_rd, _point_in, _rd_out)
                # New image -> working point FIRST (fast, results appear), then RD sweep.
                active_id.change(compute_working_point, _point_in, _point_out).then(
                    compute_rd, _point_in, _rd_out)

            # ---------------- DCT Lab ----------------
            with gr.Tab("DCT Lab"):
                gr.HTML(cards.page_header(
                    "DCT Transform Explorer",
                    "01 pixel block → 02 level shift → 03 DCT → 04 quantize → "
                    "05 zigzag → 06 reconstruction"))

                dct_rc = gr.State((None, None))
                with gr.Row(elem_classes=["sc-grid"]):
                    with gr.Column(scale=3, elem_classes=["card"]):
                        gr.HTML(cards.card_title(
                            "Click anywhere on the DEGRADED (compressed) image to inspect that 8×8 block"))
                        dct_preview = gr.Image(show_label=False,
                                               elem_classes=["image-frame", "clickable"])
                        gr.HTML(cards.caption(
                            "This preview is the image AFTER JPEG compression at the current "
                            "quality — the red box marks the block shown in the 01–06 panels "
                            "below. Click anywhere to move it."))
                    with gr.Column(scale=2):
                        with gr.Column(elem_classes=["card"]):
                            gr.HTML(cards.card_title("Block Settings"))
                            with gr.Row(elem_classes=["sc-bpp"]):
                                dct_bpp = gr.Slider(_BPP_MIN, _BPP_MAX, value=0.5, step=0.01,
                                                    label="Target BPP")
                                dct_bpp_num = gr.Number(value=0.5, precision=2, label="Manual bpp",
                                                        elem_classes=["sc-num"],
                                                        minimum=_BPP_MIN, maximum=_BPP_MAX)
                        with gr.Column(elem_classes=["card"]):
                            gr.HTML(cards.card_title("Pipeline"))
                            dct_pipeline = gr.HTML()

                with gr.Row(elem_classes=["sc-grid", "cols-6"]):
                    _dct_panels = []
                    _dct_specs = [
                        ("01 · Original 8×8 pixel block", ""),
                        ("02 · Level-shifted (-128)", ""),
                        ("03 · DCT coefficient matrix", "DC = top-left (red border)"),
                        ("04 · Quantized DCT matrix", "gray = zeroed by quantization"),
                        ("05 · Zigzag scan order", ""),
                        ("06 · Reconstructed (after IDCT)", None),  # dynamic caption
                    ]
                    for title, cap in _dct_specs:
                        with gr.Column(elem_classes=["card"]):
                            gr.HTML(cards.card_title(title))
                            _dct_panels.append(gr.Image(show_label=False,
                                                        elem_classes=["image-frame"]))
                            if cap:
                                gr.HTML(cards.caption(cap))
                            elif cap is None:
                                dct_p6_cap = gr.HTML()

                with gr.Column(elem_classes=["card"]):
                    gr.HTML(cards.card_title("Zigzag-scanned coefficients (1D)"))
                    dct_zigzag = gr.Image(show_label=False, elem_classes=["diagram-frame", "wide"])
                    dct_zz_cap = gr.HTML()

                gr.HTML(cards.page_header(
                    "Which frequency bands does DCT actually keep?",
                    "Whole image, reconstructed four times — each time keeping only one "
                    "frequency zone's coefficients across every 8×8 block. Unquantized "
                    "(independent of target BPP), the DCT-side counterpart of the DWT "
                    "Lab's LL/HL/LH/HH single-level bands."))
                with gr.Row(elem_classes=["sc-grid", "cols-4"]):
                    _dct_zone_out = []
                    for _ in range(4):
                        with gr.Column(elem_classes=["card"]):
                            _zt = gr.HTML()
                            _zi = gr.Image(show_label=False, elem_classes=["image-frame"])
                            _dct_zone_out += [_zt, _zi]

                _dct_out = [dct_preview, dct_pipeline, dct_rc,
                            _dct_panels[0], _dct_panels[1], _dct_panels[2],
                            _dct_panels[3], _dct_panels[4], _dct_panels[5],
                            dct_p6_cap, dct_zigzag, dct_zz_cap]
                _dct_bpp_in = [active_img, active_id, dct_bpp, dct_rc]

                dct_bpp.change(lambda v: round(float(v), 2), dct_bpp, dct_bpp_num)
                dct_bpp.release(dct_from_bpp, _dct_bpp_in, _dct_out)
                _wire_manual_bpp(dct_bpp_num, dct_bpp,
                                 (dct_from_bpp, _dct_bpp_in, _dct_out))
                dct_preview.select(dct_from_click, [active_img, active_id, dct_bpp], _dct_out)
                active_id.change(dct_from_bpp, _dct_bpp_in, _dct_out)
                active_id.change(dct_zones, [active_img, active_id], _dct_zone_out)

            # ---------------- DWT Lab ----------------
            with gr.Tab("DWT Lab"):
                gr.HTML(cards.page_header(
                    "DWT Transform Explorer",
                    "Multi-level wavelet decomposition — subbands are tiled "
                    "from real coefficient slices, never overlapped."))

                with gr.Row(elem_classes=["sc-grid", "cols-3"]):
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Wavelet Family"))
                        dwt_wav = gr.Dropdown(config.WAVELET_UI_OPTIONS,
                                              value=config.WAVELET_DEFAULT_FILTER,
                                              label="Filter family")
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Quantization Mode"))
                        dwt_qmode = gr.Dropdown([_QMODE_SCALAR, _QMODE_UNIFORM],
                                                value=_QMODE_SCALAR, label="Mode")
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Target Bit-Rate"))
                        dwt_step_ctrl = gr.Dropdown([_DWT_AUTO, _DWT_MANUAL],
                                                    value=_DWT_AUTO, label="Step control")
                        with gr.Row(elem_classes=["sc-bpp"], visible=True) as dwt_bpp_row:
                            dwt_bpp = gr.Slider(_BPP_MIN, _BPP_MAX, value=0.5, step=0.01,
                                                label="Target BPP")
                            dwt_bpp_num = gr.Number(value=0.5, precision=2, label="Manual bpp",
                                                    elem_classes=["sc-num"],
                                                    minimum=_BPP_MIN, maximum=_BPP_MAX)
                        with gr.Row(visible=False) as dwt_step_row:
                            dwt_step = gr.Slider(_DWT_STEP_MIN, _DWT_STEP_MAX, value=20.0,
                                                 step=0.5, label="Step (Δ)")

                with gr.Column(elem_classes=["card"]):
                    gr.HTML(cards.card_title("Quantization"))
                    dwt_quant = gr.HTML()

                with gr.Row(elem_classes=["sc-grid", "cols-4"]):
                    _dwt_band_out = []
                    for _ in range(4):
                        with gr.Column(elem_classes=["card"]):
                            _bt = gr.HTML()
                            _bi = gr.Image(show_label=False, elem_classes=["image-frame"])
                            _dwt_band_out += [_bt, _bi]

                with gr.Column(elem_classes=["card"]):
                    gr.HTML(cards.card_title("Multiresolution decomposition mosaic"))
                    dwt_mosaic = gr.Image(show_label=False, elem_classes=["diagram-frame"])
                    dwt_cap = gr.HTML()

                gr.HTML(cards.page_header(
                    "4-Band PR-QMF (M = 4 filter bank)",
                    "A genuinely different filter bank from everything above — not the "
                    "octave/recursive wavelet split, but a uniform, critically-sampled "
                    "4-band split per axis (4×4 = 16 subbands, one level, matching the "
                    "course's M-Band Subband Coding diagram)."))
                with gr.Row(elem_classes=["sc-grid", "cols-3"]):
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("PR-QMF Prototype Filter"))
                        prqmf_proto = gr.Dropdown(
                            [(lbl, pid) for pid, lbl in pr_qmf.PRQMF_UI_OPTIONS],
                            value="prqmf4", label="Prototype")
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Perfect-reconstruction check"))
                        prqmf_check = gr.HTML()
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Energy, lowest 4 bands"))
                        prqmf_energy = gr.HTML()

                with gr.Column(elem_classes=["card"]):
                    gr.HTML(cards.card_title(
                        "4×4 = 16 uniform subbands (B{row}{col}, 0 = lowest frequency quarter per axis)"))
                    prqmf_mosaic = gr.Image(show_label=False, elem_classes=["diagram-frame"])
                    gr.HTML(cards.caption(
                        "Unlike the octave wavelet mosaic above, every tile here is the "
                        "same size — a uniform (not multiresolution) filter bank."))

                _dwt_out = [dwt_quant] + _dwt_band_out + [dwt_mosaic, dwt_cap]
                _dwt_in = [active_img, active_id, dwt_wav, dwt_qmode, dwt_step_ctrl,
                           dwt_bpp, dwt_step]
                _prq_out = [prqmf_check, prqmf_energy, prqmf_mosaic]
                _prq_in = [active_img, active_id, prqmf_proto]

                dwt_step_ctrl.change(
                    lambda m: (gr.update(visible=m == _DWT_AUTO),
                               gr.update(visible=m != _DWT_AUTO)),
                    dwt_step_ctrl, [dwt_bpp_row, dwt_step_row])
                dwt_bpp.change(lambda v: round(float(v), 2), dwt_bpp, dwt_bpp_num)
                for ev in (dwt_wav.change, dwt_qmode.change, dwt_step_ctrl.change,
                           dwt_bpp.release, dwt_step.release):
                    ev(dwt_recompute, _dwt_in, _dwt_out)
                _wire_manual_bpp(dwt_bpp_num, dwt_bpp,
                                 (dwt_recompute, _dwt_in, _dwt_out))
                prqmf_proto.change(pr_qmf_recompute, _prq_in, _prq_out)
                active_id.change(dwt_recompute, _dwt_in, _dwt_out)
                active_id.change(pr_qmf_recompute, _prq_in, _prq_out)

            # ---------------- Lossless ----------------
            with gr.Tab("Lossless"):
                gr.HTML(cards.page_header(
                    "Lossless Coding",
                    "Pixel-exact reconstruction (zlib DEFLATE on the raw "
                    "samples, no DCT/DWT quantization) — for fingerprint and "
                    "biomedical images where discarding any detail is "
                    "unacceptable. Unlike the other tabs there is no "
                    "target-bitrate control: lossless coding can't trade "
                    "quality for rate, so the bit-rate is simply whatever the "
                    "entropy coder achieves."))
                with gr.Row(elem_classes=["sc-grid", "cols-3"]):
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Bit-rate (achieved)"))
                        ll_bpp = gr.HTML()
                        gr.HTML(cards.caption(
                            "Compressed size in bits ÷ pixel count. The raw "
                            "image is 8.000 bpp; everything below that is "
                            "redundancy DEFLATE removed with zero loss."))
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Losslessness check"))
                        ll_check = gr.HTML()
                        ll_pill = gr.HTML()
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Size"))
                        ll_size = gr.HTML()
                with gr.Row(elem_classes=["sc-grid"]):
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Original (uploaded)"))
                        ll_orig = gr.Image(show_label=False, elem_classes=["image-frame"])
                    with gr.Column(elem_classes=["card"]):
                        gr.HTML(cards.card_title("Reconstructed (decoded from the lossless stream)"))
                        ll_recon = gr.Image(show_label=False, elem_classes=["image-frame"])
                        gr.HTML(cards.caption(
                            "If lossless coding is working, this is bit-for-bit "
                            "identical to the original — see MSE above."))

                _ll_out = [ll_bpp, ll_check, ll_pill, ll_size, ll_orig, ll_recon]
                _ll_in = [active_img, active_id]
                active_id.change(run_lossless, _ll_in, _ll_out)

            # ---------------- Theory ----------------
            with gr.Tab("Theory"):
                gr.HTML(cards.page_header(
                    "Theory", "Quick reference for the concepts this lab visualizes."))
                gr.HTML(_theory_html())

        # Badge: "Analysis ready" once an image is loaded.
        active_id.change(
            lambda: gr.update(value='<span class="badge ready">Analysis ready</span>'),
            None, badge)
        home_btn.click(
            lambda: gr.update(value='<span class="badge">Waiting for image</span>'),
            None, badge)

    return demo


if __name__ == "__main__":
    _favicon = Path(__file__).resolve().parent / "assets" / "favicon.ico"
    build_ui().launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=True,
        favicon_path=str(_favicon) if _favicon.exists() else None,
    )
