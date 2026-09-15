# -*- coding: utf-8 -*-
"""HTML metric-card generation: the custom cards the lab interface uses
instead of a Gradio Dataframe (the default Dataframe was both visually weak
and had a layout bug that clipped labels).

Numbers are never rounded internally — only here, at display time. Consistent
precision: bpp 3 decimals, PSNR 2, SSIM 4, MSE 3 (adaptive), ratio 2,
size 2, very small errors in scientific notation.
"""
from __future__ import annotations

import html as _html

import numpy as np

from src.viz.style import (ACCENT_PURPLE, CRITICAL, POSITIVE, TEXT_MUTED,
                           WARNING)

# ---------------------------------------------------------------------------
# Formatting — rounding happens ONLY here
# ---------------------------------------------------------------------------
def fmt_psnr(v: float) -> str:
    """NaN and +Inf mean different things and are never shown the same way:
    +Inf = MSE 0, a truly lossless reconstruction ("inf (lossless)"); NaN =
    the metric is undefined in this region (e.g. the ROI covers the whole
    image so the background is empty) -> "N/A", never "lossless"."""
    if np.isnan(v):
        return "N/A"
    if np.isinf(v):
        return "∞ (lossless)"
    return f"{v:.2f} dB"


def fmt_ssim(v: float) -> str:
    return "N/A" if np.isnan(v) else f"{v:.4f}"


def fmt_mse(v: float) -> str:
    return f"{v:.3e}" if (0 < abs(v) < 0.001) else f"{v:.3f}"


def fmt_bpp(v: float) -> str:
    return f"{v:.3f}"


def fmt_ratio(v: float) -> str:
    return "∞" if not np.isfinite(v) else f"{v:.2f} : 1"


def fmt_size_kb(bytes_: float) -> str:
    kb = bytes_ / 1024.0
    return f"{kb:.2f} KB" if kb < 1024 else f"{kb / 1024.0:.2f} MB"


def fmt_reduction(v: float) -> str:
    """Size-reduction percentage — comes from the single-source
    size_reduction_pct field of quality.calculate_metrics() (never
    recomputed with a second formula)."""
    return "N/A" if np.isnan(v) else f"{v:.1f}%"


def fmt_sci(v: float) -> str:
    return f"{v:.2e}"


def _esc(s: str) -> str:
    return _html.escape(str(s))


# ---------------------------------------------------------------------------
# Old React app design-system primitives (Card / Metric / page-header / pill)
# ---------------------------------------------------------------------------
def card(title: str, body_html: str) -> str:
    """A `.card` with a `● UPPERCASE` title and arbitrary HTML body."""
    return (f'<div class="card"><p class="card-title"><span class="dot"></span>'
            f'{_esc(title)}</p>{body_html}</div>')


def metric(label: str, value: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<span class="sub">{_esc(sub)}</span>' if sub else ""
    tone_cls = tone if tone in ("accent", "violet") else ""
    return (f'<div class="metric"><span class="k">{_esc(label)}</span>'
            f'<span class="v {tone_cls}">{_esc(value)}</span>{sub_html}</div>')


def metric_grid(items: list[tuple]) -> str:
    """items: list of (label, value) or (label, value, sub) or
    (label, value, sub, tone)."""
    cells = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 else ""
        tone = it[3] if len(it) > 3 else ""
        cells.append(metric(label, value, sub, tone))
    return f'<div class="metric-grid">{"".join(cells)}</div>'


def page_header(h2: str, p: str = "") -> str:
    p_html = f"<p>{_esc(p)}</p>" if p else ""
    return f'<div class="page-header"><h2>{_esc(h2)}</h2>{p_html}</div>'


def card_title(title: str) -> str:
    """Just the `● UPPERCASE` title bar — for a `gr.Column(elem_classes=["card"])`
    that also holds live Gradio components."""
    return f'<p class="card-title"><span class="dot"></span>{_esc(title)}</p>'


def pill(text_html: str, good: bool = False) -> str:
    return f'<span class="pill{" good" if good else ""}">{text_html}</span>'


def image_type_pill(image_type: str) -> str:
    return pill(f'Detected image type: <strong>{_esc(image_type)}</strong>')


def caption(text: str) -> str:
    return f'<p class="caption">{_esc(text)}</p>'


def reference_check(title: str, rows: list[tuple]) -> str:
    """A `.card` holding the Compare tab's "is the PSNR in the reference range?"
    verdict. rows = [(label, expected_text, verdict_text, ok_bool), ...]."""
    cells = []
    for label, expected, verdict, ok in rows:
        mark = "✓" if ok else "⚠"
        col = POSITIVE if ok else WARNING
        cells.append(
            f'<div class="metric"><span class="k">{_esc(label)}</span>'
            f'<span class="v" style="color:{col}">{mark} {_esc(expected)}</span>'
            f'<span class="sub">{_esc(verdict)}</span></div>')
    return (f'<div class="card"><p class="card-title"><span class="dot"></span>'
            f'{_esc(title)}</p><div class="metric-grid">{"".join(cells)}</div></div>')


# ---------------------------------------------------------------------------
# Delta — colour it when direction is meaningful, otherwise neutral
# ---------------------------------------------------------------------------
def _delta_span(delta: float, unit: str, higher_is_better: bool | None,
                decimals: int = 2) -> str:
    if higher_is_better is None:
        color = TEXT_MUTED
    elif (delta > 0) == higher_is_better:
        color = POSITIVE
    elif delta == 0:
        color = TEXT_MUTED
    else:
        color = WARNING
    sign = "+" if delta >= 0 else ""
    return (f'<span style="color:{color};font-size:13px;font-family:var(--lab-mono)">'
           f'{sign}{delta:.{decimals}f}{unit}</span>')


# ---------------------------------------------------------------------------
# Metric card: shared by JPEG/DCT, Wavelet/DWT, JPEG2000
# ---------------------------------------------------------------------------
def _metric_row(label: str, value: str, delta: str | None = None) -> str:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return (f'<div class="metric-row"><span class="metric-label">{label}</span>'
           f'<span class="metric-value-wrap"><span class="metric-value">{value}</span>'
           f'{delta_html}</span></div>')


def compact_metric_card(
    title: str, accent: str, metrics: dict, *, compare_to: dict | None = None,
    param_label: str = "",
) -> str:
    """Primary (Level-1) view: PSNR/SSIM/BPP only. MSE / ratio / size /
    estimate-label belong in metric_card() i.e. the "Details" panel — built
    from the same `metrics` dict, no double computation."""
    d_psnr = d_ssim = d_bpp = None
    if compare_to is not None and np.isfinite(metrics["psnr"]) and np.isfinite(compare_to["psnr"]):
        d_psnr = _delta_span(metrics["psnr"] - compare_to["psnr"], " dB", True)
        d_ssim = _delta_span(metrics["ssim"] - compare_to["ssim"], "", True, 4)
        d_bpp = _delta_span(metrics["bpp"] - compare_to["bpp"], "", None, 3)
    param_html = f'<div class="metric-param">{_esc(param_label)}</div>' if param_label else ""
    rows = "".join([
        _metric_row("PSNR", fmt_psnr(metrics["psnr"]), d_psnr),
        _metric_row("SSIM", fmt_ssim(metrics["ssim"]), d_ssim),
        _metric_row("BPP", fmt_bpp(metrics["bpp"]), d_bpp),
    ])
    return f"""
<div class="lab-card metric-card" style="border-top:3px solid {accent}">
  <div class="metric-card-head">
    <span class="method-dot" style="background:{accent}"></span>
    <span class="metric-title">{_esc(title)}</span>
  </div>
  {param_html}
  {rows}
</div>"""


def target_rate_header_html(target_bpp: float) -> str:
    """Shows the target rate as one large value on the primary screen;
    the bisection detail / matched points move to Details via
    target_summary_html()."""
    return f"""
<div class="target-rate-header">
  <span class="target-rate-label">TARGET RATE</span>
  <span class="target-rate-value">{fmt_bpp(target_bpp)}<span class="unit"> bpp</span></span>
</div>"""


def category_badge_html(label: str) -> str:
    return f'<span class="category-badge">Category: {_esc(label)}</span>'


def metric_card(
    title: str, accent: str, metrics: dict, *, compare_to: dict | None = None,
    size_badge: str | None = None, param_label: str = "",
) -> str:
    """metrics: calculate_metrics() output. When compare_to is given, delta
    rows are added (e.g. the Wavelet card vs JPEG). This is the "Details"
    panel — it carries the secondary metrics (MSE / compression ratio /
    size)."""
    row = _metric_row
    d_psnr = d_ssim = d_bpp = None
    if compare_to is not None and np.isfinite(metrics["psnr"]) and np.isfinite(compare_to["psnr"]):
        d_psnr = _delta_span(metrics["psnr"] - compare_to["psnr"], " dB", True)
        d_ssim = _delta_span(metrics["ssim"] - compare_to["ssim"], "", True, 4)
        d_bpp = _delta_span(metrics["bpp"] - compare_to["bpp"], "", None, 3)

    badge_html = f'<span class="size-badge">{_esc(size_badge)}</span>' if size_badge else ""
    param_html = f'<div class="metric-param">{_esc(param_label)}</div>' if param_label else ""

    rows = "".join([
        row("PSNR", fmt_psnr(metrics["psnr"]), d_psnr),
        row("SSIM", fmt_ssim(metrics["ssim"]), d_ssim),
        row("MSE", fmt_mse(metrics["mse"])),
        row("BPP", fmt_bpp(metrics["bpp"]), d_bpp),
        row("COMPRESSION RATIO", fmt_ratio(metrics["compression_ratio"])),
        row("REDUCTION", fmt_reduction(metrics["size_reduction_pct"])),
        row("SIZE", fmt_size_kb(metrics["compressed_size_bytes"])),
    ])
    return f"""
<div class="lab-card metric-card" style="border-top:3px solid {accent}">
  <div class="metric-card-head">
    <span class="method-dot" style="background:{accent}"></span>
    <span class="metric-title">{_esc(title)}</span>
    {badge_html}
  </div>
  {param_html}
  {rows}
</div>"""


def target_summary_html(target_bpp: float, points: list[tuple[str, str, float]]) -> str:
    """points: [(label, color, actual_bpp), ...] — the matched working points."""
    rows = "".join(
        f'<div class="target-row"><span class="method-dot" style="background:{color}"></span>'
        f'<span class="target-label">{_esc(label)}</span>'
        f'<span class="target-value">{fmt_bpp(bpp)} bpp</span></div>'
        for label, color, bpp in points
    )
    diff = max(bpp for _, _, bpp in points) - min(bpp for _, _, bpp in points) if points else 0.0
    return f"""
<div class="lab-card target-card">
  <div class="section-label">TARGET RATE</div>
  <div class="target-value-hero">{fmt_bpp(target_bpp)} <span class="unit">bpp</span></div>
  <div class="target-caption">matched APPROXIMATELY by bisection — the achieved value need not equal the target</div>
  {rows}
  <div class="target-row target-diff"><span class="target-label">Difference</span>
  <span class="target-value">{fmt_bpp(diff)} bpp</span></div>
</div>"""


def status_badge(text: str, kind: str = "neutral") -> str:
    colors = {"positive": POSITIVE, "warning": WARNING, "critical": CRITICAL, "neutral": TEXT_MUTED}
    color = colors.get(kind, TEXT_MUTED)
    return (f'<span class="status-badge" style="color:{color};border-color:{color}">'
           f'{_esc(text)}</span>')


def validation_card_html(max_err: float, mse_val: float, tol: float = 1e-3) -> str:
    passed = max_err < tol
    badge = status_badge("PASS" if passed else "FAIL", "positive" if passed else "critical")
    return f"""
<div class="lab-card validation-card">
  <div class="section-label">TRANSFORM VALIDATION — DWT &rarr; IDWT (no quantization)</div>
  <div class="validation-head">{badge}</div>
  <div class="metric-row"><span class="metric-label">Max Absolute Error</span>
  <span class="metric-value mono">{fmt_sci(max_err)}</span></div>
  <div class="metric-row"><span class="metric-label">MSE</span>
  <span class="metric-value mono">{fmt_sci(mse_val)}</span></div>
  <div class="metric-param">Tolerance: {fmt_sci(tol)} — an error at numerical-precision level confirms the transform is invertible.</div>
</div>"""


def sparsity_card_html(stats: dict) -> str:
    return f"""
<div class="lab-card sparsity-card">
  <div class="section-label">COEFFICIENT SPARSITY</div>
  <div class="sparsity-compare">
    <div class="sparsity-col">
      <div class="sparsity-col-title">BEFORE QUANTIZATION (raw)</div>
      <div class="sparsity-big">{stats['raw_nonzero_pct']:.1f}%</div>
      <div class="metric-label">non-zero</div>
    </div>
    <div class="sparsity-arrow">&rarr;</div>
    <div class="sparsity-col">
      <div class="sparsity-col-title">AFTER QUANTIZATION</div>
      <div class="sparsity-big" style="color:{ACCENT_PURPLE}">{stats['nonzero_pct']:.1f}%</div>
      <div class="metric-label">non-zero</div>
    </div>
  </div>
  <div class="metric-row"><span class="metric-label">Sparsity (zero ratio)</span>
  <span class="metric-value mono">{stats['sparsity_pct']:.1f}%</span></div>
  <div class="metric-row"><span class="metric-label">Total coefficients</span>
  <span class="metric-value mono">{stats['total']:,}</span></div>
  <div class="metric-param">Raw float coefficients are almost never exactly zero; the
  sparsity that explains the compression appears AFTER QUANTIZATION.</div>
</div>"""


def empty_state_html(text: str, caption: str = "") -> str:
    caption_html = f'<div class="empty-caption">{_esc(caption)}</div>' if caption else ""
    return f"""
<div class="lab-card empty-state">
  <div class="empty-title">{_esc(text)}</div>
  {caption_html}
</div>"""


def measurement_strip_html(items: list[tuple[str, str]]) -> str:
    """A single horizontal strip that reads like a scientific instrument
    (not a stack of cards). items: [(label, value), ...]."""
    cells = "".join(
        f'<div class="measurement-item"><span class="measurement-label">{_esc(l)}</span>'
        f'<span class="measurement-value">{_esc(v)}</span></div>'
        for l, v in items
    )
    return f'<div class="measurement-strip">{cells}</div>'


def selected_block_line_html(block_size: int, row: int, col: int,
                             region: tuple[int, int, int, int] | None = None) -> str:
    """'BLOCK 8x8 . Row 48 . Col 43'."""
    txt = f"{block_size}×{block_size} · Row {row} · Col {col}"
    if region:
        x0, y0, x1, y1 = region
        txt += f" · pixels x{x0}–{x1}, y{y0}–{y1}"
    return f'<div class="selected-block-line"><span class="sb-label">Selected block</span>{_esc(txt)}</div>'


def filter_bank_flow_html(wavelet_name: str) -> str:
    """An original HTML/CSS flow diagram of the DWT analysis filter bank's
    2D separable structure — row filtering -> downsample -> column
    filtering -> downsample -> LL/LH/HL/HH. CSS-only hover tooltips, no JS."""
    def _node(label, tip, extra_cls=""):
        return (f'<div class="flow-node {extra_cls}">{_esc(label)}'
               f'<span class="flow-tip">{_esc(tip)}</span></div>')

    def _arrow():
        return '<div class="flow-arrow">↓</div>'

    return f"""
<div class="filter-flow">
  {_node("SOURCE IMAGE", "The image / channel being analysed")}
  {_arrow()}
  <div class="flow-row">
    {_node("Rows: low-pass h", f"{wavelet_name} analysis filter — along the rows", "flow-lo")}
    {_node("Rows: high-pass g", f"{wavelet_name} analysis filter — along the rows", "flow-hi")}
  </div>
  {_arrow()}
  <div class="flow-row">
    {_node("↓2 (columns)", "Downsample columns by 2")}
    {_node("↓2 (columns)", "Downsample columns by 2")}
  </div>
  {_arrow()}
  <div class="flow-row flow-row-4">
    {_node("Cols: low → LL", "Approximation (low frequency)", "flow-ll")}
    {_node("Cols: high → LH", "Horizontal detail", "flow-lh")}
    {_node("Cols: low → HL", "Vertical detail", "flow-hl")}
    {_node("Cols: high → HH", "Diagonal detail", "flow-hh")}
  </div>
  {_arrow()}
  <div class="flow-row flow-row-4">
    <span class="flow-band flow-ll">LL</span>
    <span class="flow-band flow-lh">LH</span>
    <span class="flow-band flow-hl">HL</span>
    <span class="flow-band flow-hh">HH</span>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# DWT decomposition explorer — selected-band card + fixed legend strip
# ---------------------------------------------------------------------------
_DWT_BAND_TITLES = {
    "LL": "Approximation — low frequency / overall image structure",
    "LH": "Horizontal detail (horizontally-oriented edges) — pywt 'horizontal detail'",
    "HL": "Vertical detail (vertically-oriented edges) — pywt 'vertical detail'",
    "HH": "Diagonal / high-frequency detail",
}


def dwt_band_legend_html() -> str:
    """A fixed strip that states plainly (not hover-dependent) what
    LL/LH/HL/HH represent. The orientation convention (LH = horizontal,
    HL = vertical) matches pywt's own cH/cV semantics (verified in
    tests/test_correctness.py)."""
    chips = "".join(
        f'<span class="dwt-legend-chip"><b>{name}</b>: {title}</span>'
        for name, title in _DWT_BAND_TITLES.items()
    )
    return f'<div class="dwt-legend">{chips}</div>'


def dwt_band_details_html(
    level: int, band_name: str, stats: dict, pixel: tuple[int, int, float] | None = None,
) -> str:
    """The real statistics of the selected single band — if a pixel was
    clicked, its raw coefficient value is shown too. `stats` is the REAL
    output of subbands.selected_band_stats(); nothing is fabricated here."""
    pixel_html = ""
    if pixel is not None:
        row, col, val = pixel
        pixel_html = f"""
<div class="dwt-pixel-box">
  <div class="dwt-pixel-title">{_esc(band_name)}{level}</div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">X</span><span class="dwt-pixel-val">{col}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Y</span><span class="dwt-pixel-val">{row}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Coefficient</span><span class="dwt-pixel-val">{val:+.4f}</span></div>
</div>"""
    return f"""
<div class="lab-card">
  <div class="metric-card-head">
    <span class="metric-title">{_esc(band_name)}{level} — {_esc(_DWT_BAND_TITLES.get(band_name, ""))}</span>
  </div>
  <div class="metric-row"><span class="metric-label">Shape</span>
  <span class="metric-value mono">{_esc(stats["shape"])}</span></div>
  <div class="metric-row"><span class="metric-label">Coefficient count</span>
  <span class="metric-value mono">{stats["count"]:,}</span></div>
  <div class="metric-row"><span class="metric-label">Min / Max</span>
  <span class="metric-value mono">{stats["min"]:.2f} / {stats["max"]:.2f}</span></div>
  <div class="metric-row"><span class="metric-label">Mean</span>
  <span class="metric-value mono">{stats["mean"]:.3f}</span></div>
  <div class="metric-row"><span class="metric-label">Std deviation</span>
  <span class="metric-value mono">{stats["std"]:.3f}</span></div>
  <div class="metric-row"><span class="metric-label">Energy (&Sigma; coeff&sup2;)</span>
  <span class="metric-value mono">{stats["energy"]:.2f}</span></div>
  <div class="metric-row"><span class="metric-label">Near-zero ratio</span>
  <span class="metric-value mono">{stats["zero_pct"]:.1f}%</span></div>
  {pixel_html}
</div>"""


def mosaic_pixel_html(level: int, band: str, row: int, col: int, value: float) -> str:
    """Info box for a clicked point on the Pyramid Coefficient Map — `value`
    comes straight from the raw DWT coefficient array, not the normalized
    mosaic pixel value."""
    return f"""
<div class="dwt-pixel-box">
  <div class="dwt-pixel-title">{_esc(band)}{level}</div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Subband</span><span class="dwt-pixel-val">{_esc(band)}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Level</span><span class="dwt-pixel-val">{level}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">X</span><span class="dwt-pixel-val">{col}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Y</span><span class="dwt-pixel-val">{row}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Coefficient</span><span class="dwt-pixel-val">{value:+.4f}</span></div>
</div>"""


def recon_pixel_html(row: int, col: int, original: float, reconstructed: float) -> str:
    """SHARED info box for the Quantized Reconstruction / Difference Map —
    original/reconstructed are read straight from the real (gray, recon)
    arrays; difference = original - reconstructed (same sign convention as
    subbands.reconstruction_diff_image)."""
    diff = original - reconstructed
    sign = "+" if diff >= 0 else ""
    return f"""
<div class="dwt-pixel-box">
  <div class="dwt-pixel-title">Pixel ({col}, {row})</div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">X</span><span class="dwt-pixel-val">{col}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Y</span><span class="dwt-pixel-val">{row}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Original</span><span class="dwt-pixel-val">{original:.2f}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Reconstruction</span><span class="dwt-pixel-val">{reconstructed:.2f}</span></div>
  <div class="dwt-pixel-row"><span class="dwt-pixel-key">Difference (original &minus; reconstruction)</span><span class="dwt-pixel-val">{sign}{diff:.2f}</span></div>
</div>"""


def summary_bar_html(items: list[tuple[str, str]]) -> str:
    """Compact one-line summary: [(label, value), ...] — DCT/DWT Lab top bar."""
    chips = "".join(
        f'<div class="summary-chip"><span class="summary-chip-label">{_esc(l)}</span>'
        f'<span class="summary-chip-value mono">{_esc(v)}</span></div>'
        for l, v in items
    )
    return f'<div class="summary-bar">{chips}</div>'


# ---------------------------------------------------------------------------
# Structural components: empty state / image chip / rate fairness /
# quality strip — instead of card-inside-card.
# ---------------------------------------------------------------------------
def hero_empty_html(title: str, caption: str) -> str:
    """The Compare tab's first-open state: one centered block, no big empty
    result panels. The Pick / Upload buttons are added outside this HTML as
    real Gradio Button components (they need interaction)."""
    return f"""
<div class="hero-empty">
  <div class="hero-title">{_esc(title)}</div>
  <div class="hero-caption">{_esc(caption)}</div>
</div>"""


def active_image_bar_html(
    name: str, category_label: str | None, width: int, height: int,
    original_width: int | None = None, original_height: int | None = None,
    file_size_bytes: int | None = None,
) -> str:
    """Contents of the single, shared 'ACTIVE IMAGE' bar used across the whole
    app (Compare / DCT / DWT / Semantic all use this same image).

    original_width/height: `_prepare()` (app.py) shrinks large images for
    speed — this is never hidden silently. (width,height) is ALWAYS the real
    analysis resolution; when the original differs an extra note is shown.

    file_size_bytes: the real on-disk size of the selected / uploaded file,
    shown explicitly (in KB) at the user's request."""
    cat_html = f'<span class="chip-category">{_esc(category_label)}</span>' if category_label else ""
    resized_note = ""
    if original_width and original_height and (original_width, original_height) != (width, height):
        resized_note = (
            f'<span class="active-image-resized" '
            f'title="Downscaled for analysis speed — original file {original_width}×{original_height}">'
            f'(original {original_width}×{original_height})</span>'
        )
    file_size_html = ""
    if file_size_bytes:
        file_size_html = (
            f'<span class="active-image-filesize">'
            f'<span class="afs-label">File size</span>{fmt_size_kb(file_size_bytes)}</span>'
        )
    return f"""
<div class="active-image-bar-content">
  <span class="active-image-tag">ACTIVE IMAGE</span>
  <span class="chip-name">{_esc(name)}</span>
  {cat_html}
  <span class="active-image-dims">{width}&times;{height}</span>
  {resized_note}
  {file_size_html}
</div>"""


def image_chip_html(name: str, category_label: str | None) -> str:
    cat_html = f'<span class="chip-category">{_esc(category_label)}</span>' if category_label else ""
    return f"""
<div class="image-chip">
  <span class="chip-name">{_esc(name)}</span>
  {cat_html}
</div>"""


def rate_fairness_html(target_bpp: float, dct_bpp: float, wav_bpp: float) -> str:
    """One-line rate-fairness summary: TARGET / JPEG / JPEG2000 / delta —
    used ONLY on the main Compare screen (DWT Lab / Semantic ROI do not call
    it); the second side is a real JPEG2000/OpenJPEG codec so the labels say
    "JPEG2000", not "Wavelet". The arithmetic is unchanged."""
    diff = abs(wav_bpp - dct_bpp)
    def item(label, value, cls=""):
        return (f'<div class="rf-item"><span class="rf-label">{_esc(label)}</span>'
               f'<span class="rf-value {cls}">{fmt_bpp(value)} bpp</span></div>')
    return f"""
<div class="rate-fairness">
  {item("Target", target_bpp)}
  {item("JPEG", dct_bpp, "accent-jpeg")}
  {item("JPEG2000", wav_bpp, "accent-wavelet")}
  {item("Difference |JPEG2000−JPEG|", diff)}
</div>"""


def quality_strip_html(dct_metrics: dict, wav_metrics: dict, wavelet_name: str = "JPEG2000 / DWT") -> str:
    """One comparison table instead of a stack of cards: Metric | JPEG |
    JPEG2000 | delta. PSNR/SSIM/BPP only (primary screen) — MSE / ratio /
    size are in Details. delta = JPEG2000 - JPEG, stated explicitly in the
    header."""
    def delta(key, decimals, unit, higher_better):
        d = wav_metrics[key] - dct_metrics[key]
        if higher_better is None:
            color = TEXT_MUTED
        elif d == 0:
            color = TEXT_MUTED
        else:
            color = POSITIVE if (d > 0) == higher_better else WARNING
        sign = "+" if d >= 0 else ""
        return f'<td class="delta" style="color:{color}">{sign}{d:.{decimals}f}{unit}</td>'

    def row(label, dct_val, wav_val, delta_html):
        return f'<tr><td>{label}</td><td>{dct_val}</td><td>{wav_val}</td>{delta_html}</tr>'

    rows = "".join([
        row("PSNR", fmt_psnr(dct_metrics["psnr"]), fmt_psnr(wav_metrics["psnr"]),
            delta("psnr", 2, " dB", True)),
        row("SSIM", fmt_ssim(dct_metrics["ssim"]), fmt_ssim(wav_metrics["ssim"]),
            delta("ssim", 4, "", True)),
        row("BPP", fmt_bpp(dct_metrics["bpp"]), fmt_bpp(wav_metrics["bpp"]),
            delta("bpp", 3, "", None)),
    ])
    return f"""
<table class="quality-strip">
  <thead><tr><th></th><th class="th-jpeg">JPEG / DCT</th>
  <th class="th-wavelet">{_esc(wavelet_name)}</th><th title="JPEG2000 minus JPEG">&Delta; (JPEG2000 &minus; JPEG)</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


# =============================================================================
# Compression Summary (compact KPI card) and Before/After slider. No new
# numeric value/formula: both read quality.calculate_metrics()'s (single
# source) output or already-computed reconstruction arrays.
# =============================================================================
def compression_summary_card_html(
    title: str, accent: str, metrics: dict, size_badge: str,
) -> str:
    """Compact COMPRESSION SUMMARY card: PSNR/SSIM/MSE/BPP/RATIO/REDUCTION
    KPI row + ORIGINAL -> COMPRESSED size arrow. `metrics` is the
    single-source output of calculate_metrics() — nothing is recomputed
    here; it carries the exact same numbers as metric_card (Details).
    `size_badge`: the caller picks 'REAL SIZE' / 'ENTROPY ESTIMATE' from
    EngineResult.is_real_codec — whether it came from a real bitstream or an
    entropy estimate is always marked."""
    def kpi(label: str, value: str) -> str:
        return (f'<div class="kpi-box"><div class="kpi-label">{_esc(label)}</div>'
               f'<div class="kpi-value">{value}</div></div>')

    kpis = "".join([
        kpi("PSNR", fmt_psnr(metrics["psnr"])),
        kpi("SSIM", fmt_ssim(metrics["ssim"])),
        kpi("MSE", fmt_mse(metrics["mse"])),
        kpi("BPP", fmt_bpp(metrics["bpp"])),
        kpi("RATIO", fmt_ratio(metrics["compression_ratio"])),
        kpi("REDUCTION", fmt_reduction(metrics["size_reduction_pct"])),
    ])
    return f"""
<div class="lab-card kpi-summary-card" style="border-top:3px solid {accent}">
  <div class="metric-card-head">
    <span class="method-dot" style="background:{accent}"></span>
    <span class="metric-title">{_esc(title)}</span>
    <span class="size-badge">{_esc(size_badge)}</span>
  </div>
  <div class="kpi-row">{kpis}</div>
  <div class="size-arrow-row">
    <span class="size-arrow-value">{fmt_size_kb(metrics["original_size_bytes"])}</span>
    <span class="size-arrow-sep">&rarr;</span>
    <span class="size-arrow-value" style="color:{accent}">{fmt_size_kb(metrics["compressed_size_bytes"])}</span>
    <span class="size-reduction-pct">{fmt_reduction(metrics["size_reduction_pct"])} reduction</span>
  </div>
</div>"""


