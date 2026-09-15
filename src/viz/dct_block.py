# -*- coding: utf-8 -*-
"""DCT Explorer: tek bir bloğun orijinal -> seviye-kaydırma -> DCT ->
kuantalama -> rekonstrüksiyon adımlarını GERÇEK görüntü verisinden hesaplar.

Sabit/dekoratif matris yok: her çağrı, kullanıcının yüklediği görüntüden
seçilen gerçek bloğu işler (mega-spec: "Do not use hard-coded matrices").
"""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

from src.engines.dct_engine import JPEG_LUMA_QTABLE, get_qtable, quality_to_qtable
from src.engines.entropy import zigzag_indices


# ---------------------------------------------------------------------------
# Compare-tab "JPEG layer" mosaic — |quantized DCT| AC-detail per 8x8 block at
# a given JPEG quality (ported from the old project's dct_analysis.py). Unlike
# a per-block-normalized view, this visibly changes as bpp/quality drops
# because quantization actually zeroes high-frequency coefficients.
# ---------------------------------------------------------------------------
def dct_block_mosaic_quantized(gray_u8: np.ndarray, jpeg_quality: int,
                               block_size: int = 8) -> np.ndarray:
    """Returns a uint8 RGB mosaic with yellow 8x8 grid lines. Display only —
    the DC term is dropped before normalizing and log1p is applied so one
    outlier block does not crush the visible range."""
    import cv2

    base = quality_to_qtable(float(np.clip(jpeg_quality, 1, 100)),
                             np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64))
    g = gray_u8.astype(np.float32) - 128.0
    h0, w0 = g.shape
    ph = ((h0 + block_size - 1) // block_size) * block_size
    pw = ((w0 + block_size - 1) // block_size) * block_size
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h0, :w0] = g
    out = np.zeros_like(padded)
    for i in range(0, ph, block_size):
        for j in range(0, pw, block_size):
            blk = padded[i:i + block_size, j:j + block_size]
            d = dctn(blk, norm="ortho")
            q = np.round(d / base)
            q[0, 0] = 0.0  # DC excluded
            out[i:i + block_size, j:j + block_size] = np.abs(q)
    out = np.log1p(out[:h0, :w0])
    mx = float(out.max()) or 1.0
    vis = (np.clip(out / mx, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    grid = (255, 220, 0)
    for x in range(block_size, w0, block_size):
        cv2.line(rgb, (x, 0), (x, h0 - 1), grid, 1)
    for y in range(block_size, h0, block_size):
        cv2.line(rgb, (0, y), (w0 - 1, y), grid, 1)
    return rgb


# ---------------------------------------------------------------------------
# DCT Lab — 8x8 numeric-grid panels + zigzag views (ported from the old
# project's backend/services/dct_analysis.py). Every panel is drawn as a
# heat-tinted grid with the real numeric value in each cell.
# ---------------------------------------------------------------------------
_ZIGZAG_8 = None


def _zigzag_order_8() -> list[tuple[int, int]]:
    global _ZIGZAG_8
    if _ZIGZAG_8 is None:
        order = []
        for diag in range(15):
            coords = [(i, diag - i) for i in range(8) if 0 <= diag - i < 8]
            if diag % 2 == 1:
                coords.reverse()
            order.extend(coords)
        _ZIGZAG_8 = order
    return _ZIGZAG_8


def matrix_grid_rgb(values: np.ndarray, *, cell_px: int = 40,
                    highlight_rc: tuple[int, int] | None = None,
                    zero_gray: bool = False, value_fmt: str = "auto") -> np.ndarray:
    """8x8 matrix as a heat-tinted grid (blue = positive, amber = negative,
    grey = zeroed-by-quantization) with the numeric value in each cell."""
    n = 8
    vals = np.asarray(values, dtype=np.float64)
    rgb = np.full((n * cell_px, n * cell_px, 3), 255, np.uint8)
    max_abs = float(np.max(np.abs(vals))) or 1.0
    for r in range(n):
        for c in range(n):
            v = float(vals[r, c])
            y0, x0 = r * cell_px, c * cell_px
            if zero_gray and abs(v) < 0.5:
                color = (228, 228, 228)
            elif v >= 0:
                t = min(1.0, abs(v) / max_abs)
                color = (int(255 - 120 * t), int(255 - 60 * t), 255)      # blue-ish
            else:
                t = min(1.0, abs(v) / max_abs)
                color = (255, int(255 - 70 * t), int(255 - 130 * t))      # amber-ish
            cv2.rectangle(rgb, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), color, -1)
            border = (190, 70, 30) if highlight_rc == (r, c) else (214, 214, 214)
            bw = 3 if highlight_rc == (r, c) else 1
            cv2.rectangle(rgb, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), border, bw)
            label = str(int(round(v))) if value_fmt in ("int", "pixel") else f"{v:.0f}"
            cv2.putText(rgb, label, (x0 + 3, y0 + cell_px - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (35, 30, 22), 1, cv2.LINE_AA)
    return rgb


def zigzag_order_grid_rgb(cell_px: int = 34) -> np.ndarray:
    """The JPEG zigzag scan order drawn as 1..64 on an 8x8 grid."""
    rgb = np.full((8 * cell_px, 8 * cell_px, 3), 246, np.uint8)
    for step, (u, v) in enumerate(_zigzag_order_8(), start=1):
        y0, x0 = u * cell_px, v * cell_px
        cv2.rectangle(rgb, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), (210, 210, 210), 1)
        col = (190, 40, 30) if step == 1 else (40, 40, 40)
        cv2.putText(rgb, str(step), (x0 + 4, y0 + cell_px - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    return rgb


def zigzag_bar_rgb(quant_block: np.ndarray, width: int = 820, height: int = 240) -> np.ndarray:
    """The 64 quantized coefficients in zigzag order as a bar chart."""
    order = _zigzag_order_8()
    q = np.asarray(quant_block, dtype=np.float64)
    vals = np.array([q[u, v] for u, v in order], dtype=np.float64)
    rgb = np.full((height, width, 3), 250, np.uint8)
    n = vals.size
    bar_w = max(2, width // n)
    max_abs = float(np.max(np.abs(vals))) or 1.0
    base_y = height - 28
    for i, v in enumerate(vals):
        x0 = i * bar_w
        bar_h = int(abs(v) / max_abs * (height - 56))
        if i == 0:
            color = (185, 84, 26)       # DC — copper
        elif i < 10:
            color = (14, 124, 123)      # low freq — teal
        elif abs(v) < 0.5:
            color = (214, 208, 190)
        else:
            color = (90, 90, 110)
        cv2.rectangle(rgb, (x0, base_y - bar_h), (x0 + bar_w - 2, base_y), color, -1)
    cv2.putText(rgb, "DC + low freq", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (185, 84, 26), 2, cv2.LINE_AA)
    cv2.putText(rgb, "high freq / zeros", (width - 230, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (90, 90, 110), 2, cv2.LINE_AA)
    nz = int(np.count_nonzero(np.abs(vals) >= 0.5))
    cv2.putText(rgb, f"non-zero: {nz}/64", (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (60, 55, 40), 2, cv2.LINE_AA)
    return rgb


def image_with_block_box(gray_u8: np.ndarray, bi: int, bj: int,
                         block_size: int = 8) -> np.ndarray:
    """The (already JPEG-degraded) image with a red rectangle on the inspected
    8x8 block."""
    rgb = cv2.cvtColor(np.ascontiguousarray(gray_u8), cv2.COLOR_GRAY2RGB)
    cv2.rectangle(rgb, (bj, bi), (bj + block_size - 1, bi + block_size - 1),
                  (220, 40, 30), 2)
    return rgb


def default_block_pos(h: int, w: int, block_size: int = 8) -> tuple[int, int]:
    bi = min(max(0, ((h // block_size) // 2) * block_size), max(0, h - block_size))
    bj = min(max(0, ((w // block_size) // 2) * block_size), max(0, w - block_size))
    return bi, bj


def jpeg_block_panels(gray_u8: np.ndarray, quality: int,
                      row: int | None, col: int | None) -> dict:
    """Assemble the DCT Lab's 01-06 panels + the 1D zigzag view for one 8x8
    block at the given JPEG quality. Pure maths — no I/O."""
    from scipy.fft import dctn, idctn

    g = np.clip(np.asarray(gray_u8), 0, 255).astype(np.uint8)
    h0, w0 = g.shape
    if row is None or col is None:
        bi, bj = default_block_pos(h0, w0)
    else:
        bi = min(max(0, (int(row) // 8) * 8), max(0, h0 - 8))
        bj = min(max(0, (int(col) // 8) * 8), max(0, w0 - 8))

    block_u8 = g[bi:bi + 8, bj:bj + 8].astype(np.float64)
    centered = block_u8 - 128.0
    coeffs = dctn(centered, norm="ortho")
    q_mat = quality_to_qtable(float(np.clip(quality, 1, 100)),
                              np.asarray(JPEG_LUMA_QTABLE, dtype=np.float64))
    quant = np.round(coeffs / q_mat)
    recon = np.clip(idctn(quant * q_mat, norm="ortho") + 128.0, 0, 255)
    recon_u8 = np.round(recon).astype(np.float64)

    b_mse = float(np.mean((block_u8 - recon_u8) ** 2))
    b_psnr = float("inf") if b_mse == 0 else 10.0 * np.log10(255.0 ** 2 / b_mse)

    order = _zigzag_order_8()
    zz = [int(quant[u, v]) for u, v in order]

    return {
        "block_row": bi, "block_col": bj,
        "block_psnr": b_psnr, "block_mse": b_mse,
        "zigzag_first16": zz[:16],
        "panels": [
            ("01 · Original 8×8 pixel block", "",
             matrix_grid_rgb(block_u8, value_fmt="pixel")),
            ("02 · Level-shifted (-128)", "",
             matrix_grid_rgb(centered, value_fmt="int")),
            ("03 · DCT coefficient matrix", "DC = top-left (red border)",
             matrix_grid_rgb(coeffs, highlight_rc=(0, 0))),
            ("04 · Quantized DCT matrix", "gray = zeroed by quantization",
             matrix_grid_rgb(quant, zero_gray=True, value_fmt="int")),
            ("05 · Zigzag scan order", "", zigzag_order_grid_rgb()),
            ("06 · Reconstructed (after IDCT)",
             f"{b_psnr:.1f} dB vs. original block (01)",
             matrix_grid_rgb(recon_u8, value_fmt="pixel")),
        ],
        "zigzag_bar": zigzag_bar_rgb(quant),
    }


def extract_block(gray: np.ndarray, block_size: int, row: int, col: int) -> np.ndarray:
    """gray görüntüden (row,col) indeksli block_size x block_size bloğu keser
    (kenar-tekrarı ile doldurulmuş ızgara üzerinden — motorla aynı yöntem)."""
    h, w = gray.shape
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    padded = np.pad(gray, ((0, ph), (0, pw)), mode="edge")
    nh, nw = padded.shape[0] // block_size, padded.shape[1] // block_size
    row = max(0, min(row, nh - 1))
    col = max(0, min(col, nw - 1))
    r0, c0 = row * block_size, col * block_size
    return padded[r0:r0 + block_size, c0:c0 + block_size], (nh, nw), (row, col)


def block_pixel_region(gray_shape: tuple[int, int], block_size: int,
                       row: int, col: int) -> tuple[int, int, int, int]:
    """(row,col) blok-ızgara indeksinden GERÇEK piksel bölgesini hesaplar
    (x0,y0,x1,y1) — extract_block ile AYNI r0=row*block_size, c0=col*block_size
    eşlemesi (mega-spec Part 11: "must correspond to exact block used by
    backend"). Orijinal (doldurulmamış) görüntü sınırlarına kırpılır."""
    h, w = gray_shape
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    nh, nw = (h + ph) // block_size, (w + pw) // block_size
    row = max(0, min(row, nh - 1))
    col = max(0, min(col, nw - 1))
    x0, y0 = col * block_size, row * block_size
    x1, y1 = min(x0 + block_size, w), min(y0 + block_size, h)
    return x0, y0, x1, y1


def draw_block_overlay(image: np.ndarray, block_size: int, row: int, col: int) -> np.ndarray:
    """Draws a blue rectangle on the source image marking the selected DCT
    block. Uses the exact same coordinate maths as extract_block — this is
    the real analysis region, not decoration."""
    import cv2

    h, w = image.shape[:2]
    x0, y0, x1, y1 = block_pixel_region((h, w), block_size, row, col)
    out = image.copy()
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    thickness = max(2, min(3, block_size // 4))
    cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)),
                 (37, 99, 235), thickness)  # #2563EB, JPEG/DCT identity
    return out


def inspect_block(
    gray: np.ndarray, block_size: int, row: int, col: int, quality: float,
    base_table: np.ndarray,
) -> dict:
    """Gerçek bir bloğu uçtan uca işler; tüm ara adımları sayısal olarak döner."""
    block, grid_shape, (row, col) = extract_block(gray, block_size, row, col)
    shifted = block.astype(np.float64) - 128.0
    coeffs = dctn(shifted, norm="ortho")

    table = quality_to_qtable(quality, get_qtable(base_table, block_size))
    quantized = np.round(coeffs / table).astype(np.int32)
    dequantized = quantized.astype(np.float64) * table
    recon = np.clip(idctn(dequantized, norm="ortho") + 128.0, 0.0, 255.0)

    err = recon - block.astype(np.float64)
    block_mse = float(np.mean(err ** 2))
    block_psnr = float("inf") if block_mse == 0 else 10.0 * np.log10(255.0 ** 2 / block_mse)
    n_nonzero = int(np.count_nonzero(quantized))

    return dict(
        block=block, shifted=shifted, coeffs=coeffs, qtable=table,
        quantized=quantized, dequantized=dequantized, recon=recon,
        grid_shape=grid_shape, position=(row, col),
        dc=float(coeffs[0, 0]), block_mse=block_mse, block_psnr=block_psnr,
        n_total=block_size * block_size, n_nonzero=n_nonzero,
    )


def block_pipeline_figure(result: dict) -> plt.Figure:
    """Orijinal blok / seviye-kaydırma / DCT katsayıları / kuantalanmış
    katsayılar / rekonstrüksiyon — pipeline'ın TÜM aşamaları (mega-spec
    "01 PIXELS → 02 LEVEL SHIFT → 03 DCT → 04 QUANTIZATION → 06 RECONSTRUCTION";
    zigzag için ayrı zigzag_figure() kullanılır, matris değil doğrusal akış)."""
    from src.viz.style import BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY

    # DCT Lab artık her Satır/Sütun/Quality değişiminde CANLI yeniden
    # hesaplanır (bkz. app.py update_dct_analysis) — önceki, yalnız düğmeyle
    # tetiklenen davranışa göre çok daha sık çağrılır. Matplotlib, döndürülen
    # Figure'ları kapatılana kadar bellekte tutar; her çağrı ÖNCE bir önceki
    # çağrılardan kalan (zaten Gradio'ya gönderilmiş, artık gereksiz)
    # figürleri kapatır — aksi halde uzun bir oturumda figür sızıntısı olur.
    plt.close("all")
    block = result["block"]
    n = block.shape[0]
    annotate = n <= 8

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.0))
    fig.patch.set_facecolor(BG_PANEL)
    panels = [
        (block, "01 Original block (pixels)", "gray", None),
        (result["shifted"], "02 Level shift (pixel−128)", "RdBu_r", None),
        (np.sign(result["coeffs"]) * np.log1p(np.abs(result["coeffs"])),
         "03 DCT coefficients (signed log)", "RdBu_r", result["coeffs"]),
        (np.sign(result["quantized"]) * np.log1p(np.abs(result["quantized"])),
         "04 Quantized coefficients", "RdBu_r", result["quantized"]),
        (result["recon"], "06 Reconstruction", "gray", None),
    ]
    for ax, (data, title, cmap, annot_src) in zip(axes, panels):
        ax.set_facecolor(BG_PANEL)
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=12, color=TEXT_PRIMARY)
        ax.set_xticks([]); ax.set_yticks([])
        if annotate:
            src = annot_src if annot_src is not None else data
            for i in range(n):
                for j in range(n):
                    val = src[i, j]
                    txt = f"{val:.0f}" if abs(val) < 1000 else f"{val:.1e}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=12.5,
                            color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY, labelsize=9)
    fig.suptitle(
        f"DC coefficient (mean brightness component) = {result['dc']:.1f}   |   "
        f"non-zero coefficients: {result['n_nonzero']}/{result['n_total']}   |   "
        f"block PSNR = {result['block_psnr']:.2f} dB",
        fontsize=12, color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    return fig


def zigzag_figure(quantized: np.ndarray) -> plt.Figure:
    """JPEG zigzag tarama sırasındaki 64 (n²) katsayıyı DOĞRUSAL bir akış
    olarak gösterir — matris değil (mega-spec: "path overlay + altında
    linear coefficient stream"). Gerçek zigzag_indices (src/engines/entropy)
    kullanılır — src/engines/entropy.py'deki bit-tahmin motoruyla AYNI
    tarama sırası; matris değeri değil, encoder'ın gördüğü gerçek sıra.
    Sıfır katsayılar ikincil/soluk renkte, DC ilk eleman vurgulanır."""
    from src.viz.style import (ACCENT_CYAN, BG_ELEVATED, BG_PANEL, BORDER,
                               TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY)

    n = quantized.shape[0]
    order = zigzag_indices(n)
    stream = quantized.reshape(-1)[order]
    nonzero = stream != 0
    n_nonzero = int(nonzero.sum())

    fig, ax = plt.subplots(figsize=(18, 2.6))
    fig.patch.set_facecolor(BG_PANEL)
    ax.set_facecolor(BG_PANEL)
    x = np.arange(len(stream))
    colors = [ACCENT_CYAN if i == 0 else (TEXT_SECONDARY if v != 0 else BG_ELEVATED)
             for i, v in enumerate(stream)]
    ax.bar(x, stream, color=colors, width=0.7, zorder=2)
    ax.axhline(0, color=BORDER, linewidth=0.8, zorder=1)
    ax.set_xlim(-1, len(stream))
    ax.set_xticks(list(range(0, len(stream), max(1, len(stream) // 16))))
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xlabel("zigzag order (0 = DC)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("coefficient", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        f"05 Zigzag scan — non-zero: {n_nonzero} / {len(stream)}  "
        f"(blue = DC — quantized, faded = zeroed)",
        fontsize=11, color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    return fig
