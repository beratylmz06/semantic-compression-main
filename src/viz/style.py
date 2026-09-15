# -*- coding: utf-8 -*-
"""Lab design system: colour palette + Matplotlib theme helper. Every figure
(RD curve, dyadic tree, filter bank, DCT/DWT heat-maps) is styled from this
single place so the styling code is not repeated in every plotting function.

Theme: "instrument panel on paper" — a warm ivory canvas with deep-teal and
burnt-copper instrument accents; dark warm text for high scientific
readability.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette — single source of truth (both the CSS in src/ui_theme.py
# and every Matplotlib figure read these).
# ---------------------------------------------------------------------------
BG_MAIN = "#F6F2E9"        # warm ivory canvas
BG_SECONDARY = "#ECE4D2"   # recessed strips
BG_PANEL = "#FFFEFA"       # cards / panels / plot background
BG_ELEVATED = "#F1E9D6"    # inputs, chips, table zebra stripe
BORDER = "#DED0AC"
BORDER_LIGHT = "#C7B688"

ACCENT_CYAN = "#0E7C7B"       # deep teal — JPEG / DCT identity
ACCENT_PURPLE = "#B5541A"     # burnt copper — Wavelet / DWT identity
ACCENT_GREEN = "#2F7D32"      # forest green — semantic ROI / positive validation
POSITIVE = "#2F7D32"
WARNING = "#A6740A"           # amber (trade-off, not an error)
CRITICAL = "#B3261E"          # real errors

TEXT_PRIMARY = "#231F16"
TEXT_SECONDARY = "#5C5642"
TEXT_MUTED = "#8B8163"
TEXT_PLOT = "#3F3A2B"

# Method identities (kept CONSISTENT across metric cards, chart series, badges)
METHOD_JPEG = ACCENT_CYAN
METHOD_WAVELET = ACCENT_PURPLE
METHOD_SEMANTIC = ACCENT_GREEN
METHOD_BASELINE = TEXT_SECONDARY
METHOD_JPEG2000 = "#7A2E8C"  # distinct plum for the real codec
METHOD_REAL_JPEG = "#1D5FA6"  # real-libjpeg cross-check (steel blue)

# DWT subband identity colours (used by the dyadic tree, the energy bar chart
# and the CSS filter-bank flow diagram — one definition, three consumers).
SUBBAND_LL = "#0E7C7B"
SUBBAND_LH = "#B5541A"
SUBBAND_HL = "#7A2E8C"
SUBBAND_HH = "#A6740A"


def lab_figure(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    """A single-axis Figure/Axes pair with the lab theme applied."""
    fig, ax = plt.subplots(figsize=figsize)
    apply_lab_style(fig, ax)
    return fig, ax


def apply_lab_style(fig: plt.Figure, ax_or_axes) -> None:
    """Apply the lab theme to an existing Figure + (single or multiple) Axes.

    Every plotting function (rd_curve_figure, dyadic_tree_figure,
    filter_bank_figure, block_pipeline_figure) calls this — one source of
    truth instead of scattered copy-pasted styling code.
    """
    fig.patch.set_facecolor(BG_PANEL)
    axes = ax_or_axes if hasattr(ax_or_axes, "__iter__") else [ax_or_axes]
    for ax in axes:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.title.set_color(TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_color(BORDER_LIGHT)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)


def style_legend(ax: plt.Axes, **kwargs) -> None:
    leg = ax.legend(frameon=False, labelcolor=TEXT_PLOT, fontsize=8, **kwargs)
    if leg is not None:
        leg.get_frame().set_facecolor(BG_PANEL)
