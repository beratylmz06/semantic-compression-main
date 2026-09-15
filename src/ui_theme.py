# -*- coding: utf-8 -*-
"""Lab design system: a Gradio theme (buttons / sliders / panels via the
supported theming API) plus the custom CSS the card components need.

Theme: "instrument panel on paper" — warm ivory canvas with a graph-paper
grid, deep-teal and burnt-copper instrument accents, pill controls and a
dark monospace header banner. All colour constants come from
src/viz/style.py, so the interface and the plots share one palette.
"""
from __future__ import annotations

import gradio as gr

from src.viz.style import (ACCENT_CYAN, ACCENT_GREEN, ACCENT_PURPLE, BG_ELEVATED,
                           BG_MAIN, BG_PANEL, BG_SECONDARY, BORDER, BORDER_LIGHT,
                           CRITICAL, METHOD_JPEG2000, METHOD_REAL_JPEG, POSITIVE,
                           SUBBAND_HH, SUBBAND_HL, SUBBAND_LH, SUBBAND_LL,
                           TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY, WARNING)

_teal = gr.themes.Color(
    c50="#E4F3F2", c100="#BFE3E1", c200="#8FCECB", c300="#5EB8B4",
    c400="#2E9A97", c500="#0E7C7B", c600="#0C6B6A", c700="#0A5251",
    c800="#084140", c900="#063130", c950="#042220",
)
_sand = gr.themes.Color(
    c50="#FBF7EE", c100="#F1E9D6", c200="#DED0AC", c300="#C7B688",
    c400="#A69C79", c500="#8B8163", c600="#6E6650", c700="#5C5642",
    c800="#3F3A2B", c900="#231F16", c950="#17140E",
)

LAB_THEME = gr.themes.Base(
    primary_hue=_teal, secondary_hue=_teal, neutral_hue=_sand,
    text_size=gr.themes.sizes.text_lg,
    radius_size=gr.themes.sizes.radius_sm,
    font=(gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui",
          "-apple-system", "Segoe UI", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "SFMono-Regular",
               "Consolas", "Liberation Mono", "monospace"),
).set(
    body_background_fill=BG_MAIN, body_background_fill_dark=BG_MAIN,
    background_fill_primary=BG_PANEL, background_fill_primary_dark=BG_PANEL,
    background_fill_secondary=BG_SECONDARY, background_fill_secondary_dark=BG_SECONDARY,
    block_background_fill=BG_PANEL, block_background_fill_dark=BG_PANEL,
    block_border_color=BORDER, block_border_color_dark=BORDER,
    block_label_background_fill=BG_ELEVATED, block_label_background_fill_dark=BG_ELEVATED,
    block_label_text_color=TEXT_SECONDARY, block_label_text_color_dark=TEXT_SECONDARY,
    block_title_text_color=TEXT_PRIMARY, block_title_text_color_dark=TEXT_PRIMARY,
    panel_background_fill=BG_PANEL, panel_background_fill_dark=BG_PANEL,
    panel_border_color=BORDER, panel_border_color_dark=BORDER,
    body_text_color=TEXT_PRIMARY, body_text_color_dark=TEXT_PRIMARY,
    body_text_color_subdued=TEXT_SECONDARY, body_text_color_subdued_dark=TEXT_SECONDARY,
    border_color_primary=BORDER, border_color_primary_dark=BORDER,
    input_background_fill="#FFFEFA", input_background_fill_dark="#FFFEFA",
    input_border_color=BORDER_LIGHT, input_border_color_dark=BORDER_LIGHT,
    input_border_color_focus=ACCENT_CYAN, input_border_color_focus_dark=ACCENT_CYAN,
    slider_color=ACCENT_CYAN, slider_color_dark=ACCENT_CYAN,
    button_primary_background_fill=ACCENT_CYAN, button_primary_background_fill_dark=ACCENT_CYAN,
    button_primary_background_fill_hover="#0A5251", button_primary_background_fill_hover_dark="#0A5251",
    button_primary_text_color="#FFFEFA", button_primary_text_color_dark="#FFFEFA",
    button_secondary_background_fill="#FFFEFA", button_secondary_background_fill_dark="#FFFEFA",
    button_secondary_border_color=ACCENT_CYAN, button_secondary_border_color_dark=ACCENT_CYAN,
    button_secondary_text_color=ACCENT_CYAN, button_secondary_text_color_dark=ACCENT_CYAN,
    button_large_radius="8px", button_small_radius="8px",
    button_large_text_weight="700", button_large_padding="11px 20px",
    button_small_text_weight="600", button_small_padding="8px 14px",
    checkbox_background_color="#FFFEFA", checkbox_background_color_dark="#FFFEFA",
    checkbox_background_color_selected=ACCENT_CYAN, checkbox_background_color_selected_dark=ACCENT_CYAN,
    checkbox_border_color=BORDER_LIGHT, checkbox_border_color_dark=BORDER_LIGHT,
    table_border_color=BORDER, table_border_color_dark=BORDER,
    table_even_background_fill="#FFFEFA", table_even_background_fill_dark="#FFFEFA",
    table_odd_background_fill=BG_ELEVATED, table_odd_background_fill_dark=BG_ELEVATED,
    error_background_fill="#FBEDEC", error_background_fill_dark="#FBEDEC",
    error_border_color=CRITICAL, error_border_color_dark=CRITICAL,
    shadow_drop="0 1px 2px rgba(35,31,22,0.06)",
    shadow_drop_lg="0 6px 20px rgba(35,31,22,0.12)",
    block_shadow="none", block_shadow_dark="none",
    block_border_width="0px", block_border_width_dark="0px",
    panel_border_width="0px", panel_border_width_dark="0px",
)

# Runs once on load: (1) force Gradio's supported light mode via ?__theme=light,
# (2) re-English-ify Gradio's own built-in widget strings (the upload dropzone,
# the file Start/Clear/Download buttons, ...) when the visitor's browser locale
# makes Gradio serve them translated. This lab is English-only, but Gradio picks
# its UI language from navigator.language with no server-side override in this
# version, so we patch the rendered text and keep it patched through re-renders.
FORCE_LIGHT_JS = """
function() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
        return;
    }

    const MAP = {
        "Dosyay\\u0131 Buraya S\\u00fcr\\u00fckle": "Drop File Here",
        "G\\u00f6r\\u00fcnt\\u00fcy\\u00fc Buraya S\\u00fcr\\u00fckle": "Drop Image Here",
        "Dosyalar\\u0131 Buraya B\\u0131rak\\u0131n": "Drop File Here",
        "Y\\u00fcklemek i\\u00e7in T\\u0131kla": "Click to Upload",
        "Y\\u00fcklemek i\\u00e7in T\\u0131klay\\u0131n": "Click to Upload",
        "veya": "or",
        "Ba\\u015flat": "Start",
        "Durdur": "Stop",
        "G\\u00f6nder": "Submit",
        "Temizle": "Clear",
        "Kald\\u0131r": "Remove",
        "\\u0130ndir": "Download",
        "Y\\u00fckleniyor...": "Loading...",
        "\\u0130\\u015fleniyor...": "Processing...",
        "Hata": "Error",
        "Kopyala": "Copy",
        "Payla\\u015f": "Share",
        "T\\u00fcm\\u00fcn\\u00fc temizle": "Clear all"
    };
    const KEYS = Object.keys(MAP);
    function fix() {
        const root = document.querySelector('.gradio-container') || document.body;
        if (!root) return;
        const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        const hits = [];
        let n;
        while ((n = w.nextNode())) {
            const t = n.nodeValue;
            if (!t || t.length > 80) continue;
            for (let i = 0; i < KEYS.length; i++) {
                if (t.indexOf(KEYS[i]) !== -1) { hits.push(n); break; }
            }
        }
        for (const node of hits) {
            let t = node.nodeValue;
            for (let i = 0; i < KEYS.length; i++) t = t.split(KEYS[i]).join(MAP[KEYS[i]]);
            node.nodeValue = t;
        }
    }
    let raf = 0;
    const obs = new MutationObserver(function() {
        if (raf) return;
        raf = requestAnimationFrame(function() {
            raf = 0;
            obs.disconnect();
            fix();
            obs.observe(document.body, {childList: true, subtree: true, characterData: true});
        });
    });
    fix();
    obs.observe(document.body, {childList: true, subtree: true, characterData: true});
    [120, 400, 1000, 2500].forEach(function(ms) { setTimeout(fix, ms); });
}
"""
# Backwards-compatible alias (app.py may import either name).
FORCE_DARK_JS = FORCE_LIGHT_JS

LAB_CSS = f"""
:root {{
  --lab-bg-main: {BG_MAIN};
  --lab-bg-secondary: {BG_SECONDARY};
  --lab-bg-panel: {BG_PANEL};
  --lab-bg-elevated: {BG_ELEVATED};
  --lab-border: {BORDER};
  --lab-border-light: {BORDER_LIGHT};
  --lab-text-primary: {TEXT_PRIMARY};
  --lab-text-secondary: {TEXT_SECONDARY};
  --lab-text-muted: {TEXT_MUTED};
  --lab-accent: {ACCENT_CYAN};
  --lab-purple: {ACCENT_PURPLE};
  --lab-positive: {POSITIVE};
  --lab-warning: {WARNING};
  --lab-critical: {CRITICAL};
  --lab-jp2k: {METHOD_JPEG2000};
  --lab-real-jpeg: {METHOD_REAL_JPEG};
  --lab-card-shadow: 0 1px 2px rgba(35,31,22,0.05), 0 4px 14px rgba(35,31,22,0.07);
  --lab-sans: 'Manrope', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --lab-mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;

  /* ---- Global type scale — one source; every rule below uses one of
     these, never a random px value. ---- */
  --text-2xs: 11.5px;  /* badge / micro uppercase label only (never body copy) */
  --text-xs: 13.5px;   /* secondary text lower bound */
  --text-sm: 15px;     /* secondary text upper bound / captions */
  --text-md: 16.5px;   /* body copy, input/label default */
  --text-lg: 20px;     /* secondary metric values */
  --text-xl: 29px;     /* primary metric values */
  --text-2xl: 25px;    /* empty-state title */
  --text-3xl: 30px;    /* product title */
}}

.gradio-container {{
  max-width: 1680px !important;
  font-family: var(--lab-sans) !important;
  color: var(--lab-text-primary) !important;
  background-color: var(--lab-bg-main) !important;
  background-image:
    linear-gradient(to right, rgba(199,182,136,0.30) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(199,182,136,0.30) 1px, transparent 1px);
  background-size: 34px 34px;
}}

/* Hide the Gradio brand footer */
footer {{ display: none !important; }}
.gradio-container .prose a[href*="gradio.app"] {{ display: none !important; }}

/* ---------------- Header — dark instrument banner ---------------- */
.lab-header {{
  position: relative;
  overflow: hidden;
  padding: 26px 32px 24px 30px;
  margin-bottom: 10px;
  border-radius: 4px 16px 16px 4px;
  background: linear-gradient(135deg, {TEXT_PRIMARY} 0%, #0A5251 100%);
  box-shadow: 0 6px 22px rgba(35,31,22,0.16);
}}
.lab-header::before {{
  content: "";
  position: absolute; inset: 0;
  background-image: repeating-linear-gradient(45deg,
    rgba(255,254,250,0.06) 0 12px, rgba(255,254,250,0) 12px 24px);
  pointer-events: none;
}}
.lab-header::after {{
  content: "";
  position: absolute; top: 0; right: 0; bottom: 0; width: 7px;
  background: {ACCENT_CYAN};
}}
.lab-header .lab-title {{
  position: relative;
  font-family: var(--lab-mono);
  font-size: var(--text-3xl); font-weight: 800; letter-spacing: 0.09em;
  line-height: 1.12; text-transform: uppercase;
  color: #FFFEFA; margin: 0;
}}
.lab-header .lab-subtitle {{
  position: relative;
  font-family: var(--lab-mono);
  font-size: var(--text-sm); font-weight: 500; letter-spacing: 0.16em;
  text-transform: uppercase;
  color: rgba(255,254,250,0.72); margin-top: 9px;
}}

/* ---------------- Tabs — underline (old React app style) ---------------- */
.tab-nav {{
  border: none !important;
  border-bottom: 1px solid var(--lab-border) !important;
  border-radius: 0 !important;
  background: transparent !important;
  padding: 0 2px !important;
  gap: 4px;
  flex-wrap: wrap;
  margin: 4px 0 18px 0;
  box-shadow: none;
}}
.tab-nav button {{
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
  color: var(--lab-text-secondary) !important;
  font-family: var(--lab-sans) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  letter-spacing: 0.02em !important;
  padding: 12px 16px !important;
  height: auto !important;
  text-transform: none;
}}
.tab-nav button:hover {{ color: var(--lab-text-primary) !important; }}
.tab-nav button.selected {{
  color: {ACCENT_CYAN} !important;
  background: transparent !important;
  border-bottom: 2px solid {ACCENT_CYAN} !important;
}}

/* ---------------- Section labels — two levels ---------------- */
.page-section-label {{
  position: relative;
  padding-left: 16px;
  font-size: 23px; font-weight: 800; letter-spacing: 0.02em;
  text-transform: uppercase; color: var(--lab-text-primary);
  margin-bottom: 14px;
}}
.page-section-label::before {{
  content: "";
  position: absolute; left: 0; top: 3px; bottom: 3px;
  width: 5px; border-radius: 2px;
  background: {ACCENT_CYAN};
}}
.section-label {{
  font-family: var(--lab-mono);
  font-size: 16px; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-primary);
  border-bottom: 1px dashed var(--lab-border-light);
  padding-bottom: 7px; margin-bottom: 12px;
}}
.frame-caption {{
  font-family: var(--lab-mono);
  font-size: 13.5px; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-secondary);
  margin-bottom: 8px;
}}

/* ---------------- Cards — left accent + asymmetric radius ---------------- */
.lab-card {{
  background: var(--lab-bg-panel);
  border: 1px solid var(--lab-border);
  border-left: 4px solid {ACCENT_CYAN};
  border-radius: 4px 14px 14px 4px;
  padding: 18px 20px;
  margin-bottom: 6px;
  box-shadow: var(--lab-card-shadow);
}}

.metric-card-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
.method-dot {{ width: 10px; height: 10px; border-radius: 3px; transform: rotate(45deg); flex-shrink: 0; }}
.metric-title {{
  font-family: var(--lab-mono);
  font-size: 15px; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--lab-text-primary); flex-grow: 1;
}}
.size-badge {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--lab-text-secondary); background: var(--lab-bg-elevated);
  border: 1px solid var(--lab-border-light);
  border-radius: 999px; padding: 3px 11px;
}}
.metric-param {{ font-size: var(--text-xs); color: var(--lab-text-muted); margin-bottom: 12px; }}

.metric-row {{
  display: flex; justify-content: space-between; align-items: baseline;
  padding: 9px 0; border-top: 1px solid var(--lab-border);
}}
.metric-row:first-of-type {{ border-top: none; }}
.metric-label {{ font-size: var(--text-sm); color: var(--lab-text-secondary); letter-spacing: 0.01em; }}
.metric-value-wrap {{ display: flex; align-items: baseline; gap: 8px; }}
.metric-value {{ font-family: var(--lab-mono); font-size: var(--text-xl); font-weight: 700; color: var(--lab-text-primary); }}
.metric-value.mono {{ font-family: var(--lab-mono); }}
.metric-delta {{ font-family: var(--lab-mono); font-size: var(--text-sm); }}

/* ---------------- Target / RD summary card ---------------- */
.target-card {{ text-align: left; }}
.target-value-hero {{
  font-family: var(--lab-mono); font-size: 34px; font-weight: 800;
  color: var(--lab-accent); margin: 4px 0 6px 0;
}}
.target-value-hero .unit {{ font-size: var(--text-sm); color: var(--lab-text-secondary); }}
.target-caption {{ font-size: var(--text-xs); color: var(--lab-text-muted); margin-bottom: 12px; line-height: 1.5; }}
.target-row {{ display: flex; align-items: center; gap: 8px; padding: 7px 0; border-top: 1px solid var(--lab-border); }}
.target-row:first-of-type {{ border-top: none; }}
.target-label {{ font-size: var(--text-sm); color: var(--lab-text-secondary); flex-grow: 1; }}
.target-value {{ font-family: var(--lab-mono); font-size: var(--text-md); color: var(--lab-text-primary); }}
.target-diff .target-value {{ color: var(--lab-text-muted); }}

/* ---------------- Status badge ---------------- */
.status-badge {{
  display: inline-block; font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.06em;
  border: 1px solid; border-radius: 999px; padding: 4px 13px; text-transform: uppercase;
}}
.validation-head {{ margin-bottom: 10px; }}

/* ---------------- Reference-range check card ---------------- */
.refcheck-card {{ padding-top: 16px; }}
.refcheck-table {{ width: 100%; border-collapse: collapse; margin-top: 4px; }}
.refcheck-table th {{
  text-align: right; font-family: var(--lab-mono);
  font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--lab-text-muted); padding: 0 0 10px 0;
}}
.refcheck-table th:first-child, .refcheck-table td:first-child {{ text-align: left; }}
.refcheck-table td {{
  font-family: var(--lab-mono); font-size: var(--text-lg); text-align: right;
  padding: 12px 0; border-top: 1px solid var(--lab-border); color: var(--lab-text-primary);
}}
.refcheck-table td:first-child {{
  font-family: var(--lab-sans); font-size: var(--text-sm); font-weight: 600;
  color: var(--lab-text-primary); text-align: left;
}}
.ref-pill {{
  display: inline-block; font-family: var(--lab-sans); font-size: var(--text-2xs); font-weight: 800;
  letter-spacing: 0.05em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
  border: 1px solid;
}}
.ref-pill.in {{ color: {POSITIVE}; border-color: {POSITIVE}; background: #E9F4E9; }}
.ref-pill.below {{ color: {WARNING}; border-color: {WARNING}; background: #FBF3E3; }}
.ref-pill.above {{ color: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; background: #E4F3F2; }}
.ref-pill.na {{ color: var(--lab-text-muted); border-color: var(--lab-border-light); background: var(--lab-bg-elevated); }}
.refcheck-note {{ font-size: var(--text-xs); color: var(--lab-text-muted); margin-top: 12px; line-height: 1.5; }}

/* ---------------- Sparsity card ---------------- */
.sparsity-compare {{ display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }}
.sparsity-col {{ flex: 1; text-align: center; }}
.sparsity-col-title {{ font-size: var(--text-2xs); color: var(--lab-text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 5px; }}
.sparsity-big {{ font-family: var(--lab-mono); font-size: 28px; font-weight: 800; color: var(--lab-text-primary); }}
.sparsity-arrow {{ color: var(--lab-text-muted); font-size: var(--text-lg); }}

/* ---------------- Trade-off / same-budget cards ---------------- */
.tradeoff-row {{ display: flex; justify-content: space-between; padding: 5px 0; font-size: var(--text-sm); }}
.tradeoff-delta {{ font-family: var(--lab-mono); font-size: var(--text-xl); font-weight: 800; margin-top: 8px; text-align: right; }}
.tradeoff-metric {{ padding: 10px 0; border-top: 1px solid var(--lab-border); }}
.tradeoff-metric:first-of-type {{ border-top: none; padding-top: 0; }}
.budget-row {{ display: flex; gap: 16px; }}
.budget-col {{ flex: 1; text-align: center; }}

/* ---------------- Empty state ---------------- */
.empty-state {{
  text-align: center; padding: 40px 18px;
  background: var(--lab-bg-panel);
  border: 1px solid var(--lab-border);
  border-left: 4px dashed var(--lab-border-light);
  border-radius: 4px;
  box-shadow: none;
}}
.empty-title {{
  font-family: var(--lab-mono);
  font-size: 16px; font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--lab-text-secondary);
}}
.empty-caption {{ font-size: var(--text-sm); color: var(--lab-text-muted); margin-top: 8px; line-height: 1.55; }}

/* ---------------- Summary bar (DCT/DWT Lab) ---------------- */
.summary-bar {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
.summary-chip {{
  background: var(--lab-bg-elevated);
  border: 1px solid var(--lab-border);
  border-top: 3px solid {ACCENT_CYAN};
  border-radius: 4px;
  padding: 9px 15px; min-width: 100px;
}}
.summary-chip-label {{ display: block; font-size: var(--text-2xs); color: var(--lab-text-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
.summary-chip-value {{ display: block; font-size: var(--text-lg); font-weight: 800; color: var(--lab-text-primary); margin-top: 3px; }}

/* ---------------- Compact target-rate hero ---------------- */
.target-rate-header {{ display: flex; align-items: baseline; gap: 10px; padding: 4px 0 12px 0; }}
.target-rate-label {{ font-family: var(--lab-mono); font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--lab-text-muted); }}
.target-rate-value {{ font-family: var(--lab-mono); font-size: 30px; font-weight: 800; color: var(--lab-accent); }}
.target-rate-value .unit {{ font-size: var(--text-sm); color: var(--lab-text-secondary); font-weight: 400; }}

/* ---------------- Category badge (image picker) ---------------- */
.category-badge {{
  display: inline-block; font-size: var(--text-xs); font-weight: 600; color: var(--lab-text-secondary);
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 999px; padding: 4px 12px; margin-top: 5px;
}}

/* ---------------- Measurement strip (instrument readout) ---------------- */
.measurement-strip {{
  display: flex; flex-wrap: wrap;
  border: 1px solid var(--lab-border);
  border-radius: 4px 14px 14px 4px;
  overflow: hidden; margin: 16px 0;
  background: var(--lab-bg-panel); box-shadow: var(--lab-card-shadow);
}}
.measurement-item {{
  flex: 1; min-width: 130px; padding: 14px 20px;
  border-top: 3px solid {ACCENT_CYAN};
  border-right: 1px dashed var(--lab-border);
}}
.measurement-item:last-child {{ border-right: none; }}
.measurement-label {{
  display: block; font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-muted);
}}
.measurement-value {{
  display: block; font-family: var(--lab-mono); font-size: var(--text-xl); font-weight: 800;
  color: var(--lab-text-primary); margin-top: 5px;
}}

/* ---------------- Source image section ---------------- */
.source-image-frame img {{
  object-fit: contain !important; max-height: 420px;
}}
.selected-block-line {{
  font-family: var(--lab-mono); font-size: var(--text-sm); color: var(--lab-text-secondary);
  margin: 8px 0 0 0;
}}
.selected-block-line .sb-label {{
  font-family: var(--lab-sans); font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-muted); margin-right: 8px;
}}

/* ---------------- Theory tab grid ---------------- */
.theory-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
@media (max-width: 900px) {{ .theory-grid {{ grid-template-columns: 1fr; }} }}
.theory-card p, .theory-card li {{ font-size: var(--text-sm) !important; line-height: 1.6; }}

/* ---------------- Method-colored image frames ---------------- */
.frame-jpeg {{ border: 2px solid {ACCENT_CYAN} !important; border-radius: 4px 12px 12px 4px; }}
.frame-wavelet {{ border: 2px solid {ACCENT_PURPLE} !important; border-radius: 4px 12px 12px 4px; }}
.frame-jp2k {{ border: 2px solid {METHOD_JPEG2000} !important; border-radius: 4px 12px 12px 4px; }}
.frame-real-jpeg {{ border: 2px solid {METHOD_REAL_JPEG} !important; border-radius: 4px 12px 12px 4px; }}
.frame-neutral {{ border: 1px solid var(--lab-border-light) !important; border-radius: 4px 12px 12px 4px; }}

/* ---------------- Empty / experiment bar / image chip / quality strip ---------------- */
.hero-empty {{
  text-align: center;
  padding: 76px 28px 60px 28px;
  max-width: 640px;
  margin: 0 auto;
  border: 2px dashed var(--lab-border-light);
  border-radius: 16px;
  background: var(--lab-bg-panel);
}}
.hero-empty .hero-title {{
  font-family: var(--lab-mono);
  font-size: var(--text-2xl); font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--lab-text-primary); margin-bottom: 14px;
}}
.hero-empty .hero-caption {{
  font-size: var(--text-md); color: var(--lab-text-secondary); line-height: 1.6;
  margin-bottom: 8px;
}}

.experiment-bar {{
  align-items: center; flex-wrap: wrap; gap: 24px;
  padding: 16px 0 20px 0; border-bottom: 1px dashed var(--lab-border);
  margin-bottom: 24px;
}}
.experiment-bar .bar-group {{ display: flex; flex-direction: column; gap: 3px; }}
.experiment-bar .bar-label {{
  font-family: var(--lab-mono);
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--lab-text-muted);
}}

.image-chip {{
  display: flex; align-items: center; gap: 9px;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 999px; padding: 8px 16px;
}}
.image-chip .chip-name {{ font-size: var(--text-md); font-weight: 700; color: var(--lab-text-primary); }}
.image-chip .chip-category {{
  font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--lab-accent);
}}

/* ---------------- Rate-fairness header ---------------- */
.rate-fairness {{ display: flex; gap: 34px; align-items: baseline; padding: 6px 0 8px 0; flex-wrap: wrap; }}
.rate-fairness .rf-item {{ display: flex; flex-direction: column; gap: 3px; }}
.rate-fairness .rf-label {{ font-family: var(--lab-mono); font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--lab-text-muted); }}
.rate-fairness .rf-value {{ font-family: var(--lab-mono); font-size: var(--text-lg); font-weight: 800; color: var(--lab-text-primary); }}
.rate-fairness .rf-value.accent-jpeg {{ color: {ACCENT_CYAN}; }}
.rate-fairness .rf-value.accent-wavelet {{ color: {ACCENT_PURPLE}; }}

/* ---------------- Quality comparison strip (as a real table) ---------------- */
.quality-strip {{
  width: 100%; border-collapse: collapse; margin-top: 8px;
  background: var(--lab-bg-panel); border: 1px solid var(--lab-border);
  border-radius: 4px 14px 14px 4px; overflow: hidden; box-shadow: var(--lab-card-shadow);
}}
.quality-strip th {{
  text-align: right; font-family: var(--lab-mono);
  font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--lab-text-secondary);
  padding: 14px 18px; background: var(--lab-bg-elevated);
  border-bottom: 2px solid var(--lab-border);
}}
.quality-strip th:first-child, .quality-strip td:first-child {{ text-align: left; }}
.quality-strip th.th-jpeg {{ color: {ACCENT_CYAN}; }}
.quality-strip th.th-wavelet {{ color: {ACCENT_PURPLE}; }}
.quality-strip td {{
  font-family: var(--lab-mono); font-size: var(--text-lg); text-align: right;
  padding: 14px 18px; border-top: 1px solid var(--lab-border); color: var(--lab-text-primary);
}}
.quality-strip tr:nth-child(even) td {{ background: var(--lab-bg-elevated); }}
.quality-strip td:first-child {{
  font-family: var(--lab-sans); font-size: var(--text-sm); font-weight: 700; color: var(--lab-text-primary);
  text-align: left; text-transform: uppercase; letter-spacing: 0.03em;
}}
.quality-strip td.delta {{ font-size: var(--text-sm); color: var(--lab-text-muted); }}

/* ---------------- Filter-bank flow diagram (DWT Lab) ---------------- */
.filter-flow {{ display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px 0; }}
.flow-row {{ display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }}
.flow-row-4 {{ gap: 10px; }}
.flow-arrow {{ color: var(--lab-text-muted); font-size: 15px; line-height: 1; }}
.flow-node {{
  position: relative; background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 4px; padding: 9px 15px; font-size: var(--text-sm); color: var(--lab-text-primary);
  cursor: default; transition: border-color 140ms ease, background 140ms ease;
}}
.flow-node:hover {{ border-color: var(--lab-accent); background: #FFFEFA; }}
.flow-node.flow-lo, .flow-node.flow-ll {{ border-left: 3px solid {SUBBAND_LL}; }}
.flow-node.flow-hi, .flow-node.flow-hh {{ border-left: 3px solid {SUBBAND_HH}; }}
.flow-node.flow-lh {{ border-left: 3px solid {SUBBAND_LH}; }}
.flow-node.flow-hl {{ border-left: 3px solid {SUBBAND_HL}; }}
.flow-tip {{
  position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
  background: #FFFEFA; border: 1px solid var(--lab-border-light);
  border-radius: 6px; padding: 6px 11px; font-size: var(--text-xs); color: var(--lab-text-secondary);
  white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 140ms ease; z-index: 5;
  box-shadow: var(--lab-card-shadow);
}}
.flow-node:hover .flow-tip {{ opacity: 1; }}
.flow-band {{
  display: inline-block; padding: 7px 18px; border-radius: 4px; font-weight: 800;
  font-size: var(--text-sm); color: #FFFEFA; letter-spacing: 0.04em;
}}
.flow-band.flow-ll {{ background: {SUBBAND_LL}; }}
.flow-band.flow-lh {{ background: {SUBBAND_LH}; }}
.flow-band.flow-hl {{ background: {SUBBAND_HL}; }}
.flow-band.flow-hh {{ background: {SUBBAND_HH}; }}

/* ---------------- Active image bar — dark pill ---------------- */
.active-image-bar {{
  align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 11px 24px; margin: 6px 0 20px 0;
  background: {TEXT_PRIMARY};
  border: none;
  border-radius: 999px;
  box-shadow: 0 4px 16px rgba(35,31,22,0.18);
}}
.active-image-bar-content {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.active-image-tag {{
  font-family: var(--lab-mono);
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: #FFFEFA; background: var(--lab-accent); padding: 4px 12px; border-radius: 999px;
}}
.active-image-bar-content .chip-name {{ font-size: var(--text-lg); font-weight: 800; color: #FFFEFA; }}
.active-image-bar-content .chip-category {{
  font-size: var(--text-xs); font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  color: #7ED0CE;
}}
.active-image-dims {{ font-family: var(--lab-mono); font-size: var(--text-sm); color: rgba(255,254,250,0.82); }}
.active-image-resized {{
  font-family: var(--lab-mono); font-size: var(--text-xs); color: rgba(255,254,250,0.58);
  border-left: 1px solid rgba(255,254,250,0.24); padding-left: 12px; cursor: help;
}}
.active-image-filesize {{
  font-family: var(--lab-mono); font-size: var(--text-sm); font-weight: 800;
  color: #FFFEFA;
  border-left: 1px solid rgba(255,254,250,0.24); padding-left: 12px;
}}
.active-image-filesize .afs-label {{
  font-family: var(--lab-sans); font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: rgba(255,254,250,0.58); margin-right: 7px;
}}

/* ---------------- DWT decomposition explorer legend — pill ---------------- */
.dwt-legend {{
  display: flex; flex-wrap: wrap; gap: 8px 20px;
  padding: 10px 20px; margin: 8px 0 14px 0;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-radius: 999px; font-size: var(--text-xs); color: var(--lab-text-secondary);
}}
.dwt-legend-chip b {{ color: var(--lab-text-primary); }}
.dwt-tree-image img {{ cursor: pointer; }}

/* ---------------- Automatic full-analysis status ---------------- */
.analysis-status {{
  display: flex; flex-wrap: wrap; gap: 4px 20px;
  padding: 10px 18px; margin: -4px 0 16px 0;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border-light);
  border-radius: 4px; font-size: var(--text-sm);
}}
.analysis-status-row {{ display: flex; align-items: center; gap: 7px; color: var(--lab-text-muted); }}
.analysis-status-row.done {{ color: var(--lab-text-secondary); }}
.analysis-status-mark {{
  font-family: var(--lab-mono); width: 15px; text-align: center; color: var(--lab-text-muted);
}}
.analysis-status-row.done .analysis-status-mark {{ color: var(--lab-positive); font-weight: 800; }}
.analysis-status-ready {{
  display: inline-block; padding: 7px 18px; margin: -4px 0 16px 0;
  background: #E9F4E9; border: 1px solid {POSITIVE};
  border-radius: 999px; font-family: var(--lab-mono);
  font-size: var(--text-xs); font-weight: 800;
  letter-spacing: 0.08em; text-transform: uppercase; color: {POSITIVE};
}}

/* Editorial spacing */
.section-gap-lg {{ margin-top: 44px; }}
.section-gap-sm {{ margin-top: 12px; }}

/* ---------------- Compression summary — KPI row ---------------- */
.kpi-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 2px; }}
.kpi-box {{
  flex: 1 1 100px; min-width: 92px;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-top: 3px solid {ACCENT_CYAN};
  border-radius: 4px; padding: 10px 12px;
}}
.kpi-label {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--lab-text-muted);
}}
.kpi-value {{
  font-family: var(--lab-mono); font-size: var(--text-lg); font-weight: 800;
  color: var(--lab-text-primary); margin-top: 4px;
}}
.size-arrow-row {{
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--lab-border);
}}
.size-arrow-value {{ font-family: var(--lab-mono); font-size: var(--text-lg); font-weight: 800; color: var(--lab-text-primary); }}
.size-arrow-sep {{ color: var(--lab-text-muted); font-size: var(--text-lg); }}
.size-reduction-pct {{ font-family: var(--lab-mono); font-size: var(--text-sm); color: {POSITIVE}; font-weight: 800; margin-left: auto; }}

/* ---------------- Before / after slider ---------------- */
.ba-wrap {{ display: flex; flex-direction: column; gap: 8px; }}
.ba-frame {{
  position: relative; width: 100%; aspect-ratio: var(--ba-ar, 4/3);
  border-radius: 4px 14px 14px 4px; overflow: hidden; border: 1px solid var(--lab-border);
  background: var(--lab-bg-elevated); box-shadow: var(--lab-card-shadow);
}}
.ba-frame img {{
  position: absolute; inset: 0; width: 100%; height: 100%;
  display: block; user-select: none; pointer-events: none;
}}
.ba-frame .ba-clip {{
  position: absolute; inset: 0; overflow: hidden;
  clip-path: inset(0 calc(100% - var(--ba-pos, 50%)) 0 0);
}}
.ba-frame .ba-divider {{
  position: absolute; top: 0; bottom: 0; width: 2px; left: var(--ba-pos, 50%);
  background: var(--lab-accent); transform: translateX(-1px);
  box-shadow: 0 0 0 1px rgba(255,254,250,0.7); pointer-events: none;
}}
.ba-frame .ba-tag {{
  position: absolute; top: 10px; font-family: var(--lab-mono);
  font-size: var(--text-2xs); font-weight: 800; letter-spacing: 0.05em;
  text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
  background: rgba(255,254,250,0.9); color: var(--lab-text-primary);
  border: 1px solid var(--lab-border-light); pointer-events: none;
}}
.ba-tag-left {{ left: 10px; }}
.ba-tag-right {{ right: 10px; }}
.ba-range {{
  -webkit-appearance: none; appearance: none; width: 100%; height: 24px;
  background: transparent; margin: 0; cursor: ew-resize;
}}
.ba-range::-webkit-slider-runnable-track {{ height: 4px; background: var(--lab-border-light); border-radius: 2px; }}
.ba-range::-webkit-slider-thumb {{
  -webkit-appearance: none; appearance: none; width: 18px; height: 18px;
  border-radius: 50%; margin-top: -7px; cursor: ew-resize;
  background: #FFFEFA; border: 3px solid var(--ba-accent, {ACCENT_CYAN});
  box-shadow: 0 1px 4px rgba(35,31,22,0.28);
}}
.ba-range::-moz-range-track {{ height: 4px; background: var(--lab-border-light); border-radius: 2px; }}
.ba-range::-moz-range-thumb {{
  width: 18px; height: 18px; border-radius: 50%; border: 3px solid var(--ba-accent, {ACCENT_CYAN});
  background: #FFFEFA; cursor: ew-resize;
}}
.ba-range:focus-visible {{ outline: 2px solid var(--ba-accent, {ACCENT_CYAN}); outline-offset: 3px; border-radius: 4px; }}

.view-toggle-row {{ margin: 6px 0 20px; }}

/* ---------------- DWT clicked-coefficient box ---------------- */
.dwt-pixel-box {{
  margin-top: 10px; padding: 10px 14px;
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-accent);
  border-left: 4px solid var(--lab-accent);
  border-radius: 4px;
}}
.dwt-pixel-title {{
  font-family: var(--lab-mono); font-size: var(--text-2xs); font-weight: 800;
  letter-spacing: 0.05em; text-transform: uppercase; color: var(--lab-accent);
  margin-bottom: 5px;
}}
.dwt-pixel-row {{ display: flex; justify-content: space-between; padding: 2px 0; }}
.dwt-pixel-key {{ font-size: var(--text-xs); color: var(--lab-text-secondary); }}
.dwt-pixel-val {{ font-family: var(--lab-mono); font-size: var(--text-sm); font-weight: 700; color: var(--lab-text-primary); }}

/* ---------------- General body / caption text ---------------- */
.gradio-container p, .gradio-container li {{ font-size: var(--text-md); line-height: 1.6; }}

/* Dataframe readability */
.gradio-container table.table td, .gradio-container .table-wrap td {{ font-size: var(--text-sm) !important; padding: 10px 12px !important; }}
.gradio-container table.table th, .gradio-container .table-wrap th {{ font-size: var(--text-xs) !important; font-weight: 800 !important; font-family: var(--lab-mono) !important; text-transform: uppercase; }}

/* ===================================================================
   OLD REACT APP DESIGN SYSTEM (warm-recoloured) — Card / Metric /
   page-header / control-row / image-frame / grid / pill / badge.
   Used by the rebuilt Compare & Theory tabs.
   =================================================================== */
.badge {{
  font-size: var(--text-2xs); font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; padding: 4px 11px; border-radius: 999px;
  border: 1px solid var(--lab-border); color: var(--lab-text-secondary);
  background: var(--lab-bg-panel);
}}
.badge.ready {{ color: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; background: #E4F3F2; }}

.sc-grid {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: stretch; }}
.sc-grid > * {{ flex: 1 1 0; min-width: 0; }}
.sc-grid.cols-3 > * {{ flex: 1 1 200px; }}
.sc-grid.cols-4 > * {{ flex: 1 1 160px; }}
.sc-grid.cols-6 > * {{ flex: 1 1 130px; }}
@media (max-width: 980px) {{
  .sc-grid, .sc-grid.cols-3, .sc-grid.cols-4 {{ flex-direction: column; }}
}}

.card {{
  background: var(--lab-bg-panel);
  border: 1px solid var(--lab-border);
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: var(--lab-card-shadow);
}}
.card-title {{
  display: flex; align-items: center;
  font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--lab-text-muted); font-weight: 700; margin: 0 0 12px;
}}
.card-title .dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: {ACCENT_CYAN}; margin-right: 8px; flex-shrink: 0;
}}

.page-header {{ margin-bottom: 2px; }}
.page-header h2 {{
  font-size: 16px; font-weight: 800; margin: 0 0 4px; letter-spacing: 0.01em;
  color: var(--lab-text-primary);
}}
.page-header p {{ font-size: 12.5px; color: var(--lab-text-secondary); margin: 0; line-height: 1.5; }}

.metric-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(108px, 1fr)); gap: 14px;
}}
.metric-grid .metric {{ display: flex; flex-direction: column; gap: 4px; }}
.metric-grid .metric .k {{
  font-size: 10px; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--lab-text-muted); font-weight: 700;
}}
.metric-grid .metric .v {{
  font-family: var(--lab-mono); font-size: 20px; font-weight: 700; color: var(--lab-text-primary);
}}
.metric-grid .metric .v.accent {{ color: {ACCENT_CYAN}; }}
.metric-grid .metric .v.violet {{ color: {ACCENT_PURPLE}; }}
.metric-grid .metric .sub {{ font-size: 11px; color: var(--lab-text-secondary); }}

.pill {{
  display: inline-block; font-size: 11px; padding: 5px 11px; border-radius: 999px;
  border: 1px solid var(--lab-border); color: var(--lab-text-secondary);
  background: var(--lab-bg-elevated);
}}
.pill strong {{ color: var(--lab-text-primary); }}
.pill.good {{ color: {POSITIVE}; border-color: {POSITIVE}; background: #E9F4E9; }}

.caption {{ font-size: 11px; color: var(--lab-text-secondary); text-align: center; margin-top: 8px; }}

.image-frame {{
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-radius: 12px; overflow: hidden;
}}
.image-frame img {{ max-width: 100%; display: block; }}
.image-frame.clickable img {{ cursor: crosshair; }}
.diagram-frame {{
  background: var(--lab-bg-elevated); border: 1px solid var(--lab-border);
  border-radius: 12px; padding: 16px; display: flex; justify-content: center;
}}
.diagram-frame img {{ width: 100%; max-width: 660px; }}
.diagram-frame.wide img {{ max-width: 860px; }}

/* Slider + number box side by side (BppControl) */
.sc-bpp {{ display: flex; align-items: flex-end; gap: 10px; }}
.sc-bpp > *:first-child {{ flex: 1; }}
.sc-bpp .sc-num {{ flex: none; width: 92px; }}

/* ---------------- Responsive ---------------- */
@media (max-width: 1439px) and (min-width: 900px) {{
  .lab-header .lab-title {{ font-size: 25px; }}
  .tab-nav button {{ font-size: 13.5px !important; padding: 8px 16px !important; }}
  .page-section-label {{ font-size: 21px; }}
  .section-label {{ font-size: 15px; }}
}}
@media (max-width: 899px) {{
  .rate-fairness {{ gap: 18px; }}
  .measurement-item {{ min-width: 45%; }}
  .quality-strip th, .quality-strip td {{ padding: 10px 12px; }}
}}
"""
