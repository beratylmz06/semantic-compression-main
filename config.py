# -*- coding: utf-8 -*-
"""Central configuration: every tunable parameter lives here.

Editing this file alone should be enough to change the experiments.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Image categories -> subfolders under data/ (the experiment scripts,
# experiments/*.py, use this list; each expects a populated folder under
# data/ and it should not be changed).
IMAGE_CATEGORIES = ["natural", "cgi", "mixed"]

# ---------------------------------------------------------------------------
# UI sample-image categories: (folder_name, display_label).
# Only used by the sample gallery in the interface; does NOT affect the
# experiment scripts.
#   classic     - 512x512 8-bit grayscale test photos, 100-300 KB
#                 (lenna, boat, couple, man)
#   synthetic   - procedurally generated 512x512 grayscale targets, 200-300 KB
#                 (experiments/prepare_synthetic.py) — near-white noise field
#                 with a Gaussian rate-distortion reference
#   fingerprint - synthetic 512x512 grayscale fingerprint
#                 (experiments/prepare_fingerprint.py) — the lossless-coding case
# ---------------------------------------------------------------------------
UI_IMAGE_CATEGORIES = [
    ("classic", "Classic Test Images (Grayscale)"),
    ("synthetic", "Synthetic Images (Grayscale)"),
    ("fingerprint", "Fingerprint (Lossless)"),
]

# ---------------------------------------------------------------------------
# DCT engine (JPEG-style)
# ---------------------------------------------------------------------------
DCT_BLOCK_SIZE = 8

# Standard JPEG luminance quantization matrix (ITU-T T.81, Annex K)
JPEG_LUMA_QTABLE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]

# Standard JPEG chroma quantization matrix (for colour images)
JPEG_CHROMA_QTABLE = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]

# Quality values swept for the RD curves (1-100, JPEG sense)
DCT_QUALITY_SWEEP = [5, 10, 20, 35, 50, 70, 85, 95]

# Selectable DCT block sizes in the UI. Only 8 is standard JPEG; 4/16 are an
# educational experiment (see src/engines/dct_engine.get_qtable).
DCT_BLOCK_SIZE_OPTIONS = [4, 8, 16]

# ---------------------------------------------------------------------------
# Wavelet engine (JPEG2000-style)
# ---------------------------------------------------------------------------
# Filter banks compared in the experiments (bior4.4 = CDF 9/7, bior2.2 = 5/3)
WAVELET_FILTER_BANKS = ["haar", "db4", "bior4.4", "bior2.2"]
WAVELET_DEFAULT_FILTER = "bior4.4"
WAVELET_LEVELS = 4

# Wavelet families selectable in the UI (Compare + DWT Lab). Each is a real
# pywt wavelet that works directly with wavedec2/waverec2.
WAVELET_UI_OPTIONS = ["haar", "db2", "db4", "db8", "db12", "sym4", "coif1", "bior2.2", "bior4.4"]
# UI slider ceiling for the decomposition level; the true limit is always
# recomputed from image size and wavelet via
# wavelet_engine.max_decomposition_level.
WAVELET_LEVEL_UI_MAX = 6

# Base quantization steps swept for the RD curves (larger step = lower bpp)
WAVELET_STEP_SWEEP = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]

# ---------------------------------------------------------------------------
# ROI bit steering (kept: match_bpp lives in src/roi/bit_allocation.py)
# ---------------------------------------------------------------------------
# Minimum overlap ratio for an 8x8 DCT block to count as "important".
ROI_BLOCK_OVERLAP_THRESHOLD = 0.10
# Foreground / background quantization hardness ratio.
ROI_BG_COARSENESS = 6.0

# ---------------------------------------------------------------------------
# Fair comparison: bit-budget matching
# ---------------------------------------------------------------------------
# Target bpp points used in the experiments
TARGET_BPP_POINTS = [0.25, 0.50, 0.80, 1.20]
# Convergence tolerance (relative): |actual-target|/target < tolerance
BPP_MATCH_TOLERANCE = 0.02
# Max iterations for the bisection over the quality parameter
BPP_MATCH_MAX_ITER = 30
