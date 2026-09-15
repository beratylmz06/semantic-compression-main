# Signal Compression Lab

An interactive Gradio lab for exploring transform-based image compression:
**DCT (JPEG-style)** and **DWT / PR-QMF (JPEG2000-style)**, driven from the same
target bit-rate, plus lossless coding. The UI layout follows the reference
project's five pages; the numbers come from this repo's own engines
(`src/engines/`, `src/compare.py`, `src/metrics/quality.py`).

## Quick start

```bash
pip install -r requirements.txt      # one time
python app.py                        # opens http://127.0.0.1:7860
```

or double-click **`run.bat`** on Windows (finds Python, installs requirements on
the first run, frees port 7860, launches, opens the browser).

CPU only. No GPU, no model downloads.

## Tabs

| Tab | What it shows |
|---|---|
| **Compare** | JPEG/DCT vs Wavelet/DWT at one target bpp — metrics (Quality-from-bpp / quant-step Δ, PSNR, MSE, SSIM, actual bpp, compression, size), reconstructions, the multiresolution wavelet mosaic, a Rate–Distortion curve, and a **reference-range check** (Lena → the practical PSNR bands; the synthetic target → the Gaussian rate–distortion ceiling; the fingerprint → wavelet-vs-DCT advantage + the lossless pointer). Auto-detects the image type. |
| **DCT Lab** | Click any 8×8 block on the JPEG-degraded image to inspect it: `01` original → `02` level-shift (−128) → `03` DCT coefficients → `04` quantized (grey = zeroed) → `05` zigzag order → `06` reconstruction, all as numeric grids, plus the 1D zigzag stream. JPEG quality is derived from the target bpp. A "Which frequency bands does DCT keep?" panel reconstructs the whole image from the DC+low / horizontal-AC / vertical-AC / diagonal-AC zones. |
| **DWT Lab** | One-level LL/HL/LH/HH bands with % energy, a labelled multiresolution mosaic, scalar (band-weighted) or uniform quantization, and step Δ from either a target bpp or a manual value. A second section runs a genuine **4-band PR-QMF** filter bank (4×4 = 16 uniform subbands) with a perfect-reconstruction check. |
| **Lossless** | zlib DEFLATE on the raw samples — pixel-exact reconstruction (MSE 0), compression ratio, for fingerprint / biomedical images. |
| **Theory** | Short reference notes on DCT, DWT, BPP, and PSNR / SSIM / MSE. |

## Sample images

Three categories in the picker (all 512×512, 8-bit grayscale):

| Folder | Category | Contents |
|---|---|---|
| `data/classic/` | Classic Test Images | `lenna`, `boat`, `couple`, `man` — natural photos, ~150 KB each |
| `data/synthetic/` | Synthetic Images | `synthetic_noise` — a near-white noise field with a faint calibration watermark, ~245 KB; the memoryless-Gaussian RD function is a real PSNR ceiling for it (`experiments/prepare_synthetic.py`) |
| `data/fingerprint/` | Fingerprint (Lossless) | `fingerprint` — a Gabor-noise ridge pattern with a loop core + delta; the lossless-coding case (`experiments/prepare_fingerprint.py`) |

The synthetic and fingerprint images are procedurally generated (seeded, byte-identical)
— re-run the two `experiments/prepare_*.py` scripts to regenerate them, or drop your own
PNGs into those folders. The **Change image** button also uploads your own PNG/JPEG or a
`.pgm` (Netpbm P2/P5); the uploaded file's size in KB is shown in the top bar.

Expected PSNR-vs-bitrate ranges (Lena literature bands, the Gaussian ceiling, the
fingerprint's lossless behaviour) live in `src/metrics/reference_ranges.py` and are
regression-tested in `tests/test_reference_ranges.py`.

## Layout

`config.py` — tunable parameters (quantization tables, wavelet options, bpp
matching). `src/engines/` — DCT, wavelet, real libjpeg/OpenJPEG wrappers,
PR-QMF. `src/analysis/` — image-type classifier, lossless codec, DCT frequency
zones. `src/viz/` — the card / metric / mosaic renderers and the design system
(`ui_theme.py`, warm "ivory / teal / copper" palette). Tests: `pytest -q`.
