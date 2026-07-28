# Calibration Scripts
### MSU Apple Grading System — JAI FS-3200T-10GE
**Last updated:** July 22, 2026

---

## Overview

This folder contains the three calibration scripts for the multispectral apple grading system.
Run them **in the order listed below**. Steps 1 and 2 are one-time setup.
Step 3 is run every time you collect a new apple run.

```
calibration_scripts/
├── calibration_analysis.py     ← STEP 1 (one-time): check exposure settings
├── calibration_pipeline.py     ← STEP 2 (one-time): build calibration maps
└── apply_calibration.py        ← STEP 3 (every run): apply calibration to apples
```

---

## Prerequisites

```
pip install numpy matplotlib pillow scipy
```

All scripts must be run from the **apple_class root directory**, not from inside calibration_scripts:

```powershell
cd S:\MSU_Research\apple_class
python calibration_scripts\calibration_pipeline.py
```

---

## STEP 1 — `calibration_analysis.py`

### What it does
Analyzes a single calibration capture folder (e.g. `cals/cal6`) and reports:
- Per-channel mean pixel value, max value, clipping percentage
- Whether the Spectralon panel is saturating any channel
- Histograms and sharpness metrics

Used during the initial calibration trials (cal1–cal6) to find exposure settings
where no channel clips and R = G = B on the Spectralon panel.

### When to use it
- When you change exposure settings and want to check for clipping
- When you set up the camera in a new environment
- When you want to compare two configurations

### How to run

```powershell
# Analyze a single calibration folder
python calibration_scripts\calibration_analysis.py cals\cal6 --rgb-exp 2500 --nir1-exp 1800 --nir2-exp 2300

# Analyze all known calibration runs (cal1 through cal6) at once
python calibration_scripts\calibration_analysis.py all

# Compare two runs side by side
python calibration_scripts\calibration_analysis.py cals\cal5 --rgb-exp 3000 --compare cals\cal6
```

### Arguments

| Argument | Description | Example |
|---|---|---|
| `folder` | Path to calibration folder | `cals/cal6` |
| `--rgb-exp` | RGB channel exposure in microseconds | `--rgb-exp 2500` |
| `--nir1-exp` | NIR1 channel exposure in microseconds | `--nir1-exp 1800` |
| `--nir2-exp` | NIR2 channel exposure in microseconds | `--nir2-exp 2300` |
| `--compare` | One or more folders to compare against | `--compare cals/cal5` |
| `all` | Analyze all known cals automatically | `all` |

### Output
Prints a report to terminal. No files saved.

---

## STEP 2 — `calibration_pipeline.py`

### What it does
The main calibration builder. Reads your captured dark frames and 9-position
white reference grid, then:
1. Averages all dark frames → `dark_avg_ch1/2/3.npy`
2. Computes illumination at each of the 9 belt positions
3. Fits a bicubic spline surface → `illumination_map_ch1/2/3.npy`
4. Computes per-pixel correction factors → `correction_map_ch1/2/3.npy`
5. Validates calibration accuracy using the center Spectralon image
6. Generates 5 diagnostic figures + a full text report

### When to use it
Run this **once** after collecting dark frames and white reference grid.
Re-run it if:
- You recaptured dark frames or white reference grid
- Camera was moved or refocused
- Lighting configuration changed
- More than one week has passed (lamp aging)

### Required input (already captured)
```
cals/final_runs/
  Black/raw_frames/ch1,ch2,ch3/   ← dark frames (84 frames, lens covered)
  white_UL/raw_frames/ch1,ch2,ch3/ ← upper-left Spectralon position
  white_UC/...                      ← upper-center
  white_UR/...                      ← upper-right
  white_ML/...                      ← middle-left
  white_C/...                       ← center
  white_MR/...                      ← middle-right
  white_LL/...                      ← lower-left
  white_LC/...                      ← lower-center
  white_LR/...                      ← lower-right
```

### How to run

```powershell
python calibration_scripts\calibration_pipeline.py
```

No arguments needed. All paths are configured inside the script.

### Output (saved to `calibration_results\`)

| File | What it is |
|---|---|
| `dark_avg_ch1.npy` | Averaged dark frame — RGB channel (float32) |
| `dark_avg_ch2.npy` | Averaged dark frame — NIR1 channel |
| `dark_avg_ch3.npy` | Averaged dark frame — NIR2 channel |
| `illumination_map_ch1.npy` | Full 2048×1536 illumination map — RGB |
| `illumination_map_ch2.npy` | Full 2048×1536 illumination map — NIR1 |
| `illumination_map_ch3.npy` | Full 2048×1536 illumination map — NIR2 |
| `correction_map_ch1/2/3.npy` | Per-pixel correction factors (0.53–1.0) |
| `fig1_dark_frames.png` | Dark frame heat maps |
| `fig2_white_grid.png` | 3×3 illumination grid heatmap |
| `fig3_illumination_profiles.png` | Belt illumination profiles |
| `fig4_correction_map.png` | Full-resolution correction maps |
| `fig5_calibration_impact.png` | Before/after on Spectralon panel |
| `calibration_report.txt` | Full numerical calibration report |

### Calibration settings used (already validated)

| Channel | Exposure | White Balance |
|---|---|---|
| RGB (CH1) | 2500 µs | Locked: R=0.4607 G=1.0000 B=1.6879 |
| NIR1 (CH2) | 1800 µs | N/A (grayscale) |
| NIR2 (CH3) | 2300 µs | N/A (grayscale) |

---

## STEP 3 — `apply_calibration.py`

### What it does
Applies the validated calibration maps to a new apple run. For every frame:
1. Loads raw JPG from `raw_frames/ch1,ch2,ch3/`
2. Subtracts dark frame
3. Divides by illumination map (removes spatial lighting inequality)
4. Applies exposure ratio correction (if apple was shot at different exposure than white ref)
5. Multiplies by 0.75 (Spectralon normalization)
6. Saves result as float32 `.npy` file (values 0.0 to 1.0 = reflectance)

Also generates:
- Before/after PNG visuals for the 5 frames with the most apple visible
- Per-frame statistics CSV
- Plain-language summary text

### When to use it
**Every time you collect a new apple run.** Run it immediately after collecting frames.

### How to run

```powershell
# run1 (RGB at 5000us, NIR unchanged)
python calibration_scripts\apply_calibration.py --run cals/apples_runs/run1 --rgb-exp 5000

# If all three exposures differ from white reference
python calibration_scripts\apply_calibration.py --run cals/apples_runs/run2 --rgb-exp 4000 --nir1-exp 1800 --nir2-exp 2300
```

### Arguments

| Argument | Required | Description | Default |
|---|---|---|---|
| `--run` | Yes | Path to the apple run folder | — |
| `--rgb-exp` | Yes | RGB exposure used during apple capture (µs) | 5000 |
| `--nir1-exp` | No | NIR1 exposure (µs) | 1800 |
| `--nir2-exp` | No | NIR2 exposure (µs) | 2300 |

### Output (saved inside the run folder)

```
cals/apples_runs/run1/
├── calibrated_frames/
│   ├── ch1/frame_000030.npy   ← RGB reflectance, float32, shape (1536, 2048)
│   ├── ch2/frame_000030.npy   ← NIR1 reflectance
│   └── ch3/frame_000030.npy   ← NIR2 reflectance
├── before_after/
│   └── frame_000064_before_after.png   ← visual comparison (5 best frames)
├── calibration_stats.csv              ← per-frame raw and calibrated statistics
└── calibration_stats_summary.txt      ← plain-language summary
```

### How to load a calibrated frame in Python

```python
import numpy as np

# Load one calibrated frame
refl = np.load('cals/apples_runs/run1/calibrated_frames/ch1/frame_000064.npy')

# refl is a 2D float32 array, shape (1536, 2048)
# Values:  0.0  = completely dark (no reflectance)
#          0.13 = typical red apple skin in RGB
#          0.33 = typical apple tissue in NIR1
#          0.75 = Spectralon white reference panel
#          1.0  = mirror / glare (should be masked)

# Simple apple mask (exclude belt background and glare)
apple_pixels = (refl > 0.05) & (refl < 0.95)
print(f'Mean apple reflectance: {refl[apple_pixels].mean():.4f}')
```

---

## Formula Reference

```
reflectance(x,y) = [ pixel(x,y) − dark(x,y) ]
                   / illumination_map(x,y)
                   × ( E_white / E_apple )
                   × 0.75

Where:
  pixel(x,y)          raw sensor value at position (x,y)  [0–255 DN]
  dark(x,y)           dark frame value                     [~8 DN]
  illumination_map     spatial illumination at (x,y)        [86–177 DN for RGB]
  E_white / E_apple   exposure ratio (1.0 if same exposure)
  0.75                Spectralon certified reflectance
```

| Channel | E_white | E_apple (run1) | Ratio |
|---|---|---|---|
| RGB | 2500 µs | 5000 µs | **0.5** |
| NIR1 | 1800 µs | 1800 µs | 1.0 |
| NIR2 | 2300 µs | 2300 µs | 1.0 |

---

## Recalibration Checklist

Redo Step 2 (re-run `calibration_pipeline.py`) if any of these change:

- [ ] Exposure settings changed
- [ ] Lighting configuration changed (lamps added, moved, or replaced)
- [ ] Camera repositioned or refocused
- [ ] More than one week since last calibration (lamp aging)

Estimated recapture time: **20 minutes** (dark frames + 9 positions)
