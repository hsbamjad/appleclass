# Drift-Corrected Calibration Pipeline
# MSU Multispectral Apple Classification System

## Overview

This folder contains the complete calibration pipeline with drift correction.
Run scripts in alphabetical order (a_ first, e_ last).

---

## PIPELINE ORDER

### a_check_lamp_stability.py
Run every 5 min until lamp is stable.
- Place panel at center
- Capture frame to cal_before_runs/
- Run this script
- When it says STABLE two times in a row -> go to Step B

### b_build_drift_corrected_maps.py
Build new illumination maps with lamp drift correction and generate 10 research-grade figures.

Required folder structure (create these from your GUI captures):

    calibration_trials/final_runs_02/   (or any named folder -- update BASE_DIR in script)
      white_C1/raw_frames/ch1/   <- CENTER captured FIRST  (drift anchor t=0)
      white_UL/raw_frames/ch1/
      white_LL/raw_frames/ch1/
      white_ML/raw_frames/ch1/
      white_C2/raw_frames/ch1/   <- CENTER captured MID    (2nd drift anchor)
      white_LC/raw_frames/ch1/
      white_UC/raw_frames/ch1/
      white_LR/raw_frames/ch1/
      white_MR/raw_frames/ch1/
      white_UR/raw_frames/ch1/
      white_C3/raw_frames/ch1/   <- CENTER captured LAST   (drift anchor t=end)

Same structure for ch2/ and ch3/.
Dark frames are reused from calibration_results_01/ or calibration_results_02/.

Capture order: C1 -> UL -> LL -> ML -> C2 -> LC -> UC -> LR -> MR -> UR -> C3
Complete all captures in < 10 minutes.

### c_per_run_panel.py  (formerly d_per_run_panel.py)  <-- Run before EVERY apple run
Compute per-run lamp scale factor.

    python c_per_run_panel.py --run run_001

Before each run:
  1. Place panel at center
  2. Capture 5 frames -> save to apple_runs/run_001/panel/ch1/ (ch2, ch3)
  3. Run this script
  4. Remove panel
  5. Start apple run

### d_apply_full_calibration.py  (formerly e_apply_full_calibration.py)  <-- Run after EVERY apple run
Apply full calibration (illumination map + per-run scale factor).

    python d_apply_full_calibration.py --run run_001

Reads:
  apple_runs/run_001/raw_frames/ch1/    <- raw apple frames
  apple_runs/run_001/scale_factors.json <- from Step C

Outputs:
  apple_runs/run_001/calibrated/ch1/    <- reflectance .npy files
  apple_runs/run_001/calibration_stats.csv

---

## DAILY SESSION WORKFLOW

    Session start:
      [1] Warm lamp 45+ min
      [2] Auto WB -> Lock WB -> Load Locked in GUI
      [3] Run a_check_lamp_stability.py until STABLE

    One-time per session (if maps need update):
      [4] 9-position capture (C1 first, C2 mid, C3 last, all < 10 min)
      [5] python b_build_drift_corrected_maps.py

    For each apple run:
      [6] Panel at center -> 5 frames -> panel folder
      [7] python c_per_run_panel.py --run run_XXX
      [8] Remove panel -> start apple run
      [9] python d_apply_full_calibration.py --run run_XXX

---

## WHAT EACH LAYER CORRECTS

    Illumination map (Step B):
      - Spatial non-uniformity (center brighter than edges)
      - Built once, reused for all runs in a session

    Per-run scale factor (Step C):
      - Overall lamp brightness at time of each run
      - Corrects between-run lamp drift
      - Applied on top of illumination map

    Together:
      - Any apple, at any belt position, in any run = same reflectance
      - Within-run drift (~10%) is proportional and preserves spectral ratios
        - The ML classifier is robust to proportional scaling

---

## FILE PATHS (edit in each script if your paths differ)

    CAL_DIR  = S:\MSU_Research\apple_class\calibration_results_02\
    RUNS_DIR = S:\MSU_Research\apple_class\apple_runs\
    BASE_DIR = S:\MSU_Research\apple_class\calibration_trials\final_runs_02\
