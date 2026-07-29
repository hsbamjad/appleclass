"""
c_per_run_panel.py
===================
STEP C -- Compute the per-run scale factor BEFORE each apple run.

Run this ONCE before every apple run.

What you do:
  1. Place Spectralon panel at CENTER of belt
  2. Capture frames using GUI (any number, more = more accurate)
  3. Save them to:  formal_runs/<CALIB_RUN>/raw_frames/ch1/  (ch2, ch3)
  4. Remove panel from belt
  5. Run this script:
       python c_per_run_panel.py --run <APPLE_RUN> --panel-dir <path_to_calib_raw_frames>
  6. Start apple run immediately

What this does:
  - Reads the panel frames from --panel-dir
  - Computes scale_factor for each channel
  - Saves scale factors to: formal_runs/<APPLE_RUN>/scale_factors.json
  - d_apply_full_calibration.py will automatically use these scale factors

Usage:
  python c_per_run_panel.py --run apples_run1_procc --panel-dir S:\\MSU_Research\\apple_class\\formal_runs\\calib_run1_procc\\raw_frames
  python c_per_run_panel.py --run apples_run2_procc --panel-dir S:\\MSU_Research\\apple_class\\formal_runs\\calib_run2_procc\\raw_frames
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

CAL_DIR  = Path(r'S:\MSU_Research\apple_class\calibration_results_02')
RUNS_DIR = Path(r'S:\MSU_Research\apple_class\formal_runs')
PANEL    = 0.75

# ── EXPOSURE SETTINGS ────────────────────────────────────────────────────────
# Panel capture MUST be at the same exposure as the white reference captures
# used to build the illumination map (calibration_results_02).
# Apple run may differ -- the ratio EXP_WHITE/EXP_APPLE is baked into the
# scale factor here so d_apply_full_calibration.py needs no extra correction.
#
# White ref (calibration / panel capture)     Apple run
#   ch1 (RGB) : 2500 us                         5000 us   ratio = 0.5
#   ch2 (NIR1): 1800 us                         1800 us   ratio = 1.0
#   ch3 (NIR2): 2300 us                         2300 us   ratio = 1.0

EXP_WHITE = {'ch1': 2500, 'ch2': 1800, 'ch3': 2300}  # exposure when panel captured
EXP_APPLE = {'ch1': 5000, 'ch2': 1800, 'ch3': 2300}  # exposure during apple run

ch_names = {'ch1': 'RGB', 'ch2': 'NIR1', 'ch3': 'NIR2'}

def to_gray(arr):
    arr = np.array(arr, dtype=np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr

def center_mean(arr):
    g = to_gray(arr)
    H, W = g.shape
    return g[H*2//5:H*3//5, W*2//5:W*3//5].mean()

parser = argparse.ArgumentParser(description='Compute per-run panel scale factor.')
parser.add_argument('--run', required=True, help='Run name, e.g. run_001')
parser.add_argument('--panel-dir', default=None, help='Override panel folder path')
args = parser.parse_args()

run_dir   = RUNS_DIR / args.run
panel_dir = Path(args.panel_dir) if args.panel_dir else run_dir / 'panel'

print('=' * 60)
print(f'  STEP C -- PER-RUN PANEL SCALE FACTOR')
print(f'  Run: {args.run}')
print('=' * 60)
print()
print('  Exposure config:')
for ch in ['ch1', 'ch2', 'ch3']:
    ratio = EXP_WHITE[ch] / EXP_APPLE[ch]
    print(f'    {ch_names[ch]:18s}: panel={EXP_WHITE[ch]}us  apple={EXP_APPLE[ch]}us  ratio={ratio:.4f}')
print()

scale_factors = {}
all_ok = True

# Record timestamp from the first ch1 panel frame before the loop
# (using frames[0] after the loop would give ch3's first frame, which is stale)
_ch1_frames = sorted((panel_dir / 'ch1').glob('*.jpg')) + sorted((panel_dir / 'ch1').glob('*.png'))
_first_frame_mtime = str(_ch1_frames[0].stat().st_mtime) if _ch1_frames else ''

for ch in ['ch1', 'ch2', 'ch3']:
    ch_dir = panel_dir / ch
    frames = sorted(ch_dir.glob('*.jpg')) + sorted(ch_dir.glob('*.png'))

    if not frames:
        print(f'  {ch_names[ch]}: ERROR -- no frames in {ch_dir}')
        all_ok = False
        continue

    # Average all panel frames
    arrays = [np.array(Image.open(f)).astype(np.float32) for f in frames]
    avg    = np.mean(arrays, axis=0)

    # Load calibration data
    dark  = to_gray(np.load(CAL_DIR / f'dark_avg_{ch}.npy'))
    illum = to_gray(np.load(CAL_DIR / f'illumination_map_{ch}.npy'))

    H, W   = to_gray(avg).shape
    cen    = center_mean(avg)
    dark_c = center_mean(dark)
    illum_c = center_mean(illum)

    net  = max(cen - dark_c, 0.001)

    # Measured reflectance at center using illumination map (same exposure as panel)
    # refl = net / illum_c * PANEL  (panel at EXP_WHITE, illum map at EXP_WHITE -- same)
    measured_refl = net / illum_c * PANEL

    # Scale factor = (lamp correction) x (exposure ratio)
    # Lamp correction: accounts for lamp being dimmer/brighter than at calibration time
    # Exposure ratio: EXP_WHITE / EXP_APPLE -- baked in here so d_ script formula is clean
    # Net effect: refl_apple = (net_apple / illum) * PANEL * scale_factor = true_reflectance
    exp_ratio    = EXP_WHITE[ch] / EXP_APPLE[ch]
    scale_factor = (PANEL / measured_refl) * exp_ratio

    err_before = (measured_refl - PANEL) / PANEL * 100
    # After correction, error is 0% by definition

    scale_factors[ch] = float(scale_factor)

    print(f'  {ch_names[ch]}:')
    print(f'    Panel frames:       {len(frames)}')
    print(f'    Center raw DN:      {cen:.2f}')
    print(f'    Illum center DN:    {illum_c:.2f}')
    print(f'    Measured refl:      {measured_refl:.4f}  (lamp error vs 0.75: {err_before:+.2f}%)')
    print(f'    Exp ratio (w/a):    {EXP_WHITE[ch]}/{EXP_APPLE[ch]} = {EXP_WHITE[ch]/EXP_APPLE[ch]:.4f}')
    print(f'    Scale factor:       {scale_factor:.5f}  (lamp correction x exp ratio)')
    print(f'    Verify (panel):     0.75 x {EXP_APPLE[ch]}/{EXP_WHITE[ch]} = {PANEL * EXP_APPLE[ch]/EXP_WHITE[ch]:.4f}  <- expected if you apply d_ to panel')
    print()

# Save scale factors
sf_path = run_dir / 'scale_factors.json'
run_dir.mkdir(parents=True, exist_ok=True)

out = {
    'run':          args.run,
    'captured_at':  _first_frame_mtime,
    'scale_factors': scale_factors,
    'panel_refl':   PANEL,
    'note':         'Apply these factors in d_apply_full_calibration.py'
}

with open(sf_path, 'w') as f:
    json.dump(out, f, indent=2)

print('-' * 60)
if all_ok:
    print(f'  Scale factors saved: {sf_path}')
    print()
    print('  *** REMOVE PANEL FROM BELT and start apple run NOW.')
    print()
    print('  After run: run  d_apply_full_calibration.py --run', args.run)
else:
    print('  ERROR: panel capture incomplete. Check folder and retry.')
print('=' * 60)
