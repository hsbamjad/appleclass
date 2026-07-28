"""
a_check_lamp_stability.py
=========================
STEP A -- Run this BEFORE starting the 9-position calibration capture.

Reads the latest frame from:
  S:\\MSU_Research\\apple_class\\calibration_trials\\cal_before_runs\\

Checks if the lamp is stable enough to begin the 9-position calibration.
Run this every 5 minutes until you see TWO CONSECUTIVE PASS results.
Then immediately run the 9-position capture.

Stability rule:
  - Error vs Tuesday baseline < 5%   (lamp is close to reference level)
  - Change from last check < 2%      (lamp is no longer drifting fast)

Usage:
  python a_check_lamp_stability.py
"""

import numpy as np
from pathlib import Path
from PIL import Image
import json, time

CAL_DIR   = Path(r'S:\MSU_Research\apple_class\calibration_results')
CHECK_DIR = Path(r'S:\MSU_Research\apple_class\calibration_trials\cal_before_runs')
HISTORY   = Path(r'S:\MSU_Research\apple_class\calibration_scripts_drift\stability_history.json')
PANEL     = 0.75
DARK      = 8.0

files = {
    'ch1': 'frame_000001_rgb.jpg',
    'ch2': 'frame_000001_nir1.jpg',
    'ch3': 'frame_000001_nir2.jpg',
}

ch_names = {'ch1': 'RGB', 'ch2': 'NIR1', 'ch3': 'NIR2'}

def center_mean(path):
    img  = np.array(Image.open(path)).astype(np.float32)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    H, W = gray.shape
    return gray[H*2//5:H*3//5, W*2//5:W*3//5].mean()

# Load history
history = []
if HISTORY.exists():
    with open(HISTORY) as f:
        history = json.load(f)

print('=' * 60)
print('  STEP A -- LAMP STABILITY CHECK')
print('  Run every 5 min until two consecutive STABLE results')
print('=' * 60)
print()

current = {}
for ch, fname in files.items():
    fpath = CHECK_DIR / fname
    if not fpath.exists():
        print(f'  {ch_names[ch]}: file not found -- {fpath}')
        continue

    raw_c  = center_mean(fpath)
    illum  = np.load(CAL_DIR / f'illumination_map_{ch}.npy').astype(np.float32)
    dark   = np.load(CAL_DIR / f'dark_avg_{ch}.npy').astype(np.float32)
    dark_g = dark.mean(axis=2) if dark.ndim == 3 else dark
    illum_g = illum.mean(axis=2) if illum.ndim == 3 else illum

    H, W   = illum_g.shape
    illum_c = illum_g[H*2//5:H*3//5, W*2//5:W*3//5].mean()
    dark_c  = dark_g[H*2//5:H*3//5, W*2//5:W*3//5].mean()

    net    = max(raw_c - dark_c, 0.001)
    refl   = net / illum_c * PANEL
    err    = (refl - PANEL) / PANEL * 100
    current[ch] = {'raw': raw_c, 'refl': float(refl), 'err': float(err)}

# Compare to last check
ts = time.strftime('%H:%M:%S')
drift_ok = True
err_ok   = True

print(f'  Time: {ts}')
print()
print(f'  {"Channel":<8} {"Raw DN":>8} {"Reflectance":>12} {"Error":>8} {"vs Last":>9} {"Status":<10}')
print(f'  {"-"*8} {"-"*8} {"-"*12} {"-"*8} {"-"*9} {"-"*10}')

for ch in ['ch1', 'ch2', 'ch3']:
    if ch not in current:
        continue
    v   = current[ch]
    err = v['err']

    vs_last = '--'
    if history:
        last = history[-1].get(ch, {})
        if 'err' in last:
            delta   = err - last['err']
            vs_last = f'{delta:+.1f}%'
            if abs(delta) > 2.0:
                drift_ok = False

    if abs(err) >= 5.0:
        err_ok = False

    status = 'STABLE' if abs(err) < 5.0 else 'DRIFTING'
    print(f'  {ch_names[ch]:<8} {v["raw"]:>8.1f} {v["refl"]:>12.4f} {err:>+7.1f}% {vs_last:>9} {status}')

print()

# Overall verdict
if not history:
    verdict = 'FIRST CHECK -- run again in 5 min to see drift rate'
    go = False
elif not err_ok:
    verdict = 'NOT READY -- error > 5%, lamp still far from reference'
    go = False
elif not drift_ok:
    verdict = 'NOT READY -- lamp changed > 2% since last check, still drifting'
    go = False
else:
    verdict = 'STABLE -- safe to begin 9-position capture NOW'
    go = True

print(f'  VERDICT: {verdict}')
if go:
    print()
    print('  *** GO: Start 9-position capture immediately.')
    print('      Capture: white_C_start -> 8 positions -> white_C_end')
    print('      Complete all captures in < 8 minutes.')
else:
    print()
    print('  Wait 5 minutes, capture new frame to cal_before_runs/, run again.')

print('=' * 60)

# Save history
entry = {ch: current[ch] for ch in current}
entry['time'] = ts
history.append(entry)
if len(history) > 20:
    history = history[-20:]
with open(HISTORY, 'w') as f:
    json.dump(history, f, indent=2)
