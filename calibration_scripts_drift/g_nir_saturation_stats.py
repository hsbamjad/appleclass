"""
g_nir_saturation_stats.py
==========================
STEP G -- Report NIR saturation and glare statistics for a run.

For each channel (RGB, NIR1, NIR2), reports:
  - Raw DN max and clip count (sensor saturation check)
  - Calibrated reflectance: min, mean, max
  - Pixels above 0.75, 0.90, 0.95, and 1.00 thresholds
  - Recommended glare mask threshold

Useful for diagnosing specular glare on apple skin and
deciding whether a glare mask is needed before model training.

Usage:
  python g_nir_saturation_stats.py --run apples_run1_procc --frame 65
  python g_nir_saturation_stats.py --run apples_run1_procc --frame 65 --all-frames
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(r'S:\MSU_Research\apple_class\formal_runs')

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='NIR saturation and glare statistics.')
parser.add_argument('--run',        required=True, help='Run folder name, e.g. apples_run1_procc')
parser.add_argument('--frame',      type=int, default=None, help='Single frame number to inspect')
parser.add_argument('--all-frames', action='store_true', help='Process all frames and show summary')
args = parser.parse_args()

run_dir = RUNS_DIR / args.run

raw_dirs = {
    'RGB'  : run_dir / 'raw_frames' / 'ch1',
    'NIR1' : run_dir / 'raw_frames' / 'ch2',
    'NIR2' : run_dir / 'raw_frames' / 'ch3',
}
cal_dirs = {
    'RGB'  : run_dir / 'calibrated' / 'ch1_rgb',
    'NIR1' : run_dir / 'calibrated' / 'ch2',
    'NIR2' : run_dir / 'calibrated' / 'ch3',
}

CH_NAMES = {'RGB': 'RGB (ch1)', 'NIR1': 'NIR1 ~800nm (ch2)', 'NIR2': 'NIR2 ~900nm (ch3)'}

THRESHOLDS = [0.75, 0.90, 0.95, 1.00]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_raw_grey(path):
    img = Image.open(path)
    return np.array(img.convert('L'))

def load_raw_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def raw_stats(arr, label):
    total = arr.size
    clipped = int((arr >= 250).sum())
    sat     = int((arr >= 255).sum())
    print(f'    Raw DN    : mean={arr.mean():.1f}  max={int(arr.max())}  '
          f'>=250: {clipped} px ({clipped/total*100:.3f}%)  '
          f'=255: {sat} px ({sat/total*100:.4f}%)')

def cal_stats(arr_f, label):
    total = arr_f.size
    print(f'    Cal refl  : mean={arr_f.mean():.5f}  min={arr_f.min():.5f}  max={arr_f.max():.5f}')
    for t in THRESHOLDS:
        n = int((arr_f >= t).sum())
        marker = '  <-- GLARE' if t == 0.95 and n > 0 else ''
        print(f'    >= {t:.2f}   : {n:7d} px  ({n/total*100:.4f}%){marker}')

def analyse_frame(fnum):
    fname = f'frame_{fnum:06d}'
    print()
    print(f'  Frame {fnum:04d}  ({fname})')
    print(f'  {"-" * 55}')

    for ch in ['RGB', 'NIR1', 'NIR2']:
        raw_path = raw_dirs[ch] / f'{fname}.jpg'
        cal_path = cal_dirs[ch] / f'{fname}.npy'

        if not raw_path.exists() or not cal_path.exists():
            print(f'    {CH_NAMES[ch]}: files missing, skipping')
            continue

        print(f'  [{CH_NAMES[ch]}]')

        if ch == 'RGB':
            raw_arr = load_raw_rgb(raw_path)
            cal_f   = np.load(cal_path).astype(np.float32)
            raw_stats(raw_arr, ch)
            cal_stats(cal_f, ch)
        else:
            raw_arr = load_raw_grey(raw_path)
            cal_f   = np.load(cal_path).astype(np.float32)
            raw_stats(raw_arr, ch)
            cal_stats(cal_f, ch)

# ── SINGLE FRAME ──────────────────────────────────────────────────────────────
if not args.all_frames:
    if args.frame is None:
        parser.error('Provide --frame <number> or use --all-frames')

    print('=' * 65)
    print(f'  STEP G -- NIR SATURATION STATS')
    print(f'  Run   : {args.run}')
    print(f'  Frame : {args.frame}')
    print('=' * 65)

    analyse_frame(args.frame)

    print()
    print('  Glare mask recommendation:')
    print('    valid = (nir1_refl < 0.95) & (nir2_refl < 0.95)')
    print('  Apply during dataset prep to exclude saturated glare pixels.')
    print('=' * 65)

# ── ALL FRAMES SUMMARY ────────────────────────────────────────────────────────
else:
    all_frames = sorted(raw_dirs['NIR1'].glob('frame_*.jpg'))
    frame_nums = [int(p.stem.split('_')[1]) for p in all_frames]

    print('=' * 65)
    print(f'  STEP G -- NIR SATURATION SUMMARY (all frames)')
    print(f'  Run    : {args.run}')
    print(f'  Frames : {len(frame_nums)}')
    print('=' * 65)

    # Accumulators
    total_px   = 0
    glare_n1   = 0   # NIR1 >= 0.95
    glare_n2   = 0   # NIR2 >= 0.95
    glare_both = 0   # both NIR1 and NIR2 >= 0.95
    sat_n1_raw = 0   # NIR1 raw = 255
    sat_n2_raw = 0   # NIR2 raw = 255
    frames_with_glare = 0

    for fnum in frame_nums:
        fname   = f'frame_{fnum:06d}'
        n1_path = cal_dirs['NIR1'] / f'{fname}.npy'
        n2_path = cal_dirs['NIR2'] / f'{fname}.npy'
        r1_path = raw_dirs['NIR1'] / f'{fname}.jpg'
        r2_path = raw_dirs['NIR2'] / f'{fname}.jpg'

        if not (n1_path.exists() and n2_path.exists()):
            continue

        n1 = np.load(n1_path).astype(np.float32)
        n2 = np.load(n2_path).astype(np.float32)
        r1 = np.array(Image.open(r1_path).convert('L'))
        r2 = np.array(Image.open(r2_path).convert('L'))

        px = n1.size
        total_px   += px
        g1 = (n1 >= 0.95).sum()
        g2 = (n2 >= 0.95).sum()
        glare_n1   += int(g1)
        glare_n2   += int(g2)
        glare_both += int(((n1 >= 0.95) & (n2 >= 0.95)).sum())
        sat_n1_raw += int((r1 >= 255).sum())
        sat_n2_raw += int((r2 >= 255).sum())
        if g1 > 0 or g2 > 0:
            frames_with_glare += 1

    print()
    print(f'  Total pixels analysed : {total_px:,}  ({len(frame_nums)} frames)')
    print()
    print(f'  NIR1 raw = 255 (sensor clip) : {sat_n1_raw:,} px  ({sat_n1_raw/total_px*100:.4f}%)')
    print(f'  NIR2 raw = 255 (sensor clip) : {sat_n2_raw:,} px  ({sat_n2_raw/total_px*100:.4f}%)')
    print()
    print(f'  NIR1 cal >= 0.95 (glare)     : {glare_n1:,} px  ({glare_n1/total_px*100:.4f}%)')
    print(f'  NIR2 cal >= 0.95 (glare)     : {glare_n2:,} px  ({glare_n2/total_px*100:.4f}%)')
    print(f'  Both NIR1+NIR2 >= 0.95       : {glare_both:,} px  ({glare_both/total_px*100:.4f}%)')
    print(f'  Frames with any glare        : {frames_with_glare} / {len(frame_nums)}')
    print()
    print(f'  Glare mask would exclude     : {max(glare_n1, glare_n2):,} px  ({max(glare_n1, glare_n2)/total_px*100:.4f}%)')
    print()
    print('  Glare mask to apply during dataset prep:')
    print('    valid = (nir1_refl < 0.95) & (nir2_refl < 0.95)')
    print('=' * 65)
