"""
f_inspect_frame.py
===================
STEP F -- Interactive per-pixel inspector for a single frame (all channels).

Opens a matplotlib window with 6 panels:
  Row 1 (RAW) : RGB (colour) | NIR1 ~800nm (grey) | NIR2 ~900nm (grey)
  Row 2 (CAL) : RGB (colour) | NIR1 ~800nm (grey) | NIR2 ~900nm (grey)

Hover your mouse over any panel to see:
  - Pixel coordinates (x, y)
  - Raw DN values (R G B or single for NIR)
  - Calibrated reflectance values

No stretching, no enhancement. Displayed as-is (reflectance x 255 for cal).

Usage:
  python f_inspect_frame.py --run apples_run1_procc --frame 65
  python f_inspect_frame.py --run apples_run2_procc --frame 72 --stretch
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')   # interactive backend -- change to 'Qt5Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── PATHS ─────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(r'S:\MSU_Research\apple_class\formal_runs')

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Interactive pixel inspector -- all channels.')
parser.add_argument('--run',     required=True,  help='Run folder name, e.g. apples_run1_procc')
parser.add_argument('--frame',   required=True,  type=int, help='Frame number to inspect')
parser.add_argument('--stretch', action='store_true',
                    help='Apply 4x display brightness boost (for visibility only -- values unchanged)')
args = parser.parse_args()

run_dir = RUNS_DIR / args.run
fname   = f'frame_{args.frame:06d}'

raw_paths = {
    'RGB'  : run_dir / 'raw_frames' / 'ch1' / f'{fname}.jpg',
    'NIR1' : run_dir / 'raw_frames' / 'ch2' / f'{fname}.jpg',
    'NIR2' : run_dir / 'raw_frames' / 'ch3' / f'{fname}.jpg',
}
cal_paths = {
    'RGB'  : run_dir / 'calibrated' / 'ch1_rgb' / f'{fname}.npy',
    'NIR1' : run_dir / 'calibrated' / 'ch2'     / f'{fname}.npy',
    'NIR2' : run_dir / 'calibrated' / 'ch3'     / f'{fname}.npy',
}

# Check files
for label, path in {**raw_paths, **cal_paths}.items():
    if not path.exists():
        raise FileNotFoundError(f'Missing: {path}')

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print(f'Loading frame {args.frame} from {args.run}...')

# Raw arrays (uint8, original DN)
raw_rgb_arr  = np.array(Image.open(raw_paths['RGB']).convert('RGB'))   # (H,W,3) uint8
raw_n1_arr   = np.array(Image.open(raw_paths['NIR1']).convert('L'))    # (H,W)   uint8
raw_n2_arr   = np.array(Image.open(raw_paths['NIR2']).convert('L'))    # (H,W)   uint8

# Calibrated arrays (float32 reflectance, 0-1)
cal_rgb_f    = np.load(cal_paths['RGB']).astype(np.float32)            # (H,W,3) float32
cal_n1_f     = np.load(cal_paths['NIR1']).astype(np.float32)           # (H,W)   float32
cal_n2_f     = np.load(cal_paths['NIR2']).astype(np.float32)           # (H,W)   float32

# Display arrays (what matplotlib shows -- optionally brightness-boosted)
boost = 4.0 if args.stretch else 1.0

def to_disp_rgb(arr_f):
    return np.clip(arr_f * 255.0 * boost, 0, 255).astype(np.uint8)

def to_disp_grey(arr_f):
    return np.clip(arr_f * 255.0 * boost, 0, 255).astype(np.uint8)

disp = {
    ('RAW', 'RGB')  : raw_rgb_arr,
    ('RAW', 'NIR1') : np.stack([raw_n1_arr]*3, axis=2),
    ('RAW', 'NIR2') : np.stack([raw_n2_arr]*3, axis=2),
    ('CAL', 'RGB')  : to_disp_rgb(cal_rgb_f),
    ('CAL', 'NIR1') : np.stack([to_disp_grey(cal_n1_f)]*3, axis=2),
    ('CAL', 'NIR2') : np.stack([to_disp_grey(cal_n2_f)]*3, axis=2),
}

ROWS = ['RAW', 'CAL']
COLS = ['RGB', 'NIR1', 'NIR2']

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor='#16161e')
fig.suptitle(
    f'{args.run}  |  Frame {args.frame:04d}  |  Hover to inspect pixel values'
    + ('  [4x DISPLAY BOOST]' if args.stretch else '  [no stretch]'),
    color='#dddddd', fontsize=13, fontweight='bold'
)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.04, right=0.96, top=0.91, bottom=0.06,
                       hspace=0.08, wspace=0.06)

axes = {}
ims  = {}

COL_COLORS = {'RGB': '#ffc83c', 'NIR1': '#8cdcff', 'NIR2': '#50c882'}
ROW_COLORS = {'RAW': '#ffa050', 'CAL': '#50c8ff'}

for r_idx, row in enumerate(ROWS):
    for c_idx, col in enumerate(COLS):
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.set_facecolor('#0d0d12')
        for sp in ax.spines.values():
            sp.set_edgecolor(COL_COLORS[col])
            sp.set_linewidth(1.5)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

        img_data = disp[(row, col)]
        im = ax.imshow(img_data, aspect='auto', interpolation='nearest')

        title_str = f'{row}  {col}'
        ax.set_title(title_str, color=COL_COLORS[col], fontsize=11,
                     fontweight='bold', pad=4)

        # Left-side row label on first column only
        if c_idx == 0:
            ax.set_ylabel(row, color=ROW_COLORS[row], fontsize=11,
                          fontweight='bold', rotation=0, labelpad=36, va='center')

        axes[(row, col)] = ax
        ims[(row, col)]  = im

# ── STATUS BAR (text at bottom) ───────────────────────────────────────────────
status = fig.text(0.5, 0.01, 'Move mouse over any panel', ha='center',
                  color='#aaaaaa', fontsize=11,
                  bbox=dict(facecolor='#1e1e2a', edgecolor='#444', pad=4))

# ── HOVER CALLBACK ────────────────────────────────────────────────────────────
def on_move(event):
    for (row, col), ax in axes.items():
        if event.inaxes is not ax:
            continue

        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        H, W = raw_rgb_arr.shape[:2]

        if not (0 <= x < W and 0 <= y < H):
            return

        # Always show both raw and cal for the hovered pixel
        raw_r = int(raw_rgb_arr[y, x, 0])
        raw_g = int(raw_rgb_arr[y, x, 1])
        raw_b = int(raw_rgb_arr[y, x, 2])
        raw_n1 = int(raw_n1_arr[y, x])
        raw_n2 = int(raw_n2_arr[y, x])

        cal_r  = cal_rgb_f[y, x, 0]
        cal_g  = cal_rgb_f[y, x, 1]
        cal_b  = cal_rgb_f[y, x, 2]
        cal_n1 = cal_n1_f[y, x]
        cal_n2 = cal_n2_f[y, x]

        if col == 'RGB':
            if row == 'RAW':
                pval = f'R={raw_r}  G={raw_g}  B={raw_b} DN'
            else:
                pval = f'R={cal_r:.4f}  G={cal_g:.4f}  B={cal_b:.4f} refl'
        elif col == 'NIR1':
            if row == 'RAW':
                pval = f'NIR1={raw_n1} DN'
            else:
                pval = f'NIR1={cal_n1:.4f} refl'
        else:
            if row == 'RAW':
                pval = f'NIR2={raw_n2} DN'
            else:
                pval = f'NIR2={cal_n2:.4f} refl'

        full = (
            f'({x}, {y})  |  {row} {col}: {pval}'
            f'   ——   '
            f'Raw: RGB({raw_r},{raw_g},{raw_b})  N1={raw_n1}  N2={raw_n2}'
            f'   |   '
            f'Cal: RGB({cal_r:.3f},{cal_g:.3f},{cal_b:.3f})  N1={cal_n1:.3f}  N2={cal_n2:.3f}'
        )
        status.set_text(full)
        fig.canvas.draw_idle()
        return

fig.canvas.mpl_connect('motion_notify_event', on_move)

print('Window open -- hover mouse over any panel to inspect pixel values.')
print('Close the window to exit.')
plt.show()
