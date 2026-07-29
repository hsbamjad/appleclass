"""
e_visualize_channels.py
========================
STEP E -- Generate side-by-side channel visualization panels.

Produces one PNG per frame showing all 6 panels:
  Row 1 (RAW) : RGB (colour) | NIR1 ~800nm (grey) | NIR2 ~900nm (grey)
  Row 2 (CAL) : RGB (colour) | NIR1 ~800nm (grey) | NIR2 ~900nm (grey)

No stretching. No enhancement. No gamma correction.
  Raw         : JPG opened as-is (0-255 DN), displayed as-is
  Calibrated  : float32 reflectance [0-1] x 255 -> uint8, displayed as-is

Outputs saved to: formal_runs/visual/

Usage:
  python e_visualize_channels.py --run apples_run1_procc
  python e_visualize_channels.py --run apples_run1_procc --frames 45 55 65 75
  python e_visualize_channels.py --run apples_run1_procc --all
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(r'S:\MSU_Research\apple_class\formal_runs')

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Visualize raw + calibrated channels.')
parser.add_argument('--run',    required=True, help='Run folder name, e.g. apples_run1_procc')
parser.add_argument('--frames', nargs='+', type=int, default=[55, 65, 75],
                    help='Frame numbers to visualize (default: 55 65 75)')
parser.add_argument('--all',    action='store_true',
                    help='Process all frames (ignores --frames)')
args = parser.parse_args()

run_dir = RUNS_DIR / args.run
out_dir = RUNS_DIR / 'visual'
out_dir.mkdir(exist_ok=True)

raw_dirs = {
    'RGB'  : run_dir / 'raw_frames' / 'ch1',
    'NIR1' : run_dir / 'raw_frames' / 'ch2',
    'NIR2' : run_dir / 'raw_frames' / 'ch3',
}
cal_dirs = {
    'RGB'  : run_dir / 'calibrated' / 'ch1_rgb',  # (H,W,3) float32
    'NIR1' : run_dir / 'calibrated' / 'ch2',       # (H,W)   float32
    'NIR2' : run_dir / 'calibrated' / 'ch3',       # (H,W)   float32
}

# ── FONT ──────────────────────────────────────────────────────────────────────
def get_fonts():
    try:
        return ImageFont.truetype('arial.ttf', 20), ImageFont.truetype('arial.ttf', 14)
    except OSError:
        d = ImageFont.load_default()
        return d, d

font_big, font_sm = get_fonts()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_raw_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def load_raw_grey(path):
    return np.array(Image.open(path).convert('L'))

def load_cal_rgb(path):
    return np.clip(np.load(path).astype(np.float32) * 255, 0, 255).astype(np.uint8)

def load_cal_nir(path):
    return np.clip(np.load(path).astype(np.float32) * 255, 0, 255).astype(np.uint8)

def grey_to_pil(arr):
    return Image.fromarray(np.stack([arr, arr, arr], axis=2))

def arr_stats_raw(arr):
    return f'mean {arr.mean():.1f} DN  max {int(arr.max())}'

def arr_stats_cal(path):
    arr = np.load(path).astype(np.float32)
    return f'refl mean {arr.mean():.4f}  max {arr.max():.4f}'

# ── LAYOUT ────────────────────────────────────────────────────────────────────
THUMB_W   = 512
GAP       = 10
TOP_TITLE = 44
ROW_LABEL = 36
BORDER    = 12
BG        = (22, 22, 30)

COL_COLORS = {'RGB': (255, 200, 60), 'NIR1': (140, 220, 255), 'NIR2': (80, 200, 130)}
ROW_COLORS = {'RAW': (255, 160, 80), 'CAL': (80, 200, 255)}

ROWS = ['RAW', 'CAL']
COLS = ['RGB', 'NIR1', 'NIR2']

# ── FRAME SELECTION ───────────────────────────────────────────────────────────
if args.all:
    all_raw = sorted(raw_dirs['RGB'].glob('frame_*.jpg'))
    frame_nums = [int(p.stem.split('_')[1]) for p in all_raw]
else:
    frame_nums = args.frames

print('=' * 65)
print('  STEP E -- CHANNEL VISUALIZATION')
print(f'  Run    : {args.run}')
print(f'  Frames : {frame_nums}')
print(f'  Output : {out_dir}')
print('=' * 65)
print()

for fnum in frame_nums:
    fname = f'frame_{fnum:06d}'

    # Check all files exist
    missing = []
    for ch in ['RGB', 'NIR1', 'NIR2']:
        ext = 'jpg'
        if not (raw_dirs[ch] / f'{fname}.{ext}').exists():
            missing.append(f'raw/{ch}')
        npy_path = cal_dirs[ch] / f'{fname}.npy'
        if not npy_path.exists():
            missing.append(f'cal/{ch}')
    if missing:
        print(f'  SKIP frame {fnum} -- missing: {", ".join(missing)}')
        continue

    # Load
    raw_rgb = load_raw_rgb(raw_dirs['RGB']  / f'{fname}.jpg')
    raw_n1  = load_raw_grey(raw_dirs['NIR1'] / f'{fname}.jpg')
    raw_n2  = load_raw_grey(raw_dirs['NIR2'] / f'{fname}.jpg')
    cal_rgb = load_cal_rgb(cal_dirs['RGB']  / f'{fname}.npy')
    cal_n1  = load_cal_nir(cal_dirs['NIR1'] / f'{fname}.npy')
    cal_n2  = load_cal_nir(cal_dirs['NIR2'] / f'{fname}.npy')

    # Thumbnails
    H_orig, W_orig = raw_rgb.shape[:2]
    scale   = THUMB_W / W_orig
    THUMB_H = int(H_orig * scale)

    def rsz(pil_img):
        return pil_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

    panels = {
        ('RAW', 'RGB')  : rsz(Image.fromarray(raw_rgb)),
        ('RAW', 'NIR1') : rsz(grey_to_pil(raw_n1)),
        ('RAW', 'NIR2') : rsz(grey_to_pil(raw_n2)),
        ('CAL', 'RGB')  : rsz(Image.fromarray(cal_rgb)),
        ('CAL', 'NIR1') : rsz(grey_to_pil(cal_n1)),
        ('CAL', 'NIR2') : rsz(grey_to_pil(cal_n2)),
    }

    stats = {
        ('RAW', 'RGB')  : arr_stats_raw(raw_rgb),
        ('RAW', 'NIR1') : arr_stats_raw(raw_n1),
        ('RAW', 'NIR2') : arr_stats_raw(raw_n2),
        ('CAL', 'RGB')  : arr_stats_cal(cal_dirs['RGB']  / f'{fname}.npy'),
        ('CAL', 'NIR1') : arr_stats_cal(cal_dirs['NIR1'] / f'{fname}.npy'),
        ('CAL', 'NIR2') : arr_stats_cal(cal_dirs['NIR2'] / f'{fname}.npy'),
    }

    # Canvas
    cw = THUMB_W
    ch = THUMB_H
    canvas_w = BORDER + 3 * cw + 2 * GAP + BORDER
    canvas_h = BORDER + TOP_TITLE + 2 * (ROW_LABEL + ch) + GAP + BORDER
    canvas   = Image.new('RGB', (canvas_w, canvas_h), BG)
    draw     = ImageDraw.Draw(canvas)

    title = f'{args.run}  |  Frame {fnum:04d}  |  All Channels  (no stretch, no enhancement)'
    tw = draw.textlength(title, font=font_big)
    draw.text(((canvas_w - tw) / 2, BORDER + 8), title, fill=(220, 220, 220), font=font_big)

    for r_idx, row in enumerate(ROWS):
        for c_idx, col in enumerate(COLS):
            x = BORDER + c_idx * (cw + GAP)
            y = BORDER + TOP_TITLE + r_idx * (ROW_LABEL + ch + GAP)

            draw.rectangle([x, y, x + cw, y + ROW_LABEL - 2], fill=(35, 35, 45))
            draw.text((x + 4, y + 2),  row, fill=ROW_COLORS[row], font=font_sm)
            draw.text((x + 48, y + 2), col, fill=COL_COLORS[col], font=font_sm)
            draw.text((x + 4, y + 18), stats[(row, col)], fill=(155, 155, 155), font=font_sm)

            img_y = y + ROW_LABEL
            canvas.paste(panels[(row, col)], (x, img_y))
            draw.rectangle([x, img_y, x + cw - 1, img_y + ch - 1],
                           outline=COL_COLORS[col], width=2)

    out_path = out_dir / f'{args.run}_frame_{fnum:04d}_allch.png'
    canvas.save(out_path)
    print(f'  Saved: {out_path.name}')

print()
print(f'  Done. Output folder: {out_dir}')
print('=' * 65)
