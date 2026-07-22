"""
apply_calibration.py
====================
Applies the validated calibration to apple run data.

Usage:
    python apply_calibration.py --run cals/apples_runs/run1 --rgb-exp 5000
    python apply_calibration.py --run cals/apples_runs/run1 --rgb-exp 5000 --nir1-exp 1800 --nir2-exp 2300

Outputs (inside the run folder):
    calibrated_frames/ch1/frame_XXXXXX.npy     float32 reflectance 0-1  (grayscale, backward-compat)
    calibrated_frames/ch1_rgb/frame_XXXXXX.npy float32 reflectance 0-1  (H x W x 3, per-channel R/G/B)
    calibrated_frames/ch2/frame_XXXXXX.npy
    calibrated_frames/ch3/frame_XXXXXX.npy
    before_after/frame_XXXXXX.png            side-by-side visual (best 5 frames)
    calibration_stats.csv                    per-frame statistics
    calibration_stats_summary.txt            plain-language summary
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Fixed paths (calibration maps never change unless you recapture) ──────────
CAL_DIR = Path(r'S:\MSU_Research\apple_class\calibration_results')

# White reference exposure (the exposure used when white ref grid was captured)
EXP_WHITE = {'ch1': 2500, 'ch2': 1800, 'ch3': 2300}
PANEL_REFL = 0.75
CH_NAMES   = {'ch1': 'RGB', 'ch2': 'NIR1', 'ch3': 'NIR2'}

# ─────────────────────────────────────────────────────────────────────────────

def to_gray(arr):
    arr = arr.astype(np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr

def load_calibration():
    """Load dark and illumination maps.
    Returns:
        dark     -- {ch: 2D grayscale float32}  used for ch1 grayscale + ch2/ch3
        illum    -- {ch: 2D float32}
        dark_rgb -- {'ch1': (H,W,3) float32}     3-channel dark for per-channel RGB calibration
    """
    dark, illum, dark_rgb = {}, {}, {}
    for i in [1, 2, 3]:
        ch = f'ch{i}'
        d  = np.load(CAL_DIR / f'dark_avg_ch{i}.npy')
        dark[ch]  = to_gray(d)
        illum[ch] = to_gray(np.load(CAL_DIR / f'illumination_map_ch{i}.npy'))
        if ch == 'ch1':
            # Keep the 3-channel version for per-channel RGB calibration
            dark_rgb['ch1'] = d.astype(np.float32)  # shape (H, W, 3)
    print('  Calibration maps loaded.')
    return dark, illum, dark_rgb

def calibrate_frame(raw_arr, ch, dark, illum, exp_apple):
    """Calibrate a single channel to grayscale reflectance. Returns (H, W) float32."""
    raw  = to_gray(raw_arr)
    net  = np.clip(raw - dark[ch], 0, None)
    ratio = EXP_WHITE[ch] / exp_apple[ch]
    refl = net / np.clip(illum[ch], 1, None) * ratio * PANEL_REFL
    return np.clip(refl, 0, 1).astype(np.float32)


def calibrate_frame_color(raw_arr, dark_3ch, illum_1ch, exp_apple_ch1):
    """Calibrate ch1 RGB image per-channel. Returns (H, W, 3) float32 reflectance.
    Each of R, G, B is calibrated independently using its own dark channel
    but the shared illumination map (same spatial lighting for all 3 colors).
    """
    raw = np.array(raw_arr).astype(np.float32)
    if raw.ndim == 2:          # safety: if somehow already grayscale
        raw = np.stack([raw, raw, raw], axis=2)
    ratio = EXP_WHITE['ch1'] / exp_apple_ch1
    refl_rgb = np.zeros_like(raw)
    for c in range(3):
        net = np.clip(raw[:, :, c] - dark_3ch[:, :, c], 0, None)
        refl_rgb[:, :, c] = net / np.clip(illum_1ch, 1, None) * ratio * PANEL_REFL
    return np.clip(refl_rgb, 0, 1).astype(np.float32)

def apple_mask(refl_nir1, threshold=0.05):
    return refl_nir1 > threshold

def frame_stats(raw_gray, refl, ch, mask):
    stats = {
        'raw_mean_full':  float(raw_gray.mean()),
        'raw_mean_apple': float(raw_gray[mask].mean()) if mask.sum() > 0 else float('nan'),
        'raw_std_apple':  float(raw_gray[mask].std())  if mask.sum() > 0 else float('nan'),
        'cal_mean_full':  float(refl.mean()),
        'cal_mean_apple': float(refl[mask].mean()) if mask.sum() > 0 else float('nan'),
        'cal_std_apple':  float(refl[mask].std())  if mask.sum() > 0 else float('nan'),
        'cal_min_apple':  float(refl[mask].min())  if mask.sum() > 0 else float('nan'),
        'cal_max_apple':  float(refl[mask].max())  if mask.sum() > 0 else float('nan'),
        'apple_pixels':   int(mask.sum()),
    }
    return stats

# ─────────────────────────────────────────────────────────────────────────────

def run(run_dir, exp_apple):
    run_dir = Path(run_dir)
    raw_dir = run_dir / 'raw_frames'

    out_cal    = run_dir / 'calibrated_frames'
    out_vis    = run_dir / 'before_after'
    out_cal.mkdir(exist_ok=True)
    out_vis.mkdir(exist_ok=True)
    for ch in ['ch1', 'ch1_rgb', 'ch2', 'ch3']:
        (out_cal / ch).mkdir(exist_ok=True)

    print(f'\nRun folder   : {run_dir}')
    print(f'Apple expo   : RGB={exp_apple["ch1"]}us  NIR1={exp_apple["ch2"]}us  NIR2={exp_apple["ch3"]}us')
    for ch in ['ch1','ch2','ch3']:
        r = EXP_WHITE[ch] / exp_apple[ch]
        print(f'  {CH_NAMES[ch]:4s} exposure ratio: {EXP_WHITE[ch]}/{exp_apple[ch]} = {r:.4f}')
    print()

    dark, illum, dark_rgb = load_calibration()
    frames = sorted((raw_dir / 'ch1').glob('*.jpg'))
    frame_names = [f.name for f in frames]
    print(f'  {len(frame_names)} frames found in ch1')

    # ── Pass 1: calibrate & save .npy, collect stats ──────────────────────────
    all_stats = []
    apple_pixel_counts = []

    print('\nApplying calibration and saving .npy files...')
    for i, fname in enumerate(frame_names):
        raw   = {ch: to_gray(np.array(Image.open(raw_dir / ch / fname)))
                 for ch in ['ch1','ch2','ch3']}
        refl  = {ch: calibrate_frame(np.array(Image.open(raw_dir / ch / fname)),
                                     ch, dark, illum, exp_apple)
                 for ch in ['ch1','ch2','ch3']}

        # Per-channel RGB calibration for ch1 (saves shape H x W x 3)
        raw_rgb_arr  = np.array(Image.open(raw_dir / 'ch1' / fname))
        refl_ch1_rgb = calibrate_frame_color(raw_rgb_arr, dark_rgb['ch1'],
                                             illum['ch1'], exp_apple['ch1'])

        # Save calibrated frames
        stem = Path(fname).stem
        for ch in ['ch1', 'ch2', 'ch3']:
            np.save(out_cal / ch / f'{stem}.npy', refl[ch])
        np.save(out_cal / 'ch1_rgb' / f'{stem}.npy', refl_ch1_rgb)

        # Apple mask from NIR1
        mask = apple_mask(refl['ch2'])
        apple_pixel_counts.append(int(mask.sum()))

        row = {'frame': fname, 'apple_pixels': int(mask.sum())}
        for ch in ['ch1', 'ch2', 'ch3']:
            s = frame_stats(raw[ch], refl[ch], ch, mask)
            for k, v in s.items():
                row[f'{ch}_{k}'] = v
        all_stats.append(row)

        if (i + 1) % 10 == 0 or i == len(frame_names) - 1:
            print(f'  [{i+1}/{len(frame_names)}] {fname}  apple_px={mask.sum():,}')

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = run_dir / 'calibration_stats.csv'
    fieldnames = list(all_stats[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_stats)
    print(f'\n  Stats CSV saved: {csv_path}')

    # ── Pass 2: before/after images for 5 best frames ─────────────────────────
    sorted_by_apple = sorted(range(len(frame_names)),
                             key=lambda i: apple_pixel_counts[i], reverse=True)
    best_5 = sorted_by_apple[:5]

    print('\nSaving before/after visuals for 5 best frames...')
    for idx in best_5:
        fname = frame_names[idx]
        stem  = Path(fname).stem

        raw_imgs     = {ch: to_gray(np.array(Image.open(raw_dir / ch / fname)))
                        for ch in ['ch1','ch2','ch3']}
        raw_rgb_img  = np.array(Image.open(raw_dir / 'ch1' / fname))  # (H,W,3) true color
        cal_imgs     = {ch: np.load(out_cal / ch / f'{stem}.npy')
                        for ch in ['ch1','ch2','ch3']}
        cal_rgb_img  = np.load(out_cal / 'ch1_rgb' / f'{stem}.npy')   # (H,W,3) color reflectance
        mask         = apple_mask(cal_imgs['ch2'])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#1a1a2e')

        for col, ch in enumerate(['ch1', 'ch2', 'ch3']):
            am = cal_imgs[ch][mask].mean() if mask.sum() > 0 else float('nan')

            # RAW row
            ax = axes[0, col]
            ax.set_facecolor('#0d1117')
            if ch == 'ch1':
                # True color for RGB raw
                disp_raw = raw_rgb_img.astype(np.float32) / 255.0
                im = ax.imshow(disp_raw)
                ax.set_title(f'{CH_NAMES[ch]} -- RAW  [TRUE COLOR]\n'
                             f'mean(apple)={raw_imgs[ch][mask].mean():.1f} DN  '
                             f'(true 0-255 scale)',
                             color='white', fontsize=10)
                ax.text(0.99, 0.01, 'RGB color (no colorbar)', transform=ax.transAxes,
                        color='#aaa', fontsize=7, ha='right', va='bottom')
            else:
                raw_g = raw_imgs[ch]
                im = ax.imshow(raw_g, cmap='gray', vmin=0, vmax=255)
                ax.set_title(f'{CH_NAMES[ch]} -- RAW\n'
                             f'mean(apple)={raw_g[mask].mean():.1f} DN  '
                             f'(true 0-255 scale)',
                             color='white', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(colors='white')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_edgecolor('#444')

            # CALIBRATED row
            ax = axes[1, col]
            ax.set_facecolor('#0d1117')
            if ch == 'ch1':
                # True color for RGB calibrated (scale to vmax=0.8 for visibility)
                disp_cal = np.clip(cal_rgb_img / 0.8, 0, 1)
                im2 = ax.imshow(disp_cal)
                ax.set_title(f'{CH_NAMES[ch]} -- CALIBRATED  [TRUE COLOR]\n'
                             f'mean reflectance(apple)={am:.4f}  '
                             f'(0=black  0.75=panel  1=mirror)',
                             color='white', fontsize=10)
                ax.text(0.99, 0.01, 'brightness scaled to 0.8', transform=ax.transAxes,
                        color='#aaa', fontsize=7, ha='right', va='bottom')
            else:
                cal_g = cal_imgs[ch]
                im2 = ax.imshow(cal_g, cmap='gray', vmin=0, vmax=0.8)
                ax.set_title(f'{CH_NAMES[ch]} -- CALIBRATED\n'
                             f'mean reflectance(apple)={am:.4f}  '
                             f'(0=black  0.75=panel  1=mirror)',
                             color='white', fontsize=10)
                cb = plt.colorbar(im2, ax=ax, fraction=0.046)
                cb.ax.tick_params(colors='white')
                cb.set_label('Reflectance', color='white', fontsize=8)
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_edgecolor('#444')

        fig.suptitle(
            f'Frame: {fname}  |  Apple pixels: {mask.sum():,}  |  '
            f'TOP: Raw (DN 0-255)   BOTTOM: Calibrated reflectance (0-1)  |  '
            f'RGB ratio: {EXP_WHITE["ch1"]}/{exp_apple["ch1"]}={EXP_WHITE["ch1"]/exp_apple["ch1"]:.2f}  NIR ratio: 1.0',
            color='white', fontsize=10, fontweight='bold'
        )
        fig.subplots_adjust(top=0.94)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = out_vis / f'{stem}_before_after.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f'  Saved {out_path.name}')

    # ── Plain-language summary ─────────────────────────────────────────────────
    best_idx   = sorted_by_apple[0]
    best_stats = all_stats[best_idx]

    lines = [
        '=' * 60,
        '  CALIBRATION APPLIED — run1 Summary',
        '=' * 60,
        '',
        f'  Frames processed       : {len(frame_names)}',
        f'  Best frame (most apple): {frame_names[best_idx]}',
        f'  Apple pixels in best   : {apple_pixel_counts[best_idx]:,}',
        '',
        '  BEST FRAME — Apple region reflectance',
        '-' * 60,
    ]
    for ch in ['ch1', 'ch2', 'ch3']:
        rm = best_stats[f'{ch}_raw_mean_apple']
        cm = best_stats[f'{ch}_cal_mean_apple']
        cs = best_stats[f'{ch}_cal_std_apple']
        cmin = best_stats[f'{ch}_cal_min_apple']
        cmax = best_stats[f'{ch}_cal_max_apple']
        lines += [
            f'  {CH_NAMES[ch]:4s}',
            f'    Raw mean (apple)       : {rm/255:.4f}  ({rm:.1f} DN)',
            f'    Calibrated mean refl   : {cm:.4f}',
            f'    Calibrated std         : {cs:.4f}',
            f'    Calibrated range       : {cmin:.4f} – {cmax:.4f}',
            '',
        ]
    lines += [
        '  CALIBRATED FILES SAVED TO',
        '-' * 60,
        f'  {out_cal}',
        f'    ch1/     RGB  reflectance .npy  -- grayscale (H x W), backward-compatible',
        f'    ch1_rgb/ RGB  reflectance .npy  -- true color (H x W x 3), R/G/B calibrated separately',
        f'    ch2/     NIR1 reflectance .npy  -- grayscale (H x W)',
        f'    ch3/     NIR2 reflectance .npy  -- grayscale (H x W)',
        '',
        '  BEFORE/AFTER VISUALS',
        '-' * 60,
        f'  {out_vis}',
        '    (5 best frames with most apple visible)',
        '',
        '  HOW TO LOAD A CALIBRATED FRAME IN PYTHON',
        '-' * 60,
        '  import numpy as np',
        '  refl = np.load("calibrated_frames/ch1/frame_000064.npy")',
        '  # refl is a 2D float32 array, shape (1536, 2048)',
        '  # values: 0.0 = completely dark, 1.0 = perfect mirror',
        '  # apple tissue is typically 0.05 – 0.50',
        '=' * 60,
    ]

    summary = '\n'.join(lines)
    with open(run_dir / 'calibration_stats_summary.txt', 'w') as f:
        f.write(summary)
    print()
    print(summary)

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply calibration to an apple run.')
    parser.add_argument('--run',      required=True, help='Path to run folder, e.g. cals/apples_runs/run1')
    parser.add_argument('--rgb-exp',  type=int, default=5000, help='RGB exposure used for apple capture (us)')
    parser.add_argument('--nir1-exp', type=int, default=1800, help='NIR1 exposure (us)')
    parser.add_argument('--nir2-exp', type=int, default=2300, help='NIR2 exposure (us)')
    args = parser.parse_args()

    exp_apple = {'ch1': args.rgb_exp, 'ch2': args.nir1_exp, 'ch3': args.nir2_exp}
    run(args.run, exp_apple)
