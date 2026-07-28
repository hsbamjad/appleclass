"""
view_frame.py
=============
Show original vs calibrated side by side for all 3 channels.

Layout:
    TOP ROW    : Original  RGB  |  Original  NIR1  |  Original  NIR2
    BOTTOM ROW : Calibrated RGB |  Calibrated NIR1  | Calibrated NIR2

Usage:
    # Open in a window (raw shown at true 0-255 scale)
    python calibration_scripts\view_frame.py --run cals/apples_runs/run1 --frame frame_000064

    # Save as PNG
    python calibration_scripts\view_frame.py --run cals/apples_runs/run1 --frame frame_000064 --save

    # Stretch raw image contrast to see detail in dark NIR frames
    python calibration_scripts\view_frame.py --run cals/apples_runs/run1 --frame frame_000064 --stretch

    # Adjust calibrated image brightness (lower vmax = brighter, default 0.8)
    python calibration_scripts\view_frame.py --run cals/apples_runs/run1 --frame frame_000064 --vmax 0.4
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

CH_NAMES  = {0: 'RGB', 1: 'NIR1', 2: 'NIR2'}
CH_KEYS   = ['ch1', 'ch2', 'ch3']
CAL_RESULTS = Path(r'S:\MSU_Research\apple_class\calibration_results')

# Exposure used when white reference was captured (for RGB exposure-ratio correction)
EXP_WHITE = {'ch1': 2500, 'ch2': 1800, 'ch3': 2300}
# Exposure used when apple run was captured
EXP_APPLE = {'ch1': 5000, 'ch2': 1800, 'ch3': 2300}
PANEL_REFL = 0.75


def to_gray(arr):
    arr = np.array(arr, dtype=np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr


def view(run_dir, frame_name, vmax=0.8, save=False, stretch=False):
    run_dir = Path(run_dir)
    stem    = Path(frame_name).stem

    if save:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.patch.set_facecolor('#111')

    for col, ch in enumerate(CH_KEYS):
        name = CH_NAMES[col]

        # ── Original raw image ───────────────────────────────────────────────────────
        raw_path = run_dir / 'raw_frames' / ch / f'{stem}.jpg'
        if not raw_path.exists():
            print(f'Missing raw frame: {raw_path}')
            return
        raw_gray = to_gray(Image.open(raw_path))   # grayscale for stats (0-255)

        # TRUE scale by default (no stretching) so image looks as dark as it really is
        # Use --stretch to amplify contrast for dark NIR images
        if stretch:
            raw_vmin = np.percentile(raw_gray, 1)
            raw_vmax = np.percentile(raw_gray, 99)
            scale_note = f'contrast stretched ({raw_vmin:.0f}-{raw_vmax:.0f} DN)'
        else:
            raw_vmin, raw_vmax = 0, 255
            scale_note = 'true scale (0-255 DN)'

        ax = axes[0, col]
        ax.set_facecolor('#0d1117')

        if ch == 'ch1':
            # RGB channel: display in true color using the full 3-channel JPEG
            raw_rgb = np.array(Image.open(raw_path))   # shape (H, W, 3)
            if stretch:
                # Per-channel contrast stretch for visibility
                lo = np.percentile(raw_rgb, 1)
                hi = np.percentile(raw_rgb, 99)
                raw_rgb = np.clip((raw_rgb.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
            else:
                raw_rgb = (raw_rgb.astype(np.float32) / 255.0)
            im = ax.imshow(raw_rgb)   # no cmap -- matplotlib uses RGB directly
            ax.set_title(f'Original {name}  [TRUE COLOR]\nmean={raw_gray.mean():.1f} DN  '
                         f'max={raw_gray.max():.0f} DN  [{scale_note}]',
                         color='white', fontsize=10)
            ax.tick_params(colors='white')
            # No DN colorbar for color image -- add a text note instead
            ax.text(0.99, 0.01, 'RGB color (no colorbar)', transform=ax.transAxes,
                    color='#aaa', fontsize=7, ha='right', va='bottom')
        else:
            # NIR channels: single-channel, show as grayscale
            im = ax.imshow(raw_gray, cmap='gray', vmin=raw_vmin, vmax=raw_vmax)
            ax.set_title(f'Original {name}\nmean={raw_gray.mean():.1f} DN  '
                         f'max={raw_gray.max():.0f} DN  [{scale_note}]',
                         color='white', fontsize=10)
            ax.tick_params(colors='white')
            cb = plt.colorbar(im, ax=ax, fraction=0.046)
            cb.set_label('DN (raw pixel value 0-255)', color='white', fontsize=8)
            cb.ax.tick_params(colors='white')

        for sp in ax.spines.values(): sp.set_edgecolor('#444')

        # ── Calibrated display ──────────────────────────────────────────────
        cal_path = run_dir / 'calibrated_frames' / ch / f'{stem}.npy'
        if not cal_path.exists():
            print(f'Missing calibrated frame: {cal_path}')
            print(f'Run apply_calibration.py first.')
            return
        refl = np.load(cal_path)   # 0.0-1.0 grayscale (used for stats)

        apple  = ((refl > 0.05) & (refl < 0.95)).sum()
        glare  = (refl >= 0.95).sum()
        print(f'  {name}: raw_mean={raw_gray.mean():.1f} DN  '
              f'cal_mean={refl.mean():.4f}  apple_px={apple:,}  '
              f'glare_px={glare:,} ({glare/refl.size*100:.2f}%)')

        ax = axes[1, col]
        ax.set_facecolor('#0d1117')

        if ch == 'ch1':
            # Calibrate each R, G, B channel independently for true color display.
            # Uses the saved dark (H,W,3) and illumination map (H,W) from calibration_results.
            dark_map  = np.load(CAL_RESULTS / 'dark_avg_ch1.npy').astype(np.float32)   # (H,W,3)
            illum_map = np.load(CAL_RESULTS / 'illumination_map_ch1.npy').astype(np.float32)  # (H,W)
            exp_ratio = EXP_WHITE['ch1'] / EXP_APPLE['ch1']   # 2500/5000 = 0.5
            raw_rgb   = np.array(Image.open(raw_path)).astype(np.float32)  # (H,W,3)
            # Apply reflectance formula per channel
            refl_rgb = np.zeros_like(raw_rgb)
            for c in range(3):
                net = np.clip(raw_rgb[:, :, c] - dark_map[:, :, c], 0, None)
                refl_rgb[:, :, c] = np.clip(net / np.clip(illum_map, 1, None) * exp_ratio * PANEL_REFL, 0, 1)
            # Display: scale by vmax so brightness matches NIR panels
            disp_rgb = np.clip(refl_rgb / vmax, 0, 1)
            im2 = ax.imshow(disp_rgb)   # true color, no cmap
            ax.set_title(f'Calibrated {name}  [TRUE COLOR]\n'
                         f'mean={refl.mean():.4f}  max={refl.max():.4f}  '
                         f'(scale: 0.0-{vmax:.1f} reflectance)',
                         color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.text(0.99, 0.01, f'brightness scaled to vmax={vmax}', transform=ax.transAxes,
                    color='#aaa', fontsize=7, ha='right', va='bottom')
        else:
            # NIR channels: single-channel grayscale
            im2 = ax.imshow(refl, cmap='gray', vmin=0, vmax=vmax)
            ax.set_title(f'Calibrated {name}\nmean={refl.mean():.4f}  '
                         f'max={refl.max():.4f}  (scale: 0.0-1.0 reflectance)',
                         color='white', fontsize=10)
            ax.tick_params(colors='white')
            cb2 = plt.colorbar(im2, ax=ax, fraction=0.046)
            cb2.set_label('Reflectance (0=dark  0.75=panel  1=glare)',
                          color='white', fontsize=8)
            cb2.ax.tick_params(colors='white')

        for sp in ax.spines.values(): sp.set_edgecolor('#444')

    fig.suptitle(
        f'Frame: {stem}  |  TOP: Raw (DN 0-255)   BOTTOM: Calibrated reflectance (0.0-1.0)  |  '
        f'Belt=black   Apple~0.13(RGB)/0.33(NIR)   White=glare',
        color='white', fontsize=10, fontweight='bold'
    )
    fig.subplots_adjust(top=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        suffix = '_stretch' if stretch else '_compare'
        out = run_dir / 'before_after' / f'{stem}{suffix}.png'
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'\nSaved: {out}')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare original vs calibrated for all 3 channels.')
    parser.add_argument('--run',   required=True,
                        help='Run folder, e.g. cals/apples_runs/run1')
    parser.add_argument('--frame', required=True,
                        help='Frame name, e.g. frame_000064')
    parser.add_argument('--vmax',  type=float, default=0.8,
                        help='Calibrated image brightness ceiling (default 0.8). '
                             'Lower = brighter.')
    parser.add_argument('--save',  action='store_true',
                        help='Save as PNG instead of opening a window')
    parser.add_argument('--stretch', action='store_true',
                        help='Stretch raw image contrast (useful for dark NIR frames). '
                             'Default is true 0-255 scale.')
    args = parser.parse_args()

    view(args.run, args.frame, vmax=args.vmax, save=args.save, stretch=args.stretch)
