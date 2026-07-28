"""
visualize_results.py
Generates raw vs calibrated comparison figures for all 4 formal runs.
NIR calibrated panels are scaled to the apple pixel range (belt masked black).
Saves to: formal_runs/visualization/
"""
import csv, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r'S:\MSU_Research\apple_class\formal_runs')
OUT  = BASE / 'visualization'
OUT.mkdir(exist_ok=True)

BG  = '#0d1117'
BG2 = '#111827'

# Best apple frames per run (highest NIR1 center reflectance)
BEST = {1: 62, 2: 70, 3: 73, 4: 67}


def dark_style(ax):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.xaxis.label.set_color('#aaa')
    ax.yaxis.label.set_color('#aaa')
    ax.title.set_color('white')


# FIG 1: Raw vs Calibrated per run
print('Generating raw vs calibrated figures...')

for run_i in range(1, 5):
    fidx = BEST[run_i]

    raw_ch1  = sorted((BASE / f'apples_run{run_i}_procc/raw_frames/ch1').glob('*.jpg'))
    raw_ch2  = sorted((BASE / f'apples_run{run_i}_procc/raw_frames/ch2').glob('*.jpg'))
    cal_rgb  = sorted((BASE / f'apples_run{run_i}_procc/calibrated/ch1_rgb').glob('*.npy'))
    cal_nir1 = sorted((BASE / f'apples_run{run_i}_procc/calibrated/ch2').glob('*.npy'))
    cal_nir2 = sorted((BASE / f'apples_run{run_i}_procc/calibrated/ch3').glob('*.npy'))

    raw_img = np.array(Image.open(raw_ch1[fidx]))
    raw_n1  = np.array(Image.open(raw_ch2[fidx])).astype(np.float32)
    if raw_n1.ndim == 3:
        raw_n1 = raw_n1.mean(axis=2)

    c_rgb  = np.load(cal_rgb[fidx])
    c_nir1 = np.load(cal_nir1[fidx])
    c_nir2 = np.load(cal_nir2[fidx])

    # Apple mask from RGB calibrated
    rgb_gray   = c_rgb.mean(axis=2)
    apple_mask = rgb_gray > 0.04

    # Scale NIR display to apple pixel range (2nd to 98th percentile)
    nir1_ap = c_nir1[apple_mask]
    nir2_ap = c_nir2[apple_mask]
    vmin_n1, vmax_n1 = np.percentile(nir1_ap, 2), np.percentile(nir1_ap, 98)
    vmin_n2, vmax_n2 = np.percentile(nir2_ap, 2), np.percentile(nir2_ap, 98)
    clipped_n1 = (nir1_ap >= 0.999).sum() / len(nir1_ap) * 100
    clipped_n2 = (nir2_ap >= 0.999).sum() / len(nir2_ap) * 100

    # Mask belt pixels to black in NIR display
    c_nir1_disp = c_nir1.copy()
    c_nir1_disp[~apple_mask] = 0
    c_nir2_disp = c_nir2.copy()
    c_nir2_disp[~apple_mask] = 0
    cal_rgb_disp = (np.clip(c_rgb, 0, 1) * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
    fig.suptitle(
        f'Run {run_i} | Frame {fidx + 1} | Best Apple Frame\n'
        f'Top: Raw camera output | Bottom: Calibrated reflectance'
        f' (NIR scaled to apple range, belt masked)',
        color='white', fontsize=11, fontweight='bold', y=0.99
    )

    # Row 1: raw
    ax = axes[0, 0]; dark_style(ax)
    ax.imshow(raw_img, aspect='auto')
    ax.set_title('RGB raw (real color)', fontsize=10)
    ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')

    ax = axes[0, 1]; dark_style(ax)
    ax.imshow(raw_n1, cmap='gray', vmin=0, vmax=255, aspect='auto')
    ax.set_title(f'NIR1 ~800nm raw (grayscale)\nmax DN={raw_n1.max():.0f}', fontsize=10)
    ax.set_xlabel('pixel x')

    ax = axes[0, 2]; dark_style(ax)
    ax.imshow(raw_img, aspect='auto')
    ax.set_title('RGB raw (real color) -- repeated for layout', fontsize=10)
    ax.imshow(cal_rgb_disp, aspect='auto')
    ax.set_xlabel('pixel x')

    # Row 2: calibrated
    ax = axes[1, 0]; dark_style(ax)
    ax.imshow(cal_rgb_disp, aspect='auto')
    ax.set_title(
        f'RGB calibrated (real color)\nrange: {c_rgb.min():.3f} to {c_rgb.max():.3f}',
        fontsize=10)
    ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')

    ax = axes[1, 1]; dark_style(ax)
    im = ax.imshow(c_nir1_disp, cmap='gray', vmin=vmin_n1, vmax=vmax_n1, aspect='auto')
    ax.set_title(
        f'NIR1 ~800nm calibrated (apple range only)\n'
        f'range: {vmin_n1:.3f} to {vmax_n1:.3f} | clipped: {clipped_n1:.3f}%',
        fontsize=9)
    ax.set_xlabel('pixel x')
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label('Reflectance', color='#aaa', fontsize=8)
    cb.ax.tick_params(colors='#aaa', labelsize=7)

    ax = axes[1, 2]; dark_style(ax)
    im = ax.imshow(c_nir2_disp, cmap='gray', vmin=vmin_n2, vmax=vmax_n2, aspect='auto')
    ax.set_title(
        f'NIR2 ~900nm calibrated (apple range only)\n'
        f'range: {vmin_n2:.3f} to {vmax_n2:.3f} | clipped: {clipped_n2:.3f}%',
        fontsize=9)
    ax.set_xlabel('pixel x')
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label('Reflectance', color='#aaa', fontsize=8)
    cb.ax.tick_params(colors='#aaa', labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT / f'run{run_i}_real_color.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved: {out_path.name}')


# FIG 2: Reflectance timeline all runs
print('\nGenerating reflectance timeline...')

fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor=BG)
fig.suptitle(
    'Calibrated Reflectance Timeline | All Frames | All Runs\n'
    'Flat = empty belt | Peaks = apples on belt',
    color='white', fontsize=12, fontweight='bold'
)

CH_COLORS = {'ch1_refl': '#f4a261', 'ch2_refl': '#a8dadc', 'ch3_refl': '#6d9dc5'}
CH_NAMES  = {'ch1_refl': 'RGB',     'ch2_refl': 'NIR1',    'ch3_refl': 'NIR2'}

for run_i in range(1, 5):
    ax = axes[run_i - 1]
    dark_style(ax)
    csv_path = BASE / f'apples_run{run_i}_procc' / 'calibration_stats.csv'
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    frames = [int(r['frame']) for r in rows]
    for ch in ['ch1_refl', 'ch2_refl', 'ch3_refl']:
        vals = [float(r[ch]) for r in rows]
        ax.plot(frames, vals, color=CH_COLORS[ch], lw=1.3, label=CH_NAMES[ch], alpha=0.9)
    ax.axhline(0.024, color='#555', lw=0.8, linestyle=':', label='Belt ref')
    ax.set_title(f'Run {run_i} ({len(rows)} frames)', color='white', fontsize=10)
    ax.set_xlabel('Frame', color='#aaa', fontsize=9)
    ax.set_ylabel('Center reflectance', color='#aaa', fontsize=9)
    ax.set_ylim(0, 0.30)
    ax.legend(fontsize=8, labelcolor='white', facecolor=BG, edgecolor='#333')

plt.tight_layout(rect=[0, 0, 1, 0.90])
out4 = OUT / 'timeline_all_runs.png'
fig.savefig(out4, dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print(f'  Saved: {out4.name}')

print(f'\nAll figures saved to: {OUT}')
