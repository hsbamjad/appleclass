"""
calibration_pipeline.py
=======================
Full spatial calibration analysis for MSU Multispectral Apple Grading System.

Reads:
  final_runs/Black/raw_frames/ch[1-3]/   - dark frames (lens covered)
  final_runs/white_[pos]/raw_frames/ch[1-3]/  - 9-position white reference

Produces:
  results/
    dark_avg_ch[1-3].npy       - averaged dark frames
    white_avg_[pos]_ch[1-3].npy- averaged white frames per position
    illumination_map_ch[1-3].npy- full-res 2D illumination map (interpolated)
    correction_map_ch[1-3].npy  - correction factor per pixel
    fig1_dark_frames.png
    fig2_white_grid.png
    fig3_illumination_profiles.png
    fig4_correction_map.png
    fig5_calibration_impact.png
    fig6_grid_stitched.png
    fig7_spline_concept.png
    calibration_report.txt
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────

BASE    = Path(r'S:\MSU_Research\apple_class\cals\final_runs')
RESULTS = Path(r'S:\MSU_Research\apple_class\calibration_results')
RESULTS.mkdir(exist_ok=True)

# Grid positions: row 0=Upper, row 1=Middle, row 2=Lower
#                col 0=Left,  col 1=Center, col 2=Right
POSITIONS = [
    ['white_UL', 'white_UC', 'white_UR'],
    ['white_ML', 'white_C',  'white_MR'],
    ['white_LL', 'white_LC', 'white_LR'],
]
POSITION_LABELS = [
    ['Upper-Left', 'Upper-Center', 'Upper-Right'],
    ['Mid-Left',   'Center',       'Mid-Right'],
    ['Lower-Left', 'Lower-Center', 'Lower-Right'],
]

CH_NAMES  = {1: 'RGB (CH1)', 2: 'NIR1 (CH2)', 3: 'NIR2 (CH3)'}
CH_COLORS = {1: '#e85d04', 2: '#7209b7', 3: '#0077b6'}
CH_CMAPS  = {1: 'Oranges', 2: 'Purples', 3: 'Blues'}

# Spectralon certified reflectance
PANEL_REFLECTANCE = 0.75

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def load_and_average(folder: Path, ch: int, max_frames: int = 50) -> np.ndarray:
    """Load all frames from ch[N] subfolder and return float32 average."""
    ch_folder = folder / 'raw_frames' / f'ch{ch}'
    if not ch_folder.exists():
        raise FileNotFoundError(f'Channel folder not found: {ch_folder}')
    frames = sorted(ch_folder.glob('*.jpg'))[:max_frames]
    if not frames:
        raise FileNotFoundError(f'No JPG frames in {ch_folder}')
    stack = []
    for f in frames:
        img = np.array(Image.open(f)).astype(np.float32)
        stack.append(img)
    avg = np.mean(stack, axis=0)
    print(f'  Loaded {len(stack):3d} frames from {ch_folder.parent.parent.name}/ch{ch}')
    return avg


def gray(img: np.ndarray) -> np.ndarray:
    """Return 2D grayscale from an image (average RGB channels or pass mono through)."""
    if img.ndim == 3:
        return img.mean(axis=2)
    return img.astype(float)


def region_mean(img: np.ndarray, grid_row: int, grid_col: int) -> float:
    """
    Measure the mean pixel value in the (grid_row, grid_col) sub-region of the image.
    The image is divided into a 3x3 grid of equal thirds.
    grid_row: 0=Upper, 1=Middle, 2=Lower
    grid_col: 0=Left,  1=Center, 2=Right
    Takes inner 60% of each sub-region to avoid panel-edge effects.
    """
    g = gray(img)
    H, W = g.shape
    row_bounds = [(0, H//3), (H//3, 2*H//3), (2*H//3, H)]
    col_bounds = [(0, W//3), (W//3, 2*W//3), (2*W//3, W)]
    r0, r1 = row_bounds[grid_row]
    c0, c1 = col_bounds[grid_col]
    rh = r1 - r0
    cw = c1 - c0
    r_pad = rh // 5
    c_pad = cw // 5
    patch = g[r0+r_pad : r1-r_pad, c0+c_pad : c1-c_pad]
    return float(patch.mean())


def center_mean(img: np.ndarray, frac: float = 0.20) -> float:
    """Mean value of the centre crop (frac of each side). Used for dark frames only."""
    H, W = gray(img).shape
    cy, cx = H//2, W//2
    my, mx = int(H*frac), int(W*frac)
    patch = gray(img)[cy-my:cy+my, cx-mx:cx+mx]
    return float(patch.mean())


def clip_pct(img: np.ndarray, thresh: int = 250) -> float:
    return float((gray(img) >= thresh).sum() / gray(img).size * 100)


def stats_str(img: np.ndarray, label: str) -> str:
    g = gray(img)
    return (f'{label:20s}  mean={g.mean():6.1f}  std={g.std():5.1f}  '
            f'min={g.min():5.1f}  max={g.max():5.1f}  clip={clip_pct(img):.2f}%')


# ─────────────────────────────────────────────────────────────────
#  STEP 1: LOAD DARK FRAMES
# ─────────────────────────────────────────────────────────────────

print('\n' + '='*65)
print('  STEP 1: Dark Frame Analysis')
print('='*65)

dark_folder = BASE / 'Black'
dark = {}
for ch in [1, 2, 3]:
    dark[ch] = load_and_average(dark_folder, ch)
    np.save(RESULTS / f'dark_avg_ch{ch}.npy', dark[ch])
    print(f'  {stats_str(dark[ch], CH_NAMES[ch])}')


# ─────────────────────────────────────────────────────────────────
#  STEP 2: LOAD WHITE REFERENCE FRAMES
# ─────────────────────────────────────────────────────────────────

print('\n' + '='*65)
print('  STEP 2: White Reference Grid Analysis')
print('='*65)

white = {}       # white[row][col][ch] = averaged image
white_mean = {}  # white_mean[ch] = 3x3 array of center means
dark_mean  = {}  # dark_mean[ch] = scalar

for ch in [1, 2, 3]:
    dark_mean[ch] = center_mean(dark[ch])
    white_mean[ch] = np.zeros((3, 3))
    white[ch] = [[None]*3 for _ in range(3)]

for row in range(3):
    for col in range(3):
        pos_name = POSITIONS[row][col]
        pos_folder = BASE / pos_name
        label = POSITION_LABELS[row][col]
        print(f'\n  Position: {label} ({pos_name})  -> measuring image region [{["Upper","Middle","Lower"][row]}-{["Left","Center","Right"][col]}]')
        for ch in [1, 2, 3]:
            w = load_and_average(pos_folder, ch)
            white[ch][row][col] = w
            np.save(RESULTS / f'white_avg_{pos_name}_ch{ch}.npy', w)
            # KEY FIX: measure the 1/3 region matching where the panel was placed
            rm = region_mean(w, row, col)
            dk = dark_mean[ch]
            white_mean[ch][row, col] = rm
            net = rm - dk
            print(f'    {CH_NAMES[ch]:12s} region mean={rm:.1f}  dark={dk:.1f}  net={net:.1f}  clip={clip_pct(w):.2f}%')


# ─────────────────────────────────────────────────────────────────
#  STEP 3: BUILD ILLUMINATION MAP (INTERPOLATED)
# ─────────────────────────────────────────────────────────────────

print('\n' + '='*65)
print('  STEP 3: Building Full-Resolution Illumination Maps')
print('='*65)

# Image dimensions from first white frame
H, W = gray(white[1][0][0]).shape

# Grid coordinates normalised to [0,1]
rows_norm = np.array([0.0, 0.5, 1.0])
cols_norm = np.array([0.0, 0.5, 1.0])

# Target full-resolution grid
x_full = np.linspace(0, 1, W)
y_full = np.linspace(0, 1, H)

illum_map  = {}
correct_map = {}

for ch in [1, 2, 3]:
    # Dark-corrected 3x3 net white values
    net_grid = white_mean[ch] - dark_mean[ch]

    # Fit a smooth 2D surface using bicubic spline (kx=2, ky=2 = quadratic OK for 3x3)
    spline = RectBivariateSpline(rows_norm, cols_norm, net_grid, kx=2, ky=2)
    illum = spline(y_full, x_full).astype(np.float32)
    illum = np.clip(illum, 1, None)   # avoid division by zero

    illum_map[ch]   = illum
    np.save(RESULTS / f'illumination_map_ch{ch}.npy', illum)

    # Correction map: divide any pixel by this to get reflectance-equivalent
    # (normalised by center value, so correction = illum / center_illum)
    center_val  = float(net_grid[1, 1])   # middle of the 3x3 grid
    correct_map[ch] = illum / center_val
    np.save(RESULTS / f'correction_map_ch{ch}.npy', correct_map[ch])

    vmin = illum.min(); vmax = illum.max()
    variation_pct = (vmax - vmin) / vmax * 100
    print(f'  {CH_NAMES[ch]:12s}  illumination range: {vmin:.1f} - {vmax:.1f}  '
          f'variation: {variation_pct:.1f}%  center: {center_val:.1f}')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 1: Dark Frame Maps
# ─────────────────────────────────────────────────────────────────

print('\n  Generating Figure 1: Dark Frame Maps ...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#1a1a2e')

for i, ch in enumerate([1, 2, 3]):
    g = gray(dark[ch])
    im = axes[i].imshow(g, cmap='hot', vmin=0, vmax=30)
    axes[i].set_title(f'{CH_NAMES[ch]}\nMean: {g.mean():.1f} DN  Max: {g.max():.0f} DN',
                       color='white', fontsize=11, pad=8)
    axes[i].tick_params(colors='white')
    for sp in axes[i].spines.values():
        sp.set_edgecolor('#444')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04).ax.tick_params(colors='white')

fig.suptitle('Dark Frame Analysis  |  Lens Covered, Same Exposure Settings',
             color='white', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / 'fig1_dark_frames.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('  Saved fig1_dark_frames.png')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 2: White Reference 3x3 Grid Heatmap
# ─────────────────────────────────────────────────────────────────

print('  Generating Figure 2: White Reference Grid ...')

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#1a1a2e')
row_labels = ['Upper', 'Middle', 'Lower']
col_labels = ['Left', 'Center', 'Right']

outer = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

for ch_idx, ch in enumerate([1, 2, 3]):
    for row in range(3):
        for col in range(3):
            ax = fig.add_subplot(outer[row, col]) if ch_idx == 0 else None

# Cleaner approach: 3 channels, each with a 3x3 table of numbers
# Use a big grid: 3 rows of channels, each row has a 3x3 annotation heatmap

fig, big_axes = plt.subplots(3, 1, figsize=(12, 14))
fig.patch.set_facecolor('#1a1a2e')

for i, ch in enumerate([1, 2, 3]):
    ax = big_axes[i]
    ax.set_facecolor('#0d1117')
    net_grid = white_mean[ch] - dark_mean[ch]
    im = ax.imshow(net_grid, cmap=CH_CMAPS[ch], aspect='auto',
                   vmin=net_grid.min()*0.9, vmax=net_grid.max()*1.05)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(col_labels, color='white', fontsize=11)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(row_labels, color='white', fontsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Annotate cells - adaptive text color: dark text on bright cells, white on dark
    vmin_g = net_grid.min(); vmax_g = net_grid.max()
    for row in range(3):
        for col in range(3):
            val = net_grid[row, col]
            pct = (val - net_grid[1, 1]) / net_grid[1, 1] * 100
            # Normalise brightness 0-1 to decide text color
            brightness = (val - vmin_g) / (vmax_g - vmin_g + 1e-6)
            txt_color = '#111111' if brightness > 0.55 else 'white'
            # Add outline/shadow for readability on mid-tone cells
            ax.text(col, row, f'{val:.0f}\n({pct:+.1f}%)',
                    ha='center', va='center', color=txt_color,
                    fontsize=11, fontweight='bold',
                    path_effects=[
                        __import__('matplotlib.patheffects', fromlist=['withStroke'])
                        .withStroke(linewidth=2,
                                    foreground='white' if txt_color == '#111111' else '#000000')
                    ])

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors='white')
    ax.set_title(f'{CH_NAMES[ch]}  |  Net Illumination (DN) per Belt Position  '
                 f'[dark-corrected]',
                 color='white', fontsize=12, pad=8)

fig.suptitle('White Reference Grid  |  9-Position Illumination Map\n'
             '(values show sensor response to Spectralon 75% panel; '
             '% relative to Center)',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RESULTS / 'fig2_white_grid.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('  Saved fig2_white_grid.png')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 3: Illumination Profiles (horizontal + vertical cross-sections)
# ─────────────────────────────────────────────────────────────────

print('  Generating Figure 3: Illumination Profiles ...')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#1a1a2e')

pos_labels_h = ['Left', 'Center', 'Right']
pos_labels_v = ['Upper', 'Middle', 'Lower']
x3 = [0, 1, 2]

for i, ch in enumerate([1, 2, 3]):
    net_grid = white_mean[ch] - dark_mean[ch]
    color = CH_COLORS[ch]

    # Horizontal (row by row)
    ax = axes[0, i]
    ax.set_facecolor('#0d1117')
    for row, rl in enumerate(row_labels):
        alpha = 1.0 - row * 0.2
        ax.plot(x3, net_grid[row, :], 'o-', color=color, alpha=alpha,
                linewidth=2, markersize=8, label=rl)
        for x, v in zip(x3, net_grid[row, :]):
            ax.annotate(f'{v:.0f}', (x, v), textcoords='offset points',
                        xytext=(0, 8), ha='center', color='white', fontsize=8)
    ax.set_xticks(x3)
    ax.set_xticklabels(pos_labels_h, color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'{CH_NAMES[ch]}\nHorizontal Profile', color='white', fontsize=10)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e',
              edgecolor='#444')
    ax.set_ylabel('Net DN (dark-corrected)', color='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

    # Vertical (column by column)
    ax = axes[1, i]
    ax.set_facecolor('#0d1117')
    for col, cl in enumerate(col_labels):
        alpha = 1.0 - col * 0.2
        ax.plot(x3, net_grid[:, col], 's--', color=color, alpha=alpha,
                linewidth=2, markersize=8, label=cl)
        for y, v in zip(x3, net_grid[:, col]):
            ax.annotate(f'{v:.0f}', (y, v), textcoords='offset points',
                        xytext=(0, 8), ha='center', color='white', fontsize=8)
    ax.set_xticks(x3)
    ax.set_xticklabels(pos_labels_v, color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'Vertical Profile', color='white', fontsize=10)
    ax.legend(fontsize=8, labelcolor='white', facecolor='#1a1a2e',
              edgecolor='#444')
    ax.set_ylabel('Net DN (dark-corrected)', color='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')

fig.suptitle('Illumination Uniformity Analysis  |  Horizontal and Vertical Gradients',
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(RESULTS / 'fig3_illumination_profiles.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('  Saved fig3_illumination_profiles.png')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 4: Full-Resolution Correction Maps
# ─────────────────────────────────────────────────────────────────

print('  Generating Figure 4: Correction Maps ...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#1a1a2e')

for i, ch in enumerate([1, 2, 3]):
    ax = axes[i]
    ax.set_facecolor('#0d1117')
    cmap = correct_map[ch]
    # Display: values < 1 = under-illuminated vs center, > 1 = over-illuminated
    im = ax.imshow(cmap, cmap='RdYlGn', vmin=0.85, vmax=1.15, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Correction factor\n(1.0 = same as center)', color='white')

    vmin_c = cmap.min(); vmax_c = cmap.max()
    ax.set_title(f'{CH_NAMES[ch]}\n'
                 f'Range: {vmin_c:.3f} - {vmax_c:.3f}  '
                 f'(±{(vmax_c-vmin_c)/2*100:.1f}%)',
                 color='white', fontsize=11)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    # Label axes
    ax.set_xlabel('Belt Width  (pixels)', color='white')
    ax.set_ylabel('Belt Length (pixels)', color='white')

fig.suptitle('Spatial Illumination Correction Map  |  Per-Pixel Calibration Factors\n'
             'Green = same illumination as center | Red = less light | Yellow = more light',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(RESULTS / 'fig4_correction_map.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('  Saved fig4_correction_map.png')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 5: Calibration Impact - before vs after
# ─────────────────────────────────────────────────────────────────

print('  Generating Figure 5: Calibration Impact Demonstration ...')

# Use center white reference image - evaluate ONLY the center panel region
# (the full image includes belt background which distorts CoV)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')

for i, ch in enumerate([1, 2, 3]):
    raw_full  = gray(white[ch][1][1]).astype(np.float32)   # center capture
    dk_full   = gray(dark[ch]).astype(np.float32)
    im_full   = illum_map[ch]

    # Calibrated reflectance on full image
    net_full = np.clip(raw_full - dk_full, 0, None)
    refl_full = np.clip(net_full / im_full * PANEL_REFLECTANCE, 0, 1)

    # Extract panel region (middle-center 1/3 of image) for statistics
    H, W = raw_full.shape
    r0, r1 = H//3, 2*H//3
    c0, c1 = W//3, 2*W//3
    rpad = (r1-r0)//5;  cpad = (c1-c0)//5
    raw_panel  = raw_full[r0+rpad:r1-rpad, c0+cpad:c1-cpad]
    refl_panel = refl_full[r0+rpad:r1-rpad, c0+cpad:c1-cpad]

    cov_raw  = raw_panel.std()  / (raw_panel.mean()  + 1e-6) * 100
    cov_cal  = refl_panel.std() / (refl_panel.mean() + 1e-6) * 100
    improvement = (cov_raw - cov_cal) / (cov_raw + 1e-6) * 100

    ax_raw  = axes[0, i]
    ax_cal  = axes[1, i]
    ax_raw.set_facecolor('#0d1117')
    ax_cal.set_facecolor('#0d1117')

    # Normalise for display
    raw_disp  = (raw_full  - raw_full.min())  / (raw_full.max()  - raw_full.min() + 1e-6)
    refl_disp = refl_full

    im1 = ax_raw.imshow(raw_disp, cmap='gray', vmin=0, vmax=1)
    ax_raw.set_title(f'{CH_NAMES[ch]}\nUncalibrated (raw panel region)\n'
                     f'CoV={cov_raw:.1f}%  mean={raw_panel.mean():.0f} DN',
                     color='white', fontsize=10)

    im2 = ax_cal.imshow(refl_disp, cmap='gray', vmin=0, vmax=1)
    ax_cal.set_title(f'Calibrated Reflectance (panel region)\n'
                     f'CoV={cov_cal:.1f}%  mean refl={refl_panel.mean():.4f}\n'
                     f'Uniformity improvement: {improvement:.1f}%',
                     color='white', fontsize=10)

    for ax, im_obj in [(ax_raw, im1), (ax_cal, im2)]:
        ax.tick_params(colors='white')
        # Draw rectangle showing panel region evaluated
        import matplotlib.patches as patches
        rect = patches.Rectangle((c0+cpad, r0+rpad),
                                  (c1-c0-2*cpad), (r1-r0-2*rpad),
                                  linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        plt.colorbar(im_obj, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='white')

fig.suptitle('Calibration Impact: Before vs After  (green box = panel evaluation region)\n'
             'Spectralon 75% panel should appear uniform in the green region after calibration.\n'
             'CoV = Coefficient of Variation: lower = more uniform = better calibration',
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(RESULTS / 'fig5_calibration_impact.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('  Saved fig5_calibration_impact.png')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 6: 9-Position White Reference Photos (Stitched Grid)
# ─────────────────────────────────────────────────────────────────

print('  Generating Figure 6: 9-Position White Reference Photo Grid ...')

import matplotlib.patches as fig6_patches

fig6, axes6 = plt.subplots(3, 3, figsize=(18, 12))
fig6.patch.set_facecolor('#1a1a2e')

center_net = {}  # needed for % annotations
for ch in [1, 2, 3]:
    center_net[ch] = white_mean[ch][1, 1] - dark_mean[ch]

row_labels6 = ['Upper', 'Middle', 'Lower']
col_labels6 = ['Left', 'Center', 'Right']

for row in range(3):
    for col in range(3):
        ax = axes6[row, col]
        ax.set_facecolor('#0d1117')

        # Use NIR1 (ch2) for clearest grayscale -- best spatial contrast
        img_arr = gray(white[2][row][col]).astype(np.float32)

        ax.imshow(img_arr, cmap='gray', vmin=0, vmax=255, aspect='auto')

        # Draw green rectangle showing the sampled region for this position
        img_H, img_W = img_arr.shape
        r0b = [0, img_H // 3, 2 * img_H // 3][row]
        r1b = [img_H // 3, 2 * img_H // 3, img_H][row]
        c0b = [0, img_W // 3, 2 * img_W // 3][col]
        c1b = [img_W // 3, 2 * img_W // 3, img_W][col]
        rh6 = r1b - r0b
        cw6 = c1b - c0b
        rpad6 = rh6 // 5
        cpad6 = cw6 // 5
        rect6 = fig6_patches.Rectangle(
            (c0b + cpad6, r0b + rpad6),
            cw6 - 2 * cpad6, rh6 - 2 * rpad6,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect6)

        net_val = white_mean[2][row, col] - dark_mean[2]
        pct = (net_val - center_net[2]) / (center_net[2] + 1e-6) * 100
        ax.set_title(
            f'{row_labels6[row]}-{col_labels6[col]}\n'
            f'Net illumination: {net_val:.0f} DN  ({pct:+.0f}% vs Center)',
            color='white', fontsize=10
        )
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

fig6.suptitle(
    'White Reference Grid -- 9 Positions (NIR1 Channel)\n'
    'Green box = region sampled for illumination measurement\n'
    'Net DN = dark-corrected average inside green box',
    color='white', fontsize=13, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig6.savefig(RESULTS / 'fig6_grid_stitched.png', dpi=130,
             bbox_inches='tight', facecolor=fig6.get_facecolor())
plt.close()
print('  Saved fig6_grid_stitched.png')


# ─────────────────────────────────────────────────────────────────
#  FIGURE 7: Spline Interpolation Concept (3-Panel)
# ─────────────────────────────────────────────────────────────────

print('  Generating Figure 7: Spline Interpolation Concept ...')

# Use NIR1 (ch2) net grid for the illustration
net_grid7 = white_mean[2] - dark_mean[2]
rows_norm7 = np.array([0.0, 0.5, 1.0])
cols_norm7 = np.array([0.0, 0.5, 1.0])

y_dense = np.linspace(0, 1, 200)
x_dense = np.linspace(0, 1, 200)
spline7  = RectBivariateSpline(rows_norm7, cols_norm7, net_grid7, kx=2, ky=2)
surface7 = spline7(y_dense, x_dense)

row_labels7 = ['Upper', 'Middle', 'Lower']
col_labels7 = ['Left', 'Center', 'Right']

fig7 = plt.figure(figsize=(18, 7))
fig7.patch.set_facecolor('#1a1a2e')

# -- Panel A: 3x3 heatmap of measured points --
ax7a = fig7.add_subplot(1, 3, 1)
ax7a.set_facecolor('#0d1117')
im7a = ax7a.imshow(net_grid7, cmap='Purples', aspect='auto',
                   vmin=net_grid7.min() * 0.85, vmax=net_grid7.max() * 1.05)
for r in range(3):
    for c in range(3):
        ax7a.text(c, r, f'{net_grid7[r, c]:.0f} DN',
                  ha='center', va='center', color='white',
                  fontsize=13, fontweight='bold')
ax7a.set_xticks([0, 1, 2])
ax7a.set_xticklabels(col_labels7, color='white')
ax7a.set_yticks([0, 1, 2])
ax7a.set_yticklabels(row_labels7, color='white')
ax7a.set_title('STEP 1: 9 Measured Points\n(one average DN per belt position)',
               color='white', fontsize=11)
for sp in ax7a.spines.values():
    sp.set_edgecolor('#444')

# -- Panel B: interpolated surface with 9 measured points overlaid --
ax7b = fig7.add_subplot(1, 3, 2)
ax7b.set_facecolor('#0d1117')
im7b = ax7b.imshow(surface7, cmap='Purples', aspect='auto',
                   vmin=net_grid7.min() * 0.85, vmax=net_grid7.max() * 1.05,
                   extent=[0, 1, 1, 0])
for r, rv in enumerate([0.0, 0.5, 1.0]):
    for c, cv in enumerate([0.0, 0.5, 1.0]):
        ax7b.scatter(cv, rv, color='lime', s=80, zorder=5)
        ax7b.text(cv, rv - 0.06, f'{net_grid7[r, c]:.0f}',
                  ha='center', color='lime', fontsize=9, fontweight='bold')
ax7b.set_xticks([0, 0.5, 1])
ax7b.set_xticklabels(col_labels7, color='white')
ax7b.set_yticks([0, 0.5, 1])
ax7b.set_yticklabels(row_labels7, color='white')
ax7b.set_title('STEP 2: Spline Fills Every Point\n(smooth surface through all 9 measurements)',
               color='white', fontsize=11)
plt.colorbar(im7b, ax=ax7b, fraction=0.046).ax.tick_params(colors='white')
for sp in ax7b.spines.values():
    sp.set_edgecolor('#444')

# -- Panel C: 3D surface --
ax7c = fig7.add_subplot(1, 3, 3, projection='3d')
XX7, YY7 = np.meshgrid(x_dense, y_dense)
ax7c.plot_surface(XX7, YY7, surface7, cmap='Purples', alpha=0.85, edgecolor='none')
for r, rv in enumerate([0.0, 0.5, 1.0]):
    for c, cv in enumerate([0.0, 0.5, 1.0]):
        ax7c.scatter(cv, rv, net_grid7[r, c], color='lime', s=60, zorder=5)
ax7c.set_xlabel('Belt Width (L->R)', color='white', fontsize=9)
ax7c.set_ylabel('Belt Length (U->L)', color='white', fontsize=9)
ax7c.set_zlabel('Illumination (DN)', color='white', fontsize=9)
ax7c.tick_params(colors='white')
ax7c.set_title('STEP 3: Full Illumination Surface\n(green dots = your 9 measurements)',
               color='white', fontsize=11)
ax7c.xaxis.pane.fill = False
ax7c.yaxis.pane.fill = False
ax7c.zaxis.pane.fill = False

fig7.suptitle(
    'Spline Interpolation: From 9 Measured Points to Full 2048x1536 Illumination Map\n'
    'NIR1 (CH2) channel -- dark-corrected net DN',
    color='white', fontsize=13, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig7.savefig(RESULTS / 'fig7_spline_concept.png', dpi=130,
             bbox_inches='tight', facecolor=fig7.get_facecolor())
plt.close()
print('  Saved fig7_spline_concept.png')


# ─────────────────────────────────────────────────────────────────
#  TEXT REPORT
# ─────────────────────────────────────────────────────────────────

print('\n  Generating calibration_report.txt ...')

lines = []
L = lines.append

L('=' * 70)
L('  MULTISPECTRAL CALIBRATION REPORT')
L('  MSU Apple Grading System  |  JAI FS-3200T-10GE')
L('  Generated automatically by calibration_pipeline.py')
L('=' * 70)
L('')
L('SECTION 1: EQUIPMENT AND SETTINGS')
L('-' * 70)
L('  Camera    : JAI FS-3200T-10GE (3-channel simultaneous)')
L('  Channel 1 : RGB (BayerRG8)   2048 x 1536 px')
L('  Channel 2 : NIR1 (Mono8)     2048 x 1536 px')
L('  Channel 3 : NIR2 (Mono8)     2048 x 1536 px')
L('  Lighting  : Halogen + LED (both on)')
L('  White Ref : Spectralon SRT-75-100 (certified 75% reflectance)')
L('')
L('  Exposure settings (same for white ref and apple capture):')
L('    CH1 (RGB)  : 2500 us')
L('    CH2 (NIR1) : 1800 us')
L('    CH3 (NIR2) : 2300 us')
L('')
L('  White Balance: LOCKED  R=0.4607  G=1.0000  B=1.6879')
L('    (determined by One-Push AWB on white panel, then locked to file)')
L('')

L('SECTION 2: DARK FRAME ANALYSIS')
L('-' * 70)
L('  Dark frames capture the camera\'s intrinsic electronic noise.')
L('  Method: lens covered with black tape, 84 frames averaged.')
L('  Formula: corrected = (raw - dark) / (white - dark) x 0.75')
L('')
for ch in [1, 2, 3]:
    g = gray(dark[ch])
    L(f'  {CH_NAMES[ch]:14s}')
    L(f'    Mean  : {g.mean():.2f} DN  (baseline pedestal)')
    L(f'    Std   : {g.std():.2f} DN  (random noise per pixel)')
    L(f'    Max   : {g.max():.0f} DN')
    if g.mean() < 30:
        L(f'    Status: GOOD - dark baseline is low and well-characterised')
    else:
        L(f'    Status: WARNING - dark values higher than expected. Check for light leaks.')
    L('')

L('SECTION 3: WHITE REFERENCE GRID ANALYSIS')
L('-' * 70)
L('  9-position grid (Upper/Middle/Lower x Left/Center/Right).')
L('  Each position: Spectralon panel placed flat on belt, 26 frames averaged.')
L('  Values shown are dark-corrected net illumination in DN.')
L('')
for ch in [1, 2, 3]:
    net_grid = white_mean[ch] - dark_mean[ch]
    L(f'  {CH_NAMES[ch]:14s}')
    L(f'         Left    Center   Right')
    for row, rl in enumerate(['Upper ', 'Middle', 'Lower ']):
        vals = '   '.join(f'{net_grid[row,col]:7.1f}' for col in range(3))
        L(f'    {rl}  {vals}')
    vmin = net_grid.min(); vmax = net_grid.max()
    variation = (vmax - vmin) / vmax * 100
    L(f'    Illumination variation across belt: {variation:.1f}%')
    if variation < 10:
        L(f'    Uniformity: EXCELLENT (<10% variation)')
    elif variation < 20:
        L(f'    Uniformity: GOOD (10-20% variation - calibration recommended)')
    else:
        L(f'    Uniformity: SIGNIFICANT (>20% - calibration is essential)')
    L('')

L('SECTION 4: CALIBRATION CORRECTION FACTORS')
L('-' * 70)
L('  The correction map is built by fitting a bicubic spline surface')
L('  through the 9 measured illumination values and interpolating to')
L('  full image resolution (2048 x 1536 pixels).')
L('')
L('  Correction formula applied to every apple pixel (x, y):')
L('')
L('    reflectance(x,y) = (apple(x,y) - dark(x,y))')
L('                       / illumination_map(x,y)')
L('                       x 0.75')
L('')
L('  For different exposures between white ref and apple capture:')
L('')
L('    reflectance(x,y) = (apple(x,y) - dark(x,y))')
L('                       / illumination_map(x,y)')
L('                       x (white_exposure / apple_exposure)')
L('                       x 0.75')
L('')
for ch in [1, 2, 3]:
    cmap = correct_map[ch]
    vmin = cmap.min(); vmax = cmap.max()
    L(f'  {CH_NAMES[ch]:14s}')
    L(f'    Correction range : {vmin:.4f} - {vmax:.4f}')
    L(f'    Max over-correct : {(vmax-1)*100:+.1f}%  (brightest belt region)')
    L(f'    Max under-correct: {(vmin-1)*100:+.1f}%  (darkest belt region)')
    L('')

L('SECTION 5: CALIBRATION IMPACT (BEFORE vs AFTER)')
L('-' * 70)
L('  The Coefficient of Variation (CoV = std/mean) of the white panel')
L('  region measures spatial uniformity. After correct calibration, CoV')
L('  should decrease (panel appears more uniform).')
L('')
for ch in [1, 2, 3]:
    raw_full  = gray(white[ch][1][1]).astype(np.float32)
    dk_full   = gray(dark[ch]).astype(np.float32)
    im_full   = illum_map[ch]
    net       = np.clip(raw_full - dk_full, 0, None)
    refl_full = np.clip(net / im_full * PANEL_REFLECTANCE, 0, 1)
    # Panel region only
    H, W = raw_full.shape
    r0, r1 = H//3, 2*H//3
    c0, c1 = W//3, 2*W//3
    rpad = (r1-r0)//5;  cpad = (c1-c0)//5
    raw_p  = raw_full[r0+rpad:r1-rpad, c0+cpad:c1-cpad]
    refl_p = refl_full[r0+rpad:r1-rpad, c0+cpad:c1-cpad]
    cov_raw  = raw_p.std()  / (raw_p.mean()  + 1e-6) * 100
    cov_cal  = refl_p.std() / (refl_p.mean() + 1e-6) * 100
    improvement = (cov_raw - cov_cal) / (cov_raw + 1e-6) * 100
    L(f'  {CH_NAMES[ch]:14s}')
    L(f'    CoV before calibration : {cov_raw:.2f}%  (panel region, raw image)')
    L(f'    CoV after  calibration : {cov_cal:.2f}%  (panel region, calibrated)')
    L(f'    Uniformity improvement : {improvement:.1f}%')
    L(f'    Mean reflectance       : {refl_p.mean():.4f}  (expected: 0.75)')
    L(f'    Error from 0.75        : {abs(refl_p.mean()-0.75)/0.75*100:.2f}%')
    L('')

L('SECTION 6: OUTPUT FILES')
L('-' * 70)
L('  dark_avg_ch[1-3].npy        - averaged dark frames (float32)')
L('  white_avg_[pos]_ch[1-3].npy - averaged white frames per position')
L('  illumination_map_ch[1-3].npy- full-res 2D illumination map')
L('  correction_map_ch[1-3].npy  - per-pixel correction factors')
L('  fig1_dark_frames.png           - dark frame heat maps')
L('  fig2_white_grid.png            - 3x3 white reference grid heatmap')
L('  fig3_illumination_profiles.png - horizontal & vertical profiles')
L('  fig4_correction_map.png        - full-res correction maps')
L('  fig5_calibration_impact.png    - before/after calibration comparison')
L('  fig6_grid_stitched.png         - 9 white reference photos in 3x3 belt layout')
L('  fig7_spline_concept.png        - spline interpolation explanation (3 panels)')
L('')
L('SECTION 7: CONCLUSION AND RECOMMENDATIONS')
L('-' * 70)
L('  Calibration sequence validated. Ready for apple data collection.')
L('')
L('  WORKFLOW FOR EACH DATA COLLECTION SESSION:')
L('  1. Connect camera, apply Load Locked WB (R=0.461 G=1.000 B=1.688)')
L('  2. Verify exposure: RGB=2500us  NIR1=1800us  NIR2=2300us')
L('  3. Verify lighting: Halogen + LED both on')
L('  4. Collect apple images')
L('  5. Apply calibration in post-processing using the .npy maps saved here')
L('')
L('  NOTE: The dark frame and illumination maps should be recaptured if:')
L('    - Exposure settings change')
L('    - Lighting configuration changes')
L('    - Camera is moved or refocused')
L('    - More than 1 week has passed (lamp aging)')
L('')
L('=' * 70)

report_text = '\n'.join(lines)
with open(RESULTS / 'calibration_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print('\n' + '='*65)
print(f'  All outputs saved to: {RESULTS}')
print('='*65)
