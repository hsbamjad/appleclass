"""
b_build_drift_corrected_maps.py
================================
STEP B -- Build drift-corrected illumination maps from 9-position capture.

Reads from BASE_DIR (see PATHS section below).
Saves maps + 10 research-grade figures to RESULTS_DIR.

Figures produced:
  fig1_dark_frames.png              -- dark frame heatmaps + histograms
  fig2_drift_correction_timeline.png-- lamp level over time + corrections applied
  fig3_white_grid_raw.png           -- 9-position grid before drift correction
  fig4_white_grid_corrected.png     -- 9-position grid after drift correction
  fig5_illumination_profiles.png    -- horizontal and vertical 3x3 profiles
  fig6_illumination_maps.png        -- full-resolution illumination maps (all channels)
  fig7_correction_factor_maps.png   -- per-pixel correction factor maps
  fig8_grid_stitched_photos.png     -- 9 actual frame photos in 3x3 belt layout
  fig9_spline_concept.png           -- spline interpolation: 9 points to full map
  fig10_calibration_impact.png      -- before vs after calibration on center capture

Data files produced:
  dark_avg_ch1.npy / ch2 / ch3       -- averaged dark frames
  illumination_map_ch1.npy / ch2/ch3 -- drift-corrected full-resolution maps
  correction_map_ch1.npy / ch2 / ch3 -- per-pixel correction factors
  drift_correction_report.txt        -- full calibration report

Usage:
  python b_build_drift_corrected_maps.py
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RectBivariateSpline
import matplotlib.patheffects as pe
import warnings, time
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r'S:\MSU_Research\apple_class\calibration_trials\final_runs_02')
RESULTS_DIR = Path(r'S:\MSU_Research\apple_class\calibration_results_02')
RESULTS_DIR.mkdir(exist_ok=True)

PANEL_REFL  = 0.75

# Grid layout: pos_name -> (grid_row, grid_col)
# row: 0=Upper 1=Middle 2=Lower   col: 0=Left 1=Center 2=Right
GRID = {
    'white_UL': (0, 0), 'white_UC': (0, 1), 'white_UR': (0, 2),
    'white_ML': (1, 0), 'white_C1': (1, 1), 'white_MR': (1, 2),
    'white_LL': (2, 0), 'white_LC': (2, 1), 'white_LR': (2, 2),
}

ROW_LABELS = ['Upper', 'Middle', 'Lower']
COL_LABELS = ['Left',  'Center', 'Right']

ANCHOR_START = 'white_C1'
ANCHOR_MID   = 'white_C2'
ANCHOR_END   = 'white_C3'

CHANNELS  = ['ch1', 'ch2', 'ch3']
CH_NAMES  = {'ch1': 'RGB',  'ch2': 'NIR1 (~800 nm)', 'ch3': 'NIR2 (~900 nm)'}
CH_COLORS = {'ch1': '#e85d04', 'ch2': '#7209b7', 'ch3': '#0077b6'}
CH_CMAPS  = {'ch1': 'Oranges',  'ch2': 'Purples', 'ch3': 'Blues'}

BG  = '#1a1a2e'   # dark background
BG2 = '#0d1117'   # slightly darker panel background

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_avg(folder: Path, ch: str, max_frames: int = 50):
    """Load and average all frames. Returns (avg_array, first_frame_mtime)."""
    ch_dir = folder / 'raw_frames' / ch
    frames = sorted(ch_dir.glob('*.jpg')) + sorted(ch_dir.glob('*.png'))
    frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f'No frames in {ch_dir}')
    arrays = [np.array(Image.open(f)).astype(np.float32) for f in frames]
    return np.mean(arrays, axis=0), frames[0].stat().st_mtime

def to_gray(arr):
    arr = np.array(arr, dtype=np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr

def region_mean(img, grid_row: int, grid_col: int) -> float:
    """Mean of inner 60% of the 1/3 sub-region matching the panel position."""
    g = to_gray(img)
    H, W = g.shape
    row_bounds = [(0, H//3), (H//3, 2*H//3), (2*H//3, H)]
    col_bounds = [(0, W//3), (W//3, 2*W//3), (2*W//3, W)]
    r0, r1 = row_bounds[grid_row]
    c0, c1 = col_bounds[grid_col]
    rh, cw = r1 - r0, c1 - c0
    patch = g[r0+rh//5 : r1-rh//5, c0+cw//5 : c1-cw//5]
    return float(patch.mean())

def interp_lamp(ts, anchors):
    """Piecewise linear interpolation of lamp level. anchors = [(ts, dn), ...]."""
    if ts <= anchors[0][0]:
        return anchors[0][1]
    if ts >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(len(anchors) - 1):
        t0, v0 = anchors[i]
        t1, v1 = anchors[i+1]
        if t0 <= ts <= t1:
            return v0 + (v1 - v0) * (ts - t0) / max(t1 - t0, 1)
    return anchors[-1][1]

def dark_style(ax):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.tick_params(colors='white')

def fmt_t(ts, t0):
    return (ts - t0) / 60.0   # minutes from t0

# ── MAIN ──────────────────────────────────────────────────────────────────────

print('=' * 70)
print('  STEP B -- DRIFT-CORRECTED ILLUMINATION MAP BUILDER')
print(f'  Source : {BASE_DIR.name}')
print(f'  Output : {RESULTS_DIR.name}')
print('=' * 70)
print()

report_lines = [
    'DRIFT-CORRECTED CALIBRATION REPORT',
    'MSU Multispectral Apple Classification System',
    f'Source folder : {BASE_DIR}',
    f'Results folder: {RESULTS_DIR}',
    '=' * 65, ''
]

# ── 1. DARK FRAMES ────────────────────────────────────────────────────────────

print('[1/4] Loading dark frames...')
dark_full = {}   # full arrays for display
dark_g    = {}   # grayscale
dark_mean_scalar = {}

for ch in CHANNELS:
    d = np.load(RESULTS_DIR / f'dark_avg_{ch}.npy').astype(np.float32)
    dark_full[ch] = d
    dark_g[ch]    = to_gray(d)
    dark_mean_scalar[ch] = dark_g[ch].mean()
    print(f'  {CH_NAMES[ch]}: dark mean = {dark_mean_scalar[ch]:.2f} DN')

# ── 2. DRIFT ANCHORS ─────────────────────────────────────────────────────────

print()
print('[2/4] Loading drift anchors (C1 / C2 / C3)...')
anchors = {ch: [] for ch in CHANNELS}
anchor_names = []

anchor_folders = [(ANCHOR_START, 'C1-Start')]
if ANCHOR_MID and (BASE_DIR / ANCHOR_MID).exists():
    anchor_folders.append((ANCHOR_MID, 'C2-Mid'))
anchor_folders.append((ANCHOR_END, 'C3-End'))

for ch in CHANNELS:
    print(f'\n  {CH_NAMES[ch]}:')
    for folder_name, label in anchor_folders:
        folder = BASE_DIR / folder_name
        arr, ts = load_avg(folder, ch)
        rm = region_mean(arr, 1, 1)
        anchors[ch].append((ts, rm))
        t_str = time.strftime('%H:%M:%S', time.localtime(ts))
        print(f'    {label:<12} ({folder_name}): {rm:.2f} DN at {t_str}')
    t0  = anchors[ch][0][0]
    v0  = anchors[ch][0][1]
    v1  = anchors[ch][-1][1]
    dur = (anchors[ch][-1][0] - t0) / 60
    drift = (v1 - v0) / v0 * 100
    print(f'    Total drift: {drift:+.1f}% over {dur:.1f} min')
    report_lines.append(f'{CH_NAMES[ch]} drift: {drift:+.1f}% over {dur:.1f} min')

if not anchor_names:
    anchor_names = [lbl for _, lbl in anchor_folders]

report_lines.append('')

# ── 3. LOAD POSITIONS + DRIFT CORRECTION ─────────────────────────────────────

print()
print('[3/4] Loading positions and applying drift correction...')

net_grid_raw  = {ch: np.full((3, 3), np.nan) for ch in CHANNELS}
net_grid_corr = {ch: np.full((3, 3), np.nan) for ch in CHANNELS}
pos_info      = {ch: [] for ch in CHANNELS}   # for timeline figure
frame_store   = {ch: {} for ch in CHANNELS}   # pos_name -> avg frame arr

for ch in CHANNELS:
    ref_level = anchors[ch][0][1]
    dm        = dark_mean_scalar[ch]
    print(f'\n  {CH_NAMES[ch]}  (ref: {ref_level:.2f} DN)')

    for pos_name, (grid_row, grid_col) in GRID.items():
        folder = BASE_DIR / pos_name
        try:
            arr, ts = load_avg(folder, ch)
        except FileNotFoundError:
            continue

        raw_rm   = region_mean(arr, grid_row, grid_col)
        lamp_now = interp_lamp(ts, anchors[ch])
        scale    = ref_level / max(lamp_now, 1)
        corr_rm  = raw_rm * scale

        net_grid_raw[ch][grid_row, grid_col]  = raw_rm - dm
        net_grid_corr[ch][grid_row, grid_col] = corr_rm - dm

        frame_store[ch][pos_name] = arr
        pos_info[ch].append({
            'name': pos_name, 'ts': ts, 'raw_rm': raw_rm,
            'lamp': lamp_now, 'scale': scale, 'corr_rm': corr_rm,
            'row': grid_row, 'col': grid_col,
            't_min': fmt_t(ts, anchors[ch][0][0])
        })

        t_str = time.strftime('%H:%M:%S', time.localtime(ts))
        print(f'    {pos_name:<12}: raw={raw_rm:7.2f}  scale={scale:.4f}  corr={corr_rm:7.2f}  ({t_str})')

# ── 4. BUILD ILLUMINATION MAPS ────────────────────────────────────────────────

print()
print('[4/4] Building illumination maps...')

sample_arr, _ = load_avg(BASE_DIR / ANCHOR_START, 'ch1')
H, W = to_gray(sample_arr).shape
rows_norm = np.array([0.0, 0.5, 1.0])
cols_norm = np.array([0.0, 0.5, 1.0])
x_full    = np.linspace(0, 1, W)
y_full    = np.linspace(0, 1, H)

illum_map      = {}
correction_map = {}

for ch in CHANNELS:
    grid = net_grid_corr[ch].copy()
    if np.isnan(grid).any():
        grid[np.isnan(grid)] = np.nanmedian(grid)
    grid = np.clip(grid, 0.1, None)

    spl  = RectBivariateSpline(rows_norm, cols_norm, grid, kx=2, ky=2)
    illum = np.clip(spl(y_full, x_full).astype(np.float32), 1.0, None)
    illum_map[ch] = illum
    np.save(RESULTS_DIR / f'illumination_map_{ch}.npy', illum)

    center_v = illum[H//2, W//2]
    corr     = center_v / np.clip(illum, 1, None)
    correction_map[ch] = corr
    np.save(RESULTS_DIR / f'correction_map_{ch}.npy', corr)

    ratio = illum.max() / max(illum.min(), 1)
    print(f'  {CH_NAMES[ch]}: min={illum.min():.1f}  max={illum.max():.1f}  ratio={ratio:.2f}x')

print()
print('  Maps saved to:', RESULTS_DIR)

# =============================================================================
# FIGURES
# =============================================================================

print()
print('Generating figures...')
print()

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Dark Frame Analysis
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(BG)
fig.suptitle('Fig 1 -- Dark Frame Analysis\nLens covered, same exposure. Represents camera electronic baseline.',
             color='white', fontsize=13, fontweight='bold')

for i, ch in enumerate(CHANNELS):
    g = dark_g[ch]
    ax_img = axes[0, i]
    ax_hist = axes[1, i]
    dark_style(ax_img)
    dark_style(ax_hist)

    im = ax_img.imshow(g, cmap='hot', vmin=0, vmax=20, aspect='auto')
    ax_img.set_title(f'{CH_NAMES[ch]}\nMean={g.mean():.2f} DN  Std={g.std():.2f} DN',
                     color='white', fontsize=10)
    plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04).ax.tick_params(colors='white')

    ax_hist.hist(g.ravel(), bins=60, color=CH_COLORS[ch], alpha=0.85, edgecolor='none')
    ax_hist.axvline(g.mean(), color='white', lw=1.5, linestyle='--', label=f'Mean={g.mean():.1f}')
    ax_hist.set_xlabel('Dark DN value', color='white')
    ax_hist.set_ylabel('Pixel count', color='white')
    ax_hist.legend(fontsize=9, labelcolor='white', facecolor=BG, edgecolor='#444')

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(RESULTS_DIR / 'fig1_dark_frames.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig1_dark_frames.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Drift Correction Timeline
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Fig 2 -- Lamp Drift Correction Timeline\n'
             'Green points = center anchors used for drift estimation. '
             'Orange = each position raw reading. Blue = corrected to C1 reference.',
             color='white', fontsize=12, fontweight='bold')

for i, ch in enumerate(CHANNELS):
    ax = axes[i]
    dark_style(ax)

    t0 = anchors[ch][0][0]
    ref_level = anchors[ch][0][1]

    # Draw interpolation curve
    t_curve = np.linspace(0, (anchors[ch][-1][0] - t0)/60, 200)
    lamp_curve = [interp_lamp(t0 + t*60, anchors[ch]) for t in t_curve]
    ax.plot(t_curve, lamp_curve, color='#aaaaaa', lw=1.5, linestyle='--',
            label='Interpolated lamp level', zorder=2)

    # Reference line
    ax.axhline(ref_level, color='lime', lw=1.5, linestyle=':',
               label=f'C1 reference ({ref_level:.1f} DN)', zorder=2)

    # Plot anchors
    for ts_a, dn_a in anchors[ch]:
        t_min = fmt_t(ts_a, t0)
        ax.scatter(t_min, dn_a, color='#00ff88', s=120, zorder=5, marker='D')

    # Plot each position
    for p in pos_info[ch]:
        ax.scatter(p['t_min'], p['raw_rm'], color='#ff8800', s=60, zorder=4, marker='o')
        ax.scatter(p['t_min'], p['corr_rm'], color='#4fc3f7', s=60, zorder=4, marker='s')
        ax.annotate(p['name'].replace('white_', ''),
                    (p['t_min'], p['raw_rm']),
                    textcoords='offset points', xytext=(4, 5),
                    color='#ffaaaa', fontsize=7)

    # Legend entries
    ax.scatter([], [], color='#00ff88', s=80, marker='D', label='Center anchor (C1/C2/C3)')
    ax.scatter([], [], color='#ff8800', s=60, marker='o', label='Position raw DN')
    ax.scatter([], [], color='#4fc3f7', s=60, marker='s', label='Position corrected to C1')

    ax.set_xlabel('Time from C1 (minutes)', color='white')
    ax.set_ylabel('Lamp level (region mean DN)', color='white')
    ax.set_title(f'{CH_NAMES[ch]}', color='white', fontsize=11)
    ax.legend(fontsize=8, labelcolor='white', facecolor=BG, edgecolor='#444')

plt.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(RESULTS_DIR / 'fig2_drift_correction_timeline.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig2_drift_correction_timeline.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: White Reference Grid -- RAW (before correction)
# ─────────────────────────────────────────────────────────────────────────────

def draw_grid_heatmap(net_grids, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(10, 13))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, color='white', fontsize=13, fontweight='bold')

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        dark_style(ax)
        ng = net_grids[ch]
        im = ax.imshow(ng, cmap=CH_CMAPS[ch], aspect='auto',
                       vmin=ng.min()*0.88, vmax=ng.max()*1.05)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(COL_LABELS, color='white', fontsize=11)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(ROW_LABELS, color='white', fontsize=11)

        vmin_g = ng.min(); vmax_g = ng.max()
        center_val = ng[1, 1]
        for row in range(3):
            for col in range(3):
                val = ng[row, col]
                pct = (val - center_val) / (abs(center_val) + 1e-6) * 100
                bright = (val - vmin_g) / (vmax_g - vmin_g + 1e-6)
                tc = '#111111' if bright > 0.55 else 'white'
                ax.text(col, row, f'{val:.1f}\n({pct:+.1f}%)',
                        ha='center', va='center', color=tc, fontsize=11, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2,
                                      foreground='white' if tc == '#111111' else '#000000')])
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors='white')
        ax.set_title(f'{CH_NAMES[ch]}  |  Net DN (dark-corrected)  |  % vs Center',
                     color='white', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved {filename}')

draw_grid_heatmap(net_grid_raw,
    'Fig 3 -- White Reference Grid: RAW Values (before drift correction)\n'
    'Values include lamp drift artifact -- positions captured at different lamp levels.',
    'fig3_white_grid_raw.png')

draw_grid_heatmap(net_grid_corr,
    'Fig 4 -- White Reference Grid: DRIFT-CORRECTED Values\n'
    'All positions normalized to C1 lamp reference level. '
    'Values now reflect true spatial illumination.',
    'fig4_white_grid_corrected.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: Illumination Profiles (horizontal + vertical)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(BG)
fig.suptitle('Fig 5 -- Illumination Uniformity Profiles (drift-corrected)\n'
             'Horizontal and vertical cross-sections of the 3x3 illumination grid.',
             color='white', fontsize=13, fontweight='bold')

x3 = [0, 1, 2]
for i, ch in enumerate(CHANNELS):
    ng = net_grid_corr[ch]
    color = CH_COLORS[ch]

    ax_h = axes[0, i]; ax_v = axes[1, i]
    dark_style(ax_h); dark_style(ax_v)

    for row, rl in enumerate(ROW_LABELS):
        alpha = 1.0 - row * 0.2
        ax_h.plot(x3, ng[row, :], 'o-', color=color, alpha=alpha, lw=2, ms=8, label=rl)
        for x, v in zip(x3, ng[row, :]):
            ax_h.annotate(f'{v:.0f}', (x, v), textcoords='offset points',
                          xytext=(0, 8), ha='center', color='white', fontsize=8)

    ax_h.set_xticks(x3); ax_h.set_xticklabels(COL_LABELS, color='white')
    ax_h.set_title(f'{CH_NAMES[ch]}\nHorizontal Profile (left to right)',
                   color='white', fontsize=10)
    ax_h.legend(fontsize=8, labelcolor='white', facecolor=BG, edgecolor='#444')
    ax_h.set_ylabel('Net DN (dark-corrected)', color='white')

    for col, cl in enumerate(COL_LABELS):
        alpha = 1.0 - col * 0.2
        ax_v.plot(x3, ng[:, col], 's--', color=color, alpha=alpha, lw=2, ms=8, label=cl)
        for y, v in zip(x3, ng[:, col]):
            ax_v.annotate(f'{v:.0f}', (y, v), textcoords='offset points',
                          xytext=(0, 8), ha='center', color='white', fontsize=8)

    ax_v.set_xticks(x3); ax_v.set_xticklabels(ROW_LABELS, color='white')
    ax_v.set_title('Vertical Profile (top to bottom)', color='white', fontsize=10)
    ax_v.legend(fontsize=8, labelcolor='white', facecolor=BG, edgecolor='#444')
    ax_v.set_ylabel('Net DN (dark-corrected)', color='white')

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(RESULTS_DIR / 'fig5_illumination_profiles.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig5_illumination_profiles.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6: Full-Resolution Illumination Maps
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Fig 6 -- Full-Resolution Illumination Maps (Drift-Corrected)\n'
             'Bicubic spline interpolated from 9 measured positions to 2048 x 1536.',
             color='white', fontsize=13, fontweight='bold')

for i, ch in enumerate(CHANNELS):
    ax = axes[i]; dark_style(ax)
    illum = illum_map[ch]
    ratio = illum.max() / max(illum.min(), 1)
    im = ax.imshow(illum, cmap=CH_CMAPS[ch], aspect='auto')
    ax.set_title(f'{CH_NAMES[ch]}\nmin={illum.min():.1f}  max={illum.max():.1f}  ratio={ratio:.2f}x',
                 color='white', fontsize=11)
    ax.set_xlabel('Belt width (pixels)', color='white')
    ax.set_ylabel('Belt length (pixels)', color='white')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('DN', color='white')

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(RESULTS_DIR / 'fig6_illumination_maps.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig6_illumination_maps.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7: Correction Factor Maps
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Fig 7 -- Spatial Illumination Correction Factor Maps\n'
             'Green = same as center (factor ~1.0). Red = receives less light (factor > 1.0).',
             color='white', fontsize=13, fontweight='bold')

for i, ch in enumerate(CHANNELS):
    ax = axes[i]; dark_style(ax)
    cmap = correction_map[ch]
    im = ax.imshow(cmap, cmap='RdYlGn', vmin=0.85, vmax=1.5, aspect='auto')
    vmin_c = cmap.min(); vmax_c = cmap.max()
    ax.set_title(f'{CH_NAMES[ch]}\nRange: {vmin_c:.3f} - {vmax_c:.3f}'
                 f'  (max boost: {(vmax_c-1)*100:.1f}%)',
                 color='white', fontsize=11)
    ax.set_xlabel('Belt width (pixels)', color='white')
    ax.set_ylabel('Belt length (pixels)', color='white')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Correction factor\n(1.0 = same as center)', color='white')

plt.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(RESULTS_DIR / 'fig7_correction_factor_maps.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig7_correction_factor_maps.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8: 9-Position White Reference Photos (NIR1, stitched grid)
# ─────────────────────────────────────────────────────────────────────────────

fig8, axes8 = plt.subplots(3, 3, figsize=(18, 13))
fig8.patch.set_facecolor(BG)
fig8.suptitle('Fig 8 -- White Reference Capture Grid (NIR1 channel)\n'
              'Green box = region sampled for illumination measurement. '
              'Values are drift-corrected net DN.',
              color='white', fontsize=13, fontweight='bold')

pos_order = [
    ['white_UL', 'white_UC', 'white_UR'],
    ['white_ML', 'white_C1', 'white_MR'],
    ['white_LL', 'white_LC', 'white_LR'],
]
ch_disp = 'ch2'   # NIR1 for clearest grayscale

for row in range(3):
    for col in range(3):
        ax = axes8[row, col]; dark_style(ax)
        pos_name = pos_order[row][col]

        if pos_name not in frame_store[ch_disp]:
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                    color='red', fontsize=14, transform=ax.transAxes)
            continue

        img_arr = to_gray(frame_store[ch_disp][pos_name])
        ax.imshow(img_arr, cmap='gray', vmin=0, vmax=255, aspect='auto')

        # Green box showing sampled region
        img_H, img_W = img_arr.shape
        r_bounds = [(0, img_H//3), (img_H//3, 2*img_H//3), (2*img_H//3, img_H)]
        c_bounds = [(0, img_W//3), (img_W//3, 2*img_W//3), (2*img_W//3, img_W)]
        r0, r1 = r_bounds[row]; c0, c1 = c_bounds[col]
        rh, cw = r1-r0, c1-c0
        rect = mpatches.Rectangle((c0+cw//5, r0+rh//5), cw-2*cw//5, rh-2*rh//5,
                                   lw=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        net_val  = net_grid_corr[ch_disp][row, col]
        net_raw  = net_grid_raw[ch_disp][row, col]
        pct      = (net_val - net_grid_corr[ch_disp][1, 1]) / (net_grid_corr[ch_disp][1, 1] + 1e-6) * 100
        ax.set_title(f'{ROW_LABELS[row]}-{COL_LABELS[col]}\n'
                     f'Raw: {net_raw:.0f} DN  Corr: {net_val:.0f} DN  ({pct:+.0f}% vs C)',
                     color='white', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig8.savefig(RESULTS_DIR / 'fig8_grid_stitched_photos.png', dpi=120,
             bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig8_grid_stitched_photos.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9: Spline Interpolation Concept (3-panel)
# ─────────────────────────────────────────────────────────────────────────────

net9    = net_grid_corr['ch2'].copy()
net9[np.isnan(net9)] = np.nanmedian(net9)
y_d     = np.linspace(0, 1, 200)
x_d     = np.linspace(0, 1, 200)
spl9    = RectBivariateSpline(rows_norm, cols_norm, net9, kx=2, ky=2)
surf9   = spl9(y_d, x_d)

fig9    = plt.figure(figsize=(18, 7))
fig9.patch.set_facecolor(BG)
fig9.suptitle('Fig 9 -- Spline Interpolation: 9 Points to Full 2048 x 1536 Illumination Map\n'
              'NIR1 channel -- drift-corrected net DN',
              color='white', fontsize=13, fontweight='bold')

# Panel A: 3x3 heatmap
ax9a = fig9.add_subplot(1, 3, 1); dark_style(ax9a)
im9a = ax9a.imshow(net9, cmap='Purples', aspect='auto',
                   vmin=net9.min()*0.85, vmax=net9.max()*1.05)
for r in range(3):
    for c in range(3):
        ax9a.text(c, r, f'{net9[r,c]:.0f} DN',
                  ha='center', va='center', color='white', fontsize=13, fontweight='bold')
ax9a.set_xticks([0,1,2]); ax9a.set_xticklabels(COL_LABELS, color='white')
ax9a.set_yticks([0,1,2]); ax9a.set_yticklabels(ROW_LABELS, color='white')
ax9a.set_title('Step 1: 9 Measured Points\n(one mean DN per belt position)',
               color='white', fontsize=11)

# Panel B: interpolated surface with scatter overlay
ax9b = fig9.add_subplot(1, 3, 2); dark_style(ax9b)
im9b = ax9b.imshow(surf9, cmap='Purples', aspect='auto',
                   vmin=net9.min()*0.85, vmax=net9.max()*1.05,
                   extent=[0, 1, 1, 0])
for r, rv in enumerate([0.0, 0.5, 1.0]):
    for c, cv in enumerate([0.0, 0.5, 1.0]):
        ax9b.scatter(cv, rv, color='lime', s=80, zorder=5)
        ax9b.text(cv, rv-0.06, f'{net9[r,c]:.0f}',
                  ha='center', color='lime', fontsize=9, fontweight='bold')
ax9b.set_xticks([0, 0.5, 1]); ax9b.set_xticklabels(COL_LABELS, color='white')
ax9b.set_yticks([0, 0.5, 1]); ax9b.set_yticklabels(ROW_LABELS, color='white')
ax9b.set_title('Step 2: Bicubic Spline Fills Frame\n(smooth surface through all 9 points)',
               color='white', fontsize=11)
plt.colorbar(im9b, ax=ax9b, fraction=0.046).ax.tick_params(colors='white')

# Panel C: 3D surface
ax9c = fig9.add_subplot(1, 3, 3, projection='3d')
ax9c.set_facecolor(BG2)
XX9, YY9 = np.meshgrid(x_d, y_d)
ax9c.plot_surface(XX9, YY9, surf9, cmap='Purples', alpha=0.85, edgecolor='none')
for r, rv in enumerate([0.0, 0.5, 1.0]):
    for c, cv in enumerate([0.0, 0.5, 1.0]):
        ax9c.scatter(cv, rv, net9[r, c], color='lime', s=60, zorder=5)
ax9c.set_xlabel('Belt width (L->R)', color='white', fontsize=9)
ax9c.set_ylabel('Belt length (U->L)', color='white', fontsize=9)
ax9c.set_zlabel('Illumination (DN)', color='white', fontsize=9)
ax9c.tick_params(colors='white')
ax9c.set_title('Step 3: Full Illumination Surface\n(green dots = your 9 measurements)',
               color='white', fontsize=11)
ax9c.xaxis.pane.fill = False
ax9c.yaxis.pane.fill = False
ax9c.zaxis.pane.fill = False

plt.tight_layout(rect=[0, 0, 1, 0.91])
fig9.savefig(RESULTS_DIR / 'fig9_spline_concept.png', dpi=130,
             bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig9_spline_concept.png')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10: Calibration Impact -- Before vs After (using C1 center frame)
# ─────────────────────────────────────────────────────────────────────────────

fig10, axes10 = plt.subplots(2, 3, figsize=(18, 11))
fig10.patch.set_facecolor(BG)
fig10.suptitle('Fig 10 -- Calibration Impact: Before vs After  (white_C1 center frame)\n'
               'Green box = panel evaluation region. '
               'CoV = Coefficient of Variation: lower is more uniform = better calibration.',
               color='white', fontsize=12, fontweight='bold')

for i, ch in enumerate(CHANNELS):
    raw_full  = to_gray(frame_store[ch]['white_C1']).astype(np.float32)
    dk_full   = dark_g[ch]
    im_full   = illum_map[ch]

    net_full  = np.clip(raw_full - dk_full, 0, None)
    refl_full = np.clip(net_full / im_full * PANEL_REFL, 0, 1)

    # Panel region (center third, inner 60%)
    r0, r1 = H//3, 2*H//3; c0, c1 = W//3, 2*W//3
    rp, cp = (r1-r0)//5, (c1-c0)//5
    raw_p  = raw_full[r0+rp:r1-rp, c0+cp:c1-cp]
    refl_p = refl_full[r0+rp:r1-rp, c0+cp:c1-cp]

    cov_raw = raw_p.std()  / (raw_p.mean()  + 1e-6) * 100
    cov_cal = refl_p.std() / (refl_p.mean() + 1e-6) * 100
    improve = (cov_raw - cov_cal) / (cov_raw + 1e-6) * 100

    ax_r = axes10[0, i]; ax_c = axes10[1, i]
    dark_style(ax_r); dark_style(ax_c)

    raw_disp = (raw_full - raw_full.min()) / (raw_full.max() - raw_full.min() + 1e-6)
    im1 = ax_r.imshow(raw_disp, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax_r.set_title(f'{CH_NAMES[ch]}\nUncalibrated (raw)\n'
                   f'CoV={cov_raw:.1f}%  mean={raw_p.mean():.0f} DN',
                   color='white', fontsize=10)

    im2 = ax_c.imshow(refl_full, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax_c.set_title(f'Calibrated reflectance\n'
                   f'CoV={cov_cal:.1f}%  mean refl={refl_p.mean():.4f}\n'
                   f'Uniformity improvement: {improve:.1f}%',
                   color='white', fontsize=10)

    for ax, im_obj in [(ax_r, im1), (ax_c, im2)]:
        rect = mpatches.Rectangle((c0+cp, r0+rp), (c1-c0-2*cp), (r1-r0-2*rp),
                                   lw=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        plt.colorbar(im_obj, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='white')

plt.tight_layout(rect=[0, 0, 1, 0.90])
fig10.savefig(RESULTS_DIR / 'fig10_calibration_impact.png', dpi=150,
              bbox_inches='tight', facecolor=BG)
plt.close()
print('  Saved fig10_calibration_impact.png')

# =============================================================================
# TEXT REPORT
# =============================================================================

print()
print('Generating calibration_report.txt...')

L = report_lines.append
L('')
L('SECTION 1: EQUIPMENT AND SETTINGS')
L('-' * 65)
L('  Camera     : JAI FS-3200T (3-channel simultaneous)')
L('  CH1 (RGB)  : BayerRG8    2048 x 1536 px')
L('  CH2 (NIR1) : Mono8 ~800nm  2048 x 1536 px')
L('  CH3 (NIR2) : Mono8 ~900nm  2048 x 1536 px')
L('  Lighting   : Halogen broadband lamp')
L('  White ref  : Spectralon SRT-75-100 (certified 75% reflectance)')
L('')

L('SECTION 2: DRIFT CORRECTION METHOD')
L('-' * 65)
L('  Three center captures (C1, C2, C3) bracket the 9-position sequence.')
L('  For each intermediate position at time t:')
L('    lamp(t) = piecewise linear interpolation of C1/C2/C3 anchor DNs')
L('    scale   = C1_DN / lamp(t)')
L('    corrected_DN = raw_DN * scale')
L('')
L('  Corrected values reflect what each position would have read')
L('  if the lamp had been at C1 level throughout the capture.')
L('')

for ch in CHANNELS:
    t0   = anchors[ch][0][0]
    v0   = anchors[ch][0][1]
    v1   = anchors[ch][-1][1]
    dur  = (anchors[ch][-1][0] - t0) / 60
    drift = (v1 - v0) / v0 * 100
    L(f'  {CH_NAMES[ch]}:  C1={v0:.2f} DN  ->  C3={v1:.2f} DN  '
      f'drift={drift:+.1f}%  span={dur:.1f} min')

L('')
L('SECTION 3: GRID ANALYSIS (DRIFT-CORRECTED)')
L('-' * 65)
for ch in CHANNELS:
    ng = net_grid_corr[ch]
    variation = (np.nanmax(ng) - np.nanmin(ng)) / np.nanmax(ng) * 100
    L(f'  {CH_NAMES[ch]}:')
    L(f'      Left     Center   Right')
    for row, rl in enumerate(['Upper ', 'Middle', 'Lower ']):
        vals = '   '.join(f'{ng[row,c]:8.2f}' for c in range(3))
        L(f'    {rl}  {vals}')
    L(f'    Illumination variation across belt: {variation:.1f}%')
    if variation < 10:
        L(f'    Uniformity: EXCELLENT (< 10% variation)')
    elif variation < 25:
        L(f'    Uniformity: MODERATE ({variation:.0f}% - calibration essential)')
    else:
        L(f'    Uniformity: SIGNIFICANT (> 25% - calibration is critical)')
    L('')

L('SECTION 4: CALIBRATION FORMULA')
L('-' * 65)
L('  reflectance(x,y) = (apple(x,y) - dark(x,y))')
L('                     / illumination_map(x,y)')
L('                     x 0.75')
L('                     x per_run_scale_factor')
L('')
L('  per_run_scale_factor = 0.75 / (panel_center_DN - dark_DN) / illum_center')
L('  Applied via d_per_run_panel.py before each apple run.')
L('')

L('SECTION 5: OUTPUT FILES')
L('-' * 65)
L('  illumination_map_ch1/ch2/ch3.npy  -- drift-corrected full-res maps')
L('  correction_map_ch1/ch2/ch3.npy    -- per-pixel correction factors')
L('  dark_avg_ch1/ch2/ch3.npy          -- dark frames (from calibration_results_01)')
L('  fig1  dark_frames.png             -- dark frame heatmaps + histograms')
L('  fig2  drift_correction_timeline   -- lamp level over time with corrections')
L('  fig3  white_grid_raw.png          -- 9-position grid before correction')
L('  fig4  white_grid_corrected.png    -- 9-position grid after correction')
L('  fig5  illumination_profiles.png   -- H and V cross-sections')
L('  fig6  illumination_maps.png       -- full-resolution illumination maps')
L('  fig7  correction_factor_maps.png  -- per-pixel correction factor maps')
L('  fig8  grid_stitched_photos.png    -- 9 actual frame photos in belt layout')
L('  fig9  spline_concept.png          -- interpolation: 9 points to full map')
L('  fig10 calibration_impact.png      -- before vs after calibration on C1 frame')
L('')
L('=' * 65)

with open(RESULTS_DIR / 'drift_correction_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print()
print('=' * 70)
print('  ALL DONE.')
print(f'  {len(list(RESULTS_DIR.glob("fig*.png")))} figures + report saved to:')
print(f'  {RESULTS_DIR}')
print()
print('  NEXT: Run  d_per_run_panel.py --run <run_name>  before each apple run.')
print('=' * 70)
