"""
a_build_illumination_maps.py
============================
STEP A -- Build spatial illumination maps from 9-position white reference.

NO lamp-drift correction (use only when lamp is voltage-regulated / stable).

Reads:
  calibration_data/dark_avg_ch[1-3].npy
  calibration_data/white_*/raw_frames/ch[1-3]/*.bmp

Writes:
  calibration_results/
    dark_avg_ch[1-3].npy
    illumination_map_ch[1-3].npy
    correction_map_ch[1-3].npy
    white_avg_<pos>_ch[1-3].npy
    fig1..fig7 + calibration_report.txt

Formula (per channel, dark-corrected net DN at 9 positions):
  net_grid[r,c] = region_mean(white[r,c]) - dark_mean
  illum_map     = bicubic spline(net_grid) -> full HxW
  correction    = illum_map / illum_map_center

Usage (from this folder):
  python a_build_illumination_maps.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: F401
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────
ROOT     = Path(r"D:\HA\appleclass")
BASE     = ROOT / "calibration_data"
RESULTS  = ROOT / "calibration_results"
RESULTS.mkdir(exist_ok=True)

PANEL_REFLECTANCE = 0.75

# White-ref exposures used when capturing the 9-position grid (µs)
EXP_WHITE = {"ch1": 2500, "ch2": 1800, "ch3": 2300}

# Grid: row 0=Upper, 1=Middle, 2=Lower | col 0=Left, 1=Center, 2=Right
POSITIONS = [
    ["white_UL", "white_UC", "white_UR"],
    ["white_ML", "white_C",  "white_MR"],
    ["white_LL", "white_LC", "white_LR"],
]
POSITION_LABELS = [
    ["Upper-Left", "Upper-Center", "Upper-Right"],
    ["Mid-Left",   "Center",       "Mid-Right"],
    ["Lower-Left", "Lower-Center", "Lower-Right"],
]
ROW_LABELS = ["Upper", "Middle", "Lower"]
COL_LABELS = ["Left", "Center", "Right"]

CH_NAMES  = {1: "RGB (CH1)", 2: "NIR1 (CH2)", 3: "NIR2 (CH3)"}
CH_COLORS = {1: "#e85d04", 2: "#7209b7", 3: "#0077b6"}
CH_CMAPS  = {1: "Oranges", 2: "Purples", 3: "Blues"}
BG, BG2   = "#1a1a2e", "#0d1117"

MAX_FRAMES = 80  # average up to this many frames per position/channel


# ── HELPERS ───────────────────────────────────────────────────────────────────

def list_frames(ch_folder: Path) -> list[Path]:
    frames = (
        sorted(ch_folder.glob("*.bmp"))
        + sorted(ch_folder.glob("*.png"))
        + sorted(ch_folder.glob("*.jpg"))
        + sorted(ch_folder.glob("*.jpeg"))
    )
    return frames


def load_and_average(folder: Path, ch: int, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Average frames from folder/raw_frames/chN. Returns float32 array."""
    ch_folder = folder / "raw_frames" / f"ch{ch}"
    if not ch_folder.exists():
        raise FileNotFoundError(f"Channel folder not found: {ch_folder}")
    frames = list_frames(ch_folder)[:max_frames]
    if not frames:
        raise FileNotFoundError(f"No image frames in {ch_folder}")
    stack = [np.array(Image.open(f), dtype=np.float32) for f in frames]
    avg = np.mean(stack, axis=0).astype(np.float32)
    print(f"  Loaded {len(stack):3d} frames from {folder.name}/ch{ch}  shape={avg.shape}")
    return avg


def gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    return img.mean(axis=2) if img.ndim == 3 else img


def region_mean(img: np.ndarray, grid_row: int, grid_col: int) -> float:
    """Mean of inner 60% of the 1/3 image region matching panel placement."""
    g = gray(img)
    H, W = g.shape
    row_bounds = [(0, H // 3), (H // 3, 2 * H // 3), (2 * H // 3, H)]
    col_bounds = [(0, W // 3), (W // 3, 2 * W // 3), (2 * W // 3, W)]
    r0, r1 = row_bounds[grid_row]
    c0, c1 = col_bounds[grid_col]
    rh, cw = r1 - r0, c1 - c0
    r_pad, c_pad = rh // 5, cw // 5
    patch = g[r0 + r_pad : r1 - r_pad, c0 + c_pad : c1 - c_pad]
    return float(patch.mean())


def center_mean(img: np.ndarray, frac: float = 0.20) -> float:
    g = gray(img)
    H, W = g.shape
    cy, cx = H // 2, W // 2
    my, mx = int(H * frac), int(W * frac)
    return float(g[cy - my : cy + my, cx - mx : cx + mx].mean())


def clip_pct(img: np.ndarray, thresh: int = 250) -> float:
    g = gray(img)
    return float((g >= thresh).sum() / g.size * 100)


def dark_style(ax):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.tick_params(colors="white")


# ── STEP 1: DARK ──────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP A: Build Illumination Maps (NO drift correction)")
print(f"  Source : {BASE}")
print(f"  Output : {RESULTS}")
print("=" * 65)

print("\n" + "=" * 65)
print("  STEP 1: Dark Frames")
print("=" * 65)

dark: dict[int, np.ndarray] = {}
dark_mean: dict[int, float] = {}

for ch in [1, 2, 3]:
    src = BASE / f"dark_avg_ch{ch}.npy"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}\n"
            "Place averaged dark frames as dark_avg_ch1.npy / ch2 / ch3 in calibration_data/"
        )
    dark[ch] = np.load(src).astype(np.float32)
    np.save(RESULTS / f"dark_avg_ch{ch}.npy", dark[ch])
    dark_mean[ch] = center_mean(dark[ch])
    g = gray(dark[ch])
    print(
        f"  {CH_NAMES[ch]:14s}  mean={g.mean():6.2f}  "
        f"center={dark_mean[ch]:.2f}  shape={dark[ch].shape}"
    )


# ── STEP 2: WHITE GRID ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 2: White Reference Grid (9 positions)")
print("=" * 65)

white: dict[int, list] = {ch: [[None] * 3 for _ in range(3)] for ch in [1, 2, 3]}
white_mean: dict[int, np.ndarray] = {ch: np.zeros((3, 3), dtype=np.float64) for ch in [1, 2, 3]}

for row in range(3):
    for col in range(3):
        pos_name = POSITIONS[row][col]
        pos_folder = BASE / pos_name
        label = POSITION_LABELS[row][col]
        if not pos_folder.exists():
            raise FileNotFoundError(f"Missing position folder: {pos_folder}")
        print(f"\n  Position: {label} ({pos_name})")
        for ch in [1, 2, 3]:
            w = load_and_average(pos_folder, ch)
            white[ch][row][col] = w
            np.save(RESULTS / f"white_avg_{pos_name}_ch{ch}.npy", w)
            rm = region_mean(w, row, col)
            white_mean[ch][row, col] = rm
            net = rm - dark_mean[ch]
            print(
                f"    {CH_NAMES[ch]:12s} region={rm:.1f}  dark={dark_mean[ch]:.1f}  "
                f"net={net:.1f}  clip={clip_pct(w):.2f}%"
            )


# ── STEP 3: ILLUMINATION MAPS ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  STEP 3: Full-Resolution Illumination Maps (bicubic spline)")
print("=" * 65)

H, W = gray(white[1][0][0]).shape
rows_norm = np.array([0.0, 0.5, 1.0])
cols_norm = np.array([0.0, 0.5, 1.0])
x_full = np.linspace(0, 1, W)
y_full = np.linspace(0, 1, H)

illum_map: dict[int, np.ndarray] = {}
correct_map: dict[int, np.ndarray] = {}

for ch in [1, 2, 3]:
    net_grid = white_mean[ch] - dark_mean[ch]
    spline = RectBivariateSpline(rows_norm, cols_norm, net_grid, kx=2, ky=2)
    illum = spline(y_full, x_full).astype(np.float32)
    illum = np.clip(illum, 1.0, None)

    center_val = float(net_grid[1, 1])
    cmap = (illum / center_val).astype(np.float32)

    illum_map[ch] = illum
    correct_map[ch] = cmap
    np.save(RESULTS / f"illumination_map_ch{ch}.npy", illum)
    np.save(RESULTS / f"correction_map_ch{ch}.npy", cmap)

    vmin, vmax = float(illum.min()), float(illum.max())
    variation_pct = (vmax - vmin) / vmax * 100
    print(
        f"  {CH_NAMES[ch]:12s}  range {vmin:.1f}-{vmax:.1f} DN  "
        f"variation {variation_pct:.1f}%  center={center_val:.1f}"
    )


# ── FIGURES ───────────────────────────────────────────────────────────────────

print("\n  Generating figures...")

# Fig 1 — dark
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG)
for i, ch in enumerate([1, 2, 3]):
    g = gray(dark[ch])
    im = axes[i].imshow(g, cmap="hot", vmin=0, vmax=max(30, float(g.max()) + 1))
    dark_style(axes[i])
    axes[i].set_title(
        f"{CH_NAMES[ch]}\nMean: {g.mean():.1f} DN  Max: {g.max():.0f} DN",
        color="white", fontsize=11, pad=8,
    )
    cb = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="white")
fig.suptitle("Dark Frame Analysis | Lens Covered", color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "fig1_dark_frames.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig1_dark_frames.png")

# Fig 2 — 3x3 net grid
fig, big_axes = plt.subplots(3, 1, figsize=(12, 14))
fig.patch.set_facecolor(BG)
for i, ch in enumerate([1, 2, 3]):
    ax = big_axes[i]
    dark_style(ax)
    net_grid = white_mean[ch] - dark_mean[ch]
    im = ax.imshow(net_grid, cmap=CH_CMAPS[ch], aspect="auto",
                   vmin=net_grid.min() * 0.9, vmax=net_grid.max() * 1.05)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(COL_LABELS, color="white", fontsize=11)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(ROW_LABELS, color="white", fontsize=11)
    vmin_g, vmax_g = net_grid.min(), net_grid.max()
    for row in range(3):
        for col in range(3):
            val = net_grid[row, col]
            pct = (val - net_grid[1, 1]) / (net_grid[1, 1] + 1e-6) * 100
            brightness = (val - vmin_g) / (vmax_g - vmin_g + 1e-6)
            txt_color = "#111111" if brightness > 0.55 else "white"
            ax.text(
                col, row, f"{val:.0f}\n({pct:+.1f}%)",
                ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white" if txt_color == "#111111" else "#000000")],
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="white")
    ax.set_title(f"{CH_NAMES[ch]} | Net Illumination (DN) [dark-corrected]", color="white", fontsize=12, pad=8)
fig.suptitle(
    "White Reference Grid | 9-Position Illumination Map\n"
    "(Spectralon 75%; % relative to Center)",
    color="white", fontsize=13, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(RESULTS / "fig2_white_grid.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig2_white_grid.png")

# Fig 3 — profiles
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(BG)
x3 = [0, 1, 2]
for i, ch in enumerate([1, 2, 3]):
    net_grid = white_mean[ch] - dark_mean[ch]
    color = CH_COLORS[ch]
    ax = axes[0, i]
    dark_style(ax)
    for row, rl in enumerate(ROW_LABELS):
        ax.plot(x3, net_grid[row, :], "o-", color=color, alpha=1.0 - row * 0.2,
                linewidth=2, markersize=8, label=rl)
    ax.set_xticks(x3); ax.set_xticklabels(COL_LABELS, color="white")
    ax.set_title(f"{CH_NAMES[ch]}\nHorizontal Profile", color="white", fontsize=10)
    ax.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor="#444")
    ax.set_ylabel("Net DN", color="white")

    ax = axes[1, i]
    dark_style(ax)
    for col, cl in enumerate(COL_LABELS):
        ax.plot(x3, net_grid[:, col], "s--", color=color, alpha=1.0 - col * 0.2,
                linewidth=2, markersize=8, label=cl)
    ax.set_xticks(x3); ax.set_xticklabels(ROW_LABELS, color="white")
    ax.set_title("Vertical Profile", color="white", fontsize=10)
    ax.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor="#444")
    ax.set_ylabel("Net DN", color="white")
fig.suptitle("Illumination Uniformity | Horizontal & Vertical Gradients",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULTS / "fig3_illumination_profiles.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig3_illumination_profiles.png")

# Fig 4 — correction maps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
for i, ch in enumerate([1, 2, 3]):
    ax = axes[i]
    dark_style(ax)
    cmap = correct_map[ch]
    im = ax.imshow(cmap, cmap="RdYlGn", vmin=0.85, vmax=1.15, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Correction factor (1.0 = center)", color="white")
    ax.set_title(
        f"{CH_NAMES[ch]}\nRange: {cmap.min():.3f} - {cmap.max():.3f}",
        color="white", fontsize=11,
    )
    ax.set_xlabel("Belt Width (px)", color="white")
    ax.set_ylabel("Belt Length (px)", color="white")
fig.suptitle("Spatial Illumination Correction Map", color="white", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULTS / "fig4_correction_map.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig4_correction_map.png")

# Fig 5 — before/after on center white
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG)
for i, ch in enumerate([1, 2, 3]):
    raw_full = gray(white[ch][1][1]).astype(np.float32)
    dk_full = gray(dark[ch]).astype(np.float32)
    im_full = illum_map[ch]
    net_full = np.clip(raw_full - dk_full, 0, None)
    refl_full = np.clip(net_full / im_full * PANEL_REFLECTANCE, 0, 1)

    Hh, Ww = raw_full.shape
    r0, r1 = Hh // 3, 2 * Hh // 3
    c0, c1 = Ww // 3, 2 * Ww // 3
    rpad, cpad = (r1 - r0) // 5, (c1 - c0) // 5
    raw_panel = raw_full[r0 + rpad : r1 - rpad, c0 + cpad : c1 - cpad]
    refl_panel = refl_full[r0 + rpad : r1 - rpad, c0 + cpad : c1 - cpad]
    cov_raw = raw_panel.std() / (raw_panel.mean() + 1e-6) * 100
    cov_cal = refl_panel.std() / (refl_panel.mean() + 1e-6) * 100
    improvement = (cov_raw - cov_cal) / (cov_raw + 1e-6) * 100

    ax_raw, ax_cal = axes[0, i], axes[1, i]
    dark_style(ax_raw); dark_style(ax_cal)
    raw_disp = (raw_full - raw_full.min()) / (raw_full.max() - raw_full.min() + 1e-6)
    im1 = ax_raw.imshow(raw_disp, cmap="gray", vmin=0, vmax=1)
    ax_raw.set_title(
        f"{CH_NAMES[ch]}\nUncalibrated panel CoV={cov_raw:.1f}%  mean={raw_panel.mean():.0f} DN",
        color="white", fontsize=10,
    )
    im2 = ax_cal.imshow(refl_full, cmap="gray", vmin=0, vmax=1)
    ax_cal.set_title(
        f"Calibrated CoV={cov_cal:.1f}%  mean refl={refl_panel.mean():.4f}\n"
        f"Uniformity improvement: {improvement:.1f}%",
        color="white", fontsize=10,
    )
    for ax, im_obj in [(ax_raw, im1), (ax_cal, im2)]:
        rect = mpatches.Rectangle(
            (c0 + cpad, r0 + rpad), (c1 - c0 - 2 * cpad), (r1 - r0 - 2 * rpad),
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        plt.colorbar(im_obj, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white")
fig.suptitle(
    "Calibration Impact (green box = panel region) | CoV lower = more uniform",
    color="white", fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig.savefig(RESULTS / "fig5_calibration_impact.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig5_calibration_impact.png")

# Fig 6 — stitched NIR1 photos
fig6, axes6 = plt.subplots(3, 3, figsize=(18, 12))
fig6.patch.set_facecolor(BG)
center_net2 = white_mean[2][1, 1] - dark_mean[2]
for row in range(3):
    for col in range(3):
        ax = axes6[row, col]
        dark_style(ax)
        img_arr = gray(white[2][row][col]).astype(np.float32)
        ax.imshow(img_arr, cmap="gray", vmin=0, vmax=255, aspect="auto")
        img_H, img_W = img_arr.shape
        r0b = [0, img_H // 3, 2 * img_H // 3][row]
        r1b = [img_H // 3, 2 * img_H // 3, img_H][row]
        c0b = [0, img_W // 3, 2 * img_W // 3][col]
        c1b = [img_W // 3, 2 * img_W // 3, img_W][col]
        rh6, cw6 = r1b - r0b, c1b - c0b
        rpad6, cpad6 = rh6 // 5, cw6 // 5
        ax.add_patch(mpatches.Rectangle(
            (c0b + cpad6, r0b + rpad6), cw6 - 2 * cpad6, rh6 - 2 * rpad6,
            linewidth=2, edgecolor="lime", facecolor="none",
        ))
        net_val = white_mean[2][row, col] - dark_mean[2]
        pct = (net_val - center_net2) / (center_net2 + 1e-6) * 100
        ax.set_title(
            f"{ROW_LABELS[row]}-{COL_LABELS[col]}\nNet {net_val:.0f} DN ({pct:+.0f}% vs C)",
            color="white", fontsize=10,
        )
fig6.suptitle(
    "White Reference Grid — 9 Positions (NIR1)\nGreen box = sampled region",
    color="white", fontsize=13, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig6.savefig(RESULTS / "fig6_grid_stitched.png", dpi=130, bbox_inches="tight", facecolor=fig6.get_facecolor())
plt.close()
print("  Saved fig6_grid_stitched.png")

# Fig 7 — spline concept (NIR1)
net_grid7 = white_mean[2] - dark_mean[2]
y_dense = np.linspace(0, 1, 200)
x_dense = np.linspace(0, 1, 200)
spline7 = RectBivariateSpline(rows_norm, cols_norm, net_grid7, kx=2, ky=2)
surface7 = spline7(y_dense, x_dense)

fig7 = plt.figure(figsize=(18, 7))
fig7.patch.set_facecolor(BG)
ax7a = fig7.add_subplot(1, 3, 1)
dark_style(ax7a)
ax7a.imshow(net_grid7, cmap="Purples", aspect="auto",
            vmin=net_grid7.min() * 0.85, vmax=net_grid7.max() * 1.05)
for r in range(3):
    for c in range(3):
        ax7a.text(c, r, f"{net_grid7[r, c]:.0f} DN", ha="center", va="center",
                  color="white", fontsize=13, fontweight="bold")
ax7a.set_xticks([0, 1, 2]); ax7a.set_xticklabels(COL_LABELS, color="white")
ax7a.set_yticks([0, 1, 2]); ax7a.set_yticklabels(ROW_LABELS, color="white")
ax7a.set_title("STEP 1: 9 Measured Points", color="white", fontsize=11)

ax7b = fig7.add_subplot(1, 3, 2)
dark_style(ax7b)
im7b = ax7b.imshow(surface7, cmap="Purples", aspect="auto",
                   vmin=net_grid7.min() * 0.85, vmax=net_grid7.max() * 1.05,
                   extent=[0, 1, 1, 0])
for r, rv in enumerate([0.0, 0.5, 1.0]):
    for c, cv in enumerate([0.0, 0.5, 1.0]):
        ax7b.scatter(cv, rv, color="lime", s=80, zorder=5)
ax7b.set_xticks([0, 0.5, 1]); ax7b.set_xticklabels(COL_LABELS, color="white")
ax7b.set_yticks([0, 0.5, 1]); ax7b.set_yticklabels(ROW_LABELS, color="white")
ax7b.set_title("STEP 2: Spline Surface", color="white", fontsize=11)
plt.colorbar(im7b, ax=ax7b, fraction=0.046).ax.tick_params(colors="white")

ax7c = fig7.add_subplot(1, 3, 3, projection="3d")
XX7, YY7 = np.meshgrid(x_dense, y_dense)
ax7c.plot_surface(XX7, YY7, surface7, cmap="Purples", alpha=0.85, edgecolor="none")
for r, rv in enumerate([0.0, 0.5, 1.0]):
    for c, cv in enumerate([0.0, 0.5, 1.0]):
        ax7c.scatter(cv, rv, net_grid7[r, c], color="lime", s=60, zorder=5)
ax7c.set_title("STEP 3: 3D Illumination", color="white", fontsize=11)
ax7c.tick_params(colors="white")
fig7.suptitle("Spline Interpolation — NIR1 dark-corrected net DN",
              color="white", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig7.savefig(RESULTS / "fig7_spline_concept.png", dpi=130, bbox_inches="tight", facecolor=fig7.get_facecolor())
plt.close()
print("  Saved fig7_spline_concept.png")


# ── REPORT ────────────────────────────────────────────────────────────────────

lines: list[str] = []
L = lines.append
L("=" * 70)
L("  MULTISPECTRAL CALIBRATION REPORT (NO DRIFT)")
L("  MSU Apple Grading | JAI FS-3200T-10GE")
L("  Script: a_build_illumination_maps.py")
L("=" * 70)
L("")
L("SECTION 1: SETTINGS")
L("-" * 70)
L(f"  Data     : {BASE}")
L(f"  Results  : {RESULTS}")
L("  White ref exposures:")
for ch, us in EXP_WHITE.items():
    L(f"    {ch}: {us} us")
L("  Panel reflectance: 0.75 (Spectralon)")
L("  Drift correction : DISABLED (stable regulated lamp)")
L("")
L("SECTION 2: DARK")
L("-" * 70)
for ch in [1, 2, 3]:
    g = gray(dark[ch])
    L(f"  {CH_NAMES[ch]:14s} mean={g.mean():.2f}  center={dark_mean[ch]:.2f}")
L("")
L("SECTION 3: WHITE NET GRID (DN)")
L("-" * 70)
for ch in [1, 2, 3]:
    net_grid = white_mean[ch] - dark_mean[ch]
    L(f"  {CH_NAMES[ch]}")
    L("         Left    Center   Right")
    for row, rl in enumerate(["Upper ", "Middle", "Lower "]):
        vals = "   ".join(f"{net_grid[row, col]:7.1f}" for col in range(3))
        L(f"    {rl}  {vals}")
    variation = (net_grid.max() - net_grid.min()) / net_grid.max() * 100
    L(f"    Variation: {variation:.1f}%")
    L("")
L("SECTION 4: APPLY FORMULA (see b_apply_calibration.py)")
L("-" * 70)
L("  reflectance(x,y) = (raw - dark) / illumination_map")
L("                     * (EXP_WHITE / EXP_APPLE)")
L("                     * 0.75")
L("")
L("  Typical: RGB white=2500us, apple=5000us -> ratio=0.5")
L("           NIR1/NIR2 same exposure -> ratio=1.0")
L("")
L("=" * 70)

report = "\n".join(lines)
(RESULTS / "calibration_report.txt").write_text(report, encoding="utf-8")
print(report)
print(f"\n  All outputs saved to: {RESULTS}")
