"""
b_apply_calibration.py
======================
STEP B -- Apply spatial calibration to apple run frames.

NO per-run lamp drift scale (stable regulated lamp).

Formula (per channel, per pixel):
  reflectance = (raw - dark) / illumination_map
                * (EXP_WHITE / EXP_APPLE)
                * PANEL_REFL

Exposure (locked workflow):
  Channel     White ref     Apple run     Ratio
  RGB (ch1)   2500 us       5000 us       0.5
  NIR1 (ch2)  1800 us       1800 us       1.0
  NIR2 (ch3)  2300 us       2300 us       1.0

Reads:
  calibration_results/dark_avg_ch*.npy
  calibration_results/illumination_map_ch*.npy
  formal_runs/<run>/raw_frames/ch*/frame_*.bmp

Writes (inside each run):
  calibrated_frames/ch1|ch2|ch3/*.npy      float32 reflectance (H,W)
  calibrated_frames/ch1_rgb/*.npy          float32 reflectance (H,W,3)
  before_after/*_before_after.png          5 best frames (unless --no-vis)
  calibration_stats.csv
  calibration_stats_summary.txt

Usage:
  cd D:\\HA\\appleclass\\calibration_scripts_01

  # one run
  python b_apply_calibration.py --run proc_run1_hc

  # all formal runs
  python b_apply_calibration.py --all

  # override exposures if needed
  python b_apply_calibration.py --run proc_run1_hc --rgb-exp 5000 --nir1-exp 1800 --nir2-exp 2300
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

ROOT     = Path(r"D:\HA\appleclass")
CAL_DIR  = ROOT / "calibration_results"
RUNS_DIR = ROOT / "formal_runs"

# White-ref exposures used when building illumination maps (must match STEP A)
EXP_WHITE = {"ch1": 2500, "ch2": 1800, "ch3": 2300}
# Defaults for apple capture
EXP_APPLE_DEFAULT = {"ch1": 5000, "ch2": 1800, "ch3": 2300}

PANEL_REFL = 0.75
CH_NAMES = {"ch1": "RGB", "ch2": "NIR1", "ch3": "NIR2"}


def list_frames(folder: Path) -> list[Path]:
    return (
        sorted(folder.glob("*.bmp"))
        + sorted(folder.glob("*.png"))
        + sorted(folder.glob("*.jpg"))
        + sorted(folder.glob("*.jpeg"))
    )


def to_gray(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return arr.mean(axis=2) if arr.ndim == 3 else arr


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.float32)


def load_calibration():
    dark, illum, dark_rgb = {}, {}, {}
    for i in [1, 2, 3]:
        ch = f"ch{i}"
        d_path = CAL_DIR / f"dark_avg_ch{i}.npy"
        i_path = CAL_DIR / f"illumination_map_ch{i}.npy"
        if not d_path.exists() or not i_path.exists():
            raise FileNotFoundError(
                f"Missing calibration maps in {CAL_DIR}\n"
                "Run a_build_illumination_maps.py first."
            )
        d = np.load(d_path).astype(np.float32)
        ill = np.load(i_path).astype(np.float32)
        dark[ch] = to_gray(d)
        illum[ch] = to_gray(ill)
        if ch == "ch1":
            dark_rgb["ch1"] = d if d.ndim == 3 else np.stack([d, d, d], axis=2)
    print("  Calibration maps loaded.")
    for ch in ["ch1", "ch2", "ch3"]:
        cy, cx = illum[ch].shape[0] // 2, illum[ch].shape[1] // 2
        print(f"    {CH_NAMES[ch]:4s} illum center = {illum[ch][cy, cx]:.1f} DN")
    return dark, illum, dark_rgb


def calibrate_gray(raw_arr, ch, dark, illum, exp_apple) -> np.ndarray:
    """Grayscale reflectance for one channel."""
    raw = to_gray(raw_arr)
    net = np.clip(raw - dark[ch], 0, None)
    ratio = EXP_WHITE[ch] / exp_apple[ch]
    refl = net / np.clip(illum[ch], 1.0, None) * ratio * PANEL_REFL
    return np.clip(refl, 0, 1).astype(np.float32)


def calibrate_rgb(raw_arr, dark_3ch, illum_1ch, exp_apple_ch1) -> np.ndarray:
    """Per-channel RGB reflectance (H,W,3). Shared spatial illum map."""
    raw = np.asarray(raw_arr, dtype=np.float32)
    if raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=2)
    ratio = EXP_WHITE["ch1"] / exp_apple_ch1
    out = np.empty_like(raw)
    denom = np.clip(illum_1ch, 1.0, None)
    for c in range(3):
        net = np.clip(raw[:, :, c] - dark_3ch[:, :, c], 0, None)
        out[:, :, c] = net / denom * ratio * PANEL_REFL
    return np.clip(out, 0, 1).astype(np.float32)


def apple_mask(refl_nir1: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    return refl_nir1 > threshold


def frame_stats(raw_gray, refl, mask) -> dict:
    return {
        "raw_mean_full": float(raw_gray.mean()),
        "raw_mean_apple": float(raw_gray[mask].mean()) if mask.any() else float("nan"),
        "raw_std_apple": float(raw_gray[mask].std()) if mask.any() else float("nan"),
        "cal_mean_full": float(refl.mean()),
        "cal_mean_apple": float(refl[mask].mean()) if mask.any() else float("nan"),
        "cal_std_apple": float(refl[mask].std()) if mask.any() else float("nan"),
        "cal_min_apple": float(refl[mask].min()) if mask.any() else float("nan"),
        "cal_max_apple": float(refl[mask].max()) if mask.any() else float("nan"),
        "apple_pixels": int(mask.sum()),
    }


def process_run(run_dir: Path, exp_apple: dict, make_vis: bool = True) -> None:
    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw_frames"
    if not (raw_dir / "ch1").exists():
        raise FileNotFoundError(f"No raw_frames/ch1 in {run_dir}")

    out_cal = run_dir / "calibrated_frames"
    out_vis = run_dir / "before_after"
    out_cal.mkdir(exist_ok=True)
    if make_vis:
        out_vis.mkdir(exist_ok=True)
    for ch in ["ch1", "ch1_rgb", "ch2", "ch3"]:
        (out_cal / ch).mkdir(exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  APPLY CALIBRATION  |  {run_dir.name}")
    print(f"{'=' * 65}")
    print(f"  Run folder : {run_dir}")
    print(
        f"  Apple expo : RGB={exp_apple['ch1']}us  "
        f"NIR1={exp_apple['ch2']}us  NIR2={exp_apple['ch3']}us"
    )
    for ch in ["ch1", "ch2", "ch3"]:
        r = EXP_WHITE[ch] / exp_apple[ch]
        print(
            f"    {CH_NAMES[ch]:4s}  white={EXP_WHITE[ch]} / apple={exp_apple[ch]}  "
            f"ratio={r:.4f}"
        )

    dark, illum, dark_rgb = load_calibration()

    frames = list_frames(raw_dir / "ch1")
    if not frames:
        raise FileNotFoundError(f"No frames in {raw_dir / 'ch1'}")
    frame_names = [f.name for f in frames]
    print(f"  {len(frame_names)} frames")

    all_stats = []
    apple_pixel_counts = []

    print("\n  Calibrating...")
    for i, fname in enumerate(frame_names):
        paths = {ch: raw_dir / ch / fname for ch in ["ch1", "ch2", "ch3"]}
        for ch, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")

        raw_rgb = load_image(paths["ch1"])
        raw = {
            "ch1": to_gray(raw_rgb),
            "ch2": to_gray(load_image(paths["ch2"])),
            "ch3": to_gray(load_image(paths["ch3"])),
        }
        refl = {
            ch: calibrate_gray(raw[ch], ch, dark, illum, exp_apple)
            for ch in ["ch1", "ch2", "ch3"]
        }
        # overwrite ch1 gray from full array for consistency with color path inputs
        refl["ch1"] = calibrate_gray(raw_rgb, "ch1", dark, illum, exp_apple)
        refl_ch1_rgb = calibrate_rgb(raw_rgb, dark_rgb["ch1"], illum["ch1"], exp_apple["ch1"])

        stem = Path(fname).stem
        for ch in ["ch1", "ch2", "ch3"]:
            np.save(out_cal / ch / f"{stem}.npy", refl[ch])
        np.save(out_cal / "ch1_rgb" / f"{stem}.npy", refl_ch1_rgb)

        mask = apple_mask(refl["ch2"])
        apple_pixel_counts.append(int(mask.sum()))

        row = {"frame": fname, "apple_pixels": int(mask.sum())}
        for ch in ["ch1", "ch2", "ch3"]:
            s = frame_stats(raw[ch], refl[ch], mask)
            for k, v in s.items():
                row[f"{ch}_{k}"] = v
        all_stats.append(row)

        if (i + 1) % 50 == 0 or i == len(frame_names) - 1:
            print(f"    [{i + 1}/{len(frame_names)}] {fname}  apple_px={mask.sum():,}")

    csv_path = run_dir / "calibration_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        w.writeheader()
        w.writerows(all_stats)
    print(f"\n  Stats CSV: {csv_path}")

    if make_vis:
        best_5 = sorted(range(len(frame_names)), key=lambda i: apple_pixel_counts[i], reverse=True)[:5]
        print("\n  Saving before/after for 5 best frames...")
        for idx in best_5:
            fname = frame_names[idx]
            stem = Path(fname).stem
            raw_imgs = {
                ch: to_gray(load_image(raw_dir / ch / fname)) for ch in ["ch1", "ch2", "ch3"]
            }
            raw_rgb_img = load_image(raw_dir / "ch1" / fname)
            cal_imgs = {ch: np.load(out_cal / ch / f"{stem}.npy") for ch in ["ch1", "ch2", "ch3"]}
            cal_rgb_img = np.load(out_cal / "ch1_rgb" / f"{stem}.npy")
            mask = apple_mask(cal_imgs["ch2"])

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.patch.set_facecolor("#1a1a2e")
            for col, ch in enumerate(["ch1", "ch2", "ch3"]):
                am = float(cal_imgs[ch][mask].mean()) if mask.any() else float("nan")
                ax = axes[0, col]
                ax.set_facecolor("#0d1117")
                if ch == "ch1":
                    ax.imshow(np.clip(raw_rgb_img / 255.0, 0, 1))
                    ax.set_title(
                        f"{CH_NAMES[ch]} RAW [COLOR]\n"
                        f"mean(apple)={raw_imgs[ch][mask].mean():.1f} DN",
                        color="white", fontsize=10,
                    )
                else:
                    im = ax.imshow(raw_imgs[ch], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(
                        f"{CH_NAMES[ch]} RAW\nmean(apple)={raw_imgs[ch][mask].mean():.1f} DN",
                        color="white", fontsize=10,
                    )
                    plt.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(colors="white")
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#444")

                ax = axes[1, col]
                ax.set_facecolor("#0d1117")
                if ch == "ch1":
                    ax.imshow(np.clip(cal_rgb_img / 0.8, 0, 1))
                    ax.set_title(
                        f"{CH_NAMES[ch]} CAL [COLOR]\nmean refl(apple)={am:.4f}",
                        color="white", fontsize=10,
                    )
                else:
                    im2 = ax.imshow(cal_imgs[ch], cmap="gray", vmin=0, vmax=0.8)
                    ax.set_title(
                        f"{CH_NAMES[ch]} CAL\nmean refl(apple)={am:.4f}",
                        color="white", fontsize=10,
                    )
                    cb = plt.colorbar(im2, ax=ax, fraction=0.046)
                    cb.ax.tick_params(colors="white")
                    cb.set_label("Reflectance", color="white", fontsize=8)
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#444")

            ratio_rgb = EXP_WHITE["ch1"] / exp_apple["ch1"]
            fig.suptitle(
                f"{fname} | apple_px={mask.sum():,} | "
                f"RGB exp ratio {EXP_WHITE['ch1']}/{exp_apple['ch1']}={ratio_rgb:.2f}",
                color="white", fontsize=10, fontweight="bold",
            )
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            out_path = out_vis / f"{stem}_before_after.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close()
            print(f"    {out_path.name}")

    best_idx = int(np.argmax(apple_pixel_counts))
    best_stats = all_stats[best_idx]
    lines = [
        "=" * 60,
        f"  CALIBRATION APPLIED — {run_dir.name}",
        "=" * 60,
        "",
        f"  Frames processed       : {len(frame_names)}",
        f"  Best frame (most apple): {frame_names[best_idx]}",
        f"  Apple pixels in best   : {apple_pixel_counts[best_idx]:,}",
        "",
        "  Exposure correction:",
    ]
    for ch in ["ch1", "ch2", "ch3"]:
        r = EXP_WHITE[ch] / exp_apple[ch]
        lines.append(
            f"    {CH_NAMES[ch]:4s}  white={EXP_WHITE[ch]} / apple={exp_apple[ch]}  ratio={r:.4f}"
        )
    lines += ["", "  BEST FRAME — Apple region reflectance", "-" * 60]
    for ch in ["ch1", "ch2", "ch3"]:
        rm = best_stats[f"{ch}_raw_mean_apple"]
        cm = best_stats[f"{ch}_cal_mean_apple"]
        cs = best_stats[f"{ch}_cal_std_apple"]
        cmin = best_stats[f"{ch}_cal_min_apple"]
        cmax = best_stats[f"{ch}_cal_max_apple"]
        lines += [
            f"  {CH_NAMES[ch]:4s}",
            f"    Raw mean (apple)       : {rm / 255:.4f}  ({rm:.1f} DN)",
            f"    Calibrated mean refl   : {cm:.4f}",
            f"    Calibrated std         : {cs:.4f}",
            f"    Calibrated range       : {cmin:.4f} – {cmax:.4f}",
            "",
        ]
    lines += [
        "  Formula:",
        "    refl = (raw - dark) / illum * (EXP_WHITE / EXP_APPLE) * 0.75",
        "",
        f"  Output: {out_cal}",
        "=" * 60,
    ]
    summary = "\n".join(lines)
    (run_dir / "calibration_stats_summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply calibration to apple run(s).")
    ap.add_argument("--run", help="Run folder name under formal_runs/, or absolute path")
    ap.add_argument("--all", action="store_true", help="Process every folder in formal_runs/")
    ap.add_argument("--rgb-exp", type=int, default=EXP_APPLE_DEFAULT["ch1"])
    ap.add_argument("--nir1-exp", type=int, default=EXP_APPLE_DEFAULT["ch2"])
    ap.add_argument("--nir2-exp", type=int, default=EXP_APPLE_DEFAULT["ch3"])
    ap.add_argument("--no-vis", action="store_true", help="Skip before/after PNGs (faster)")
    args = ap.parse_args()

    if not args.all and not args.run:
        ap.error("Provide --run NAME or --all")

    exp_apple = {"ch1": args.rgb_exp, "ch2": args.nir1_exp, "ch3": args.nir2_exp}

    # Sanity: print ratio check for RGB 2500/5000
    r_rgb = EXP_WHITE["ch1"] / exp_apple["ch1"]
    print("Exposure check:")
    print(f"  RGB  white={EXP_WHITE['ch1']} / apple={exp_apple['ch1']} = {r_rgb:.4f}  "
          f"{'(OK: 0.5)' if abs(r_rgb - 0.5) < 1e-6 else '(custom)'}")
    for ch in ["ch2", "ch3"]:
        r = EXP_WHITE[ch] / exp_apple[ch]
        print(f"  {CH_NAMES[ch]:4s} white={EXP_WHITE[ch]} / apple={exp_apple[ch]} = {r:.4f}  "
              f"{'(OK: 1.0)' if abs(r - 1.0) < 1e-6 else '(custom)'}")

    runs: list[Path] = []
    if args.all:
        runs = sorted(
            [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "raw_frames" / "ch1").exists()],
            key=lambda p: p.name,
        )
        if not runs:
            raise SystemExit(f"No runs found in {RUNS_DIR}")
    else:
        p = Path(args.run)
        runs = [p if p.is_absolute() or p.exists() else RUNS_DIR / args.run]

    for run_dir in runs:
        process_run(run_dir, exp_apple, make_vis=not args.no_vis)

    print(f"\nDone. Processed {len(runs)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
