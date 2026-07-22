"""
calibration_analysis.py
=======================
Multispectral White Reference Panel Analysis Tool
MSU Apple Grading Project

Usage:
    python calibration_analysis.py <folder_path> [--rgb-exp N] [--nir1-exp N] [--nir2-exp N] [--label NAME]

Examples:
    python calibration_analysis.py cals/cal6
    python calibration_analysis.py cals/cal6 --rgb-exp 2500 --nir1-exp 1800 --nir2-exp 2300 --label "Hal+LED+AWB"
    python calibration_analysis.py all
    python calibration_analysis.py cals/cal6 --compare cals/cal5 cals/cal4

The script expects files named:
    frame_000001_rgb.jpg
    frame_000001_nir1.jpg
    frame_000001_nir2.jpg
"""

import argparse
import sys
import os
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# Panel ROI: fraction of image height/width used as the center crop
# Default assumes panel fills roughly the center 40% of the frame
PANEL_CROP_FRACTION = 0.20   # each side — so panel = centre 40% of frame

# Clipping thresholds
CLIP_HARD  = 250   # pixel value counted as clipped
CLIP_NEAR  = 230   # pixel value counted as near-clip (warning zone)

# Verdict thresholds
VERDICT_CLIP_FAIL   = 5.0    # % clip → NOT USABLE
VERDICT_CLIP_WARN   = 1.0    # % clip → MARGINAL
VERDICT_NEAR_WARN   = 20.0   # % near-clip → CAUTION
VERDICT_TOO_DARK    = 80.0   # gray mean below this → TOO DARK


# ─────────────────────────────────────────────
#  CORE ANALYSIS
# ─────────────────────────────────────────────

def load_panel_region(path: str):
    """Load image and crop to the center panel ROI. Returns (full_img, panel_crop)."""
    img = np.array(Image.open(path))
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    my = int(H * PANEL_CROP_FRACTION)
    mx = int(W * PANEL_CROP_FRACTION)
    panel = img[cy - my:cy + my, cx - mx:cx + mx]
    return img, panel


def channel_stats(arr: np.ndarray) -> dict:
    """Compute statistics for a single-channel 2D array."""
    a = arr.astype(float)
    return {
        "mean":     round(float(a.mean()), 1),
        "std":      round(float(a.std()),  1),
        "min":      int(a.min()),
        "max":      int(a.max()),
        "clip_pct": round(float((a >= CLIP_HARD).sum() / a.size * 100), 2),
        "near_pct": round(float((a >= CLIP_NEAR).sum() / a.size * 100), 1),
    }


def verdict(stats: dict) -> str:
    """Return a one-line verdict string based on statistics."""
    if stats["clip_pct"] > VERDICT_CLIP_FAIL:
        return "NOT USABLE - severe clipping"
    if stats["clip_pct"] > VERDICT_CLIP_WARN:
        return f"MARGINAL - {stats['clip_pct']}% pixels clipping"
    if stats["near_pct"] > VERDICT_NEAR_WARN:
        return f"CAUTION - {stats['near_pct']}% near clip threshold"
    if stats["mean"] < VERDICT_TOO_DARK:
        return f"TOO DARK - mean {stats['mean']} below {VERDICT_TOO_DARK}"
    return "USABLE"


def analyze_band(path: str, band: str, exposure: int) -> dict:
    """
    Analyze one image file (one band).
    Returns a dict of all statistics for the panel region.
    """
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    img, panel = load_panel_region(path)
    is_color = img.ndim == 3

    result = {
        "band":     band,
        "exposure": exposure,
        "path":     path,
    }

    if is_color:
        # Per-channel stats
        channels = {}
        for i, ch in enumerate(["R", "G", "B"]):
            channels[ch] = channel_stats(panel[:, :, i])
        result["channels"] = channels

        # Overall grayscale (average of channels)
        gray = panel.mean(axis=2)
        result["gray"] = channel_stats(gray)
        result["verdict"] = verdict(result["gray"])

        # Balance check: are R, G, B means equal? (target for white reference)
        means = [channels[c]["mean"] for c in ["R", "G", "B"]]
        result["channel_balance"] = round(max(means) - min(means), 1)

    else:
        # Grayscale (NIR)
        gray_stats = channel_stats(panel)
        result["gray"] = gray_stats
        result["verdict"] = verdict(gray_stats)

    return result


def analyze_folder(folder: str, rgb_exp: int = 0, nir1_exp: int = 0,
                   nir2_exp: int = 0, label: str = "") -> dict:
    """Analyze all three bands in a calibration folder."""
    bands = {
        "rgb":  (os.path.join(folder, "frame_000001_rgb.jpg"),  rgb_exp),
        "nir1": (os.path.join(folder, "frame_000001_nir1.jpg"), nir1_exp),
        "nir2": (os.path.join(folder, "frame_000001_nir2.jpg"), nir2_exp),
    }
    results = {}
    for band, (path, exp) in bands.items():
        results[band] = analyze_band(path, band, exp)

    return {
        "folder": folder,
        "label":  label or os.path.basename(folder),
        "bands":  results,
    }


# ─────────────────────────────────────────────
# ---------------------------------------------
#  PRINTING
# ---------------------------------------------

def print_band(result: dict):
    band = result.get("band", "?").upper()
    exp  = result.get("exposure", 0)
    print(f"\n  {band}  ({exp} us)")

    if "error" in result:
        print(f"    ERROR: {result['error']}")
        return

    gray = result["gray"]
    print(f"    Gray mean     : {gray['mean']}")
    print(f"    Gray max      : {gray['max']}")
    print(f"    Std dev       : {gray['std']}  (illumination spread - lower = more uniform)")
    print(f"    Clip >={CLIP_HARD}    : {gray['clip_pct']}%")
    print(f"    Near clip >={CLIP_NEAR}: {gray['near_pct']}%")

    if "channels" in result:
        print(f"    Per channel:")
        for ch, s in result["channels"].items():
            clip_flag = "  <-- CLIPPING" if s["clip_pct"] > 1 else ""
            print(f"      {ch}: mean={s['mean']:<7} max={s['max']:<5} clip={s['clip_pct']}%{clip_flag}")
        bal = result.get("channel_balance", None)
        bal_note = "BALANCED" if bal is not None and bal < 10 else f"UNBALANCED (spread={bal})"
        print(f"    R/G/B balance : {bal_note}")

    print(f"    Verdict: {result['verdict']}")


def print_report(analysis: dict):
    print()
    print("=" * 65)
    print(f"  CALIBRATION PANEL ANALYSIS")
    print(f"  Folder : {analysis['folder']}")
    print(f"  Label  : {analysis['label']}")
    print("=" * 65)

    for band_key in ["rgb", "nir1", "nir2"]:
        res = analysis["bands"].get(band_key)
        if res:
            print_band(res)

    print()
    print("-" * 65)
    print("  SUMMARY")
    print("-" * 65)
    for band_key in ["rgb", "nir1", "nir2"]:
        res = analysis["bands"].get(band_key, {})
        v = res.get("verdict", "No data")
        exp = res.get("exposure", 0)
        print(f"  {band_key.upper():<6} ({exp:>4} us)  -->  {v}")
    print()


def print_comparison_table(analyses: list):
    """Print a side-by-side comparison of multiple calibration runs."""
    print()
    print("=" * 80)
    print("  COMPARISON TABLE - RGB CHANNEL CLIPPING")
    print("=" * 80)
    header = f"{'Name':<8} {'Label':<18} {'Exp':>5}  {'R_mean':>7} {'R_clip%':>8}  {'G_mean':>7} {'G_clip%':>8}  {'B_mean':>7} {'B_clip%':>8}"
    print(header)
    print("-" * 80)

    for a in analyses:
        rgb = a["bands"].get("rgb", {})
        if "error" in rgb or "channels" not in rgb:
            print(f"{os.path.basename(a['folder']):<8}  (RGB not available)")
            continue
        ch = rgb["channels"]
        name  = os.path.basename(a["folder"])
        label = a["label"]
        exp   = rgb.get("exposure", 0)
        print(f"{name:<8} {label:<18} {exp:>5}  "
              f"{ch['R']['mean']:>7} {ch['R']['clip_pct']:>7}%  "
              f"{ch['G']['mean']:>7} {ch['G']['clip_pct']:>7}%  "
              f"{ch['B']['mean']:>7} {ch['B']['clip_pct']:>7}%")

    print()
    print("  NIR BANDS")
    print("-" * 80)
    nir_header = f"{'Name':<8} {'Label':<18}  {'N1_exp':>6} {'N1_mean':>8} {'N1_clip':>8}  {'N2_exp':>6} {'N2_mean':>8} {'N2_clip':>8}"
    print(nir_header)
    print("-" * 80)
    for a in analyses:
        name  = os.path.basename(a["folder"])
        label = a["label"]
        n1 = a["bands"].get("nir1", {})
        n2 = a["bands"].get("nir2", {})
        n1g = n1.get("gray", {})
        n2g = n2.get("gray", {})
        print(f"{name:<8} {label:<18}  "
              f"{n1.get('exposure',0):>6} {n1g.get('mean','--'):>8} {n1g.get('clip_pct','--'):>7}%  "
              f"{n2.get('exposure',0):>6} {n2g.get('mean','--'):>8} {n2g.get('clip_pct','--'):>7}%")
    print()


# ─────────────────────────────────────────────
#  BATCH: ALL KNOWN CALS
# ─────────────────────────────────────────────

# Recorded settings for each calibration run
KNOWN_CALS = {
    "cal1": {"rgb_exp": 5000, "nir1_exp": 2100, "nir2_exp": 2500, "label": "Hal+LED  noWB"},
    "cal2": {"rgb_exp": 4000, "nir1_exp":    0, "nir2_exp":    0, "label": "Hal+LED  noWB"},
    "cal3": {"rgb_exp": 3000, "nir1_exp": 2100, "nir2_exp": 2500, "label": "Hal only noWB"},
    "cal4": {"rgb_exp": 4000, "nir1_exp": 2100, "nir2_exp": 2500, "label": "Hal only AWB "},
    "cal5": {"rgb_exp": 3000, "nir1_exp": 1800, "nir2_exp": 2300, "label": "Hal only AWB "},
    "cal6": {"rgb_exp": 2500, "nir1_exp": 1800, "nir2_exp": 2300, "label": "Hal+LED AWB [CHOSEN]"},
}


def run_all_known(base_folder: str):
    """Run analysis on all known cal folders under base_folder."""
    results = []
    for cal_name, cfg in KNOWN_CALS.items():
        folder = os.path.join(base_folder, cal_name)
        if not os.path.isdir(folder):
            print(f"  Skipping {cal_name} — folder not found: {folder}")
            continue
        a = analyze_folder(
            folder,
            rgb_exp  = cfg["rgb_exp"],
            nir1_exp = cfg["nir1_exp"],
            nir2_exp = cfg["nir2_exp"],
            label    = cfg["label"],
        )
        print_report(a)
        results.append(a)
    if results:
        print_comparison_table(results)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze white reference panel calibration images."
    )
    parser.add_argument(
        "folder", nargs="?",
        help="Path to calibration folder (e.g. cals/cal6). "
             "Use 'all' to analyze all known cals under --base."
    )
    parser.add_argument("--rgb-exp",  type=int, default=0,
                        help="RGB exposure in microseconds")
    parser.add_argument("--nir1-exp", type=int, default=0,
                        help="NIR1 exposure in microseconds")
    parser.add_argument("--nir2-exp", type=int, default=0,
                        help="NIR2 exposure in microseconds")
    parser.add_argument("--label", type=str, default="",
                        help="Human-readable label for this run")
    parser.add_argument("--compare", nargs="+", metavar="FOLDER",
                        help="Additional folders to compare against the main folder")
    parser.add_argument("--base", type=str,
                        default=r"S:\MSU_Research\apple_class\cals",
                        help="Base directory containing cal1, cal2, ... folders (used with 'all')")

    args = parser.parse_args()

    if args.folder is None or args.folder.lower() == "all":
        print(f"\nRunning all known calibration folders under: {args.base}")
        run_all_known(args.base)
        return

    # Single folder analysis
    a = analyze_folder(
        args.folder,
        rgb_exp  = args.rgb_exp,
        nir1_exp = args.nir1_exp,
        nir2_exp = args.nir2_exp,
        label    = args.label,
    )
    print_report(a)

    # Comparison mode
    if args.compare:
        all_analyses = [a]
        for comp_folder in args.compare:
            # Try to look up known settings
            name = os.path.basename(comp_folder)
            cfg  = KNOWN_CALS.get(name, {})
            comp = analyze_folder(
                comp_folder,
                rgb_exp  = cfg.get("rgb_exp",  0),
                nir1_exp = cfg.get("nir1_exp", 0),
                nir2_exp = cfg.get("nir2_exp", 0),
                label    = cfg.get("label",    name),
            )
            all_analyses.append(comp)
        print_comparison_table(all_analyses)


if __name__ == "__main__":
    main()
