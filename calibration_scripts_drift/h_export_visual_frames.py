"""
h_export_visual_frames.py
===========================
Export calibrated reflectance .npy files into full-resolution 8-bit visual images (JPG/PNG)
for annotation in CVAT, Roboflow, or Label Studio.

Output location per run:
  S:\\MSU_Research\\apple_class\\formal_runs\\<run_name>\\calibrated_visual_frames\\
    ├── rgb/     (Calibrated RGB true-color JPEGs)
    ├── nir1/    (Calibrated NIR1 710nm JPEGs)
    └── nir2/    (Calibrated NIR2 800nm JPEGs)

Usage:
  python h_export_visual_frames.py --run run1_fresh_hc
  python h_export_visual_frames.py --all
  python h_export_visual_frames.py --all --stride 3
"""

import argparse
import glob
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Paths
FORMAL_RUNS_DIR = Path(r'S:\MSU_Research\apple_class\formal_runs')


def parse_args():
    parser = argparse.ArgumentParser(description='Export calibrated .npy arrays to visual images for annotation.')
    parser.add_argument('--run', type=str, help='Run folder name, e.g. run1_fresh_hc')
    parser.add_argument('--all', action='store_true', help='Process all run folders in formal_runs')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride (default: 1 for all frames, 3 for every 3rd)')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format (jpg or png)')
    return parser.parse_args()


def export_run(run_dir: Path, stride: int = 1, fmt: str = 'jpg'):
    print(f"\n{'='*70}")
    print(f" Processing Run: {run_dir.name}")
    print(f"{'='*70}")

    cal_dir = run_dir / 'calibrated'
    if not cal_dir.exists():
        print(f" [SKIP] No 'calibrated' folder found in {run_dir.name}")
        return

    out_base = run_dir / 'calibrated_visual_frames'
    out_rgb = out_base / 'rgb'
    out_nir1 = out_base / 'nir1'
    out_nir2 = out_base / 'nir2'

    out_rgb.mkdir(parents=True, exist_ok=True)
    out_nir1.mkdir(parents=True, exist_ok=True)
    out_nir2.mkdir(parents=True, exist_ok=True)

    # 1. RGB (ch1_rgb if exists, else ch1)
    ch1_dir = cal_dir / 'ch1_rgb'
    if not ch1_dir.exists():
        ch1_dir = cal_dir / 'ch1'

    rgb_files = sorted(list(ch1_dir.glob('*.npy')))
    nir1_files = sorted(list((cal_dir / 'ch2').glob('*.npy')))
    nir2_files = sorted(list((cal_dir / 'ch3').glob('*.npy')))

    if not rgb_files and not nir1_files and not nir2_files:
        print(f" [SKIP] No .npy files found in {cal_dir}")
        return

    # Frame list based on available files & stride
    sample_files = rgb_files if rgb_files else (nir1_files if nir1_files else nir2_files)
    selected_files = sample_files[::stride]

    print(f" Total frames found: {len(sample_files)} | Exporting {len(selected_files)} frames (stride={stride})")

    exported_count = 0
    for idx, sample_p in enumerate(selected_files, 1):
        stem = sample_p.stem  # e.g., frame_000062

        # --- Process RGB ---
        rgb_npy = ch1_dir / f"{stem}.npy"
        if rgb_npy.exists():
            arr_rgb = np.load(rgb_npy).astype(np.float32)
            # Apply sRGB gamma correction (gamma=2.2) for bright human visualization
            arr_rgb_gamma = np.power(np.clip(arr_rgb, 0, 1), 1.0 / 2.2)
            img_rgb_8u = np.clip(arr_rgb_gamma * 255.0, 0, 255).astype(np.uint8)
            if img_rgb_8u.ndim == 3:
                img_rgb_bgr = cv2.cvtColor(img_rgb_8u, cv2.COLOR_RGB2BGR)
            else:
                img_rgb_bgr = img_rgb_8u
            cv2.imwrite(str(out_rgb / f"{stem}.{fmt}"), img_rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # --- Process NIR1 (CH2) ---
        nir1_npy = (cal_dir / 'ch2') / f"{stem}.npy"
        if nir1_npy.exists():
            arr_n1 = np.load(nir1_npy).astype(np.float32)
            # Apply gamma correction to NIR1 for crisp display
            arr_n1_gamma = np.power(np.clip(arr_n1, 0, 1), 1.0 / 2.2)
            img_n1_8u = np.clip(arr_n1_gamma * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_nir1 / f"{stem}.{fmt}"), img_n1_8u, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # --- Process NIR2 (CH3) ---
        nir2_npy = (cal_dir / 'ch3') / f"{stem}.npy"
        if nir2_npy.exists():
            arr_n2 = np.load(nir2_npy).astype(np.float32)
            # Apply gamma correction to NIR2 for crisp display
            arr_n2_gamma = np.power(np.clip(arr_n2, 0, 1), 1.0 / 2.2)
            img_n2_8u = np.clip(arr_n2_gamma * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_nir2 / f"{stem}.{fmt}"), img_n2_8u, [cv2.IMWRITE_JPEG_QUALITY, 95])

        exported_count += 1
        if exported_count % 20 == 0 or exported_count == len(selected_files):
            print(f"   --> Exported {exported_count}/{len(selected_files)} frames...")

    print(f" [DONE] Saved visual frames to: {out_base}")


def main():
    args = parse_args()
    if not args.run and not args.all:
        print("Error: Must specify either --run <run_name> or --all")
        sys.exit(1)

    if args.all:
        run_dirs = sorted([d for d in FORMAL_RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith('cal_') and d.name != 'visual'])
        print(f"Found {len(run_dirs)} formal run directories to process.")
        for r_dir in run_dirs:
            export_run(r_dir, stride=args.stride, fmt=args.format)
    else:
        run_dir = FORMAL_RUNS_DIR / args.run
        if not run_dir.exists():
            print(f"Error: Run directory does not exist: {run_dir}")
            sys.exit(1)
        export_run(run_dir, stride=args.stride, fmt=args.format)

    print(f"\n{'='*70}\n ALL EXPORTS COMPLETED SUCCESSFULLY!\n{'='*70}\n")


if __name__ == '__main__':
    main()
