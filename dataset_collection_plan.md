# Dataset Collection Plan
### MSU Multispectral Apple Grading System
**Date finalized:** July 23, 2026
**System:** JAI FS-3200T-10GE | Halogen + LED | Spectralon SRT-75-100

---

## 1. Objective

Collect a balanced, annotated multispectral dataset of apples on a conveyor belt for training a YOLO-based apple grading model. The dataset must support classification of apples into three commercial grades.

---

## 2. Apple Classes

| Class | Commercial meaning | Label in YOLO |
|---|---|---|
| **Fresh** | Market-ready apples, no visible defects, good color/shape | `fresh` |
| **Processing** | Acceptable quality but not market-grade (minor blemish, off-size, etc.) | `processing` |
| **Cull** | Rejected apples -- significant defects, bruising, disease, misshapen | `cull` |

---

## 3. Locked Camera Settings

All data must be collected with these exact settings. Do not change between sessions.

| Parameter | Value |
|---|---|
| RGB exposure (apple capture) | **5000 µs** |
| RGB exposure (white reference) | **2500 µs** |
| NIR1 exposure | **1800 µs** (same for white ref and apple) |
| NIR2 exposure | **2300 µs** (same for white ref and apple) |
| Lighting | **Halogen + LED both ON, room lights OFF** |
| White balance | **Locked after white reference** (R=0.461 G=1.000 B=1.688) |
| White reference panel | **Spectralon SRT-75-100 (certified 75%)** |
| Lamp warmup time | **8-10 minutes minimum** (not 5 -- halogens need more time to fully stabilize) |
| Room lights | **OFF during all captures** (white reference AND apple runs) |

> **Critical:** Do not change exposure, lighting, or white balance mid-session or between sessions. The calibration is only valid for these settings.
> **Room lights must be OFF.** Room lights add uncontrolled ambient illumination that is not accounted for in the calibration. Today's test showed room lights ON caused +13.8% error in RGB reflectance. Always collect with room lights OFF.

---

## 4. Run Plan

### 4a. Overview

| Run type | Runs | Apples/run | Apple source | Total apples |
|---|---|---|---|---|
| Fresh (class runs) | 4 | 9 fresh | All unique | 36 |
| Processing (class runs) | 4 | 9 processing | All unique | 36 |
| Cull (class runs) | 4 | 9 cull | All unique | 36 |
| Mixed (validation/test runs) | 2 | 3+3+3 | All unique | 18 (6 per class) |
| **Total** | **14 runs** | | | **126 unique apples** |

### 4b. Apple count breakdown

| Class | From class runs | From mixed runs | **Total unique apples** |
|---|---|---|---|
| Fresh | 36 | 6 | **42** |
| Processing | 36 | 6 | **42** |
| Cull | 36 | 6 | **42** |
| **Grand total** | **108** | **18** | **126 apples** |

> Use all unique apples -- do not reuse the same apple in multiple runs.
> This gives the model maximum variety across specimen size, shape, and color distribution.

### 4c. Within-class variety

When selecting apples for each run, aim to include variety:
- **Different sizes** -- small, medium, large within same grade
- **Different color distribution** -- fully red, partial green, striped
- **Different shapes** -- round, slightly oblong, flattened
- **Processing/cull:** Different defect types -- bruise, russet, misshapen, small size, stem damage

### 4d. Mixed run apple placement

For mixed runs, arrange by belt lane to simplify annotation:
```
Left 3 positions  → fresh
Center 3 positions → processing
Right 3 positions → cull
```
This way you know the class of each position without checking every apple individually.

---

## 5. Frames and Instances

### 5a. Raw frames

| Run type | Runs | Frames/run (avg) | Total raw frames |
|---|---|---|---|
| Class runs | 12 | 65 | 780 |
| Mixed runs | 2 | 65 | 130 |
| **Total** | **14** | | **910 raw frames** |

### 5b. Annotation strategy

Do not annotate every frame -- adjacent frames are nearly identical (apple moves ~5-8 pixels per frame). Annotating all 910 frames wastes time with negligible benefit.

**Rule: annotate every 3rd frame, skip first 4 and last 4 of each run.**

| Step | Frames per run |
|---|---|
| Total frames | 65 |
| Remove first 4 + last 4 (entry/exit, partially visible) | -8 |
| Usable frames | 57 |
| Annotate every 3rd | **~19 frames per run** |

Total annotated frames: 14 runs × 19 = **~266 frames**

### 5c. Instances

Average apple instances visible per annotated frame: **~7**
(Not all 9 apples fully visible in every frame -- some near edges)

| | Annotated frames | Instances/frame | Total instances |
|---|---|---|---|
| Fresh (4 runs) | 76 | 7 | **532** |
| Processing (4 runs) | 76 | 7 | **532** |
| Cull (4 runs) | 76 | 7 | **532** |
| Mixed (2 runs) | 38 | 7 | **266** |
| **Total** | **266** | | **~1,862 instances** |

Per class including mixed share (~88 per class from mixed):
- Fresh: **~620 instances**
- Processing: **~620 instances**
- Cull: **~620 instances**

Class ratio: **1:1:1 (balanced)**

---

## 6. Train / Val / Test Split

> **Critical rule: split by RUN, not by frame.**
> If you split by frame, consecutive frames from the same apple appear in both train and test → the model memorizes, not generalizes.

| Split | Runs | Which runs | Annotated frames | Instances |
|---|---|---|---|---|
| **Train** | 9 | 3 runs × each class | 171 | ~1,197 |
| **Val** | 3 | 1 run × each class | 57 | ~399 |
| **Test** | 2 | Both mixed runs | 38 | ~266 |
| **Total** | **14** | | **266** | **~1,862** |

Split ratio: **64% train / 21% val / 14% test**

The two mixed runs are held out as test set because they represent the real deployment scenario (all 3 classes present simultaneously on the belt).

---

## 7. Annotation Instructions

### 7a. What to annotate on

**Annotate on raw JPEG frames** (in `raw_frames/ch1/` -- the RGB channel).
- Raw frames are true-color JPEGs, easier to see clearly
- Bounding boxes have identical pixel coordinates on calibrated frames
- No re-annotation needed for calibrated data

### 7b. Tool

Recommended: **CVAT** (free, browser-based) or **Roboflow** (easier export to YOLO format).

### 7c. Annotation workflow per run

1. Create a new task in CVAT for the run
2. Set class label for the entire task (e.g., all boxes in this run = `fresh`)
3. Open each annotated frame (every 3rd frame)
4. Draw tight bounding boxes around each fully or mostly visible apple (>60% visible)
5. Skip apples that are cut off by the frame edge by more than 40%
6. Do not annotate the belt, background, or panel

### 7d. Bounding box quality

- Box should be tight around the apple (not too loose)
- Include the full apple silhouette
- For apples touching each other: draw separate overlapping boxes, one per apple
- Consistent box size matters -- if an apple is half behind another, still draw its box

---

## 8. Folder Structure

```
apple_class/
│
├── calibration_results/            ← dark maps, illumination maps (DO NOT TOUCH)
├── calibration_scripts/            ← all pipeline scripts
│
├── dataset/
│   ├── raw_runs/
│   │   ├── fresh_run1/             ← raw_frames/, calibrated_frames/
│   │   ├── fresh_run2/
│   │   ├── fresh_run3/
│   │   ├── fresh_run4/
│   │   ├── processing_run1/
│   │   ├── processing_run2/
│   │   ├── processing_run3/
│   │   ├── processing_run4/
│   │   ├── cull_run1/
│   │   ├── cull_run2/
│   │   ├── cull_run3/
│   │   ├── cull_run4/
│   │   ├── mixed_run1/
│   │   └── mixed_run2/
│   │
│   ├── annotations/                ← CVAT exports, one XML/JSON per run
│   │
│   └── yolo_dataset/               ← final YOLO-format dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
```

---

## 9. Per-Session Collection Protocol

### Start of session (daily)

Full calibration was completed **July 22, 2026** and results are saved in `calibration_results/`.
**You do not need to redo it every session** as long as the physical setup is unchanged.

Minimum required at the start of every collection day:

1. Turn on halogen + LED -- let warm up for **5 minutes minimum**
2. Point camera at Spectralon panel at center position
3. Visually confirm **R ≈ G ≈ B** (white balance lock still holding)
4. Start apple runs

**Optional but recommended (adds 5 min, high confidence):**
- Capture one white reference image at center position
- Load in Python: `np.load('ch1/white_center.npy').mean()` should read ~0.75 after calibration
- If it reads within 0.75 ± 2% (0.735 - 0.765) → proceed
- If outside that range → do full recalibration (see below)

### When to redo FULL calibration (9-position grid)

Only needed if any of these occur:
- Camera was physically moved or bumped
- Lamps were replaced, repositioned, or one died
- Belt was realigned
- Quick panel check reads **outside 0.735 - 0.765**
- More than **2 weeks** since last calibration (lamp aging)

### Per run

1. Place 9 apples of the target class on the belt
2. Start recording (all 3 channels simultaneously)
3. Run belt at standard speed
4. Stop recording when all apples have passed through
5. Note run ID, class, and any observations in the session log
6. Run `apply_calibration.py` to generate calibrated frames

### After collection

1. Run `validate_calibration.py` on at least one run per session to confirm quality
2. Back up raw_frames immediately after each session
3. Do not delete raw_frames -- keep originals permanently

---

## 10. Run Naming Convention

```
{class}_{runnumber}

Examples:
  fresh_run1
  fresh_run2
  processing_run1
  cull_run3
  mixed_run1
```

---

## 11. Session Schedule (Suggested)

| Day | Runs | Apples needed |
|---|---|---|
| Day 1 | fresh_run1, fresh_run2, processing_run1, processing_run2 | 18 fresh + 18 processing |
| Day 2 | fresh_run3, fresh_run4, processing_run3, processing_run4 | 18 fresh + 18 processing |
| Day 3 | cull_run1, cull_run2, cull_run3, cull_run4 | 36 cull |
| Day 4 | mixed_run1, mixed_run2 | 6 fresh + 6 processing + 6 cull |

> Calibration (white reference capture) must be done at the start of each day (~5 min).
> Dark frames only need to be re-captured if exposure settings change (they won't -- settings are locked).

---

## 12. Quality Checklist Per Run

Before moving to the next run, verify:

- [ ] All 9 apples were on the belt during capture
- [ ] Belt ran at consistent speed (no stops or jams)
- [ ] Lighting was stable (no flicker visible in frames)
- [ ] Frame count is 60-70 (not cut short)
- [ ] No apple fell off the belt mid-run
- [ ] Calibrated frames look correct (apples visible, belt is dark)
- [ ] Run folder is named correctly and saved to correct location

---

## 13. Summary Numbers

```
Total runs:              14  (12 class + 2 mixed)
Total raw frames:        910
Annotated frames:        ~266  (every 3rd, skip entry/exit)
Total instances:         ~1,862
Per class (balanced):    ~620 each
Train instances:         ~1,197  (9 runs)
Val instances:           ~399    (3 runs, 1 per class)
Test instances:          ~266    (2 mixed runs)
Unique apples total:     126  (all separate specimens)
```

---

*Document finalized: July 23, 2026*
*Calibration validated: July 22, 2026 (run1, settings_audit all checks passed)*
*Ready for full data collection.*
