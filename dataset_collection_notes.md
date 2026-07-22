# Apple Multispectral Dataset Collection — Notes

## Dataset Size Target
- **Total: 1,500–2,000 annotated instances** (masks in AnyLabeling)
- Fresh: ~500–700 instances
- Processing: ~500–600 instances
- Cull: ~400–500 instances ← hardest to collect, most critical
- Translates to ~350–450 physical apples × ~5–8 frames each

## Physical Data Collection
- **No individual apple markers needed** for training data
- **No physical size measurements needed** for training (size estimated geometrically from mask later)
- Markers + size GT only needed in the **evaluation phase** (separate experiment)
- Sort apples into 3 physical bins (Fresh / Processing / Cull) by expert visual inspection **before** running

## Data Collection Strategy — Hybrid (Final Decision)
- **50% bin-wise runs** — run one class per run, annotation is fast (class already known)
- **50% mixed runs** — physically grab 2–3 apples from each bin, place together on belt, annotate per-instance in AnyLabeling
- **Val and Test sets = 100% mixed, always** — simulates real deployment
- More mixed data in training = better generalization and robustness (stronger than 70/30)

## Apple Density
- Include **dense-pack runs** — load apples close together/touching on belt (same speed, just tighter spacing)
- Simulates high-throughput (3 apple/s) deployment where apples are densely packed
- Not about belt speed — just physical placement on belt

## Conveyor Speed for Data Collection
- Collect at **1 apple/s** — pick one speed, stay consistent
- No need to collect at multiple speeds
- Belt speed does not affect apple appearance per frame (shutter handles this)

## Camera FPS
- **Set to 30fps** — camera connects at ~27fps, that is fine and expected
- Use 27fps consistently across all sessions, never vary
- Match collection fps to deployment fps — always

## Camera Shutter Speed
- Set shutter at your **maximum deployment speed (3 apple/s)** and lock it
- Use that same shutter setting for all data collection

## Train/Val/Test Split
- Split by **apple run/batch, NOT by frame** — avoid data leakage
- Hold out entire runs for val/test (e.g., Fresh Run 4 = test, Runs 1–3 = train)
- Suggested split: 70% train / 15% val / 15% test
- Val and Test carved from the 50% mixed data — not additional data on top
- Train = all bin-wise + bulk of mixed
- Val = subset of mixed only
- Test = subset of mixed only (never touched until final evaluation)

## Annotation (AnyLabeling)
- Annotate: polygon mask per apple + class label per instance
- Export format: COCO JSON
- Bin-wise runs: just draw masks, class is already known from the bin
- Mixed runs: draw mask + assign class per instance (requires expert eye)

## What NOT to Do (For Training Phase)
- Do not track individual apples across frames
- Do not measure physical size of apples
- Do not number/marker individual apples
- Do not vary fps between sessions
- Do not vary belt speed between sessions

## Evaluation Phase (Later — Separate Experiment)
- Number apples with marker
- Measure physical size (caliper) and GT class per apple
- Run through system, compare system output vs. GT
- Report: F1, grading accuracy, size estimation error
