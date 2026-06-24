# Apple Classification — Research Project

A machine-learning pipeline for classifying apple varieties (and/or quality/defect grading) from image data.

## Project Layout

```
apple_class/
├── data/
│   ├── raw/          # Original images — NOT committed (gitignored)
│   ├── processed/    # Pre-processed tensors / splits
│   └── splits/       # train / val / test CSV manifests
├── src/
│   ├── dataset.py    # PyTorch Dataset / DataLoader helpers
│   ├── model.py      # Model architecture definitions
│   ├── train.py      # Training loop
│   ├── evaluate.py   # Evaluation & metrics
│   └── utils.py      # Shared utilities
├── configs/
│   └── default.yaml  # Hyperparameters & paths
├── notebooks/        # Exploratory analysis (NOT committed)
├── outputs/
│   ├── logs/         # Training logs
│   └── figures/      # Saved plots
├── requirements.txt
├── setup.py          # (optional) Installable package
└── README.md
```

## Quick Start

```bash
# 1. Clone on the data system
git clone <repo-url>
cd apple_class

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Point configs/default.yaml at your data path, then:
python src/train.py --config configs/default.yaml
```

## Workflow

| Step | Command |
|------|---------|
| Preprocess / build splits | `python src/dataset.py` |
| Train | `python src/train.py` |
| Evaluate | `python src/evaluate.py` |

## Requirements

See [requirements.txt](requirements.txt).
