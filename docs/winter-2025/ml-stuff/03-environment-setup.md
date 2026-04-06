# Environment Setup

## Prerequisites

- **Python 3.12+** -- installed at `C:\Users\gavin\AppData\Local\Programs\Python\Python312\python.exe`
- **Git** -- for cloning dataset repos
- **uv** -- fast Python package manager (already installed on Gavin's machine)

## Setup from Scratch

```bash
cd "C:\Uni Stuff\CAPSTONE\capstone\ml"

# Create virtual environment
uv venv

# Activate it
source .venv/Scripts/activate    # Git Bash on Windows
# .venv\Scripts\activate         # Windows cmd
# source .venv/bin/activate      # Linux/Mac

# Install the package + all dev dependencies
uv pip install -e ".[dev]"
```

This installs 124 packages including:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.11.0 | Neural network framework |
| numpy | 2.4.4 | Array operations |
| scipy | 1.17.1 | .mat file loading, signal resampling |
| pandas | 3.0.2 | CSV loading, tabular data |
| scikit-learn | 1.8.0 | Baseline models, metrics, splits |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical plots |
| pyyaml | 6.0.3 | Config file parsing |
| tqdm | 4.67.3 | Progress bars |
| h5py | 3.16.0 | Fallback for MATLAB v7.3 HDF5 files |
| pytest | 9.0.2 | Testing (dev) |
| jupyter | 1.1.1 | Notebooks (dev) |
| ruff | 0.15.9 | Linting (dev) |

## Why uv Instead of pip or conda

- **Speed**: uv resolves and installs dependencies 10-100x faster than pip. The full install takes ~70 seconds vs. several minutes with pip.
- **Already available**: uv was already installed on the dev machine.
- **Lockfile**: `uv.lock` (if generated) pins exact versions for reproducibility.
- **Native venv**: `uv venv` creates standard Python venvs, no conda channels or environment.yml needed.

## Downloading Datasets

After setting up the environment:

```bash
# Download the two primary datasets (clones from GitHub)
python scripts/download_data.py --dataset nhoyh
python scripts/download_data.py --dataset microchip

# Or download all at once (will print manual instructions for non-GitHub datasets)
python scripts/download_data.py --dataset all

# List available datasets
python scripts/download_data.py --list
```

Datasets are saved to `ml/data/raw/<dataset_name>/` and are git-ignored.

## GPU Support

- **Local machine**: No NVIDIA GPU detected. Training runs on CPU, which is adequate for the current dataset size (~370 windows). A 5-epoch training run takes about 15 seconds.
- **Google Colab**: The code auto-detects CUDA via `falldet/utils/device.py`. Upload the `ml/` directory to Colab and it will use the GPU automatically.
- **Cloud VM**: Same auto-detection. Just ensure CUDA PyTorch is installed.

The `get_device()` utility in `falldet/utils/device.py` checks in order: CUDA > MPS (Apple Silicon) > CPU.

## Verifying the Setup

Run this after setup to confirm everything works:

```bash
cd ml

# Quick smoke test -- 5 epoch training run
python scripts/train.py --config configs/default.yaml --training.epochs=5

# Expected output:
# Device: cpu
# Loading and harmonizing datasets...
# Loaded 349 records from nhoyh
# Loaded 20 records from microchip
# ...
# Best validation AUC: ~0.94
```

If this runs without errors, the environment is correctly set up.
