# Repository Structure

## Top Level

```
capstone/                           # git repo root
  .gitignore                        # excludes ml/data/, ml/outputs/, __pycache__, .venv
  3B/                               # course docs from BME 362 (Fall 2025) -- untouched
  winter-2025/
    data-sources.txt                # curated list of public IMU fall detection dataset URLs
  docs/
    winter-2025/
      ml-stuff/                     # THIS documentation suite
  ml/                               # ALL ML infrastructure lives here
```

## The `ml/` Directory in Detail

```
ml/
  pyproject.toml                    # Python project config + dependencies (uv-compatible)
  README.md                         # Quick-start guide
  .venv/                            # virtual environment (git-ignored, created by `uv venv`)

  configs/
    default.yaml                    # master config: all hyperparams, dataset selection, model choice
    experiment/                     # (empty) for per-experiment YAML overrides

  falldet/                          # the core Python package -- installed as editable via `pip install -e .`
    __init__.py

    data/                           # dataset loading, harmonization, preprocessing
      __init__.py
      registry.py                   # central metadata for all datasets (rates, URLs, label maps)
      download.py                   # (placeholder) dataset download utilities
      unified.py                    # harmonize multiple datasets to common format (50Hz, 6-axis)
      preprocessing.py              # 60s windowing, z-score normalization, subject-based splits
      dataset.py                    # PyTorch Dataset class + DataLoader factory

      loaders/                      # one module per dataset source
        __init__.py
        nhoyh.py                    # HR_IMU dataset loader (.mat files, 21 subjects)
        microchip.py                # Microchip SAMD21 loader (CSV files, accel-only)
        edgefall.py                 # (not yet implemented) IEEE EdgeFall loader
        ur_fall.py                  # (not yet implemented) UR Fall loader

    features/                       # (not yet implemented) feature engineering
      __init__.py
      time_domain.py                # planned: magnitude, jerk, SMA, stats
      freq_domain.py                # planned: FFT, spectral energy, band ratios
      extract.py                    # planned: orchestrator for feature extraction

    models/                         # model architectures
      __init__.py
      cnn1d.py                      # 1D-CNN -- the first and currently only neural network
      factory.py                    # model registry: name -> class mapping
      baseline.py                   # (not yet implemented) sklearn logistic regression
      lstm.py                       # (not yet implemented) BiLSTM model
      transformer.py                # (not yet implemented) patch-based Transformer

    training/                       # training loop infrastructure
      __init__.py
      trainer.py                    # Trainer class: train/val loop, early stopping, checkpointing
      losses.py                     # BCEWithLogitsLoss (weighted) and FocalLoss

    evaluation/                     # model evaluation
      __init__.py
      metrics.py                    # TPR, FPR, AUC, confusion matrix, threshold sweep
      threshold.py                  # (not yet implemented) dedicated threshold analysis
      report.py                     # (not yet implemented) markdown/HTML eval report generator

    tracking/                       # experiment logging
      __init__.py
      logger.py                     # CSV + JSON logger (zero external dependencies)

    utils/                          # shared utilities
      __init__.py
      config.py                     # YAML config loader with CLI dot-notation overrides
      seed.py                       # set random seeds for reproducibility
      device.py                     # auto-detect CPU/CUDA/MPS

  scripts/                          # CLI entry points (run with `python scripts/xxx.py`)
    download_data.py                # download datasets from GitHub or print manual instructions
    train.py                        # main training script
    explore_data.py                 # (not yet implemented) dataset stats + plots
    evaluate.py                     # (not yet implemented) evaluate checkpoint on test set
    predict.py                      # (not yet implemented) single-packet inference demo

  notebooks/                        # (empty) for Jupyter exploration notebooks
  tests/                            # (empty) for pytest unit tests

  data/                             # LOCAL ONLY -- git-ignored, never committed
    raw/                            # downloaded datasets as-is
      nhoyh/                        # 21 subject directories, each with fall/ and non-fall/ .mat files
      microchip/                    # falldataset/ directory with fall-XX-acc.csv and adl-XX-acc.csv
    processed/                      # (empty) for preprocessed data caches
    splits/                         # train/val/test split metadata
      split_info.json               # subject assignments and window counts per split
      norm_stats.json               # per-channel mean and std from training set

  outputs/                          # LOCAL ONLY -- git-ignored
    best.pt                         # best model checkpoint (by val AUC)
    last.pt                         # last epoch checkpoint
    config.json                     # config snapshot for this run
    metrics.csv                     # per-epoch training metrics
    summary.json                    # run summary with best metrics and timing
    checkpoints/                    # (empty) for named checkpoint organization
    logs/                           # (empty) for future structured logs
    figures/                        # (empty) for generated plots
```

## Package Import Structure

Because `falldet` is installed as an editable package (`uv pip install -e .`), imports work cleanly from anywhere:

```python
from falldet.data.unified import load_and_harmonize
from falldet.data.preprocessing import build_dataset
from falldet.data.dataset import FallDetectionDataset, create_dataloader
from falldet.models.factory import create_model
from falldet.training.trainer import Trainer
from falldet.evaluation.metrics import compute_metrics
from falldet.utils.config import load_config
from falldet.utils.seed import set_seed
from falldet.utils.device import get_device
```

The scripts in `scripts/` add the parent directory to `sys.path` as a fallback, but the editable install is the primary mechanism.

## What's Git-Tracked vs Git-Ignored

**Tracked** (committed to git):
- All Python source code in `falldet/`, `scripts/`
- Config files in `configs/`
- Documentation in `docs/`
- `pyproject.toml`, `README.md`, `.gitignore`

**Ignored** (local only):
- `ml/data/` -- raw and processed datasets (too large, download via script)
- `ml/outputs/` -- model checkpoints, training logs (generated per run)
- `ml/.venv/` -- virtual environment (recreated via `uv venv && uv pip install -e ".[dev]"`)
- `__pycache__/` -- Python bytecode cache
