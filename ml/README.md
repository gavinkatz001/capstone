# Fall Detection ML Pipeline

ML training infrastructure for IMU-based fall detection. Takes 60-second windows of
6-axis IMU data (3-accel + 3-gyro) and classifies them as fall / no-fall.

## Architecture

```
Wearable (60s IMU packets) → Gateway → Cloud Inference (this model)
```

## Quick Start

```bash
cd ml

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"

# Activate the environment
source .venv/Scripts/activate    # Windows (Git Bash)
# or: .venv\Scripts\activate     # Windows (cmd)
# or: source .venv/bin/activate  # Linux/Mac

# Download datasets
python scripts/download_data.py --dataset nhoyh
python scripts/download_data.py --dataset microchip

# Explore the data
jupyter notebook notebooks/01_data_exploration.ipynb

# Train a model
python scripts/train.py --config configs/default.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/best.pt --split test

# Run inference on a single 60s packet
python scripts/predict.py --checkpoint outputs/best.pt --input path/to/packet.csv
```

## Project Structure

```
ml/
  configs/          - YAML configs for experiments
  falldet/          - Core Python package
    data/           - Dataset loading, preprocessing, windowing
    features/       - Time-domain and frequency-domain feature extraction
    models/         - Model architectures (baseline, CNN, LSTM, Transformer)
    training/       - Training loop, losses, schedulers
    evaluation/     - Metrics, threshold selection, reports
    tracking/       - Experiment logging (CSV/JSON)
    utils/          - Config loader, seeding, device detection
  scripts/          - CLI entry points
  notebooks/        - Jupyter notebooks for exploration
  tests/            - Unit tests
  data/             - Downloaded datasets (git-ignored)
  outputs/          - Checkpoints, logs, figures (git-ignored)
```

## Engineering Targets (from QFD)

| Metric | Marginal | Ideal |
|--------|----------|-------|
| True Positive Rate | >= 90% | >= 95% |
| False Positive Rate | <= 15% | <= 10% |
| Fall types detected | >= 3 | >= 4 |

## Datasets

| Dataset | Sampling Rate | Axes | Placement | Priority |
|---------|--------------|------|-----------|----------|
| nhoyh (HR_IMU) | 50 Hz | accel + gyro | Wrist | Primary |
| Microchip | 100 Hz | accel + gyro | Unknown | High |
| IEEE EdgeFall | TBD | accel + gyro | Neck | High |
| UR Fall | 60 Hz | accel only | Body | Low |
