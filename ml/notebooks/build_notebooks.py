"""Build Colab-ready .ipynb notebooks from clean cell source.

Run with: python build_notebooks.py

Produces:
  - fall_detection_train.ipynb         (end-to-end training pipeline)
  - fall_detection_inference_demo.ipynb (load checkpoint, run on a 60s window)
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook_metadata():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "colab": {
            "provenance": [],
            "toc_visible": True,
        },
        "accelerator": "GPU",
    }


# ============================================================================
# TRAINING NOTEBOOK
# ============================================================================

train_cells = []

train_cells.append(md('''
# Fall Detection Training Pipeline

End-to-end training of a 1D-CNN to detect falls from 60-second windows of IMU data (3-axis accelerometer + 3-axis gyroscope).

## System context

Our wearable device records IMU motion data in 60-second packets and sends them through a gateway to a cloud service. **This notebook trains the model that runs in the cloud.**

```
[Wearable]  --BLE-->  [Gateway]  --WiFi-->  [Cloud + THIS MODEL]
  60s IMU packets          Relay              Fall detection
```

## How to run in Google Colab

1. Upload this `.ipynb` to Colab
2. **Runtime -> Change runtime type -> Hardware accelerator -> GPU**
3. Runtime -> Run all

Expected runtime: ~5 minutes on T4 GPU, ~20 minutes on CPU.

## Engineering targets (from QFD)

| Metric | Marginal | Ideal |
|--------|----------|-------|
| True Positive Rate (TPR) | >= 90% | >= 95% |
| False Positive Rate (FPR) | <= 15% | <= 10% |
| Fall types detected | >= 3 | >= 4 |

## What this notebook does

1. Downloads two public IMU fall detection datasets from GitHub
2. Harmonizes them to a common format (50 Hz, 6 channels, 60-second windows)
3. Splits train/val/test by subject (prevents leakage)
4. Trains a 1D-CNN with BCE loss and class-weighted sampling
5. Selects the best operating threshold on the validation set
6. Evaluates on the held-out test set and reports against QFD targets
7. Saves the checkpoint (optionally to Google Drive)
'''))

train_cells.append(md("## 0. Check GPU and environment"))

train_cells.append(code('''
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"CUDA version:    {torch.version.cuda}")
else:
    print("WARNING: No GPU detected.")
    print("  In Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU")
'''))

train_cells.append(md('''
## 1. Download datasets

We use two public fall detection datasets:

- **HR_IMU (nhoyh)**: 349 recordings from 21 subjects at 50 Hz, wrist-mounted, with 6-axis IMU. Our primary dataset.
- **Microchip**: 20 short recordings with accelerometer only, 100 Hz. Supplementary.

Both are cloned from public GitHub repositories. Total size ~30 MB.
'''))

train_cells.append(code('''
import os
import subprocess

os.makedirs("data/raw", exist_ok=True)

DATASETS_TO_DOWNLOAD = [
    ("nhoyh", "https://github.com/nhoyh/HR_IMU_falldetection_dataset", "data/raw/nhoyh/subject_01"),
    ("microchip", "https://github.com/MicrochipTech/ml-Fall-Detection-SAMD21-IMU", "data/raw/microchip/falldataset"),
]

for name, url, check_path in DATASETS_TO_DOWNLOAD:
    target = f"data/raw/{name}"
    if os.path.exists(check_path):
        print(f"[{name}] already exists, skipping")
        continue
    print(f"[{name}] cloning {url} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, target],
        check=True,
        capture_output=True,
    )
    print(f"[{name}] done")

print("\\nAll datasets ready.")
'''))

train_cells.append(md('''
## 2. Configuration

All hyperparameters live in this single dict. Edit and re-run from here to experiment.
'''))

train_cells.append(code('''
CONFIG = {
    "seed": 42,

    "data": {
        "datasets": ["nhoyh", "microchip"],
        "target_rate_hz": 50,         # resample everything to this rate
        "window_sec": 60,             # window length (architectural constraint)
        "stride_sec": 30,             # 50% overlap for training data
        "val_fraction": 0.15,         # fraction of subjects for validation
        "test_fraction": 0.15,        # fraction of subjects for test
    },

    "augmentation": {
        "enabled": True,
        "noise_std_accel": 0.01,      # Gaussian noise sigma in g
        "noise_std_gyro": 1.0,        # Gaussian noise sigma in deg/s
        "scale_range": (0.9, 1.1),    # per-axis random scaling
    },

    "model": {
        "channels": [32, 64, 128, 128],
        "kernel_sizes": [7, 5, 5, 3],
        "dropout": 0.3,
    },

    "training": {
        "epochs": 100,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,               # early stopping patience
        "loss": "bce",                # "bce" or "focal"
        "focal_gamma": 2.0,
    },

    "evaluation": {
        "target_tpr": 0.95,
        "max_fpr": 0.10,
    },
}

WINDOW_SAMPLES = int(CONFIG["data"]["window_sec"] * CONFIG["data"]["target_rate_hz"])
print(f"Window size: {WINDOW_SAMPLES} samples ({CONFIG['data']['window_sec']}s at {CONFIG['data']['target_rate_hz']} Hz)")
'''))

train_cells.append(md("## 3. Imports and reproducibility"))

train_cells.append(code('''
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Seed: {CONFIG['seed']}")
'''))

train_cells.append(md('''
## 4. Dataset registry and loaders

Each dataset has its own on-disk format. A loader function converts each dataset into a list of `SampleRecord` objects with a common schema.
'''))

train_cells.append(code('''
# Dataset-specific metadata. The label maps translate dataset-specific activity
# names into our standardized fall type taxonomy.
DATASETS = {
    "nhoyh": {
        "name": "HR_IMU Fall Detection Dataset",
        "sampling_rate_hz": 50,
        "placement": "wrist",
        "has_gyro": True,
        "fall_label_map": {
            "Fall forward": "forward",
            "Fall forward knees": "forward",
            "Fall backward": "backward",
            "Fall sideward": "lateral",
            "Fall sitting chair": "seated",
            "Fall from bed": "seated",
        },
    },
    "microchip": {
        "name": "Microchip ML Fall Detection SAMD21 IMU",
        "sampling_rate_hz": 100,
        "placement": "unknown",
        "has_gyro": False,
        "fall_label_map": {"fall": "other_fall"},
    },
}


@dataclass
class SampleRecord:
    """A single continuous recording from one activity."""
    dataset: str
    subject_id: str
    activity: str               # raw label
    is_fall: bool               # binary label
    fall_type: str | None       # standardized: "forward", "backward", "lateral", "seated", ...
    accel: np.ndarray           # (T, 3) in g
    gyro: np.ndarray | None     # (T, 3) in deg/s, or None
    sampling_rate_hz: float
    placement: str
    metadata: dict = field(default_factory=dict)
'''))

train_cells.append(code('''
# nhoyh: 21 subject directories, each with fall/ and non-fall/ containing .mat files.
# Each .mat file has: ax, ay, az (accel in g) and droll, dpitch, dyaw (gyro in deg/s).

_NHOYH_FALL_FILE_MAP = {
    "fall1": "Fall forward",
    "fall2": "Fall forward knees",
    "fall3": "Fall backward",
    "fall4": "Fall sideward",
    "fall5": "Fall sitting chair",
    "fall6": "Fall from bed",
}


def load_nhoyh(raw_dir: Path) -> list[SampleRecord]:
    info = DATASETS["nhoyh"]
    records = []

    for subject_dir in sorted(raw_dir.glob("subject_*")):
        subject_id = subject_dir.name

        for phase, is_fall in [("fall", True), ("non-fall", False)]:
            phase_dir = subject_dir / phase
            if not phase_dir.exists():
                continue

            for mat_file in sorted(phase_dir.glob("*.mat")):
                stem = mat_file.stem
                if is_fall:
                    activity = _NHOYH_FALL_FILE_MAP.get(stem, stem.capitalize())
                    fall_type = info["fall_label_map"].get(activity)
                else:
                    activity = stem.capitalize()
                    fall_type = None

                data = scipy.io.loadmat(str(mat_file))
                accel = np.column_stack([
                    data["ax"].ravel(), data["ay"].ravel(), data["az"].ravel()
                ]).astype(np.float32)
                gyro = np.column_stack([
                    data["droll"].ravel(), data["dpitch"].ravel(), data["dyaw"].ravel()
                ]).astype(np.float32)

                records.append(SampleRecord(
                    dataset="nhoyh",
                    subject_id=subject_id,
                    activity=activity,
                    is_fall=is_fall,
                    fall_type=fall_type,
                    accel=accel,
                    gyro=gyro,
                    sampling_rate_hz=info["sampling_rate_hz"],
                    placement=info["placement"],
                ))
    return records


print("nhoyh loader defined")
'''))

train_cells.append(code('''
# microchip: falldataset/ contains fall-XX-acc.csv and adl-XX-acc.csv files.
# CSV columns: Svtotal, Ax, Ay, Az. No gyro data.

def load_microchip(raw_dir: Path) -> list[SampleRecord]:
    info = DATASETS["microchip"]
    records = []

    dataset_dir = raw_dir / "falldataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"falldataset/ not found in {raw_dir}")

    for csv_file in sorted(dataset_dir.glob("*-acc.csv")):
        stem = csv_file.stem  # e.g. "fall-01-acc"
        parts = stem.split("-")
        category = parts[0]   # "fall" or "adl"
        sample_num = parts[1] # "01", "02", ...

        is_fall = category == "fall"
        fall_type = info["fall_label_map"].get(category) if is_fall else None

        df = pd.read_csv(csv_file)
        accel = df[["Ax", "Ay", "Az"]].values.astype(np.float32)

        records.append(SampleRecord(
            dataset="microchip",
            subject_id=f"microchip_{sample_num}",
            activity=category,
            is_fall=is_fall,
            fall_type=fall_type,
            accel=accel,
            gyro=None,  # no gyro available
            sampling_rate_hz=info["sampling_rate_hz"],
            placement=info["placement"],
        ))
    return records


print("microchip loader defined")
'''))

train_cells.append(md('''
## 5. Harmonization

Convert all `SampleRecord`s to a common `UnifiedSample` format:
- 50 Hz sampling rate (resampled via `scipy.signal.resample`)
- 6 channels: `[ax, ay, az, gx, gy, gz]`
- If a dataset has no gyro, fill with zeros and flag `has_gyro=False`
'''))

train_cells.append(code('''
@dataclass
class UnifiedSample:
    dataset: str
    subject_id: str
    activity: str
    is_fall: bool
    fall_type: str | None
    data: np.ndarray              # (T, 6) at target_rate_hz
    has_gyro: bool
    sampling_rate_hz: float       # always target_rate_hz after harmonization
    placement: str
    duration_sec: float
    metadata: dict = field(default_factory=dict)


def _resample_to_target(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    if abs(source_rate - target_rate) < 0.1:
        return signal
    n_target = int(len(signal) * target_rate / source_rate)
    if n_target < 1:
        return signal[:1]
    return resample(signal, n_target, axis=0).astype(np.float32)


def harmonize_record(record: SampleRecord, target_rate: float) -> UnifiedSample:
    accel = record.accel
    if record.gyro is not None:
        gyro = record.gyro
        has_gyro = True
    else:
        gyro = np.zeros_like(accel)
        has_gyro = False

    data = np.column_stack([accel, gyro]).astype(np.float32)
    data = _resample_to_target(data, record.sampling_rate_hz, target_rate)

    return UnifiedSample(
        dataset=record.dataset,
        subject_id=record.subject_id,
        activity=record.activity,
        is_fall=record.is_fall,
        fall_type=record.fall_type,
        data=data,
        has_gyro=has_gyro,
        sampling_rate_hz=target_rate,
        placement=record.placement,
        duration_sec=len(data) / target_rate,
    )


def load_and_harmonize(raw_base: Path, dataset_names: list[str]) -> list[UnifiedSample]:
    all_records = []
    for ds in dataset_names:
        ds_dir = raw_base / ds
        if ds == "nhoyh":
            recs = load_nhoyh(ds_dir)
        elif ds == "microchip":
            recs = load_microchip(ds_dir)
        else:
            print(f"  warning: no loader for '{ds}'")
            continue
        print(f"  {ds}: {len(recs)} records")
        all_records.extend(recs)

    target_rate = CONFIG["data"]["target_rate_hz"]
    return [harmonize_record(r, target_rate) for r in all_records]


print("Harmonization functions defined")
'''))

train_cells.append(code('''
print("Loading and harmonizing datasets...")
samples = load_and_harmonize(Path("data/raw"), CONFIG["data"]["datasets"])

n_falls = sum(1 for s in samples if s.is_fall)
n_adl = len(samples) - n_falls
durations = [s.duration_sec for s in samples]

print(f"\\nTotal: {len(samples)} samples")
print(f"  Falls: {n_falls}")
print(f"  ADL:   {n_adl}")
print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s (mean {np.mean(durations):.1f}s)")
print(f"  With gyro:    {sum(1 for s in samples if s.has_gyro)}")
print(f"  Without gyro: {sum(1 for s in samples if not s.has_gyro)}")
'''))

train_cells.append(md('''
## 6. Windowing, normalization, and subject-based splits

- Long recordings are split into overlapping 60-second windows
- Short recordings are zero-padded to 60 seconds (with `valid_length` tracking)
- Train/val/test split is done **by subject** to prevent data leakage
- Z-score normalization stats are computed on the training set only
'''))

train_cells.append(code('''
@dataclass
class Window:
    data: np.ndarray              # (window_samples, 6)
    label: int                    # 1 = fall, 0 = no-fall
    valid_length: int             # real samples before zero-padding
    dataset: str
    subject_id: str
    activity: str
    fall_type: str | None
    has_gyro: bool


def create_windows(
    samples: list[UnifiedSample],
    window_sec: float,
    stride_sec: float,
    rate_hz: float,
) -> list[Window]:
    window_len = int(window_sec * rate_hz)
    stride_len = int(stride_sec * rate_hz)
    windows = []

    for sample in samples:
        T = len(sample.data)
        if T >= window_len:
            # Sliding window for long recordings
            start = 0
            while start + window_len <= T:
                chunk = sample.data[start:start + window_len].copy()
                windows.append(Window(
                    data=chunk, label=int(sample.is_fall), valid_length=window_len,
                    dataset=sample.dataset, subject_id=sample.subject_id,
                    activity=sample.activity, fall_type=sample.fall_type,
                    has_gyro=sample.has_gyro,
                ))
                start += stride_len
        else:
            # Zero-pad short recordings
            padded = np.zeros((window_len, sample.data.shape[1]), dtype=np.float32)
            padded[:T] = sample.data
            windows.append(Window(
                data=padded, label=int(sample.is_fall), valid_length=T,
                dataset=sample.dataset, subject_id=sample.subject_id,
                activity=sample.activity, fall_type=sample.fall_type,
                has_gyro=sample.has_gyro,
            ))
    return windows


def compute_normalization_stats(windows: list[Window]) -> dict:
    """Compute per-channel mean/std from training windows only."""
    all_data = np.concatenate([w.data[:w.valid_length] for w in windows], axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_windows(windows: list[Window], stats: dict) -> list[Window]:
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    for w in windows:
        w.data = (w.data - mean) / std
    return windows


def subject_based_split(
    windows: list[Window], val_frac: float, test_frac: float, seed: int
) -> tuple[list[Window], list[Window], list[Window]]:
    """Split by subject so no subject appears in more than one split."""
    rng = np.random.RandomState(seed)
    subjects = sorted(set(w.subject_id for w in windows))
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test:n_test + n_val])
    train_subjects = set(subjects[n_test + n_val:])

    train = [w for w in windows if w.subject_id in train_subjects]
    val = [w for w in windows if w.subject_id in val_subjects]
    test = [w for w in windows if w.subject_id in test_subjects]
    return train, val, test


def build_dataset(samples, config):
    dc = config["data"]
    windows = create_windows(
        samples, dc["window_sec"], dc["stride_sec"], dc["target_rate_hz"]
    )
    print(f"Created {len(windows)} windows ({sum(w.label for w in windows)} falls)")

    train, val, test = subject_based_split(
        windows, dc["val_fraction"], dc["test_fraction"], config["seed"]
    )
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    stats = compute_normalization_stats(train)
    train = normalize_windows(train, stats)
    val = normalize_windows(val, stats)
    test = normalize_windows(test, stats)

    return train, val, test, stats
'''))

train_cells.append(code('''
train_windows, val_windows, test_windows, norm_stats = build_dataset(samples, CONFIG)

def _split_stats(name, split):
    n_fall = sum(w.label for w in split)
    n_adl = len(split) - n_fall
    subjects = len(set(w.subject_id for w in split))
    print(f"  {name:5s}: {len(split):4d} windows ({n_fall:3d} fall / {n_adl:3d} adl) across {subjects} subjects")

print()
_split_stats("train", train_windows)
_split_stats("val",   val_windows)
_split_stats("test",  test_windows)

print(f"\\nWindow tensor shape: {train_windows[0].data.shape}  (samples, channels)")
print(f"Norm mean: {[f'{m:+.3f}' for m in norm_stats['mean']]}")
print(f"Norm std:  {[f'{s:+.3f}' for s in norm_stats['std']]}")
'''))

train_cells.append(md('''
## 7. PyTorch Dataset and DataLoader

Wraps the list of `Window`s as a PyTorch Dataset. Returns tensors in **channels-first format** `(6, 3000)` because that's what `Conv1d` expects.

Class imbalance is handled in two ways:
1. `WeightedRandomSampler` oversamples the minority class in each batch
2. `BCEWithLogitsLoss` uses `pos_weight = n_negative / n_positive`
'''))

train_cells.append(code('''
class FallDetectionDataset(Dataset):
    def __init__(self, windows: list[Window], augment: bool = False, aug_config: dict | None = None):
        self.windows = windows
        self.augment = augment
        self.aug_config = aug_config or {}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        data = w.data.copy()
        if self.augment:
            data = self._augment(data, w.valid_length)
        # Channels-first for Conv1d: (channels, time)
        x = torch.from_numpy(data.T).float()
        y = torch.tensor([w.label], dtype=torch.float32)
        return x, y

    def _augment(self, data: np.ndarray, valid_length: int) -> np.ndarray:
        cfg = self.aug_config
        valid = data[:valid_length]

        # Gaussian noise
        if cfg.get("noise_std_accel", 0) > 0:
            noise_a = np.random.randn(valid_length, 3) * cfg["noise_std_accel"]
            noise_g = np.random.randn(valid_length, 3) * cfg.get("noise_std_gyro", 0)
            valid[:, :3] += noise_a.astype(np.float32)
            valid[:, 3:] += noise_g.astype(np.float32)

        # Per-axis random scaling
        scale_range = cfg.get("scale_range", (1.0, 1.0))
        if scale_range[0] != scale_range[1]:
            scales = np.random.uniform(scale_range[0], scale_range[1], size=(1, 6))
            valid *= scales.astype(np.float32)

        data[:valid_length] = valid
        return data

    def get_labels(self) -> np.ndarray:
        return np.array([w.label for w in self.windows])


def compute_pos_weight(dataset: FallDetectionDataset) -> float:
    labels = dataset.get_labels()
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def create_dataloader(dataset, batch_size, shuffle, weighted=False):
    sampler = None
    if weighted and shuffle:
        labels = dataset.get_labels()
        class_counts = np.bincount(labels.astype(int))
        weights = 1.0 / class_counts[labels.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# Build datasets and loaders
aug_cfg = CONFIG["augmentation"] if CONFIG["augmentation"]["enabled"] else None
train_ds = FallDetectionDataset(train_windows, augment=CONFIG["augmentation"]["enabled"], aug_config=aug_cfg)
val_ds = FallDetectionDataset(val_windows)
test_ds = FallDetectionDataset(test_windows)

bs = CONFIG["training"]["batch_size"]
train_loader = create_dataloader(train_ds, bs, shuffle=True, weighted=True)
val_loader = create_dataloader(val_ds, bs, shuffle=False)
test_loader = create_dataloader(test_ds, bs, shuffle=False)

# Verify shapes
x_sample, y_sample = next(iter(train_loader))
print(f"Batch x shape: {tuple(x_sample.shape)}  (batch, channels, time)")
print(f"Batch y shape: {tuple(y_sample.shape)}")
print(f"Labels in first batch: {y_sample.squeeze().tolist()[:16]}")
'''))

train_cells.append(md('''
## 8. Model: 1D-CNN

A compact 1D convolutional network on raw time-series:

```
Input (B, 6, 3000)
  -> Conv1d(6, 32, k=7, s=2) + BN + ReLU    (B, 32, 1500)
  -> Conv1d(32, 64, k=5, s=2) + BN + ReLU   (B, 64, 750)
  -> Conv1d(64, 128, k=5, s=2) + BN + ReLU  (B, 128, 375)
  -> Conv1d(128, 128, k=3, s=2) + BN + ReLU (B, 128, 188)
  -> AdaptiveAvgPool1d(1)                   (B, 128, 1)
  -> Dropout(0.3) + Linear(128, 1)          (B, 1)  logits
```

The large initial kernel (size 7) is sized to capture fall impact signatures, which typically last 0.1-0.3 seconds (at 50 Hz = 5-15 samples).
'''))

train_cells.append(code('''
class CNN1D(nn.Module):
    def __init__(self, in_channels=6, channels=None, kernel_sizes=None, dropout=0.3):
        super().__init__()
        channels = channels or [32, 64, 128, 128]
        kernel_sizes = kernel_sizes or [7, 5, 5, 3]

        layers = []
        prev_ch = in_channels
        for ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=ks, stride=2, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


model = CNN1D(
    in_channels=6,
    channels=CONFIG["model"]["channels"],
    kernel_sizes=CONFIG["model"]["kernel_sizes"],
    dropout=CONFIG["model"]["dropout"],
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: CNN1D ({n_params:,} parameters)")
print(model)
'''))

train_cells.append(md('''
## 9. Loss functions and metrics
'''))

train_cells.append(code('''
class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples via (1 - p_t)^gamma."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def get_loss_fn(name: str, pos_weight: float, focal_gamma: float) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    elif name == "focal":
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    raise ValueError(f"Unknown loss: {name}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Confusion matrix + derived metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "tpr": float(tpr), "fpr": float(fpr),
        "precision": float(precision), "specificity": float(specificity),
        "f1": float(f1), "roc_auc": float(roc_auc),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_tpr: float = 0.95, n_steps: int = 100):
    """Find threshold achieving target TPR with minimum FPR.
    Falls back to max F2 if target TPR is unreachable.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    best_thr = 0.5
    best_fpr = float("inf")

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        if m["tpr"] >= target_tpr and m["fpr"] < best_fpr:
            best_fpr = m["fpr"]
            best_thr = float(t)

    if best_fpr == float("inf"):
        # Fall back to max F2 (recall-weighted F-score)
        best_f2 = -1.0
        for t in thresholds:
            m = compute_metrics(y_true, y_prob, threshold=t)
            p, r = m["precision"], m["tpr"]
            f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
            if f2 > best_f2:
                best_f2 = f2
                best_thr = float(t)

    return best_thr, compute_metrics(y_true, y_prob, best_thr)
'''))

train_cells.append(md('''
## 10. Trainer: loop, early stopping, checkpointing

The trainer runs training and validation, tracks the best validation AUC, keeps the best model state in memory, and stops early if validation stops improving.
'''))

train_cells.append(code('''
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def is_best(self, score: float) -> bool:
        return score >= self.best_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device

        tc = config["training"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=tc["epochs"])
        self.early_stopping = EarlyStopping(patience=tc["patience"])
        self.epochs = tc["epochs"]

        self.history: list[dict] = []
        self.best_metrics: dict = {}
        self.best_state_dict = None

    def train(self):
        for epoch in range(self.epochs):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_metrics = self._validate()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["roc_auc"],
                "val_tpr": val_metrics["tpr"],
                "val_fpr": val_metrics["fpr"],
                "lr": lr,
                "time_sec": time.time() - t0,
            }
            self.history.append(row)

            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auc={val_metrics['roc_auc']:.4f} | "
                f"tpr={val_metrics['tpr']:.3f} fpr={val_metrics['fpr']:.3f} | "
                f"lr={lr:.5f} | {row['time_sec']:.1f}s"
            )

            if self.early_stopping.is_best(val_metrics["roc_auc"]):
                self.best_metrics = val_metrics
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if self.early_stopping.step(val_metrics["roc_auc"]):
                print(f"Early stopping at epoch {epoch}")
                break

        return self.best_metrics, self.history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []
        n_batches = 0
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy().ravel())
            n_batches += 1

        metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics
'''))

train_cells.append(md("## 11. Run training"))

train_cells.append(code('''
pos_weight = compute_pos_weight(train_ds)
print(f"Positive class weight (for class imbalance): {pos_weight:.2f}")

criterion = get_loss_fn(
    CONFIG["training"]["loss"],
    pos_weight=pos_weight,
    focal_gamma=CONFIG["training"]["focal_gamma"],
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    config=CONFIG,
    device=DEVICE,
)

print(f"\\nTraining for up to {CONFIG['training']['epochs']} epochs (early stopping patience={CONFIG['training']['patience']})")
print("-" * 110)
best_metrics, history = trainer.train()
print("-" * 110)

# Restore best weights
if trainer.best_state_dict is not None:
    model.load_state_dict(trainer.best_state_dict)

print(f"\\nBest validation AUC: {best_metrics['roc_auc']:.4f}")
print(f"Best validation TPR: {best_metrics['tpr']:.3f}")
print(f"Best validation FPR: {best_metrics['fpr']:.3f}")
'''))

train_cells.append(md('''
## 12. Evaluation on the test set

Two-step threshold selection:
1. Use the validation set to find the threshold that achieves target TPR with minimum FPR
2. Apply that threshold to the test set for the final metrics

This prevents overfitting to the test set.
'''))

train_cells.append(code('''
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.extend(probs)
        all_labels.extend(y.numpy().ravel())
    return np.array(all_labels), np.array(all_probs)


# Step 1: pick threshold on validation set
val_y, val_probs = get_predictions(model, val_loader, DEVICE)
best_threshold, val_metrics_at_thr = find_best_threshold(
    val_y, val_probs, target_tpr=CONFIG["evaluation"]["target_tpr"]
)
print(f"Threshold selected from val set (targeting TPR >= {CONFIG['evaluation']['target_tpr']}): {best_threshold:.3f}")
print(f"  Val TPR: {val_metrics_at_thr['tpr']:.3f}, Val FPR: {val_metrics_at_thr['fpr']:.3f}")

# Step 2: apply that threshold to the test set
test_y, test_probs = get_predictions(model, test_loader, DEVICE)
test_metrics = compute_metrics(test_y, test_probs, threshold=best_threshold)

print("\\n========== Final Test Metrics ==========")
print(f"Threshold:    {best_threshold:.3f}")
print(f"TPR (recall): {test_metrics['tpr']:.3f}")
print(f"FPR:          {test_metrics['fpr']:.3f}")
print(f"Precision:    {test_metrics['precision']:.3f}")
print(f"Specificity:  {test_metrics['specificity']:.3f}")
print(f"F1:           {test_metrics['f1']:.3f}")
print(f"ROC AUC:      {test_metrics['roc_auc']:.4f}")
print(f"Confusion:    TP={test_metrics['tp']} FP={test_metrics['fp']} TN={test_metrics['tn']} FN={test_metrics['fn']}")

# Check against QFD targets
print("\\n========== QFD Target Check ==========")
target_tpr = CONFIG["evaluation"]["target_tpr"]
max_fpr = CONFIG["evaluation"]["max_fpr"]
tpr_ok = test_metrics["tpr"] >= target_tpr
fpr_ok = test_metrics["fpr"] <= max_fpr
print(f"TPR >= {target_tpr}: {'PASS' if tpr_ok else 'FAIL'} (got {test_metrics['tpr']:.3f})")
print(f"FPR <= {max_fpr}: {'PASS' if fpr_ok else 'FAIL'} (got {test_metrics['fpr']:.3f})")
'''))

train_cells.append(md("## 13. Visualizations"))

train_cells.append(code('''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training curves
ax = axes[0]
epochs_ = [h["epoch"] for h in history]
ax.plot(epochs_, [h["train_loss"] for h in history], label="train loss")
ax.plot(epochs_, [h["val_loss"] for h in history], label="val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Curves")
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: ROC curve
ax = axes[1]
fpr_arr, tpr_arr, _ = roc_curve(test_y, test_probs)
auc_val = auc(fpr_arr, tpr_arr)
ax.plot(fpr_arr, tpr_arr, label=f"ROC (AUC={auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
ax.scatter([test_metrics["fpr"]], [test_metrics["tpr"]], color="red", s=80, zorder=5, label="operating point")
ax.axhline(y=CONFIG["evaluation"]["target_tpr"], color="green", linestyle=":", alpha=0.5, label=f"target TPR={CONFIG['evaluation']['target_tpr']}")
ax.axvline(x=CONFIG["evaluation"]["max_fpr"], color="orange", linestyle=":", alpha=0.5, label=f"max FPR={CONFIG['evaluation']['max_fpr']}")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (test set)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: Confusion matrix
ax = axes[2]
cm = np.array([[test_metrics["tn"], test_metrics["fp"]],
               [test_metrics["fn"], test_metrics["tp"]]])
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: ADL", "Pred: Fall"])
ax.set_yticklabels(["True: ADL", "True: Fall"])
thresh_vis = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh_vis else "black", fontsize=16)
ax.set_title(f"Confusion Matrix @ threshold={best_threshold:.2f}")

plt.tight_layout()
plt.show()
'''))

train_cells.append(md('''
## 14. Save checkpoint

Saves the model weights, config, normalization stats, best threshold, and test metrics into a single `.pt` file.

To persist across Colab sessions, optionally mount Google Drive.
'''))

train_cells.append(code('''
# Grab a few real test windows to bundle with the checkpoint.
# These are in pre-normalization form so the inference notebook can demonstrate
# the full preprocessing + predict pipeline on realistic data.
mean_arr = np.array(norm_stats["mean"], dtype=np.float32)
std_arr = np.array(norm_stats["std"], dtype=np.float32)

def _denormalize(window_data):
    """Undo the z-score normalization to recover raw (g, deg/s) units."""
    return window_data * std_arr + mean_arr

sample_fall = next((w for w in test_windows if w.label == 1), None)
sample_adl = next((w for w in test_windows if w.label == 0), None)

sample_windows = {}
if sample_fall is not None:
    sample_windows["fall"] = {
        "data": _denormalize(sample_fall.data),  # (3000, 6) in raw units
        "activity": sample_fall.activity,
        "fall_type": sample_fall.fall_type,
        "subject_id": sample_fall.subject_id,
        "dataset": sample_fall.dataset,
        "valid_length": sample_fall.valid_length,
    }
if sample_adl is not None:
    sample_windows["adl"] = {
        "data": _denormalize(sample_adl.data),
        "activity": sample_adl.activity,
        "fall_type": sample_adl.fall_type,
        "subject_id": sample_adl.subject_id,
        "dataset": sample_adl.dataset,
        "valid_length": sample_adl.valid_length,
    }

checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": CONFIG,
    "norm_stats": norm_stats,
    "best_threshold": best_threshold,
    "test_metrics": test_metrics,
    "history": history,
    "sample_windows": sample_windows,
}

torch.save(checkpoint, "fall_detection_model.pt")
print("Saved checkpoint to fall_detection_model.pt")
print(f"  Test AUC: {test_metrics['roc_auc']:.4f}")
print(f"  Test TPR: {test_metrics['tpr']:.3f}")
print(f"  Test FPR: {test_metrics['fpr']:.3f}")
print(f"  Threshold: {best_threshold:.3f}")
print(f"  Sample windows bundled: {list(sample_windows.keys())}")

# Optional: save to Google Drive for persistence across Colab sessions
# Uncomment to use:
#
# from google.colab import drive
# drive.mount("/content/drive")
# import shutil
# shutil.copy("fall_detection_model.pt", "/content/drive/MyDrive/fall_detection_model.pt")
# print("Saved to Google Drive")
'''))

train_cells.append(md('''
## Done

The trained model is in `fall_detection_model.pt`. You can:

1. Download it from Colab's file browser (Files tab on the left)
2. Use the `fall_detection_inference_demo.ipynb` notebook to load it and run inference on single 60s windows
3. Re-run this notebook with different config (edit Cell 2 and run from there) to experiment

## What to iterate on

- **Run more epochs**: default is 100 with patience 15
- **Try focal loss**: set `CONFIG["training"]["loss"] = "focal"`
- **Tune augmentation**: adjust noise, scaling ranges
- **Add more datasets**: extend the loader with EdgeFall, UR Fall, or your own data
- **Try other architectures**: replace CNN1D with an LSTM or Transformer
'''))


# Build and write the training notebook
train_nb = nbf.v4.new_notebook()
train_nb.cells = train_cells
train_nb.metadata = notebook_metadata()

train_path = HERE / "fall_detection_train.ipynb"
with open(train_path, "w", encoding="utf-8") as f:
    nbf.write(train_nb, f)
print(f"Wrote {train_path} ({len(train_cells)} cells)")


# ============================================================================
# INFERENCE DEMO NOTEBOOK
# ============================================================================

infer_cells = []

infer_cells.append(md('''
# Fall Detection Inference Demo

This notebook demonstrates how to use a trained fall detection model on a **single 60-second IMU packet** -- simulating what the cloud service does for each incoming packet from a wearable.

## Prerequisites

1. Run `fall_detection_train.ipynb` first to produce `fall_detection_model.pt`
2. Either:
   - Upload `fall_detection_model.pt` to Colab (Files tab on the left), OR
   - Mount Google Drive and load from there

## What this notebook does

1. Loads a trained checkpoint (model weights + normalization stats + operating threshold)
2. Defines a `predict(window)` function that takes a `(3000, 6)` IMU window and returns fall probability
3. Runs inference on synthetic example windows to demonstrate the API
'''))

infer_cells.append(code('''
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
'''))

infer_cells.append(md('''
## 1. Model architecture

This must match the architecture used during training. It's the same `CNN1D` class from the training notebook.
'''))

infer_cells.append(code('''
class CNN1D(nn.Module):
    def __init__(self, in_channels=6, channels=None, kernel_sizes=None, dropout=0.3):
        super().__init__()
        channels = channels or [32, 64, 128, 128]
        kernel_sizes = kernel_sizes or [7, 5, 5, 3]

        layers = []
        prev_ch = in_channels
        for ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=ks, stride=2, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
'''))

infer_cells.append(md('''
## 2. Load the checkpoint

Point `CHECKPOINT_PATH` to wherever you uploaded `fall_detection_model.pt`.
'''))

infer_cells.append(code('''
# Option A: File uploaded directly to Colab runtime
CHECKPOINT_PATH = "fall_detection_model.pt"

# Option B: Mount Google Drive and load from there
# from google.colab import drive
# drive.mount("/content/drive")
# CHECKPOINT_PATH = "/content/drive/MyDrive/fall_detection_model.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

config = checkpoint["config"]
norm_stats = checkpoint["norm_stats"]
best_threshold = checkpoint["best_threshold"]
test_metrics = checkpoint["test_metrics"]

print("Checkpoint loaded.")
print(f"  Best threshold: {best_threshold:.3f}")
print(f"  Test TPR:       {test_metrics['tpr']:.3f}")
print(f"  Test FPR:       {test_metrics['fpr']:.3f}")
print(f"  Test AUC:       {test_metrics['roc_auc']:.4f}")
print(f"  Norm mean:      {[f'{m:+.3f}' for m in norm_stats['mean']]}")
print(f"  Norm std:       {[f'{s:+.3f}' for s in norm_stats['std']]}")

# Rebuild model and load weights
model = CNN1D(
    in_channels=6,
    channels=config["model"]["channels"],
    kernel_sizes=config["model"]["kernel_sizes"],
    dropout=config["model"]["dropout"],
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\\nModel is ready for inference.")
'''))

infer_cells.append(md('''
## 3. The `predict` function

This is what the cloud service would call for each incoming 60s packet from the gateway.

Input: numpy array of shape `(3000, 6)` with columns `[ax, ay, az, gx, gy, gz]`
- `ax, ay, az`: accelerometer in g
- `gx, gy, gz`: gyroscope angular velocity in deg/s
- 3000 samples = 60 seconds at 50 Hz

Output: dict with `is_fall`, `confidence`, `logit`, `latency_ms`
'''))

infer_cells.append(code('''
import time


def predict(window: np.ndarray, model, norm_stats, threshold):
    """Run fall detection on a single 60-second window.

    Args:
        window: np.ndarray of shape (3000, 6) -- raw IMU data at 50 Hz
                Columns: [ax, ay, az, gx, gy, gz]
                Units:   accel in g, gyro in deg/s
        model: trained CNN1D in eval mode
        norm_stats: dict with 'mean' and 'std' (6-element lists from training)
        threshold: classification threshold (picked during training on val set)

    Returns:
        dict with keys:
            is_fall (bool):    prediction
            confidence (float): sigmoid probability in [0, 1]
            logit (float):      raw logit (pre-sigmoid)
            latency_ms (float): inference time in milliseconds
    """
    # 1) Normalize to match training distribution
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    normed = (window.astype(np.float32) - mean) / std

    # 2) Convert to channels-first tensor with batch dim: (1, 6, 3000)
    x = torch.from_numpy(normed.T).float().unsqueeze(0).to(DEVICE)

    # 3) Forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        logit = model(x).item()
    latency_ms = (time.perf_counter() - t0) * 1000

    confidence = float(torch.sigmoid(torch.tensor(logit)))

    return {
        "is_fall": confidence >= threshold,
        "confidence": confidence,
        "logit": logit,
        "latency_ms": latency_ms,
    }


print("predict() function defined")
'''))

infer_cells.append(md('''
## 4. Demo on real test windows

The training notebook bundles one fall and one ADL window from the held-out test set into the checkpoint (stored in raw units -- NOT pre-normalized). We use those here to demonstrate the full preprocessing + inference pipeline on realistic data.

If the bundled windows are missing (older checkpoint), we fall back to a synthetic demo.
'''))

infer_cells.append(code('''
WINDOW_SAMPLES = 3000  # 60s @ 50Hz
N_CHANNELS = 6

sample_windows = checkpoint.get("sample_windows", {})

if "fall" in sample_windows and "adl" in sample_windows:
    # Use real bundled test windows
    adl_info = sample_windows["adl"]
    fall_info = sample_windows["fall"]

    adl_window = np.array(adl_info["data"], dtype=np.float32)
    fall_window = np.array(fall_info["data"], dtype=np.float32)

    print("Using real test windows from checkpoint:")
    print(f"  ADL:  dataset={adl_info['dataset']}, subject={adl_info['subject_id']}, activity={adl_info['activity']} (valid {adl_info['valid_length']}/{WINDOW_SAMPLES} samples)")
    print(f"  Fall: dataset={fall_info['dataset']}, subject={fall_info['subject_id']}, activity={fall_info['activity']}, type={fall_info['fall_type']} (valid {fall_info['valid_length']}/{WINDOW_SAMPLES} samples)")
    using_real_data = True
else:
    # Fallback: synthetic data
    print("No bundled sample windows found. Generating synthetic demo data.")
    print("(Re-run the training notebook to bundle real windows in the checkpoint.)")
    np.random.seed(0)
    adl_window = np.random.randn(WINDOW_SAMPLES, N_CHANNELS).astype(np.float32) * 0.1
    adl_window[:, 2] += 1.0  # gravity

    fall_window = np.random.randn(WINDOW_SAMPLES, N_CHANNELS).astype(np.float32) * 0.1
    fall_window[:, 2] += 1.0
    fall_window[1400:1430, :3] += np.random.randn(30, 3).astype(np.float32) * 3.0
    fall_window[1400:1430, 3:] += np.random.randn(30, 3).astype(np.float32) * 100.0
    using_real_data = False

# Run inference on both windows
result_adl = predict(adl_window, model, norm_stats, best_threshold)
result_fall = predict(fall_window, model, norm_stats, best_threshold)

print()
print(f"ADL window:  confidence={result_adl['confidence']:.3f}  is_fall={result_adl['is_fall']}  ({result_adl['latency_ms']:.1f} ms)")
print(f"Fall window: confidence={result_fall['confidence']:.3f}  is_fall={result_fall['is_fall']}  ({result_fall['latency_ms']:.1f} ms)")
print(f"\\nOperating threshold: {best_threshold:.3f}")
'''))

infer_cells.append(md("## 5. Visualize the two windows"))

infer_cells.append(code('''
fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
time_axis = np.arange(WINDOW_SAMPLES) / 50.0  # seconds

source_tag = "real test data" if using_real_data else "synthetic"

# Accel row
axes[0, 0].plot(time_axis, adl_window[:, :3])
axes[0, 0].set_title(f"ADL window ({source_tag}) | confidence={result_adl['confidence']:.3f}")
axes[0, 0].set_ylabel("Accel (g)")
axes[0, 0].legend(["ax", "ay", "az"], loc="upper right", fontsize=8)
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(time_axis, fall_window[:, :3])
axes[0, 1].set_title(f"Fall window ({source_tag}) | confidence={result_fall['confidence']:.3f}")
axes[0, 1].legend(["ax", "ay", "az"], loc="upper right", fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Gyro row
axes[1, 0].plot(time_axis, adl_window[:, 3:])
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Gyro (deg/s)")
axes[1, 0].legend(["gx", "gy", "gz"], loc="upper right", fontsize=8)
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(time_axis, fall_window[:, 3:])
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].legend(["gx", "gy", "gz"], loc="upper right", fontsize=8)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
'''))

infer_cells.append(md('''
## 6. Latency benchmark

Measure end-to-end inference latency across many windows. This is what the cloud service would see per packet.

Target: <= 1 second per packet (comfortable for a 60s-cadence system).
'''))

infer_cells.append(code('''
# Warm up
for _ in range(3):
    _ = predict(adl_window, model, norm_stats, best_threshold)

# Benchmark
n_runs = 50
latencies = []
for _ in range(n_runs):
    r = predict(adl_window, model, norm_stats, best_threshold)
    latencies.append(r["latency_ms"])

latencies = np.array(latencies)
print(f"Inference latency over {n_runs} runs:")
print(f"  mean:   {latencies.mean():.2f} ms")
print(f"  median: {np.median(latencies):.2f} ms")
print(f"  p95:    {np.percentile(latencies, 95):.2f} ms")
print(f"  min:    {latencies.min():.2f} ms")
print(f"  max:    {latencies.max():.2f} ms")
'''))

infer_cells.append(md('''
## Integration notes

To plug this into the cloud service:

1. Load the checkpoint **once** at service startup
2. Call `predict(window, model, norm_stats, best_threshold)` per incoming packet
3. The gateway must send windows in the same format: `(3000, 6)` numpy array, 50 Hz, accel in g, gyro in deg/s
4. If the wearable samples at a different rate, resample to 50 Hz before calling `predict`
5. If a fall is detected, route the alert to nursing staff with the confidence score and timestamp

The model is deterministic given fixed weights and no dropout (`model.eval()`), so the same window always produces the same prediction.
'''))

# Build and write the inference notebook
infer_nb = nbf.v4.new_notebook()
infer_nb.cells = infer_cells
infer_nb.metadata = notebook_metadata()

infer_path = HERE / "fall_detection_inference_demo.ipynb"
with open(infer_path, "w", encoding="utf-8") as f:
    nbf.write(infer_nb, f)
print(f"Wrote {infer_path} ({len(infer_cells)} cells)")

print("\nDone.")
