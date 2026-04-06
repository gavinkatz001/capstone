"""Windowing, normalization, augmentation, and train/val/test splitting."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from falldet.data.unified import UnifiedSample

WINDOW_SAMPLES = None  # set dynamically based on config


@dataclass
class Window:
    """A single fixed-length window ready for model input."""

    data: np.ndarray  # (window_samples, 6) -- [ax, ay, az, gx, gy, gz]
    label: int  # 1 = fall, 0 = no-fall
    valid_length: int  # how many samples are real (rest is padding)
    dataset: str
    subject_id: str
    activity: str
    fall_type: str | None
    has_gyro: bool
    metadata: dict = field(default_factory=dict)


def create_windows(
    samples: list[UnifiedSample],
    window_sec: float = 60.0,
    stride_sec: float = 30.0,
    rate_hz: float = 50.0,
) -> list[Window]:
    """Segment unified samples into fixed-length windows.

    - Long recordings (>window_sec): sliding window with stride
    - Short recordings (<window_sec): zero-pad to window_sec

    A window is labeled as fall if the source sample is a fall.
    """
    window_len = int(window_sec * rate_hz)
    stride_len = int(stride_sec * rate_hz)
    windows = []

    for sample in samples:
        T = len(sample.data)

        if T >= window_len:
            # Sliding window
            start = 0
            while start + window_len <= T:
                chunk = sample.data[start : start + window_len].copy()
                windows.append(Window(
                    data=chunk,
                    label=int(sample.is_fall),
                    valid_length=window_len,
                    dataset=sample.dataset,
                    subject_id=sample.subject_id,
                    activity=sample.activity,
                    fall_type=sample.fall_type,
                    has_gyro=sample.has_gyro,
                ))
                start += stride_len
        else:
            # Pad short recordings
            padded = np.zeros((window_len, sample.data.shape[1]), dtype=np.float32)
            padded[:T] = sample.data
            windows.append(Window(
                data=padded,
                label=int(sample.is_fall),
                valid_length=T,
                dataset=sample.dataset,
                subject_id=sample.subject_id,
                activity=sample.activity,
                fall_type=sample.fall_type,
                has_gyro=sample.has_gyro,
            ))

    return windows


def compute_normalization_stats(windows: list[Window]) -> dict:
    """Compute per-channel mean and std from a list of windows (training set only)."""
    all_data = np.concatenate([w.data[: w.valid_length] for w in windows], axis=0)
    mean = all_data.mean(axis=0).tolist()
    std = all_data.std(axis=0).tolist()
    # Avoid division by zero
    std = [s if s > 1e-8 else 1.0 for s in std]
    return {"mean": mean, "std": std}


def normalize_windows(windows: list[Window], stats: dict) -> list[Window]:
    """Apply z-score normalization to windows using precomputed stats."""
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    for w in windows:
        w.data = (w.data - mean) / std

    return windows


def subject_based_split(
    windows: list[Window],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Window], list[Window], list[Window]]:
    """Split windows by subject ID to prevent data leakage.

    Returns (train, val, test) lists of Windows.
    """
    rng = np.random.RandomState(seed)

    # Collect all unique subjects
    subjects = sorted(set(w.subject_id for w in windows))
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test : n_test + n_val])
    train_subjects = set(subjects[n_test + n_val :])

    train = [w for w in windows if w.subject_id in train_subjects]
    val = [w for w in windows if w.subject_id in val_subjects]
    test = [w for w in windows if w.subject_id in test_subjects]

    return train, val, test


def save_split_info(
    train: list[Window],
    val: list[Window],
    test: list[Window],
    output_path: Path,
) -> None:
    """Save split metadata for reproducibility."""
    info = {
        "train": {
            "n_windows": len(train),
            "n_falls": sum(1 for w in train if w.label == 1),
            "n_adl": sum(1 for w in train if w.label == 0),
            "subjects": sorted(set(w.subject_id for w in train)),
        },
        "val": {
            "n_windows": len(val),
            "n_falls": sum(1 for w in val if w.label == 1),
            "n_adl": sum(1 for w in val if w.label == 0),
            "subjects": sorted(set(w.subject_id for w in val)),
        },
        "test": {
            "n_windows": len(test),
            "n_falls": sum(1 for w in test if w.label == 1),
            "n_adl": sum(1 for w in test if w.label == 0),
            "subjects": sorted(set(w.subject_id for w in test)),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)


def build_dataset(
    samples: list[UnifiedSample],
    window_sec: float = 60.0,
    stride_sec: float = 30.0,
    rate_hz: float = 50.0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    splits_dir: Path | None = None,
) -> tuple[list[Window], list[Window], list[Window], dict]:
    """Full pipeline: window -> split -> normalize.

    Returns (train_windows, val_windows, test_windows, norm_stats).
    """
    # Create windows
    windows = create_windows(samples, window_sec, stride_sec, rate_hz)
    print(f"Created {len(windows)} windows ({sum(w.label for w in windows)} falls)")

    # Split by subject
    train, val, test = subject_based_split(windows, val_fraction, test_fraction, seed)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Compute normalization from training set only
    norm_stats = compute_normalization_stats(train)

    # Apply normalization to all splits
    train = normalize_windows(train, norm_stats)
    val = normalize_windows(val, norm_stats)
    test = normalize_windows(test, norm_stats)

    # Save split info
    if splits_dir:
        save_split_info(train, val, test, splits_dir / "split_info.json")
        # Save normalization stats
        with open(splits_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

    return train, val, test, norm_stats
