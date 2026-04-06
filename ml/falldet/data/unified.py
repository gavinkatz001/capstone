"""Harmonize multiple datasets into a common format.

Target unified format:
  - Sampling rate: 50 Hz
  - Channels: 6 floats per sample [ax, ay, az, gx, gy, gz]
  - Units: accel in g's, gyro in deg/s
  - If no gyro, fill with zeros and flag has_gyro=False
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample

from falldet.data.loaders.nhoyh import SampleRecord

TARGET_RATE_HZ = 50


@dataclass
class UnifiedSample:
    """A harmonized sample in the common format."""

    dataset: str
    subject_id: str
    activity: str
    is_fall: bool
    fall_type: str | None
    data: np.ndarray  # (T, 6) -- [ax, ay, az, gx, gy, gz] at TARGET_RATE_HZ
    has_gyro: bool
    sampling_rate_hz: float  # always TARGET_RATE_HZ after harmonization
    placement: str
    duration_sec: float
    metadata: dict = field(default_factory=dict)


def _resample_to_target(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Resample a (T, C) signal from source_rate to target_rate."""
    if abs(source_rate - target_rate) < 0.1:
        return signal  # already at target rate

    n_samples_target = int(len(signal) * target_rate / source_rate)
    if n_samples_target < 1:
        return signal[:1]

    return resample(signal, n_samples_target, axis=0).astype(np.float32)


def harmonize_record(record: SampleRecord, target_rate: float = TARGET_RATE_HZ) -> UnifiedSample:
    """Convert a SampleRecord to a UnifiedSample at the target sampling rate."""
    accel = record.accel  # (T, 3)

    if record.gyro is not None:
        gyro = record.gyro  # (T, 3)
        has_gyro = True
    else:
        gyro = np.zeros_like(accel)
        has_gyro = False

    # Stack into (T, 6)
    data = np.column_stack([accel, gyro]).astype(np.float32)

    # Resample if needed
    data = _resample_to_target(data, record.sampling_rate_hz, target_rate)

    duration_sec = len(data) / target_rate

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
        duration_sec=duration_sec,
        metadata=record.metadata,
    )


def harmonize_all(
    records: list[SampleRecord], target_rate: float = TARGET_RATE_HZ
) -> list[UnifiedSample]:
    """Harmonize a list of SampleRecords to unified format."""
    return [harmonize_record(r, target_rate) for r in records]


def load_and_harmonize(raw_base_dir: Path, datasets: list[str]) -> list[UnifiedSample]:
    """Load and harmonize multiple datasets.

    Args:
        raw_base_dir: Path to data/raw/ directory.
        datasets: List of dataset names to load (e.g. ["nhoyh", "microchip"]).

    Returns:
        Combined list of UnifiedSamples.
    """
    all_records = []

    for ds_name in datasets:
        ds_dir = raw_base_dir / ds_name
        if not ds_dir.exists():
            print(f"Warning: Dataset directory not found: {ds_dir}, skipping.")
            continue

        if ds_name == "nhoyh":
            from falldet.data.loaders.nhoyh import load
        elif ds_name == "microchip":
            from falldet.data.loaders.microchip import load
        else:
            print(f"Warning: No loader for '{ds_name}', skipping.")
            continue

        records = load(ds_dir)
        print(f"Loaded {len(records)} records from {ds_name}")
        all_records.extend(records)

    unified = harmonize_all(all_records)
    print(f"Harmonized {len(unified)} total samples to {TARGET_RATE_HZ} Hz")

    return unified
