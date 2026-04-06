"""Loader for the Microchip ML Fall Detection SAMD21 IMU Dataset.

Dataset structure:
  data/raw/microchip/
    falldataset/
      fall-01-acc.csv, fall-02-acc.csv, ...
      adl-01-acc.csv, adl-02-acc.csv, ...

Each CSV has columns: Svtotal, Ax, Ay, Az
  - Svtotal: signal vector total (magnitude)
  - Ax, Ay, Az: accelerometer in g's
  - No gyro data available
  - ~194 rows per file at 100 Hz = ~1.94 seconds per sample

Note: This dataset has accel only. Gyro will be set to None.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from falldet.data.loaders.nhoyh import SampleRecord
from falldet.data.registry import get_dataset_info


def load(raw_dir: Path) -> list[SampleRecord]:
    """Load all samples from the Microchip dataset.

    Args:
        raw_dir: Path to data/raw/microchip/ containing falldataset/.

    Returns:
        List of SampleRecord, one per CSV file.
    """
    info = get_dataset_info("microchip")
    records = []

    dataset_dir = raw_dir / "falldataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"falldataset/ not found in {raw_dir}")

    for csv_file in sorted(dataset_dir.glob("*-acc.csv")):
        stem = csv_file.stem  # e.g. "fall-01-acc"
        parts = stem.split("-")
        category = parts[0]  # "fall" or "adl"
        sample_num = parts[1]  # "01", "02", etc.

        is_fall = category == "fall"
        activity = category
        fall_type = info["fall_label_map"].get(category) if is_fall else None

        df = pd.read_csv(csv_file)
        accel = df[["Ax", "Ay", "Az"]].values.astype(np.float32)

        records.append(SampleRecord(
            dataset="microchip",
            subject_id=f"microchip_{sample_num}",
            activity=activity,
            is_fall=is_fall,
            fall_type=fall_type,
            accel=accel,
            gyro=None,  # no gyro in this dataset
            sampling_rate_hz=info["sampling_rate_hz"],
            placement=info["placement"],
            metadata={"source_file": str(csv_file.relative_to(raw_dir))},
        ))

    return records
