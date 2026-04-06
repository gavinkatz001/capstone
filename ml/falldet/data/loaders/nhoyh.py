"""Loader for the HR_IMU Fall Detection Dataset (nhoyh).

Dataset structure:
  data/raw/nhoyh/
    subject_01/
      fall/
        fall1.mat, fall2.mat, ...
      non-fall/
        walk.mat, chair.mat, ...
    subject_02/
    ...

Each .mat file contains:
  time: (N, 6) -- [year, month, day, hour, minute, second]
  ax, ay, az: (N, 1) -- accelerometer in g's
  droll, dpitch, dyaw: (N, 1) -- gyro angular velocity in deg/s
  w, x, y, z: (N, 1) -- quaternion (not used)
  heart: (N, 1) -- heart rate (not used)
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io

from falldet.data.registry import get_dataset_info


@dataclass
class SampleRecord:
    """A single continuous recording from one activity."""

    dataset: str
    subject_id: str
    activity: str  # raw label from filename
    is_fall: bool
    fall_type: str | None  # standardized fall type or None for ADL
    accel: np.ndarray  # (T, 3) in g's
    gyro: np.ndarray | None  # (T, 3) in deg/s or None
    sampling_rate_hz: float
    placement: str
    metadata: dict = field(default_factory=dict)


# Map filenames to activity labels for fall files
_FALL_FILE_MAP = {
    "fall1": "Fall forward",
    "fall2": "Fall forward knees",
    "fall3": "Fall backward",
    "fall4": "Fall sideward",
    "fall5": "Fall sitting chair",
    "fall6": "Fall from bed",
}


def _load_mat_file(path: Path) -> dict[str, np.ndarray]:
    """Load a .mat file, trying scipy first then h5py for v7.3 files."""
    try:
        return scipy.io.loadmat(str(path))
    except NotImplementedError:
        import h5py

        data = {}
        with h5py.File(str(path), "r") as f:
            for key in f.keys():
                data[key] = np.array(f[key]).T  # h5py transposes MATLAB arrays
        return data


def load(raw_dir: Path) -> list[SampleRecord]:
    """Load all samples from the nhoyh dataset.

    Args:
        raw_dir: Path to data/raw/nhoyh/ containing subject_01/, subject_02/, etc.

    Returns:
        List of SampleRecord, one per .mat file.
    """
    info = get_dataset_info("nhoyh")
    records = []

    subject_dirs = sorted(raw_dir.glob("subject_*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found in {raw_dir}")

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name  # e.g. "subject_01"

        # Process fall files
        fall_dir = subject_dir / "fall"
        if fall_dir.exists():
            for mat_file in sorted(fall_dir.glob("*.mat")):
                stem = mat_file.stem  # e.g. "fall1"
                activity = _FALL_FILE_MAP.get(stem, f"Unknown fall ({stem})")
                fall_type = info["fall_label_map"].get(activity)

                data = _load_mat_file(mat_file)
                accel = np.column_stack([
                    data["ax"].ravel(),
                    data["ay"].ravel(),
                    data["az"].ravel(),
                ])
                gyro = np.column_stack([
                    data["droll"].ravel(),
                    data["dpitch"].ravel(),
                    data["dyaw"].ravel(),
                ])

                records.append(SampleRecord(
                    dataset="nhoyh",
                    subject_id=subject_id,
                    activity=activity,
                    is_fall=True,
                    fall_type=fall_type,
                    accel=accel.astype(np.float32),
                    gyro=gyro.astype(np.float32),
                    sampling_rate_hz=info["sampling_rate_hz"],
                    placement=info["placement"],
                    metadata={"source_file": str(mat_file.relative_to(raw_dir))},
                ))

        # Process non-fall files
        nf_dir = subject_dir / "non-fall"
        if nf_dir.exists():
            for mat_file in sorted(nf_dir.glob("*.mat")):
                stem = mat_file.stem  # e.g. "walk", "chair"
                # Capitalize for label lookup
                activity = stem.capitalize()

                data = _load_mat_file(mat_file)
                accel = np.column_stack([
                    data["ax"].ravel(),
                    data["ay"].ravel(),
                    data["az"].ravel(),
                ])
                gyro = np.column_stack([
                    data["droll"].ravel(),
                    data["dpitch"].ravel(),
                    data["dyaw"].ravel(),
                ])

                records.append(SampleRecord(
                    dataset="nhoyh",
                    subject_id=subject_id,
                    activity=activity,
                    is_fall=False,
                    fall_type=None,
                    accel=accel.astype(np.float32),
                    gyro=gyro.astype(np.float32),
                    sampling_rate_hz=info["sampling_rate_hz"],
                    placement=info["placement"],
                    metadata={"source_file": str(mat_file.relative_to(raw_dir))},
                ))

    return records
