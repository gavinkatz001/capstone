# Data Pipeline

This document traces the full journey of data from raw public datasets to PyTorch tensors ready for training.

## Pipeline Overview

```
[Raw Datasets]          [Per-Dataset Loaders]        [Harmonization]
  nhoyh/ (.mat)    -->   nhoyh.py -> SampleRecord     |
  microchip/ (.csv) -->  microchip.py -> SampleRecord  +--> unified.py -> UnifiedSample (50Hz, 6-axis)
  edgefall/ (.csv)  -->  (not yet)                     |
  ur_fall/ (.csv)   -->  (not yet)                     |

[Preprocessing]                    [PyTorch Dataset]
  preprocessing.py                   dataset.py
  - 60s windowing                    - FallDetectionDataset
  - z-score normalization            - channels-first (6, 3000)
  - subject-based split              - weighted sampling
  --> Window objects                  - augmentation on-the-fly
                                     --> DataLoader batches
```

## Stage 1: Raw Datasets

### Dataset Inventory

| Dataset | Files | Format | Sampling Rate | Channels | Subjects | Falls | ADL | Status |
|---------|-------|--------|---------------|----------|----------|-------|-----|--------|
| nhoyh (HR_IMU) | 349 .mat files | MATLAB | 50 Hz | ax,ay,az,droll,dpitch,dyaw | 21 | 104 | 245 | **Loaded** |
| Microchip | 20 .csv files | CSV | 100 Hz | Ax,Ay,Az (accel only) | N/A | 10 | 10 | **Loaded** |
| IEEE EdgeFall | ~1500 CSVs | CSV | Unknown | accel + gyro | Unknown | Many | Many | Not yet |
| UR Fall | ~70 CSVs | CSV | 60 Hz | accel only | Unknown | 30 | 40 | Not yet |

### Datasets We Evaluated and Skipped

- **Kaggle Walker Fall Detection**: Walker-mounted sensor, not body-worn. Different motion dynamics, would confuse the model.
- **Mendeley dataset**: Contains only ADL data, no falls. Useless for supervised learning.
- **IEEE DataPort keyword page**: Just an index page, not a dataset.
- **FRDR Canadian dataset**: Was being backed up/migrated at evaluation time. Worth revisiting.

### nhoyh Dataset Details

This is the primary dataset. Structure on disk:

```
data/raw/nhoyh/
  subject_01/
    fall/
      fall1.mat  -- "Fall forward" (1089 samples = 21.8s at 50Hz)
      fall2.mat  -- "Fall forward knees"
      fall3.mat  -- "Fall backward"
      fall4.mat  -- "Fall sideward"
      fall6.mat  -- "Fall from bed" (note: fall5 is missing for some subjects)
    non-fall/
      walk.mat, chair.mat, clap.mat, cloth.mat, eat.mat, hair.mat,
      shoe.mat, stair.mat, teeth.mat, wash.mat, write.mat, zip.mat
  subject_02/
  ...
  subject_21/
```

Each `.mat` file contains these MATLAB arrays (all shape `(N, 1)` except time):
- `time` (N, 6): [year, month, day, hour, minute, second]
- `ax`, `ay`, `az`: accelerometer in g's
- `droll`, `dpitch`, `dyaw`: gyroscope angular velocity in deg/s
- `w`, `x`, `y`, `z`: quaternion orientation (we ignore these)
- `heart`: heart rate sensor (we ignore this)

The loader extracts only accel + gyro columns. Typical recording duration is 15-50 seconds.

### Microchip Dataset Details

```
data/raw/microchip/
  falldataset/
    fall-01-acc.csv through fall-10-acc.csv
    adl-01-acc.csv through adl-10-acc.csv
```

Each CSV has columns: `Svtotal, Ax, Ay, Az` (no headers in some files, detected by pandas).
- `Svtotal`: signal vector total (magnitude) -- we don't use this, compute our own
- `Ax, Ay, Az`: accelerometer in g's
- **No gyroscope data**
- ~194 rows per file at 100 Hz = ~1.9 seconds per sample
- No subject IDs (treated as independent samples)

## Stage 2: Per-Dataset Loaders

**Location**: `falldet/data/loaders/nhoyh.py` and `falldet/data/loaders/microchip.py`

Each loader has a single function: `load(raw_dir: Path) -> list[SampleRecord]`

The `SampleRecord` dataclass (defined in `nhoyh.py`, shared by all loaders):

```python
@dataclass
class SampleRecord:
    dataset: str            # "nhoyh" or "microchip"
    subject_id: str         # "subject_01" or "microchip_01"
    activity: str           # raw label like "Fall forward" or "walk"
    is_fall: bool           # binary label
    fall_type: str | None   # standardized: "forward", "backward", "lateral", "seated", etc.
    accel: np.ndarray       # (T, 3) in g's
    gyro: np.ndarray | None # (T, 3) in deg/s, or None if dataset has no gyro
    sampling_rate_hz: float # original rate before resampling
    placement: str          # "wrist", "unknown", etc.
    metadata: dict          # source file path, etc.
```

### How to Add a New Dataset Loader

1. Create `falldet/data/loaders/your_dataset.py`
2. Implement `def load(raw_dir: Path) -> list[SampleRecord]`
3. Add metadata to `falldet/data/registry.py` including `fall_label_map` and `adl_label_map`
4. Add an `elif` branch in `falldet/data/unified.py::load_and_harmonize()`
5. Add the dataset name to `configs/default.yaml` under `data.datasets`

## Stage 3: Harmonization

**Location**: `falldet/data/unified.py`

Converts all `SampleRecord`s to `UnifiedSample`s with a common format:

- **Target rate**: 50 Hz (resampled via `scipy.signal.resample`)
- **Channels**: 6 floats per timestep: `[ax, ay, az, gx, gy, gz]`
- **Units**: accel in g's, gyro in deg/s
- **Missing gyro**: If a dataset has no gyro (e.g., Microchip), the gyro channels are filled with 0.0 and `has_gyro=False` is set

### Why 50 Hz?

- The nhoyh dataset (our largest, 349 records) is already at 50 Hz -- no resampling needed for the majority of data
- Human fall dynamics have frequency content below 20 Hz. Nyquist frequency of 50 Hz captures everything meaningful.
- Higher rates (100 Hz, 128 Hz) would double or triple data volume and training time without improving detection accuracy.
- The Microchip dataset at 100 Hz is downsampled 2:1. UR Fall at 60 Hz would be downsampled ~1.2:1.

### Resampling Details

Uses `scipy.signal.resample(signal, n_target_samples, axis=0)` which applies a Fourier-based resampling (anti-aliasing included). This preserves the frequency content up to the Nyquist of the target rate.

## Stage 4: Windowing and Preprocessing

**Location**: `falldet/data/preprocessing.py`

### Windowing Strategy

All recordings are segmented into 60-second windows (3000 samples at 50 Hz).

**For recordings >= 60 seconds**:
- Sliding window with configurable stride (default: 30s = 50% overlap for training)
- This creates more training examples from limited data

**For recordings < 60 seconds** (this is ALL current data -- max duration is 51.6s):
- The recording is placed at the start of a 60-second buffer
- The rest is zero-padded
- `valid_length` tracks how many samples are real data
- This matches deployment: in production, a fall might only occupy a few seconds of a 60-second packet

**Labels**: A window inherits the `is_fall` label from its source recording. If the source is a fall, the entire window is labeled as a fall.

### Current Data Volume

After windowing with the nhoyh + microchip datasets:

```
Total windows: 369
  Falls: 114 (30.9%)
  ADL: 255 (69.1%)

Split (subject-based):
  Train: 267 (82 falls, 185 ADL)
  Val: 37 (13 falls, 24 ADL)
  Test: 65 (19 falls, 46 ADL)
```

This is a small dataset. Augmentation and additional dataset loaders will be important for improving generalization.

### Normalization

Per-channel z-score normalization:
- Mean and std are computed **from the training set only** (to prevent data leakage)
- Applied identically to train, val, and test sets
- Stats are saved to `data/splits/norm_stats.json` for use at inference time

Current normalization stats (from training set):

```
Mean: [-0.0420, 0.0117, -0.0178, 0.1724, -0.0728, 0.9173]
Std:  [ 0.2993, 0.3876,  0.3110, 79.876, 49.341, 73.311]
```

Note the large difference in scale between accel channels (order ~0.3) and gyro channels (order ~50-80). This is expected: accel is in g's, gyro is in deg/s. The z-score normalization brings them to the same scale.

### Subject-Based Splitting

**Critical for honest evaluation.** Windows from the same subject NEVER appear in both train and test sets.

Why this matters: Each person has a unique gait pattern, sensor attachment angle, and movement style. If the same subject's data is in both train and test, the model can memorize subject-specific patterns instead of learning general fall signatures. This would give inflated test accuracy that doesn't generalize to new users.

Current split (seed=42):
- 21 nhoyh subjects shuffled, then divided: ~15 train, ~3 val, ~3 test
- Microchip samples (no subject IDs) are assigned pseudo-IDs and split accordingly

## Stage 5: PyTorch Dataset

**Location**: `falldet/data/dataset.py`

### FallDetectionDataset

Wraps a `list[Window]` as a PyTorch Dataset. Key behavior:

- Returns tensors in **channels-first format**: `(6, 3000)` -- required by Conv1d
- Each `__getitem__` returns `(x, y, metadata)`:
  - `x`: `torch.float32` tensor of shape `(6, 3000)`
  - `y`: `torch.float32` tensor of shape `(1,)` -- 0.0 or 1.0
  - `metadata`: dict with dataset name, subject ID, activity, fall type, valid length

### Data Augmentation

Applied on-the-fly during training (not pre-computed). Currently implemented:

- **Gaussian noise**: sigma=0.01g for accel, 1.0 deg/s for gyro
- **Random scaling**: each axis independently scaled by [0.9, 1.1]

Planned but not yet implemented:
- Time jitter (shift window start)
- Time warping (stretch/compress via interpolation)
- 3D rotation (simulate different sensor orientations)

### Class Imbalance Handling

Falls are the minority class (~31% of windows). Two mechanisms:

1. **Weighted sampling** (`WeightedRandomSampler`): Oversamples falls so each batch has roughly equal representation. Enabled by default in the training DataLoader.

2. **Weighted loss** (`pos_weight` in `BCEWithLogitsLoss`): Scales the loss for positive (fall) samples by `n_negative / n_positive`. Current pos_weight = 2.26.

### DataLoader Configuration

- Batch size: 32 (configurable)
- `num_workers=0` on Windows (multiprocessing has issues with Windows file handles)
- `pin_memory=True` if CUDA is available
- Custom `collate_fn` to keep metadata as a list of dicts

## The Complete Pipeline in Code

```python
from pathlib import Path
from falldet.data.unified import load_and_harmonize
from falldet.data.preprocessing import build_dataset
from falldet.data.dataset import FallDetectionDataset, create_dataloader, compute_pos_weight

# Stage 1-3: Load raw data -> SampleRecords -> UnifiedSamples
samples = load_and_harmonize(Path("data/raw"), ["nhoyh", "microchip"])

# Stage 4: UnifiedSamples -> Windows (60s, normalized, split)
train_win, val_win, test_win, norm_stats = build_dataset(
    samples, window_sec=60.0, stride_sec=30.0
)

# Stage 5: Windows -> PyTorch DataLoaders
train_ds = FallDetectionDataset(train_win, augment=True, aug_config={...})
val_ds = FallDetectionDataset(val_win)
train_loader = create_dataloader(train_ds, batch_size=32, weighted_sampling=True)
val_loader = create_dataloader(val_ds, batch_size=32, shuffle=False)

# Each batch: x.shape = (32, 6, 3000), y.shape = (32, 1)
```
