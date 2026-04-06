"""PyTorch Dataset and DataLoader factory for fall detection."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from falldet.data.preprocessing import Window


class FallDetectionDataset(Dataset):
    """PyTorch Dataset wrapping a list of Windows.

    Returns tensors in channels-first format (6, T) for Conv1d compatibility.
    """

    def __init__(self, windows: list[Window], augment: bool = False, aug_config: dict | None = None):
        self.windows = windows
        self.augment = augment
        self.aug_config = aug_config or {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        w = self.windows[idx]
        data = w.data.copy()  # (T, 6)

        if self.augment:
            data = self._apply_augmentation(data, w.valid_length)

        # Transpose to channels-first: (6, T)
        x = torch.from_numpy(data.T).float()
        y = torch.tensor([w.label], dtype=torch.float32)

        meta = {
            "dataset": w.dataset,
            "subject_id": w.subject_id,
            "activity": w.activity,
            "fall_type": w.fall_type or "",
            "valid_length": w.valid_length,
            "has_gyro": w.has_gyro,
        }

        return x, y, meta

    def _apply_augmentation(self, data: np.ndarray, valid_length: int) -> np.ndarray:
        """Apply data augmentation to the valid portion of the window."""
        cfg = self.aug_config
        valid = data[:valid_length]

        # Gaussian noise
        if cfg.get("noise_std_accel", 0) > 0:
            noise_a = np.random.randn(valid_length, 3) * cfg["noise_std_accel"]
            noise_g = np.random.randn(valid_length, 3) * cfg.get("noise_std_gyro", 0)
            valid[:, :3] += noise_a.astype(np.float32)
            valid[:, 3:] += noise_g.astype(np.float32)

        # Random scaling per axis
        scale_range = cfg.get("scale_range", [1.0, 1.0])
        if scale_range[0] != scale_range[1]:
            scales = np.random.uniform(scale_range[0], scale_range[1], size=(1, 6))
            valid *= scales.astype(np.float32)

        data[:valid_length] = valid
        return data

    def get_labels(self) -> np.ndarray:
        """Return array of all labels (for computing class weights)."""
        return np.array([w.label for w in self.windows])


def compute_pos_weight(dataset: FallDetectionDataset) -> float:
    """Compute positive class weight for BCEWithLogitsLoss."""
    labels = dataset.get_labels()
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def create_dataloader(
    dataset: FallDetectionDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    weighted_sampling: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with optional weighted sampling for class imbalance."""
    sampler = None
    if weighted_sampling and shuffle:
        labels = dataset.get_labels()
        class_counts = np.bincount(labels.astype(int))
        weights = 1.0 / class_counts[labels.astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_collate_fn,
    )


def _collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Custom collate that keeps metadata as a list of dicts."""
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)
