"""Generate the standalone Colab notebook for the verified six-axis baseline."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md("""
# Preliminary six-axis fall detector

This notebook runs standalone in Google Colab. It trains a **preliminary research baseline** from the verified [HR_IMU wrist dataset](https://github.com/nhoyh/HR_IMU_falldetection_dataset). Each input is 4 seconds of 50 Hz motion data with columns `ax, ay, az` in g and `gx, gy, gz` in degrees/second. A positive window is centered on the strongest acceleration event in a fall recording; negative windows come from non-fall recordings. The source records have fall/ADL labels but no precise impact timestamps, so the peak is a heuristic, not ground truth timing.

The source assessment in `ml/DATA_SOURCE_PLAN.md` explains why the other links are not automatically merged yet. The adapter contract below is ready for verified six-axis sources: each record must carry a source, original participant ID, placement, native rate, units, six-channel array, and recording label. Subject IDs are prefixed with the source name before splitting.

**Colab:** upload this notebook, select a GPU runtime if available, then Run all. It downloads the HR_IMU repository and saves `preliminary_fall_model.pt`. This is a 4-second event detector; a future 60-second packet service should scan multiple 4-second windows and calibrate its packet-level false-alarm rate separately.
""")

md("## 1. Setup and download")
code("""
import copy
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SEED = 42
RATE_HZ = 50
WINDOW_SEC = 4
WINDOW_LEN = RATE_HZ * WINDOW_SEC
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 8
DATA_ROOT = Path('data')
NHOYH_ROOT = DATA_ROOT / 'nhoyh'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

if not (NHOYH_ROOT / 'subject_01').exists():
    DATA_ROOT.mkdir(exist_ok=True)
    subprocess.run([
        'git', 'clone', '--depth', '1',
        'https://github.com/nhoyh/HR_IMU_falldetection_dataset',
        str(NHOYH_ROOT),
    ], check=True)
print('Dataset ready:', NHOYH_ROOT)
""")

md("## 2. Inspect and harmonize records")
code("""
CHANNELS = ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
FALL_TYPE = {
    'fall1': 'forward', 'fall2': 'backward', 'fall3': 'lateral',
    'fall4': 'forward', 'fall5': 'backward', 'fall6': 'lateral',
}

@dataclass
class Record:
    source: str
    subject_id: str
    recording_id: str
    placement: str
    data: np.ndarray  # float32 (time, 6), g and degrees/second, 50 Hz
    label: int
    activity: str
    fall_type: str | None

def harmonize_six_axis(values, source_rate_hz, accel_unit='g', gyro_unit='deg/s'):
    # Validate units/channels, convert units, and downsample to 50 Hz.
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6 or len(x) < 20:
        raise ValueError(f'Expected at least 20 rows x 6 channels, got {x.shape}')
    if not np.isfinite(x).all() or source_rate_hz < RATE_HZ:
        raise ValueError('Nonfinite values or source rate below 50 Hz')
    x = x.copy()
    if accel_unit == 'm/s2':
        x[:, :3] /= 9.80665
    elif accel_unit != 'g':
        raise ValueError(f'Unverified acceleration unit: {accel_unit}')
    if gyro_unit == 'rad/s':
        x[:, 3:] *= 180 / np.pi
    elif gyro_unit != 'deg/s':
        raise ValueError(f'Unverified gyroscope unit: {gyro_unit}')
    if source_rate_hz != RATE_HZ:
        from fractions import Fraction
        ratio = Fraction(RATE_HZ / source_rate_hz).limit_denominator(1000)
        x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
    return np.asarray(x, dtype=np.float32)

def load_nhoyh(root):
    records = []
    for path in sorted(root.glob('subject_*/*/*.mat')):
        is_fall = path.parent.name == 'fall'
        if path.parent.name not in ('fall', 'non-fall'):
            continue
        mat = scipy.io.loadmat(path, variable_names=[
            'ax', 'ay', 'az', 'droll', 'dpitch', 'dyaw'])
        values = np.column_stack([np.ravel(mat[key]) for key in (
            'ax', 'ay', 'az', 'droll', 'dpitch', 'dyaw')])
        records.append(Record(
            source='nhoyh',
            subject_id=f'nhoyh:{path.parent.parent.name}',
            recording_id=str(path.relative_to(root)),
            placement='left_wrist',
            data=harmonize_six_axis(values, 50, 'g', 'deg/s'),
            label=int(is_fall),
            activity=path.stem,
            fall_type=FALL_TYPE.get(path.stem) if is_fall else None,
        ))
    if not records:
        raise RuntimeError(f'No recordings found under {root}')
    return records

records = load_nhoyh(NHOYH_ROOT)
inventory = pd.DataFrame([{
    'source': r.source, 'subject': r.subject_id, 'label': r.label,
    'activity': r.activity, 'seconds': len(r.data) / RATE_HZ,
} for r in records])
display(inventory.groupby(['source', 'label']).agg(
    recordings=('activity', 'size'), subjects=('subject', 'nunique'),
    min_sec=('seconds', 'min'), median_sec=('seconds', 'median'),
    max_sec=('seconds', 'max')).round(2))
assert set(inventory.label) == {0, 1}
""")

md("""
## 3. Subject split, then windowing

All windows from one participant stay in one split. For fall recordings, only a window around the largest acceleration magnitude is labeled positive. The rest of that recording is **unlabeled** and excluded. Non-fall recordings yield non-overlapping 4-second negative windows plus one window around their strongest motion. This avoids padding ~20-second trials to 60 seconds and avoids labeling pre-fall motion as an impact. Peak selection is done independently within each recording, after subject splitting.
""")
code("""
subjects = sorted({r.subject_id for r in records})
train_subjects, held_subjects = train_test_split(
    subjects, test_size=0.30, random_state=SEED, shuffle=True)
val_subjects, test_subjects = train_test_split(
    held_subjects, test_size=0.50, random_state=SEED, shuffle=True)
split_subjects = {
    'train': set(train_subjects), 'val': set(val_subjects),
    'test': set(test_subjects),
}
assert not (split_subjects['train'] & split_subjects['val'])
assert not (split_subjects['train'] & split_subjects['test'])
assert not (split_subjects['val'] & split_subjects['test'])

def window_record(record):
    x = record.data
    if len(x) < WINDOW_LEN:
        return []
    if record.label:
        # The wrist dataset documents gravity removal. Strongest dynamic acceleration
        # is a plausible event anchor, but may not match the exact impact time.
        peak = int(np.argmax(np.linalg.norm(x[:, :3], axis=1)))
        start = int(np.clip(peak - WINDOW_LEN // 2, 0, len(x) - WINDOW_LEN))
        starts = [start]
    else:
        starts = list(range(0, len(x) - WINDOW_LEN + 1, WINDOW_LEN))
        # Include the strongest ADL motion as a hard negative, even when
        # the non-overlapping grid misses it.
        peak = int(np.argmax(np.linalg.norm(x[:, :3], axis=1)))
        peak_start = int(np.clip(peak - WINDOW_LEN // 2, 0, len(x) - WINDOW_LEN))
        starts = sorted(set(starts + [peak_start]))
    return [(x[s:s + WINDOW_LEN].copy(), record.label, record) for s in starts]

splits = {}
for name, members in split_subjects.items():
    splits[name] = [window for record in records if record.subject_id in members
                    for window in window_record(record)]
    labels = [w[1] for w in splits[name]]
    print(name, 'subjects=', len(members), 'windows=', len(labels),
          'falls=', sum(labels), 'ADLs=', len(labels) - sum(labels))
    if set(labels) != {0, 1}:
        raise RuntimeError(f'{name} needs both classes; change SEED or split method')
""")

md("## 4. Normalize from training windows only")
code("""
train_values = np.concatenate([w[0] for w in splits['train']], axis=0)
norm_mean = train_values.mean(axis=0).astype(np.float32)
norm_std = train_values.std(axis=0).astype(np.float32)
norm_std = np.where(norm_std < 1e-6, 1.0, norm_std).astype(np.float32)
print('mean:', norm_mean.round(3).tolist())
print('std: ', norm_std.round(3).tolist())

class WindowDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.labels = np.array([r[1] for r in rows], dtype=np.int64)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        x, label, _ = self.rows[i]
        x = ((x - norm_mean) / norm_std).T.copy()
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)

datasets = {name: WindowDataset(rows) for name, rows in splits.items()}
counts = np.bincount(datasets['train'].labels, minlength=2)
weights = 1.0 / counts[datasets['train'].labels]
sampler = WeightedRandomSampler(
    torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
loaders = {
    'train': DataLoader(datasets['train'], batch_size=BATCH_SIZE, sampler=sampler),
    'val': DataLoader(datasets['val'], batch_size=BATCH_SIZE, shuffle=False),
    'test': DataLoader(datasets['test'], batch_size=BATCH_SIZE, shuffle=False),
}
""")

md("## 5. Train the 1D CNN")
code("""
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        previous = 6
        for channels, kernel in zip((32, 64, 128), (7, 5, 5)):
            blocks += [nn.Conv1d(previous, channels, kernel, stride=2,
                                 padding=kernel // 2), nn.BatchNorm1d(channels),
                       nn.ReLU()]
            previous = channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.25), nn.Linear(128, 1))
    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

model = CNN1D().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

@torch.no_grad()
def predict_loader(loader):
    model.eval()
    labels, probabilities = [], []
    for x, y in loader:
        probability = torch.sigmoid(model(x.to(device))).cpu().numpy()
        labels.extend(y.numpy())
        probabilities.extend(probability)
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities)

best_auc = -1.0
best_state = None
best_epoch = None
stale = 0
history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for x, y in loaders['train']:
        optimizer.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    val_y, val_p = predict_loader(loaders['val'])
    val_auc = roc_auc_score(val_y, val_p)
    history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                    'val_auc': float(val_auc)})
    print(f'epoch {epoch:02d}  loss={np.mean(losses):.4f}  val_auc={val_auc:.4f}')
    if val_auc > best_auc + 1e-4:
        best_auc = val_auc
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        stale = 0
    else:
        stale += 1
        if stale >= PATIENCE:
            print('Early stopping')
            break
model.load_state_dict(best_state)
""")

md("""
## 6. Choose threshold on validation, evaluate test once

The selected threshold maximizes validation sensitivity subject to a validation false-positive rate at most 10%. If none meets that constraint, it minimizes the distance to the target of 95% sensitivity and 10% false-positive rate. These are window-level metrics on simulated falls; they do not establish safety in real use.
""")
code("""
def metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {
        'tpr': tp / (tp + fn) if tp + fn else 0.0,
        'fpr': fp / (fp + tn) if fp + tn else 0.0,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'auc': float(roc_auc_score(y, p)),
    }

val_y, val_p = predict_loader(loaders['val'])
thresholds = sorted(set([0.0, 1.0] + val_p.tolist()))
choices = [(t, metrics(val_y, val_p, t)) for t in thresholds]
feasible = [(t, m) for t, m in choices if m['fpr'] <= 0.10]
if feasible:
    threshold, val_metrics = max(feasible, key=lambda tm: (tm[1]['tpr'], -tm[1]['fpr'], tm[0]))
else:
    threshold, val_metrics = min(choices, key=lambda tm: (max(0, 0.95-tm[1]['tpr']) +
                                                      max(0, tm[1]['fpr']-0.10)))
test_y, test_p = predict_loader(loaders['test'])
test_metrics = metrics(test_y, test_p, threshold)
print('Best epoch:', best_epoch, 'threshold:', round(float(threshold), 4))
print('Validation:', val_metrics)
print('Held-out test:', test_metrics)
print('Test subjects:', sorted(split_subjects['test']))

# Recording-level view: one positive window per fall recording, many negatives
# per ADL recording. Aggregate ADL window predictions with max (any alert).
test_rows = splits['test']
record_preds = {}
for (_, label, record), p in zip(test_rows, test_p):
    entry = record_preds.setdefault(record.recording_id, {
        'label': label, 'source': record.source, 'probs': []})
    entry['probs'].append(float(p))
rec_y = np.array([item['label'] for item in record_preds.values()])
rec_p = np.array([max(item['probs']) for item in record_preds.values()])
print('Recording-level:', metrics(rec_y, rec_p, threshold))
""")

md("## 7. Save checkpoint and use one window")
code("""
checkpoint = {
    'model_state_dict': model.cpu().state_dict(),
    'model_name': 'CNN1D_4sec',
    'channels': CHANNELS,
    'rate_hz': RATE_HZ,
    'window_seconds': WINDOW_SEC,
    'accel_unit': 'g',
    'gyro_unit': 'deg/s',
    'norm_mean': norm_mean.tolist(),
    'norm_std': norm_std.tolist(),
    'threshold': float(threshold),
    'seed': SEED,
    'best_epoch': best_epoch,
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'split_subjects': {k: sorted(v) for k, v in split_subjects.items()},
    'source': 'https://github.com/nhoyh/HR_IMU_falldetection_dataset',
}
torch.save(checkpoint, 'preliminary_fall_model.pt')
print('Saved preliminary_fall_model.pt')

def predict_window(raw_window):
    # Input: 200x6 raw values at 50 Hz, accel g, gyro deg/s.
    x = np.asarray(raw_window, dtype=np.float32)
    if x.shape != (WINDOW_LEN, 6) or not np.isfinite(x).all():
        raise ValueError(f'Expected finite {(WINDOW_LEN, 6)} array')
    z = ((x - norm_mean) / norm_std).T.copy()
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.from_numpy(z[None]))).item()
    return {'fall_probability': probability, 'is_fall': probability >= threshold}

print('Example:', predict_window(splits['test'][0][0]))
try:
    from google.colab import files
    files.download('preliminary_fall_model.pt')
except ImportError:
    pass
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'colab': {'name': 'preliminary_fall_training.ipynb', 'provenance': []},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
})
output = ROOT / 'preliminary_fall_training.ipynb'
nbf.write(notebook, output)
print(output)
