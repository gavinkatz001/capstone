"""Generate the standalone Colab notebook for the multi-source six-axis baseline."""

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

This standalone Colab notebook trains a **preliminary research baseline** on harmonized accelerometer and gyroscope recordings. The common input is 4 seconds at 50 Hz with columns `ax, ay, az` in g and `gx, gy, gz` in degrees/second. Fall recordings supply a window around the strongest motion event; non-fall recordings supply spaced windows and a hard negative. Peak timing is a heuristic, not a ground truth impact timestamp.

The source adapters cover [HR_IMU](https://github.com/nhoyh/HR_IMU_falldetection_dataset), [FallAllD](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living), [UP-Fall / HAR-UP](https://sites.google.com/up.edu.mx/har-up/), [JasonKau](https://github.com/jasonkau/fall-detection-dataset-IMU), [SisFall](https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/), [SmartFallMM](https://github.com/txst-cs-smartfall/SmartFallMM-Dataset), and [UNIVPM](https://pmc.ncbi.nlm.nih.gov/articles/PMC6660610/). HR_IMU and JasonKau can be fetched by this notebook. Other archives must be placed under `data/` as described below. A source is never silently omitted: the inventory prints its status. FallAllD requires an explicit manifest because the IEEE archive's raw layout was not available for verification. SmartFallMM watch units are provisional inferences from inspected values because its README omits them.

**Colab:** upload this notebook, optionally mount Drive containing staged datasets, set the paths below, then Run all. The notebook saves `preliminary_fall_model.pt`. A future 60-second packet service should scan multiple 4-second windows and calibrate its packet-level false-alarm rate separately.
""")

md("## 1. Setup and download")
code("""
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/fall_data') if USE_DRIVE else Path('data')
NHOYH_ROOT = DATA_ROOT / 'nhoyh'
JASON_ROOT = DATA_ROOT / 'jasonkau'
UPFALL_ROOT = DATA_ROOT / 'upfall'
SISFALL_ROOT = DATA_ROOT / 'sisfall'
SMART_ROOT = DATA_ROOT / 'smartfallmm'
UNIVPM_ROOT = DATA_ROOT / 'univpm'
FALLALLD_ROOT = DATA_ROOT / 'fallalld'
FETCH_JASONKAU = True
# Watch-unit inference from inspected files; override if device documentation differs.
SMART_ACCEL_UNIT = 'm/s2'
SMART_GYRO_UNIT = 'rad/s'

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
if FETCH_JASONKAU and not JASON_ROOT.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/jasonkau/fall-detection-dataset-IMU',
                    str(JASON_ROOT)], check=True)
try:
    import openpyxl
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'openpyxl'], check=True)
""")

md("## 2. Stage and inspect the source archives")
md("""
Place extracted files in `data/upfall`, `data/sisfall`, `data/smartfallmm`, `data/univpm`, and `data/fallalld`. In Colab, mount Drive and change the root paths above to folders in Drive. The notebook walks nested directories, so the original archive root may remain inside each folder. UP-Fall expects per-trial CSVs named with `S#A#T#`; SisFall expects `F01_SA01_R01.txt` style names; SmartFallMM expects its GitHub `young/` and `old/` directories; UNIVPM expects the Data in Brief article's `Fall/` and `ADL/` text folders. For FallAllD, create `fallalld_manifest.csv` with columns `file,subject_id,placement,label,rate_hz,accel_unit,gyro_unit,ax,ay,az,gx,gy,gz`; the six axis fields name columns in each staged CSV. `label` is 1 for fall, 0 for ADL. This manifest is required until the gated IEEE archive is inspected.

SmartFallMM's README documents timestamps and sample rates but not units. Inspected watch samples have ~9.8 m/s² at rest and gyro magnitudes consistent with rad/s, so those units are provisional defaults. The adapter pairs matching watch files, keeps the longest continuous overlap, and skips pairs with less than 4 seconds of shared data. JasonKau has no participant IDs, so it is used for training only and cannot contribute to subject-independent validation or test results.
""")
code((ROOT / 'multisource_loaders.py').read_text(encoding='utf-8'))
code("""
source_specs = [
    ('nhoyh', NHOYH_ROOT, lambda: load_nhoyh(NHOYH_ROOT)),
    ('jasonkau', JASON_ROOT, lambda: load_jasonkau(JASON_ROOT)),
    ('upfall', UPFALL_ROOT, lambda: load_upfall(UPFALL_ROOT)),
    ('sisfall', SISFALL_ROOT, lambda: load_sisfall(SISFALL_ROOT)),
    ('univpm', UNIVPM_ROOT, lambda: load_univpm(UNIVPM_ROOT)),
    ('fallalld', FALLALLD_ROOT, lambda: load_fallalld_manifest(FALLALLD_ROOT)),
    ('smartfallmm', SMART_ROOT, lambda: load_smartfallmm(
        SMART_ROOT, 'watch', SMART_ACCEL_UNIT, SMART_GYRO_UNIT)),
]
records = []
for name, root, loader in source_specs:
    if not root.exists():
        print(f'{name}: NOT STAGED at {root}')
        continue
    if name == 'smartfallmm' and (SMART_ACCEL_UNIT is None or SMART_GYRO_UNIT is None):
        raise ValueError('SmartFallMM needs explicit watch acceleration and gyro units')
    added = loader()
    if not added:
        print(f'{name}: no compatible six-axis records found under {root}')
        continue
    records.extend(added)
    print(f'{name}: {len(added)} records, {sum(r.label for r in added)} falls')
if not records:
    raise RuntimeError('No usable records were loaded')
inventory = pd.DataFrame([{
    'source': r.source, 'subject': r.subject_id, 'label': r.label,
    'placement': r.placement, 'activity': r.activity,
    'seconds': len(r.data) / RATE_HZ,
} for r in records])
display(inventory.groupby(['source', 'placement', 'label']).agg(
    recordings=('activity', 'size'), subjects=('subject', 'nunique'),
    min_sec=('seconds', 'min'), median_sec=('seconds', 'median'),
    max_sec=('seconds', 'max')).round(2))
assert set(inventory.label) == {0, 1}
""")

md("""
## 3. Subject split, then windowing

All windows from one participant stay in one split, and each source's participants are split separately. JasonKau has no participant IDs and contributes training windows only. For fall recordings, one window around the largest acceleration magnitude is labeled positive; the rest is excluded. Non-fall recordings yield spaced negative windows plus the strongest motion. This avoids labeling pre-fall motion as an impact. The same six-axis shape does not make wrist and waist distributions identical, so results are reported by source and placement.
""")
code("""
split_subjects = {'train': set(), 'val': set(), 'test': set()}
rng = np.random.default_rng(SEED)
for source in sorted({r.source for r in records}):
    source_rows = [r for r in records if r.source == source]
    if source_rows[0].split_policy == 'train_only':
        split_subjects['train'].update(r.subject_id for r in source_rows)
        continue
    subjects = sorted({r.subject_id for r in source_rows})
    if len(subjects) < 3:
        raise ValueError(f'{source} needs at least 3 subjects for subject-level splits')
    shuffled = rng.permutation(subjects).tolist()
    n_test = max(1, round(len(subjects) * 0.15))
    n_val = max(1, round(len(subjects) * 0.15))
    split_subjects['test'].update(shuffled[:n_test])
    split_subjects['val'].update(shuffled[n_test:n_test+n_val])
    split_subjects['train'].update(shuffled[n_test+n_val:])
assert not (split_subjects['train'] & split_subjects['val'])
assert not (split_subjects['train'] & split_subjects['test'])
assert not (split_subjects['val'] & split_subjects['test'])

def window_record(record):
    x = record.data
    if len(x) < WINDOW_LEN:
        return []
    if record.label:
        # Use change from the recording's median gravity vector as an event anchor.
        motion = x[:, :3] - np.median(x[:, :3], axis=0)
        peak = int(np.argmax(np.linalg.norm(motion, axis=1)))
        start = int(np.clip(peak - WINDOW_LEN // 2, 0, len(x) - WINDOW_LEN))
        starts = [start]
    else:
        starts = list(range(0, len(x) - WINDOW_LEN + 1, WINDOW_LEN))
        # Include the strongest ADL motion as a hard negative, even when
        # the non-overlapping grid misses it.
        motion = x[:, :3] - np.median(x[:, :3], axis=0)
        peak = int(np.argmax(np.linalg.norm(motion, axis=1)))
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
        raise RuntimeError(f'{name} needs both classes; inspect subject labels or change SEED')
""")

md("## 4. Remove static acceleration and normalize from training windows only")
code("""
def preprocess_window(x):
    # Center each acceleration axis within this window. This removes the static
    # gravity offset from sources that retained it and also applies at inference.
    x = np.asarray(x, dtype=np.float32).copy()
    x[:, :3] -= np.median(x[:, :3], axis=0)
    return x

train_values = np.concatenate([preprocess_window(w[0]) for w in splits['train']], axis=0)
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
        x = ((preprocess_window(x) - norm_mean) / norm_std).T.copy()
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
for source in sorted({r.source for _, _, r in splits['test']}):
    indices = [i for i, (_, _, r) in enumerate(splits['test']) if r.source == source]
    if len(set(test_y[indices])) == 2:
        print('Test source', source, metrics(test_y[indices], test_p[indices], threshold))
    else:
        print('Test source', source, 'one class only; AUC undefined')
for placement in sorted({r.placement for _, _, r in splits['test']}):
    indices = [i for i, (_, _, r) in enumerate(splits['test']) if r.placement == placement]
    if len(set(test_y[indices])) == 2:
        print('Test placement', placement, metrics(test_y[indices], test_p[indices], threshold))
    else:
        print('Test placement', placement, 'one class only; AUC undefined')

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
    'accel_preprocess': 'subtract_per_window_axis_median',
    'norm_mean': norm_mean.tolist(),
    'norm_std': norm_std.tolist(),
    'threshold': float(threshold),
    'seed': SEED,
    'best_epoch': best_epoch,
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'split_subjects': {k: sorted(v) for k, v in split_subjects.items()},
    'sources': sorted({r.source for r in records}),
    'placements': sorted({r.placement for r in records}),
}
torch.save(checkpoint, 'preliminary_fall_model.pt')
print('Saved preliminary_fall_model.pt')

def predict_window(raw_window):
    # Input: 200x6 raw values at 50 Hz, accel g, gyro deg/s.
    x = np.asarray(raw_window, dtype=np.float32)
    if x.shape != (WINDOW_LEN, 6) or not np.isfinite(x).all():
        raise ValueError(f'Expected finite {(WINDOW_LEN, 6)} array')
    z = ((preprocess_window(x) - norm_mean) / norm_std).T.copy()
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
