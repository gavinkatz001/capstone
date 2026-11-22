"""Standalone source adapters embedded into preliminary_fall_training.ipynb."""

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample_poly

CHANNELS = ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
FALL_TYPE = {'fall1': 'forward', 'fall2': 'backward', 'fall3': 'lateral',
             'fall4': 'forward', 'fall5': 'backward', 'fall6': 'lateral'}


@dataclass
class Record:
    source: str
    subject_id: str
    recording_id: str
    placement: str
    data: np.ndarray
    label: int
    activity: str
    fall_type: str | None = None
    split_policy: str = 'subject'


def harmonize_six_axis(values, source_rate_hz, accel_unit='g', gyro_unit='deg/s'):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 6 or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite Nx6 signal with at least 20 rows, got {x.shape}')
    if not np.isfinite(source_rate_hz) or source_rate_hz <= 0:
        raise ValueError('Invalid source sampling rate')
    x = x.copy()
    if accel_unit == 'm/s2':
        x[:, :3] /= 9.80665
    elif accel_unit == 'mg':
        x[:, :3] /= 1000
    elif accel_unit != 'g':
        raise ValueError(f'Unknown acceleration unit: {accel_unit}')
    if gyro_unit == 'rad/s':
        x[:, 3:] *= 180 / np.pi
    elif gyro_unit != 'deg/s':
        raise ValueError(f'Unknown gyroscope unit: {gyro_unit}')
    if abs(source_rate_hz - RATE_HZ) > 0.01:
        ratio = Fraction(float(RATE_HZ / source_rate_hz)).limit_denominator(1000)
        x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
    return np.asarray(x, dtype=np.float32)


def from_timestamps(times, values, accel_unit, gyro_unit, max_gap=0.5):
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    if len(t) != len(x) or len(t) < 20 or not np.isfinite(t).all():
        raise ValueError('Invalid timestamps')
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    if len(t) < 20 or np.any(np.diff(t) > max_gap):
        raise ValueError('Timestamps have gaps or too few samples')
    grid = np.arange(t[0], t[-1], 1 / RATE_HZ)
    if len(grid) < WINDOW_LEN:
        raise ValueError('Recording shorter than one window')
    y = np.column_stack([np.interp(grid, t, x[:, j]) for j in range(6)])
    return harmonize_six_axis(y, RATE_HZ, accel_unit, gyro_unit)


def load_nhoyh(root):
    records = []
    for path in sorted(Path(root).glob('subject_*/*/*.mat')):
        if path.parent.name not in ('fall', 'non-fall'):
            continue
        mat = scipy.io.loadmat(path, variable_names=list(CHANNELS[:3]) +
                               ['droll', 'dpitch', 'dyaw'])
        values = np.column_stack([np.ravel(mat[key]) for key in
                                  ('ax', 'ay', 'az', 'droll', 'dpitch', 'dyaw')])
        fall = path.parent.name == 'fall'
        records.append(Record('nhoyh', f'nhoyh:{path.parent.parent.name}',
                              f'nhoyh:{path.relative_to(root)}', 'wrist',
                              harmonize_six_axis(values, 50), int(fall), path.stem,
                              FALL_TYPE.get(path.stem) if fall else None))
    return records


def load_jasonkau(root):
    records = []
    for path in sorted(Path(root).rglob('*.xlsx')):
        if not any(s in path.parts for s in ('Fall Events', 'Daily Activities')):
            continue
        table = pd.read_excel(path, header=None).iloc[:, :12].apply(pd.to_numeric, errors='coerce')
        if table.shape[1] < 8 or table.iloc[:, 1:8].isna().any().any():
            raise ValueError(f'Unexpected JasonKau sheet: {path}')
        # Raw magnitude (column 1) tracks the norm of columns 2:5 and is ~1000 at rest.
        magnitude = np.linalg.norm(table.iloc[:, 2:5].to_numpy(), axis=1)
        reference = table.iloc[:, 1].to_numpy()
        if not np.allclose(np.median(magnitude), np.median(reference), rtol=0.2):
            raise ValueError(f'JasonKau magnitude disagrees with axes: {path}')
        values = table.iloc[:, [2, 3, 4, 5, 6, 7]].to_numpy()
        fall = 'Fall Events' in path.parts
        records.append(Record('jasonkau', 'jasonkau:unknown',
                              f'jasonkau:{path.relative_to(root)}', 'unknown',
                              harmonize_six_axis(values, 100, 'mg', 'deg/s'),
                              int(fall), path.stem, split_policy='train_only'))
    return records


def load_sisfall(root):
    records = []
    for path in sorted(Path(root).rglob('*.txt')):
        match = re.fullmatch(r'([FD]\d\d)_(S[AE]\d\d)_R\d\d', path.stem, re.I)
        if not match:
            continue
        raw = pd.read_csv(path, sep=r'[,;]', header=None, engine='python').iloc[:, :9]
        raw = raw.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        if raw.shape[1] != 9 or raw.isna().any().any():
            raise ValueError(f'Unexpected SisFall file: {path}')
        # ADXL345: 256 LSB/g at +/-16g; ITG3200: 14.375 LSB/(deg/s).
        values = np.column_stack([raw.iloc[:, :3].to_numpy() / 256,
                                  raw.iloc[:, 6:9].to_numpy() / 14.375])
        fall = match.group(1).upper().startswith('F')
        records.append(Record('sisfall', f'sisfall:{match.group(2).upper()}',
                              f'sisfall:{path.relative_to(root)}', 'waist',
                              harmonize_six_axis(values, 200), int(fall),
                              match.group(1).upper()))
    return records


def load_upfall(root, placement='wrist'):
    # Published synchronized CSV: timestamp followed by 7 channels per IMU.
    starts = {'ankle': 1, 'pocket': 8, 'waist': 15, 'neck': 22, 'wrist': 29}
    start = starts[placement]
    records = []
    for path in sorted(Path(root).rglob('*.csv')):
        match = re.search(r'S(?:ubject)?(\d+)A(?:ctivity)?(\d+)T(?:rial)?(\d+)',
                          str(path).replace(' ', ''), re.I)
        if not match:
            continue
        table = pd.read_csv(path, header=None, low_memory=False)
        # A header row is accepted; the remaining synchronized dataset has 43 columns.
        if pd.to_numeric(table.iloc[0, 0], errors='coerce') != pd.to_numeric(table.iloc[0, 0], errors='coerce'):
            table = table.iloc[1:]
        if table.shape[1] < 43:
            raise ValueError(f'UP-Fall expected at least 43 columns: {path}')
        times = pd.to_numeric(table.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
        values = table.iloc[:, [start, start+1, start+2, start+3, start+4, start+5]]
        values = values.apply(pd.to_numeric, errors='coerce').to_numpy()
        if not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError(f'Non-numeric UP-Fall IMU values: {path}')
        # Publisher timestamp is milliseconds; synchronized CSV is ~18 Hz.
        times = (times - times[0]) / 1000
        if np.median(np.diff(times)) < 0.001:
            raise ValueError(f'UP-Fall timestamp unit differs from expected ms: {path}')
        activity = int(match.group(2))
        if activity not in range(1, 12):
            continue
        records.append(Record('upfall', f'upfall:S{int(match.group(1)):02d}',
                              f'upfall:{path.relative_to(root)}', placement,
                              from_timestamps(times, values, 'g', 'deg/s'),
                              int(activity <= 5), f'A{activity:02d}'))
    return records


def load_univpm(root):
    records = []
    for path in sorted(Path(root).rglob('*.txt')):
        category = next((p.lower() for p in path.parts if p.lower() in ('fall', 'adl')), None)
        match = re.fullmatch(r'S(\d+)_(\d+)', path.stem, re.I)
        if category is None or match is None:
            continue
        table = pd.read_csv(path, sep=';', engine='python', header=None)
        numeric = table.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        if numeric.shape[1] < 7 or numeric.iloc[:, :7].isna().any().any():
            raise ValueError(f'Unexpected UNIVPM file: {path}')
        values = numeric.iloc[:, 1:7].to_numpy()
        times = numeric.iloc[:, 0].to_numpy()
        records.append(Record('univpm', f'univpm:S{int(match.group(1)):02d}',
                              f'univpm:{path.relative_to(root)}', 'waist',
                              from_timestamps(times, values, 'g', 'deg/s'),
                              int(category == 'fall'), path.parent.parent.name))
    return records


def load_smartfallmm(root, device='watch', accel_unit='m/s2', gyro_unit='rad/s'):
    if accel_unit is None or gyro_unit is None:
        raise ValueError('Set SmartFallMM device-specific accel and gyro units after checking source files')
    records = []
    paths = (p for p in Path(root).rglob('*.csv')
             if p.parent.name == device and p.parent.parent.name == 'accelerometer'
             and p.parent.parent.parent.name in ('young', 'old'))
    for path in sorted(paths):
        match = re.fullmatch(r'S(\d+)A(\d+)T(\d+)', path.stem, re.I)
        if not match:
            continue
        gyro_path = Path(str(path).replace('/accelerometer/', '/gyroscope/').replace('\\accelerometer\\', '\\gyroscope\\'))
        if not gyro_path.exists():
            continue
        acc = pd.read_csv(path, header=None)
        gyro = pd.read_csv(gyro_path, header=None)
        if acc.shape[1] not in (4, 6) or gyro.shape[1] not in (4, 6):
            raise ValueError(f'Unexpected SmartFallMM columns: {path}')
        def sensor_time(table):
            if table.shape[1] == 6:
                t = pd.to_numeric(table.iloc[:, 0], errors='coerce').to_numpy() / 1000
            else:
                parsed = pd.to_datetime(table.iloc[:, 0], errors='coerce')
                if parsed.isna().any():
                    raise ValueError(f'Invalid SmartFallMM timestamp: {path}')
                t = parsed.astype('datetime64[ns]').astype('int64').to_numpy() / 1e9
            v = table.iloc[:, -3:].apply(pd.to_numeric, errors='coerce').to_numpy()
            if not np.isfinite(t).all() or not np.isfinite(v).all():
                raise ValueError(f'Invalid SmartFallMM signal: {path}')
            return t, v
        ta, a = sensor_time(acc)
        tg, g = sensor_time(gyro)
        def segments(t):
            boundaries = np.r_[0, np.flatnonzero(np.diff(t) > 0.5) + 1, len(t)]
            return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
        candidates = []
        for a0, a1 in segments(ta):
            for g0, g1 in segments(tg):
                start, end = max(ta[a0], tg[g0]), min(ta[a1-1], tg[g1-1])
                if end - start >= WINDOW_SEC:
                    candidates.append((end-start, start, end, a0, a1, g0, g1))
        if not candidates:
            continue
        _, start, end, a0, a1, g0, g1 = max(candidates)
        ta, a = ta[a0:a1], a[a0:a1]
        tg, g = tg[g0:g1], g[g0:g1]
        grid = np.arange(start, end, 1 / RATE_HZ)
        values = np.column_stack([np.interp(grid, ta, a[:, j]) for j in range(3)] +
                                 [np.interp(grid, tg, g[:, j]) for j in range(3)])
        activity = int(match.group(2))
        if activity not in range(1, 15):
            continue
        records.append(Record('smartfallmm', f'smartfallmm:S{int(match.group(1)):02d}',
                              f'smartfallmm:{path.relative_to(root)}', device,
                              harmonize_six_axis(values, RATE_HZ, accel_unit, gyro_unit),
                              int(activity >= 10), f'A{activity:02d}'))
    return records


def load_fallalld_manifest(root):
    # IEEE archive structure is access gated. User-supplied manifest is explicit,
    # auditable mapping from verified raw CSVs to the common six-axis contract.
    manifest = Path(root) / 'fallalld_manifest.csv'
    if not manifest.exists():
        return []
    rows = pd.read_csv(manifest)
    required = {'file', 'subject_id', 'placement', 'label', 'rate_hz',
                'accel_unit', 'gyro_unit', *CHANNELS}
    if not required.issubset(rows.columns):
        raise ValueError(f'FallAllD manifest missing {sorted(required - set(rows.columns))}')
    records = []
    for row in rows.itertuples(index=False):
        path = Path(root) / row.file
        table = pd.read_csv(path)
        columns = [getattr(row, c) for c in CHANNELS]
        if not set(columns).issubset(table.columns):
            raise ValueError(f'FallAllD columns missing in {path}')
        label = str(row.label).lower()
        if label not in ('0', '1', '0.0', '1.0'):
            raise ValueError(f'FallAllD label must be 0 or 1: {path}')
        records.append(Record('fallalld', f'fallalld:{row.subject_id}',
                              f'fallalld:{row.file}', str(row.placement),
                              harmonize_six_axis(table[columns].to_numpy(), row.rate_hz,
                                                 row.accel_unit, row.gyro_unit),
                              int(float(label)), str(getattr(row, 'activity', 'unknown'))))
    return records
