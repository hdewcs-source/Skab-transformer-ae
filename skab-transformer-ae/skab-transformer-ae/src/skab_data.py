"""Data loading and windowing for the Skoltech Anomaly Benchmark (SKAB).

Protocol
--------
* Training data: the anomaly-free recording, split chronologically into a
  training part (first 80 percent of rows) and a validation part (last 20 percent).
  The validation part contains only normal data and is used only to calibrate
  the decision threshold, never the test labels.
* Test data: every recording in the valve1 folder. Each recording is scaled and
  windowed on its own, so no window crosses a recording boundary.
* Scaling: Min-Max scaler fitted on the training part only.
* Window label: the label of the last time step in the window ("last") or an
  anomaly flag if any time step in the window is anomalous ("any").
* end_index: row index (within its recording) of the last time step of each test
  window. Used for per-recording calibration on the first rows of each recording.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


def read_recording(path):
    return pd.read_csv(path, sep=";")


def _natural_key(path):
    stem = Path(path).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def make_windows(values, labels, window, label_mode="last"):
    """Return windows of shape (n, window, d) and one label per window."""
    if len(values) < window:
        raise ValueError("Recording is shorter than the window size")
    n = len(values) - window + 1
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    x = values[idx]
    window_labels = labels[idx]
    if label_mode == "last":
        y = window_labels[:, -1]
    elif label_mode == "any":
        y = window_labels.max(axis=1)
    else:
        raise ValueError("label_mode must be 'last' or 'any'")
    return x, y.astype(int)


def load_skab(data_root, window=30, label_mode="last", val_fraction=0.2):
    """Load SKAB from a local clone of https://github.com/waico/SKAB.

    data_root is the folder that contains anomaly-free/ and valve1/.
    """
    root = Path(data_root)
    normal = read_recording(root / "anomaly-free" / "anomaly-free.csv")
    normal_values = normal[FEATURES].values

    split = int(len(normal_values) * (1.0 - val_fraction))
    raw_train, raw_val = normal_values[:split], normal_values[split:]

    scaler = MinMaxScaler().fit(raw_train)
    zeros = lambda a: np.zeros(len(a), dtype=int)
    x_train, _ = make_windows(scaler.transform(raw_train), zeros(raw_train), window)
    x_val, _ = make_windows(scaler.transform(raw_val), zeros(raw_val), window)

    xs, ys, rec_ids, end_idx = [], [], [], []
    files = sorted((root / "valve1").glob("*.csv"), key=_natural_key)
    for k, f in enumerate(files):
        df = read_recording(f)
        values = scaler.transform(df[FEATURES].values)
        x, y = make_windows(values, df["anomaly"].values.astype(int), window, label_mode)
        xs.append(x)
        ys.append(y)
        rec_ids.append(np.full(len(y), k))
        end_idx.append(np.arange(window - 1, len(df)))

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": np.concatenate(xs),
        "y_test": np.concatenate(ys),
        "recording_id": np.concatenate(rec_ids),
        "end_index": np.concatenate(end_idx),
        "n_recordings": len(files),
    }
