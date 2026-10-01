# -*- coding: utf-8 -*-
"""UCI HAR data: download, load, subject-wise split, scale.

The subject-wise split is the load-bearing part. Windows overlap by 50%, so a
random split puts near-duplicate windows on both sides of it, and a model can
learn a person rather than an activity. Subjects 1-20 train, 21-25 validate,
26-30 test, and nothing crosses.
"""
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")

SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]

SEQ_LEN, N_CHANNELS, N_CLASSES = 128, 9, 6

TRAIN_SUBJECTS = range(1, 21)
VAL_SUBJECTS = range(21, 26)
TEST_SUBJECTS = range(26, 31)


def download(root="data"):
    """Fetch and unpack. The archive contains a nested zip, which is the one
    thing about this download that surprises people."""
    root = Path(root)
    base = root / "UCI HAR Dataset"
    if (base / "train" / "Inertial Signals").exists():
        return base
    root.mkdir(parents=True, exist_ok=True)

    import requests
    outer = root / "har.zip"
    if not outer.exists():
        print("downloading UCI HAR ...")
        r = requests.get(URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(outer, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)

    with zipfile.ZipFile(outer) as z:
        z.extractall(root)
    nested = root / "UCI HAR Dataset.zip"
    if nested.exists():
        with zipfile.ZipFile(nested) as z:
            z.extractall(root)

    # the nested archive sometimes unpacks one level deeper
    if not (base / "train").exists():
        inner = base / "UCI HAR Dataset"
        if (inner / "train").exists():
            base = inner
    if not (base / "train" / "Inertial Signals").exists():
        raise RuntimeError("extraction layout unexpected under %s" % base)
    return base


def _split_arrays(base, split):
    xs = [pd.read_csv(base / split / "Inertial Signals" / f"{s}_{split}.txt",
                      sep=r"\s+", header=None).values.astype(np.float32)
          for s in SIGNALS]
    X = np.stack(xs, axis=-1)                       # (N, 128, 9)
    y = pd.read_csv(base / split / f"y_{split}.txt",
                    header=None).values.ravel().astype(np.int64) - 1   # 0-5
    subj = pd.read_csv(base / split / f"subject_{split}.txt",
                       header=None).values.ravel().astype(np.int64)
    return X, y, subj


def load_all(root="data"):
    """The archive's own train/test division is by subject too, but with a
    different membership. We ignore it, pool everything, and re-split by our
    own subject lists so train, val and test are defined in one place."""
    base = download(root)
    Xa, ya, sa = _split_arrays(base, "train")
    Xb, yb, sb = _split_arrays(base, "test")
    return (np.concatenate([Xa, Xb]), np.concatenate([ya, yb]),
            np.concatenate([sa, sb]))


def make_splits(root="data", batch_size=64, seed=0, verbose=True):
    """-> dict of DataLoaders plus the raw test tensors for measurement."""
    X, y, subj = load_all(root)

    masks = {k: np.isin(subj, list(v)) for k, v in
             (("train", TRAIN_SUBJECTS), ("val", VAL_SUBJECTS),
              ("test", TEST_SUBJECTS))}

    # the assertions are the point: a silent split error invalidates everything
    assert masks["train"].sum() + masks["val"].sum() + masks["test"].sum() == len(y), \
        "splits do not cover every window"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (masks[a] & masks[b]).any(), "subject overlap between %s and %s" % (a, b)

    scaler = StandardScaler().fit(X[masks["train"]].reshape(-1, N_CHANNELS))

    def prep(k):
        Xs = scaler.transform(X[masks[k]].reshape(-1, N_CHANNELS)
                              ).reshape(-1, SEQ_LEN, N_CHANNELS)
        return torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(y[masks[k]])

    parts = {k: prep(k) for k in masks}

    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(TensorDataset(*parts["train"]), batch_size=batch_size,
                            shuffle=True, generator=g),
        "val": DataLoader(TensorDataset(*parts["val"]), batch_size=batch_size),
        "test": DataLoader(TensorDataset(*parts["test"]), batch_size=batch_size),
    }

    if verbose:
        for k in ("train", "val", "test"):
            print("%-5s %5d windows  subjects %s"
                  % (k, masks[k].sum(),
                     sorted(int(s) for s in set(subj[masks[k]]))))
        tr = parts["train"][0]
        print("train mean %.4f std %.4f  (should be ~0 and ~1)"
              % (tr.mean(), tr.std()))
        te = parts["test"][0]
        print("test  mean %.4f          (near 0, NOT exactly 0 -- exactly 0 means leakage)"
              % te.mean())

    return loaders, parts
