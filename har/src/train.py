# -*- coding: utf-8 -*-
"""One training run: (model, seed) -> checkpoint plus metrics.

Model selection happens on the VALIDATION subjects. The test subjects are
touched exactly once, at the end. Pick the best epoch by test score and the test
set has quietly become a validation set, and the headline number is optimistic.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from models import build, n_params

EPOCHS, LR, MAX_NORM = 50, 1e-3, 1.0
PATIENCE = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return {"macro_f1": f1_score(t, p, average="macro"),
            "accuracy": accuracy_score(t, p)}, p, t


def _hms(s):
    """Seconds as 1m04s / 42.3s -- readable at a glance in a long run."""
    return "%dm%02ds" % (int(s) // 60, int(s) % 60) if s >= 60 else "%.1fs" % s


def train_one(name, seed, loaders, device="cpu", epochs=EPOCHS, lr=LR,
              out_dir="checkpoints", use_mlflow=True, verbose=True,
              log_every=1):
    set_seed(seed)
    model = build(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(out_dir) / f"{name}_seed{seed}.pt"

    best_f1, best_epoch, stale, history = -1.0, -1, 0, []
    import time
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        total = 0.0
        for xb, yb in loaders["train"]:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            # clipping saves you from EXPLODING gradients. It does nothing for
            # vanishing ones -- that is what the gates are for.
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            total += loss.item() * xb.size(0)

        train_loss = total / len(loaders["train"].dataset)
        val, _, _ = evaluate(model, loaders["val"], device)
        history.append({"epoch": ep, "train_loss": train_loss, **val})

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = val["macro_f1"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1

        if verbose and (log_every and (ep % log_every == 0 or ep == 1)):
            print("  ep%02d/%d  loss %.4f  val F1 %.4f  %-13s %s"
                  % (ep, epochs, train_loss, val["macro_f1"],
                     "*best" if stale == 0 else "no gain %d/%d" % (stale, PATIENCE),
                     _hms(time.perf_counter() - t_ep)), flush=True)

        if stale >= PATIENCE:
            if verbose:
                print("  early stop at ep%02d -- no gain for %d epochs "
                      "(best ep%02d, val F1 %.4f)"
                      % (ep, PATIENCE, best_epoch, best_f1), flush=True)
            break

    train_time = time.perf_counter() - t_start
    if verbose:
        print("  trained %d epochs in %s (best ep%02d)"
              % (len(history), _hms(train_time), best_epoch), flush=True)

    # the single look at the test subjects
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test, preds, trues = evaluate(model, loaders["test"], device)

    row = {"model": name, "seed": seed, "best_epoch": best_epoch,
           "val_macro_f1": round(best_f1, 4),
           "macro_f1": round(test["macro_f1"], 4),
           "accuracy": round(test["accuracy"], 4),
           "params": n_params(model),
           "train_time_s": round(train_time, 1),
           "checkpoint": str(ckpt)}

    if use_mlflow:
        _log_mlflow(name, seed, row, history, lr, epochs)

    np.save(Path(out_dir) / f"{name}_seed{seed}_preds.npy",
            np.stack([preds, trues]))
    (Path(out_dir) / f"{name}_seed{seed}_history.json").write_text(
        json.dumps(history, indent=1))
    return row, model


def _log_mlflow(name, seed, row, history, lr, epochs):
    """Twelve runs across four models and three seeds is exactly what MLflow is
    for. Logging here is less work than assembling the comparison by hand."""
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=f"{name}_seed{seed}"):
        mlflow.log_params({"model": name, "seed": seed, "lr": lr,
                           "max_epochs": epochs, "hidden": 64, "layers": 2,
                           "dropout": 0.4, "params": row["params"]})
        for h in history:
            mlflow.log_metrics({"train_loss": h["train_loss"],
                                "val_macro_f1": h["macro_f1"]}, step=h["epoch"])
        mlflow.log_metrics({"test_macro_f1": row["macro_f1"],
                            "test_accuracy": row["accuracy"],
                            "best_epoch": row["best_epoch"],
                            "train_time_s": row["train_time_s"]})
