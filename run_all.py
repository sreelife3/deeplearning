# -*- coding: utf-8 -*-
"""The whole study: four models, three seeds, FP32 and INT8, one command.

    python run_all.py --selftest      shapes, latency and quantisation only,
                                      no dataset, ~30 seconds
    python run_all.py                 the full twelve runs
    python run_all.py --models gru --seeds 1     one run, for a quick check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import torch

import models as M
import measure as ME

MODELS = ["rnn", "lstm", "gru", "cnn1d"]
SEEDS = [1, 2, 3]


def selftest():
    """Everything that does not need the dataset. Run this first -- it catches
    the transpose bug, a broken quantisation set and a mis-shaped classifier in
    about thirty seconds, before you spend an hour training."""
    print("SELF-TEST  (no dataset required)\n" + "-" * 62)
    x = torch.randn(8, M.SEQ_LEN, M.N_CHANNELS)
    one = torch.randn(1, M.SEQ_LEN, M.N_CHANNELS)
    ok = True

    for name in MODELS:
        m = M.build(name).eval()
        with torch.no_grad():
            out = m(x)
        shape_ok = out.shape == (8, M.N_CLASSES)

        q, converted, skipped = ME.quantise(M.build(name).eval())
        with torch.no_grad():
            qout = q(x)
        fp32_kb, int8_kb = ME.size_kb(m), ME.size_kb(q)
        med, p95 = ME.latency_ms(m, one, warmup=5, reps=30)
        qmed, _ = ME.latency_ms(q, one, warmup=5, reps=30)

        print("%-6s out%-12s params %8s  fp32 %7.1f KB  int8 %7.1f KB  "
              "(%.2fx)  %.2f ms -> %.2f ms"
              % (name, str(tuple(out.shape)), f"{M.n_params(m):,}",
                 fp32_kb, int8_kb, fp32_kb / max(int8_kb, 1e-9), med, qmed))
        print("        converted: %-22s not covered: %s"
              % (", ".join(converted) or "nothing",
                 ", ".join(skipped) or "-"))
        if not shape_ok:
            print("        !! wrong output shape"); ok = False
        if qout.shape != out.shape:
            print("        !! quantised output shape differs"); ok = False

    # the transpose guard must actually fire
    try:
        M.build("cnn1d")(torch.randn(4, M.N_CHANNELS, M.SEQ_LEN))
        print("!! CNN1D accepted (batch, channels, time) -- the assert is not working")
        ok = False
    except AssertionError:
        print("\ntranspose guard fires correctly on (batch, channels, time)")

    print("-" * 62)
    print("SELF-TEST %s" % ("PASSED" if ok else "FAILED"))
    print("\nExpected results, not bugs -- and both belong in the write-up:")
    print("  * cnn1d barely shrinks: dynamic quantisation does not cover Conv1d.")
    print("  * rnn barely shrinks either: plain nn.RNN is not covered, even "
          "though\n    torch accepts the request for it without complaint. "
          "That is why quantise()\n    reports what was converted rather than "
          "what was asked for.")
    return 0 if ok else 1


def main(args):
    import data as D
    from train import train_one

    from train import _hms
    import time

    total_runs = len(args.models) * len(args.seeds)
    print("HAR study: %d models x %d seeds = %d runs"
          % (len(args.models), len(args.seeds), total_runs))
    print("torch %s | threads %d | max %d epochs | batch %d | patience 8"
          % (torch.__version__, torch.get_num_threads(), args.epochs, args.batch))
    print("-" * 64)

    loaders, parts = D.make_splits(root=args.data, batch_size=args.batch)
    one = parts["test"][0][:1]                      # a single window

    Path("results").mkdir(exist_ok=True)
    rows = []
    done = 0
    t_study = time.perf_counter()
    for name in args.models:
        for seed in args.seeds:
            done += 1
            print("\n[%2d/%d] %-6s seed %d %s"
                  % (done, total_runs, name, seed, "-" * 34), flush=True)

            reused = _finished(name, seed, args.epochs) if args.resume else None
            if reused:
                row, model = _reload(name, seed, loaders, reused)
                print("  reusing the finished run on disk (%d epochs, best ep%02d)"
                      " — re-measuring only" % (reused["epochs"],
                                                     row["best_epoch"]), flush=True)
            else:
                row, model = train_one(name, seed, loaders, epochs=args.epochs,
                                       use_mlflow=not args.no_mlflow,
                                       log_every=args.log_every)

            fp32 = ME.profile(model.eval(), one)
            rows.append({**row, **fp32})

            q, converted, skipped = ME.quantise(model)
            qmetrics, preds, trues = _eval_quantised(q, loaders["test"])
            int8 = ME.profile(q, one, quantised=True, params=row["params"])
            rows.append({**row, **int8, **qmetrics,
                         "quantised_layers": ";".join(converted),
                         "not_quantised_layers": ";".join(skipped)})

            print("  fp32  F1 %.4f  %5.1f KB  %.2f ms"
                  % (row["macro_f1"], fp32["size_kb"], fp32["latency_med_ms"]))
            print("  int8  F1 %.4f  %5.1f KB  %.2f ms"
                  % (qmetrics["macro_f1"], int8["size_kb"], int8["latency_med_ms"]))

            # Write after EVERY run, not at the end. The first version of this
            # script accumulated rows in memory and wrote the CSV once, so a
            # machine restart three-quarters of the way through a sweep threw
            # away every completed run. Checkpoints survived; the measurements
            # did not.
            pd.DataFrame(rows).to_csv("results/raw.csv", index=False)

            # a running clock, and an estimate that is honest about being one:
            # it assumes the remaining runs look like the ones already done,
            # which is wrong whenever the next model is a different shape.
            spent = time.perf_counter() - t_study
            print("  elapsed %s | %d of %d runs | rough estimate remaining %s"
                  % (_hms(spent), done, total_runs,
                     _hms(spent / done * (total_runs - done))), flush=True)

    Path("results/environment.json").write_text(
        json.dumps(ME.environment(), indent=1))

    import report as R
    summary = R.aggregate()
    print("\n" + R.table(summary))
    for a, b, gap, spread in R.indistinguishable(summary):
        print("\nNOTE: %s and %s differ by %.4f, inside a seed spread of %.4f."
              " Report them as indistinguishable." % (a, b, gap, spread))
    R.pareto_plot(summary)
    best = summary[summary.precision == "fp32"].sort_values(
        "macro_f1_mean", ascending=False).iloc[0]
    stats = R.confusion_plot(
        "checkpoints/%s_seed%d_preds.npy" % (best.model, args.seeds[0]),
        title="%s — confusion on held-out subjects" % best.model.upper())
    print("\nSITTING/STANDING confusions: %s and %s, %.0f%% of all errors"
          % (stats["sitting_as_standing"], stats["standing_as_sitting"],
             100 * stats["sit_stand_share_of_errors"]))
    print("\nwrote results/raw.csv, results/summary.csv, figures/")
    print("total wall clock %s" % _hms(time.perf_counter() - t_study))
    return 0


def _finished(name, seed, epochs, out_dir="checkpoints", patience=8):
    """Is there a GENUINELY finished run for (name, seed) on disk?

    The three files must all exist, and the history has to show a run that
    actually ran to a stopping condition under the CURRENT epoch budget: either
    it hit the cap, or it early-stopped (patience epochs with no improvement).

    That last test is the one that matters. A smoke run left behind by
    `--epochs 3` produces a perfectly valid checkpoint, history and predictions
    file. Treating it as complete would silently fold a three-epoch model into
    the study alongside models that trained for twenty-five, and every
    comparison in the results table would be quietly wrong.
    """
    p = Path(out_dir) / ("%s_seed%d" % (name, seed))
    pt, hist, preds = Path(str(p) + ".pt"), Path(str(p) + "_history.json"), \
        Path(str(p) + "_preds.npy")
    if not (pt.exists() and hist.exists() and preds.exists()):
        return None
    h = json.loads(hist.read_text())
    if not h:
        return None
    best = max(h, key=lambda r: r["macro_f1"])
    ran, stale = len(h), len(h) - best["epoch"]
    if ran >= epochs or stale >= patience:
        return {"epochs": ran, "best_epoch": best["epoch"],
                "val_macro_f1": round(best["macro_f1"], 4), "history": h}
    return None


def _reload(name, seed, loaders, info, out_dir="checkpoints"):
    """Rebuild a finished run's row from disk, then re-measure it.

    Training is the expensive part and it is already done. Evaluation,
    quantisation and the latency harness together take seconds, and re-running
    them means every number in the table was measured in one session on one
    machine rather than stitched across a reboot.
    """
    from train import evaluate
    ckpt = Path(out_dir) / ("%s_seed%d.pt" % (name, seed))
    model = M.build(name)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    test, _, _ = evaluate(model, loaders["test"])
    return {"model": name, "seed": seed, "best_epoch": info["best_epoch"],
            "val_macro_f1": info["val_macro_f1"],
            "macro_f1": round(test["macro_f1"], 4),
            "accuracy": round(test["accuracy"], 4),
            "params": M.n_params(model),
            "train_time_s": None, "checkpoint": str(ckpt)}, model


def _eval_quantised(q, loader):
    from sklearn.metrics import accuracy_score, f1_score
    q.eval()
    P, T = [], []
    with torch.no_grad():
        for xb, yb in loader:
            P.append(q(xb).argmax(1).numpy()); T.append(yb.numpy())
    p, t = np.concatenate(P), np.concatenate(T)
    return {"macro_f1": round(f1_score(t, p, average="macro"), 4),
            "accuracy": round(accuracy_score(t, p), 4)}, p, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished runs already on disk; retrain only "
                         "what is missing or was cut short")
    ap.add_argument("--log-every", type=int, default=1,
                    help="print every Nth epoch (0 silences per-epoch lines)")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else main(a))
