# -*- coding: utf-8 -*-
"""Regenerate the README's Results section from results/summary.csv.

Hand-copying numbers out of a CSV into a README is how a repository ends up
claiming something its own results file contradicts. This writes the table, the
findings and the caveats straight from the data, so the prose cannot drift away
from the numbers.

    python make_report.py            # rewrite README.md between the markers
    python make_report.py --print    # print the section, change nothing
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

import report as R

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

NAMES = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "cnn1d": "CNN1D"}


def _fmt_env(path="results/environment.json"):
    p = Path(path)
    if not p.exists():
        return ""
    e = json.loads(p.read_text())
    return ("All timings measured on %s, %s core(s), **%d thread**, torch %s, "
            "Python %s." % (e.get("processor") or e.get("machine", "unknown"),
                            e.get("cpu_count", "?"), e.get("threads_used", 1),
                            e.get("torch", "?"), e.get("python", "?")))


def _latency_lines(summary):
    """What INT8 did to LATENCY.

    The size column was already reported above and is the easy half. This is the
    half that inverts the usual expectation, so it is derived here rather than
    left for a reader to spot in the table: on a CPU with no well-optimised int8
    recurrent kernels, the conversion can cost more time than it saves space.
    """
    out, rows = [], []
    for m in summary.model.unique():
        f = summary[(summary.model == m) & (summary.precision == "fp32")]
        q = summary[(summary.model == m) & (summary.precision == "int8")]
        if f.empty or q.empty:
            continue
        f, q = f.iloc[0], q.iloc[0]
        rows.append((m,
                     float(f.latency_med_ms), float(q.latency_med_ms),
                     float(q.latency_med_ms) / max(float(f.latency_med_ms), 1e-9),
                     float(f.size_kb) / max(float(q.size_kb), 1e-9)))
    if not rows:
        return out

    worst = max(rows, key=lambda r: r[3])
    if worst[3] > 1.5:
        out.append("- **INT8 costs latency on this CPU, it does not save it.** "
                   "%s goes %.2f ms → %.2f ms per window, **%.1f× slower**, "
                   "while shrinking %.2f×. Compression and speed are separate "
                   "questions and this run separates them."
                   % (NAMES.get(worst[0], worst[0]), worst[1], worst[2],
                      worst[3], worst[4]))

    # The causal test: did the slowdown land on exactly the models that were
    # actually converted? If the slowest-converted model is still slower than
    # the slowest un-converted one, the conversion is what did it -- and that is
    # a stronger statement than "INT8 was slower here".
    shrank = [r for r in rows if r[4] >= 1.5]
    kept = [r for r in rows if r[4] < 1.5]
    if shrank and kept and min(r[3] for r in shrank) > max(r[3] for r in kept):
        out.append("- **The slowdown tracks the conversion, not the precision "
                   "flag.** The %s that actually shrank (%s) slowed by "
                   "%.1f–%.1f×; the %s that dynamic quantisation left alone "
                   "(%s) stayed within %.1f×. That is evidence the int8 "
                   "recurrent kernels are the cause, not measurement drift."
                   % ("models" if len(shrank) > 1 else "model",
                      ", ".join(NAMES.get(r[0], r[0]) for r in shrank),
                      min(r[3] for r in shrank), max(r[3] for r in shrank),
                      "models" if len(kept) > 1 else "model",
                      ", ".join(NAMES.get(r[0], r[0]) for r in kept),
                      max(r[3] for r in kept)))
    return out


def _latency_spread(raw):
    """Worst within-model fp32 latency spread across seeds, as a ratio.

    Three seeds of one architecture differ in their weights and in nothing that
    affects timing. So this number is a direct read on how much the MACHINE was
    interfering: tight means the latency column can be trusted, wide means it
    cannot, and either way it is measured rather than assumed.
    """
    if raw.empty or "latency_med_ms" not in raw:
        return None
    g = (raw[raw.precision == "fp32"]
         .groupby("model").latency_med_ms.agg(["min", "max"]))
    g["ratio"] = g["max"] / g["min"].clip(lower=1e-9)
    return g.sort_values("ratio", ascending=False)


def _inversions(summary):
    """Recurrent pairs where FEWER parameters measured SLOWER, by over 1.5x."""
    rec = summary[(summary.precision == "fp32")
                  & summary.model.isin(["rnn", "lstm", "gru"])].sort_values("params")
    out = []
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            a, b = rec.iloc[i], rec.iloc[j]          # a has FEWER params
            if a.latency_med_ms > b.latency_med_ms * 1.5:
                out.append((a, b))
    return out


def _pair(a, b):
    return ("%s vs %s, %s vs %s parameters, %.2f ms vs %.2f ms"
            % (NAMES.get(a.model, a.model), NAMES.get(b.model, b.model),
               format(int(a.params), ","), format(int(b.params), ","),
               a.latency_med_ms, b.latency_med_ms))


def _kernel_lines(summary, raw, tight=1.5):
    """The parameter-count inversion, stated as a FINDING -- but only once the
    measurement is tight enough to have earned it.

    The same observation is a finding or an open question depending entirely on
    how noisy the run was, so the threshold decides which section it lands in
    rather than the author deciding after seeing the number.
    """
    g = _latency_spread(raw)
    inv = _inversions(summary)
    if g is None or g.empty or not inv or g.ratio.max() >= tight:
        return []
    return ["- **Parameter count does not predict CPU latency; the kernel "
            "does.** %s. Across three seeds every model timed within "
            "**%.2f×** of itself, so this ordering is a reproducible property "
            "of the PyTorch CPU kernels on this machine, not load during the "
            "run. Sizing an edge budget from parameter counts would have "
            "picked the wrong model."
            % ("; ".join(_pair(a, b) for a, b in inv), g.ratio.max())]


def _anomalies(summary, raw, spread_factor=2.0):
    """Things this run cannot explain, stated rather than smoothed over.

    Every item here is derived from the data. A results section that only ever
    reports what it set out to measure is not a measurement, it is a press
    release -- and the first person to notice the odd number in the table should
    be the author, not the reader.
    """
    notes = []
    g = _latency_spread(raw)

    # 1. How noisy is the latency column, within one model, across seeds?
    if g is not None:
        for m, r in g[g.ratio >= spread_factor].iterrows():
            notes.append("- **The latency column is noisier than the accuracy "
                         "column.** %s measured %.2f-%.2f ms across three "
                         "seeds (**%.1f×**) on a model whose weights differ but "
                         "whose shape does not. Identical architectures should "
                         "time the same; re-measure on an otherwise idle "
                         "machine before quoting single latency figures."
                         % (NAMES.get(m, m), r["min"], r["max"], r["ratio"]))

    # 2. The parameter-count inversion, but ONLY while the run is too noisy to
    #    tell a kernel difference from background load. Once the spread is
    #    tight, _kernel_lines promotes it to a finding instead.
    if g is not None and not g.empty and g.ratio.max() >= spread_factor:
        for a, b in _inversions(summary):
            notes.append("- **%s is slower than %s despite having fewer "
                         "parameters** (%s). With a seed spread of up to %.1f× "
                         "this run cannot separate a kernel difference from "
                         "background load; `python run_all.py --resume` "
                         "re-measures from the saved checkpoints without "
                         "retraining and would settle it."
                         % (NAMES.get(a.model, a.model),
                            NAMES.get(b.model, b.model), _pair(a, b),
                            g.ratio.max()))
    return notes


def _findings(summary, raw):
    """Statements that follow from the numbers, not from expectation."""
    out = []
    fp = summary[summary.precision == "fp32"].sort_values("macro_f1_mean",
                                                          ascending=False)
    best = fp.iloc[0]
    out.append("- **Most accurate (fp32):** %s at macro-F1 **%.4f** "
               "(range %.4f across %d seeds)."
               % (NAMES.get(best.model, best.model), best.macro_f1_mean,
                  best.macro_f1_range, int(best.n_seeds)))

    fast = fp.sort_values("latency_med_ms").iloc[0]
    slow = fp.sort_values("latency_med_ms").iloc[-1]
    out.append("- **Fastest (fp32):** %s at **%.2f ms** per window, versus "
               "%.2f ms for %s — a **%.0f×** difference for a "
               "macro-F1 gap of %.4f."
               % (NAMES.get(fast.model, fast.model), fast.latency_med_ms,
                  slow.latency_med_ms, NAMES.get(slow.model, slow.model),
                  slow.latency_med_ms / max(fast.latency_med_ms, 1e-9),
                  abs(fast.macro_f1_mean - slow.macro_f1_mean)))

    pairs = R.indistinguishable(summary)
    if pairs:
        for a, b, gap, spread in pairs:
            out.append("- **%s and %s are indistinguishable here:** they differ "
                       "by %.4f, inside a seed spread of %.4f. Any ranking "
                       "between them is noise."
                       % (NAMES.get(a, a), NAMES.get(b, b), gap, spread))
    else:
        out.append("- Every pair of models differs by more than the seed "
                   "spread, so the ranking above is a real ordering rather "
                   "than noise.")

    # what quantisation did, per model, read off the result
    for m in summary.model.unique():
        f = summary[(summary.model == m) & (summary.precision == "fp32")]
        q = summary[(summary.model == m) & (summary.precision == "int8")]
        if f.empty or q.empty:
            continue
        f, q = f.iloc[0], q.iloc[0]
        ratio = f.size_kb / max(q.size_kb, 1e-9)
        dF1 = q.macro_f1_mean - f.macro_f1_mean
        conv = ""
        rows = raw[(raw.model == m) & (raw.precision == "int8")]
        if not rows.empty and "quantised_layers" in rows:
            got = str(rows.iloc[0].get("quantised_layers") or "")
            missed = str(rows.iloc[0].get("not_quantised_layers") or "")
            if missed and missed != "nan":
                conv = (" Dynamic quantisation converted %s but **not %s**, "
                        "which is why." % (got.replace(";", ", "),
                                           missed.replace(";", ", ")))
            elif ratio < 1.2:
                conv = (" Only %s was converted — `Conv1d` is not covered "
                        "by dynamic quantisation." % got.replace(";", ", "))
        out.append("- **%s under INT8:** %.1f KB → %.1f KB (**%.2f×**), "
                   "macro-F1 %+.4f.%s"
                   % (NAMES.get(m, m), f.size_kb, q.size_kb, ratio, dF1, conv))

    out += _latency_lines(summary)
    out += _kernel_lines(summary, raw)
    return "\n".join(out)


def build(summary_csv="results/summary.csv", raw_csv="results/raw.csv"):
    if not Path(summary_csv).exists():
        raise SystemExit("no %s yet — run `python run_all.py` first" % summary_csv)
    summary = pd.read_csv(summary_csv)
    raw = pd.read_csv(raw_csv) if Path(raw_csv).exists() else pd.DataFrame()

    parts = [R.table(summary), ""]
    env = _fmt_env()
    if env:
        parts += [env, ""]
    parts += ["### What the numbers say", "", _findings(summary, raw), ""]

    notes = _anomalies(summary, raw)
    if notes:
        parts += ["### What this run cannot explain", "",
                  "\n".join(notes), ""]

    figs = []
    if Path("figures/pareto.png").exists():
        figs.append("![Accuracy against latency](figures/pareto.png)\n\n"
                    "*Filled markers are FP32, hollow are INT8; marker size is "
                    "the model's size on disk. The dashed line is the Pareto "
                    "frontier — the points nothing else beats on both axes at "
                    "once.*")
    if Path("figures/confusion.png").exists():
        figs.append("![Confusion matrix](figures/confusion.png)\n\n"
                    "*Held-out subjects. The static postures are where the "
                    "errors concentrate: SITTING and STANDING differ by "
                    "orientation, not motion, which is the hardest distinction "
                    "for an accelerometer to make.*")
    if figs:
        parts += ["### Figures", ""] + ["\n\n".join(figs), ""]
    return "\n".join(parts).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the section instead of rewriting README.md")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()

    section = build()
    if a.show:
        print(section)
        return 0

    p = Path(a.readme)
    text = p.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("markers %s / %s not found in %s" % (START, END, a.readme))
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    p.write_text(head + START + "\n" + section + END + tail, encoding="utf-8")
    print("updated %s between the markers (%d chars)" % (a.readme, len(section)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
