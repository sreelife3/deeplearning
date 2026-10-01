# -*- coding: utf-8 -*-
"""Size, parameters, latency and INT8 quantisation.

Latency is the part the course notebook gets wrong for this question. It times
batches of 64 through the test loader and divides by the sample count, which
answers "how fast can I clear a backlog" -- a server question. A wearable
classifies one window at a time, and batching amortises per-call overhead across
64 samples, so that number is far better than the device will ever see.

Everything here measures the deployment case: batch size 1, one thread, warm-up
discarded, median and p95 rather than mean.
"""
import os
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn

# What we ASK torch to quantise. Conv1d is deliberately absent -- dynamic
# quantisation does not cover it, which is why the CNN barely shrinks: a finding
# to report, not a bug to hide. nn.RNN is present and is also not covered, but
# torch accepts the request silently, so quantise() checks the result instead of
# trusting the request.
QUANTISABLE = {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def size_kb(model):
    """Bytes on disk. This is the number that matters on a device, and the one
    quantisation actually moves."""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path) / 1024.0
    finally:
        os.unlink(path)


def latency_ms(model, sample, warmup=20, reps=200, threads=1):
    """Median and p95 milliseconds for ONE window.

    sample: a tensor shaped (1, 128, 9).
    """
    assert sample.shape[0] == 1, "latency is measured at batch size 1"
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(sample)
            ts = np.empty(reps)
            for i in range(reps):
                t0 = time.perf_counter()
                model(sample)
                ts[i] = (time.perf_counter() - t0) * 1000.0
    finally:
        torch.set_num_threads(prev)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def quantise(model):
    """INT8 dynamic post-training quantisation.

    Weights are quantised ahead of time, activations on the fly per forward
    pass, so no calibration data is needed.

    Returns (quantised model, converted, skipped). The last two are read off the
    RESULT, not off what we asked for -- asking for a layer type is not the same
    as getting it. Plain nn.RNN is the case in point: it is in the request set,
    torch accepts the request without complaint, and the module comes back
    untouched. Reporting the request would have claimed a 4x saving that the
    file size flatly contradicts.
    """
    model = model.eval()
    q = torch.quantization.quantize_dynamic(model, QUANTISABLE, dtype=torch.qint8)

    # Compare the two models module by module, by NAME. Scanning types instead
    # is tempting and wrong: the quantised LSTM is also called "LSTM", and the
    # conversion adds internal modules of its own that are not layers at all.
    after = dict(q.named_modules())
    converted, skipped = set(), set()
    for name, m in model.named_modules():
        if type(m) not in QUANTISABLE:
            continue
        new = after.get(name)
        if new is not None and "quantized" in type(new).__module__:
            converted.add(type(m).__name__)
        else:
            skipped.add(type(m).__name__)
    return q, sorted(converted), sorted(skipped)


def profile(model, sample, quantised=False, params=None, **kw):
    """One row of the results table.

    Pass `params` for a quantised model. Counting its parameters directly gives
    zero: quantisation packs the weights into buffers that are no longer
    nn.Parameter, so the count is an artefact of the representation, not a
    property of the network. The parameter count belongs to the architecture and
    does not change when the weights are made narrower -- what changes is the
    size on disk, which is measured separately.
    """
    med, p95 = latency_ms(model, sample, **kw)
    return {
        "precision": "int8" if quantised else "fp32",
        "params": n_params(model) if params is None else params,
        "size_kb": round(size_kb(model), 1),
        "latency_med_ms": round(med, 3),
        "latency_p95_ms": round(p95, 3),
    }


def environment():
    """Record what the latency numbers mean. A timing without its conditions is
    not a measurement."""
    import platform
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "threads_used": 1,
    }
