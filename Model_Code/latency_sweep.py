"""
Inference latency sweep for the hybrid study models
Author: Aryan Kulkarni

hybrid_study.py times one batch size (the 1,000-point test set). This script times
every model across batch sizes and thread counts, so latency can be compared against
parameter count and multiply-accumulate (MAC) count. Weights do not affect latency,
so each model is timed at its seeded initialization.

To keep run order from biasing the comparison, every repeat times all (model, batch)
pairs in a freshly shuffled order. The reported median is the median of the per-repeat
medians, with a 95% interval across repeats.
"""

import json
import random
import time

import numpy as np
import torch
from torch import nn

from hybrid_study import MODELS

#ADJUSTABLE VALUES
#----------------------------------
BATCHES = [1, 8, 64, 512, 1000, 4096]
THREADS = [1, 4]
WARMUP = 50
CALLS = 1000
REPEATS = 10
SEED = 0
#-----------------------------------


def count_macs_and_ops(model, x):
    """MACs from every nn.Linear, plus the number of tensor ops on the forward path."""
    macs = 0
    ops = 0

    def linear_hook(module, inp, out):
        nonlocal macs
        macs += module.in_features * module.out_features

    hooks = [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]

    class OpCounter(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            nonlocal ops
            ops += 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), OpCounter():
        model(x)
    for h in hooks:
        h.remove()
    return macs, ops


def count_kernels(model, x):
    """ATen operators actually executed (below the Python layer), via torch.profiler."""
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        model(x)
    return sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))


def time_once(model, x):
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        timings = []
        for _ in range(CALLS):
            t0 = time.perf_counter()
            model(x)
            timings.append(time.perf_counter() - t0)
    return 1e6 * np.array(timings)


def main():
    rng = random.Random(SEED)
    models, results = {}, {"meta": {"torch": torch.__version__, "batches": BATCHES, "threads": THREADS,
                                    "calls": CALLS, "repeats": REPEATS}, "models": {}}
    for name, cls in MODELS.items():
        torch.manual_seed(SEED)
        models[name] = cls().eval()
        macs, ops = count_macs_and_ops(models[name], torch.rand(1, 1))
        results["models"][name] = {"params": sum(p.numel() for p in models[name].parameters()),
                                   "macs_per_sample": macs, "ops_per_call": ops,
                                   "kernels_per_call": count_kernels(models[name], torch.rand(1, 1)),
                                   "timings": {}}
    inputs = {b: torch.rand(b, 1) for b in BATCHES}

    for threads in THREADS:
        torch.set_num_threads(threads)
        medians = {(m, b): [] for m in MODELS for b in BATCHES}
        pooled = {(m, b): [] for m in MODELS for b in BATCHES}
        for rep in range(REPEATS):
            order = list(medians)
            rng.shuffle(order)
            for m, b in order:
                us = time_once(models[m], inputs[b])
                medians[(m, b)].append(float(np.median(us)))
                pooled[(m, b)].append(us)
            print(f"threads={threads} repeat {rep + 1}/{REPEATS} done", flush=True)
        for (m, b), meds in medians.items():
            allus = np.concatenate(pooled[(m, b)])
            results["models"][m]["timings"][f"{threads}|{b}"] = {
                "median_us": float(np.median(meds)),
                "ci_lo_us": float(np.percentile(meds, 2.5)), "ci_hi_us": float(np.percentile(meds, 97.5)),
                "p10_us": float(np.percentile(allus, 10)), "p90_us": float(np.percentile(allus, 90)),
                "repeat_medians_us": meds}

    for m, e in results["models"].items():
        print(f"{m:28s} params={e['params']} macs={e['macs_per_sample']} ops={e['ops_per_call']} "
              f"kernels={e['kernels_per_call']} b1={e['timings']['1|1']['median_us']:.1f}us "
              f"b4096={e['timings']['1|4096']['median_us']:.0f}us", flush=True)
    with open("latency_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
