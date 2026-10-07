"""
Batch-1 inference latency with torch.compile
Author: Aryan Kulkarni

Eager PyTorch pays a dispatch cost per tensor operation. torch.compile traces each model and
fuses its operations into fewer kernels, so this measures how much of the operator-count effect
survives compilation. Same protocol as latency_sweep.py: single thread, 10 repeats in shuffled
order, 1,000 timed calls each, median of the per-repeat medians with a 95% interval.
"""

import json
import random

import numpy as np
import torch

from hybrid_study import MODELS
from latency_sweep import CALLS, REPEATS, SEED, WARMUP, time_once


def main():
    torch.set_num_threads(1)
    x = torch.rand(1, 1)
    compiled = {}
    for name, cls in MODELS.items():
        torch.manual_seed(SEED)
        model = torch.compile(cls().eval())
        with torch.no_grad():
            for _ in range(WARMUP):  # first calls trigger compilation
                model(x)
        compiled[name] = model

    rng = random.Random(SEED)
    medians = {name: [] for name in MODELS}
    for rep in range(REPEATS):
        order = list(MODELS)
        rng.shuffle(order)
        for name in order:
            medians[name].append(float(np.median(time_once(compiled[name], x))))
        print(f"repeat {rep + 1}/{REPEATS} done", flush=True)

    results = {"meta": {"torch": torch.__version__, "calls": CALLS, "repeats": REPEATS}, "models": {}}
    for name, meds in medians.items():
        results["models"][name] = {"median_us": float(np.median(meds)),
                                   "ci_lo_us": float(np.percentile(meds, 2.5)),
                                   "ci_hi_us": float(np.percentile(meds, 97.5)),
                                   "repeat_medians_us": meds}
        print(f"{name:28s} compiled b1={results['models'][name]['median_us']:.1f}us", flush=True)
    with open("compile_latency_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
