"""
Long-training check on the chained task
Author: Aryan Kulkarni

At 100 epochs the single-paradigm models are still improving on the chained task, so their gap
to the hybrids could be undertraining rather than a lower ceiling. This trains Sequential,
Parallel, H-A and H-B for 300 epochs with the main study's protocol (same split, seeds,
optimizer, standardized targets) and records test MSE every epoch in original units.
"""

import json

import numpy as np
import torch

import hybrid_study as hs

#ADJUSTABLE VALUES
#----------------------------------
EPOCHS = 300
MODELS = ["Sequential", "Parallel", "H-A Parallel of Sequentials", "H-B Staged Modular Chain"]
#-----------------------------------


def plateau_epoch(history, tol=0.01):
    """First epoch after which test MSE never improves by more than tol (relative)."""
    best_after = np.minimum.accumulate(np.array(history)[::-1])[::-1]
    for i, h in enumerate(history):
        if best_after[i] >= h * (1 - tol):
            return i + 1
    return len(history)


def main():
    torch.set_num_threads(1)
    X, envs = hs.make_environments()
    split = np.random.default_rng(hs.DATA_SEED).permutation(hs.DATA)
    tr, te = split[: int(0.8 * hs.DATA)], split[int(0.8 * hs.DATA):]
    x_tr, x_te = (torch.tensor(X[i], dtype=torch.float32) for i in (tr, te))
    y_tr, y_te = (torch.tensor(envs["Chain"][i], dtype=torch.float32) for i in (tr, te))
    y_tr, y_te, y_sd = hs.standardize(y_tr, y_te)

    results = {}
    for name in MODELS:
        runs = [hs.run(hs.MODELS[name], x_tr, y_tr, x_te, y_te, s, y_sd, epochs=EPOCHS) for s in hs.SEEDS]
        hist = np.array([r["history"] for r in runs])
        summary = {
            "mse_100": (float(hist[:, 99].mean()), float(hist[:, 99].std())),
            "mse_300": (float(hist[:, -1].mean()), float(hist[:, -1].std())),
            "best": (float(hist.min(axis=1).mean()), float(hist.min(axis=1).std())),
            "plateau_epoch": (float(np.mean([plateau_epoch(h) for h in hist])),
                              float(np.std([plateau_epoch(h) for h in hist]))),
            "history_mean": hist.mean(axis=0).tolist(),
        }
        results[name] = summary
        print(f"{name:28s} mse@100={summary['mse_100'][0]:.3f} mse@300={summary['mse_300'][0]:.3f}"
              f"±{summary['mse_300'][1]:.3f} plateau={summary['plateau_epoch'][0]:.0f}", flush=True)

    with open("long_training_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
