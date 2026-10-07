"""
Main study: benchmarking single vs. hybrid neural architectures (corrected protocol)
Author: Aryan Kulkarni

Fixes from Benchmark 1:
  - models train on the true x (scaled to [0, 1]) instead of unrelated random inputs
  - every model runs in one framework (PyTorch) with one training loop
  - 80/20 train/test split, metrics reported on held-out test data
  - 5 seeds per (model, task), reported as mean +/- std
  - targets are standardized with the training set's mean and std; every reported MSE is
    converted back to original units, so it still compares to the noise floors (1, 4, 0.25)

Every model is kept at 385 trainable parameters (H-C: 387, the closest its wiring allows),
the same budget as Benchmark 1.
"""

import json
import time

import numpy as np
import torch
from torch import nn

#ADJUSTABLE VALUES
#----------------------------------
DATA = 5000
DATA_SEED = 42
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 100
BATCH = 32
#-----------------------------------

torch.set_num_threads(1)  # single thread so timing compares architectures, not scheduling


def make_environments():
    """Same equations and noise as Benchmark 1, generated with the same seed."""
    np.random.seed(DATA_SEED)
    np.random.rand(DATA, 1)  # keep the RNG stream identical to the notebook
    X = np.linspace(0, 10, DATA).reshape(-1, 1)
    y_linear = 2 * X + np.random.normal(0, 1, (DATA, 1))
    y_curved = 0.5 * X**2 + np.sin(X) + np.random.normal(0, 2, (DATA, 1))
    stage_a = X * np.cos(0.5 * X)
    stage_b = np.sin(stage_a)
    stage_c = np.sign(np.sin(X)) * 0.5
    y_chain = stage_a + stage_b + stage_c + np.random.normal(0, 0.5, (DATA, 1))
    return X / 10.0, {"Linear": y_linear, "Curved": y_curved, "Chain": y_chain}


def mlp(n_in, hidden, n_out=1):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


#Single architectures (same wiring as Benchmark 1)
class Sequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(1, 128)

    def forward(self, x):
        return self.net(x)


class Parallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(1, 64)
        self.branch_b = nn.Linear(1, 64)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        merged = torch.cat([torch.relu(self.branch_a(x)), torch.relu(self.branch_b(x))], dim=1)
        return self.output(merged)


class Modular(nn.Module):
    """Modularnet from Benchmark 1: a self-contained hidden -> output block."""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


#Hybrid architectures
class ParallelOfSequentials(nn.Module):
    """H-A: two branches, each a 2-layer sequential stack (depth + breadth)."""
    def __init__(self, h=12):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.branch_a = branch()
        self.branch_b = branch()
        self.output = nn.Linear(2 * h, 1)

    def forward(self, x):
        return self.output(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1))


class StagedModularChain(nn.Module):
    """H-B: one module per stage of the chained equation, with skip links from x.
    A sees x; B sees (x, a); C sees (x, a, b); output = a + b + c."""
    def __init__(self, widths=(33, 32, 31)):  # widths chosen so the total is exactly 385
        super().__init__()
        self.module_a = mlp(1, widths[0])
        self.module_b = mlp(2, widths[1])
        self.module_c = mlp(3, widths[2])

    def forward(self, x):
        a = self.module_a(x)
        b = self.module_b(torch.cat([x, a], dim=1))
        c = self.module_c(torch.cat([x, a, b], dim=1))
        return a + b + c


class TrunkParallelHeads(nn.Module):
    """H-C: shared sequential trunk, then two parallel heads averaged."""
    def __init__(self, h1=20, h2=15):  # no two-head width pair gives exactly 385; 20/15 gives 387
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(1, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.head_1 = nn.Linear(h2, 1)
        self.head_2 = nn.Linear(h2, 1)

    def forward(self, x):
        z = self.trunk(x)
        return (self.head_1(z) + self.head_2(z)) / 2


MODELS = {
    "Sequential": Sequential,
    "Parallel": Parallel,
    "Modular": Modular,
    "H-A Parallel of Sequentials": ParallelOfSequentials,
    "H-B Staged Modular Chain": StagedModularChain,
    "H-C Trunk + Parallel Heads": TrunkParallelHeads,
}


def standardize(y_tr, y_te):
    """Scale targets with training-set statistics only; returns the scale to undo it."""
    mean, sd = y_tr.mean().item(), y_tr.std(unbiased=False).item()
    return (y_tr - mean) / sd, (y_te - mean) / sd, sd


def run(model_cls, x_tr, y_tr, x_te, y_te, seed, y_sd=1.0, epochs=EPOCHS):
    """y_tr / y_te are standardized; y_sd converts MSE back to original units (MSE * sd^2)."""
    torch.manual_seed(seed)
    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
    criterion = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed))

    history = []
    steps = 0
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            steps += 1
        model.eval()
        with torch.no_grad():
            history.append(criterion(model(x_te), y_te).item() * y_sd ** 2)
    train_time = time.perf_counter() - start

    #inference latency: predict the full 1,000-point test set, median of 50 calls
    with torch.no_grad():
        timings = []
        for _ in range(50):
            t0 = time.perf_counter()
            model(x_te)
            timings.append(time.perf_counter() - t0)
    test_mse = history[-1]
    #converged = first epoch whose test MSE is within 5% of the run's best
    best = min(history)
    converge_epoch = next(i + 1 for i, h in enumerate(history) if h <= best * 1.05)
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "time_s": train_time,
        "ms_per_step": 1000 * train_time / steps,
        "infer_ms": 1000 * float(np.median(timings)),
        "test_mse": test_mse,
        "r2": 1 - test_mse / (y_te.var(unbiased=False).item() * y_sd ** 2),
        "converge_epoch": converge_epoch,
        "history": history,
    }


def main():
    X, environments = make_environments()
    split = np.random.default_rng(DATA_SEED).permutation(DATA)
    tr, te = split[: int(0.8 * DATA)], split[int(0.8 * DATA):]
    x_tr, x_te = (torch.tensor(X[i], dtype=torch.float32) for i in (tr, te))

    results = {}
    for env, y in environments.items():
        y_tr, y_te = (torch.tensor(y[i], dtype=torch.float32) for i in (tr, te))
        y_tr, y_te, y_sd = standardize(y_tr, y_te)
        for name, cls in MODELS.items():
            runs = [run(cls, x_tr, y_tr, x_te, y_te, s, y_sd) for s in SEEDS]
            summary = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
                       for k in ("time_s", "ms_per_step", "infer_ms", "test_mse", "r2", "converge_epoch")}
            summary["params"] = runs[0]["params"]
            summary["history_mean"] = np.mean([r["history"] for r in runs], axis=0).tolist()
            results[f"{env}|{name}"] = summary
            print(f"{env:7s} {name:28s} params={summary['params']:4d} "
                  f"mse={summary['test_mse'][0]:8.3f}±{summary['test_mse'][1]:6.3f} "
                  f"r2={summary['r2'][0]:6.3f} ms/step={summary['ms_per_step'][0]:.3f} "
                  f"infer={summary['infer_ms'][0]:.3f}ms conv={summary['converge_epoch'][0]:.0f}",
                  flush=True)

    with open("hybrid_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
