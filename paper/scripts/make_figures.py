"""
Builds every figure and results table in the paper from the raw experiment outputs.

Inputs  (paper/results/): hybrid_results.json  (Model_Code/hybrid_study.py)
                          latency_results.json (Model_Code/latency_sweep.py)
Outputs (paper/figures/, paper/tables/): PDF figures and LaTeX table bodies.

Run from the paper/ directory:  python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "Model_Code"))
from hybrid_study import make_environments  # noqa: E402

FIG, TAB, RES = ROOT / "figures", ROOT / "tables", ROOT / "results"
TASKS = ["Linear", "Curved", "Chain"]
TASK_LABEL = {"Linear": "Linear", "Curved": "Curved", "Chain": "Chained"}
MODELS = ["Sequential", "Parallel", "Modular",
          "H-A Parallel of Sequentials", "H-B Staged Modular Chain", "H-C Trunk + Parallel Heads"]
SHORT = dict(zip(MODELS, ["Sequential", "Parallel", "Modular", "H-A", "H-B", "H-C"]))
N_SEEDS = 5

#Validated categorical palette (fixed slot order), plus marker/dash so identity survives grayscale print
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKER = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P"]))
DASH = dict(zip(MODELS, ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

COL_W, PAGE_W = 3.5, 7.16  # IEEE two-column widths in inches
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

#Benchmark 1 (pilot) values, copied from the final cell of Model_Code/Colab_Research_Workspace.ipynb
#(train_time_s, final_train_mse). Sequential/Parallel ran in Keras, Modular in PyTorch.
PILOT = {
    "Sequential": {"Linear": (29.382, 37.796), "Curved": (29.462, 239.638), "Chain": (30.082, 10.386)},
    "Parallel":   {"Linear": (30.313, 37.833), "Curved": (30.932, 240.527), "Chain": (30.378, 10.444)},
    "Modular":    {"Linear": (18.671, 38.158), "Curved": (17.724, 245.678), "Chain": (18.603, 10.547)},
}


def load():
    hybrid = json.loads((RES / "hybrid_results.json").read_text())
    latency = json.loads((RES / "latency_results.json").read_text())
    return hybrid, latency


def get(hybrid, task, model, key):
    return hybrid[f"{task}|{model}"][key]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_tasks():
    X, envs = make_environments()
    x = X.ravel() * 10
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.75))
    for ax, task in zip(axes, TASKS):
        ax.scatter(x, envs[task].ravel(), s=1.2, color="#2a78d6", alpha=0.35, linewidths=0, rasterized=True)
        ax.set_title(f"({'abc'[TASKS.index(task)]}) {TASK_LABEL[task]}")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_tasks")


def fig_curves(hybrid):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.05))
    for ax, task in zip(axes, TASKS):
        for m in MODELS:
            h = np.array(get(hybrid, task, m, "history_mean"))
            ax.plot(np.arange(1, len(h) + 1), h, color=COLOR[m], linestyle=DASH[m], label=SHORT[m],
                    marker=MARKER[m], markevery=12, markersize=3.5)
        ax.set_yscale("log")
        ax.set_title(f"({'abc'[TASKS.index(task)]}) {TASK_LABEL[task]}")
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Test MSE (mean of 5 seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_curves")


def fig_tradeoff(hybrid):
    """Accuracy vs. training cost: relative test MSE against ms per training step."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.1))
    for ax, task in zip(axes, TASKS):
        best = min(get(hybrid, task, m, "test_mse")[0] for m in MODELS)
        for m in MODELS:
            (mse, mse_sd), (step, step_sd) = get(hybrid, task, m, "test_mse"), get(hybrid, task, m, "ms_per_step")
            ax.errorbar(step, mse / best, xerr=step_sd, yerr=mse_sd / best, fmt=MARKER[m], color=COLOR[m],
                        markersize=5, markeredgecolor="white", markeredgewidth=0.6, elinewidth=0.8,
                        capsize=1.5, label=SHORT[m])
        ax.set_title(f"({'abc'[TASKS.index(task)]}) {TASK_LABEL[task]}")
        ax.set_xlabel("Training cost (ms / step)")
        ax.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=":")
    axes[0].set_ylabel("Test MSE / best on task")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_tradeoff")


def fig_latency_batch(latency):
    lat = latency["models"]
    batches = latency["meta"]["batches"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.2))
    for ax, threads in zip(axes, latency["meta"]["threads"]):
        for m in MODELS:
            t = [lat[m]["timings"][f"{threads}|{b}"] for b in batches]
            med = np.array([v["median_us"] for v in t])
            lo = med - np.array([v["p10_us"] for v in t])
            hi = np.array([v["p90_us"] for v in t]) - med
            ax.errorbar(batches, med, yerr=[lo, hi], color=COLOR[m], linestyle=DASH[m], marker=MARKER[m],
                        markersize=3.5, elinewidth=0.6, capsize=1.2, label=SHORT[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch size (samples per forward call)")
        ax.set_title(f"({'ab'[latency['meta']['threads'].index(threads)]}) "
                     f"{threads} CPU thread{'s' if threads > 1 else ''}")
    axes[0].set_ylabel(r"Inference latency ($\mu$s, median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_latency_batch")

    #single-column, single-thread variant for the 6-page build
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))  # compact-only figure; kept short for the 6-page fit
    for m in MODELS:
        med = [lat[m]["timings"][f"1|{b}"]["median_us"] for b in batches]
        ax.plot(batches, med, color=COLOR[m], linestyle=DASH[m], marker=MARKER[m], markersize=3.5, label=SHORT[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size (samples per forward call)")
    ax.set_ylabel(r"Latency ($\mu$s, median, 1 thread)")
    ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=6.3, columnspacing=0.8, handlelength=2.2)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    save(fig, "fig_latency_batch_1t")


def fig_ops_latency(latency):
    """Batch-1 latency is predicted by operator count, not parameter count."""
    lat = latency["models"]
    ops = np.array([lat[m]["ops_per_call"] for m in MODELS])
    us = np.array([lat[m]["timings"]["1|1"]["median_us"] for m in MODELS])
    fit = stats.linregress(ops, us)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    xs = np.linspace(ops.min() - 1, ops.max() + 1, 50)
    ax.plot(xs, fit.intercept + fit.slope * xs, color=MUTED, linewidth=0.9, linestyle="--", zorder=1,
            label=f"OLS fit ($R^2$ = {fit.rvalue**2:.2f})")
    for m, o, u in zip(MODELS, ops, us):
        t = lat[m]["timings"]["1|1"]
        if "ci_lo_us" in t:  # 95% interval across shuffled repeats
            ax.errorbar(o, u, yerr=[[u - t["ci_lo_us"]], [t["ci_hi_us"] - u]], color=COLOR[m], elinewidth=0.8,
                        capsize=2, fmt="none")
        ax.plot(o, u, MARKER[m], color=COLOR[m], markersize=6, markeredgecolor="white", markeredgewidth=0.7)
        ax.annotate(f"{SHORT[m]} ({lat[m]['params']} p)", (o, u), xytext=(5, -9 if m == "Modular" else 1),
                    textcoords="offset points",
                    fontsize=6.5, color=INK)
    ax.set_xlabel("Tensor operations per forward call")
    ax.set_ylabel(r"Batch-1 latency ($\mu$s, 1 thread)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(ops.min() - 1, ops.max() + 4)
    save(fig, "fig_ops_latency")
    return fit


def fig_pilot():
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    width = 0.26
    x = np.arange(len(TASKS))
    labels = {"Sequential": "Sequential (Keras)", "Parallel": "Parallel (Keras)", "Modular": "Modular (PyTorch)"}
    for i, m in enumerate(PILOT):
        vals = [PILOT[m][t][0] for t in TASKS]
        bars = ax.bar(x + (i - 1) * width, vals, width - 0.03, color=COLOR[m], label=labels[m],
                      edgecolor="white", linewidth=0.5, hatch=["", "////", "...."][i])
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=1.5, color=INK)
    ax.set_xticks(x, [TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("Training time, 100 epochs (s)")
    ax.set_ylim(0, 37)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2), fontsize=6.3,
              columnspacing=0.8, handlelength=1.4)
    save(fig, "fig_pilot")


def fmt(mean_sd, digits=3):
    m, s = mean_sd
    return f"{m:.{digits}f}\\,$\\pm$\\,{s:.{digits}f}"


def table_main(hybrid):
    """One block per task. Bold = best mean; dagger = Welch t-test vs Sequential, p < 0.05."""
    lines = []
    for task in TASKS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{TASK_LABEL[task]} task}}}}\\\\")
        best_mse = min(get(hybrid, task, m, "test_mse")[0] for m in MODELS)
        best_inf = min(get(hybrid, task, m, "infer_ms")[0] for m in MODELS)
        best_step = min(get(hybrid, task, m, "ms_per_step")[0] for m in MODELS)
        base = get(hybrid, task, "Sequential", "test_mse")
        for m in MODELS:
            mse = get(hybrid, task, m, "test_mse")
            digits = 3 if mse[0] < 10 else 2
            cell = fmt(mse, digits)
            if f"{mse[0]:.{digits}f}" == f"{best_mse:.{digits}f}":  # ties at shown precision
                cell = f"\\textbf{{{cell}}}"
            if m != "Sequential":
                p = stats.ttest_ind_from_stats(mse[0], mse[1] * np.sqrt(N_SEEDS / (N_SEEDS - 1)), N_SEEDS,
                                               base[0], base[1] * np.sqrt(N_SEEDS / (N_SEEDS - 1)), N_SEEDS,
                                               equal_var=False).pvalue
                if p < 0.05:
                    cell += "$^\\dagger$"
            step = get(hybrid, task, m, "ms_per_step")
            inf = get(hybrid, task, m, "infer_ms")
            step_c = fmt(step, 3) if f"{step[0]:.3f}" != f"{best_step:.3f}" else f"\\textbf{{{fmt(step, 3)}}}"
            inf_c = fmt(inf, 3) if f"{inf[0]:.3f}" != f"{best_inf:.3f}" else f"\\textbf{{{fmt(inf, 3)}}}"
            r2 = get(hybrid, task, m, "r2")[0]
            conv = get(hybrid, task, m, "converge_epoch")
            lines.append(f"{SHORT[m]} & {cell} & {r2:.3f} & {step_c} & {inf_c} & {conv[0]:.0f}\\,$\\pm$\\,{conv[1]:.0f}\\\\")
        if task != TASKS[-1]:
            lines.append("\\midrule")
    (TAB / "tab_main_body.tex").write_text("\n".join(lines) + "\n")


def table_arch(latency):
    depth = {"Sequential": 2, "Parallel": 2, "Modular": 2, "H-A Parallel of Sequentials": 3,
             "H-B Staged Modular Chain": 6, "H-C Trunk + Parallel Heads": 3}
    lines = []
    for m in MODELS:
        e = latency["models"][m]
        lines.append(f"{SHORT[m]} & {e['params']} & {e['macs_per_sample']} & {depth[m]} & {e['ops_per_call']}"
                     f" & {e['kernels_per_call']}"
                     f" & {e['timings']['1|1']['median_us']:.1f} & {e['timings']['1|4096']['median_us']:.0f}\\\\")
    (TAB / "tab_arch_body.tex").write_text("\n".join(lines) + "\n")


def table_pilot():
    lines = []
    fw = {"Sequential": "Keras", "Parallel": "Keras", "Modular": "PyTorch"}
    for m in PILOT:
        cells = " & ".join(f"{PILOT[m][t][0]:.1f} & {PILOT[m][t][1]:.2f}" for t in TASKS)
        lines.append(f"{m} ({fw[m]}) & {cells}\\\\")
    (TAB / "tab_pilot_body.tex").write_text("\n".join(lines) + "\n")


def macros(hybrid, latency, fit):
    """Numbers quoted in the prose, so the text can never drift from the results."""
    lat = latency["models"]
    out = {}
    for task in TASKS:
        for m in MODELS:
            key = (task + SHORT[m]).replace("-", "")
            out[f"mse{key}"] = f"{get(hybrid, task, m, 'test_mse')[0]:.3f}"
            out[f"rtwo{key}"] = f"{get(hybrid, task, m, 'r2')[0]:.3f}"
            out[f"step{key}"] = f"{get(hybrid, task, m, 'ms_per_step')[0]:.3f}"
            out[f"inf{key}"] = f"{get(hybrid, task, m, 'infer_ms')[0]:.3f}"
    for m in MODELS:
        k = SHORT[m].replace("-", "")
        out[f"latOne{k}"] = f"{lat[m]['timings']['1|1']['median_us']:.1f}"
        out[f"latBig{k}"] = f"{lat[m]['timings']['1|4096']['median_us']:.0f}"
        out[f"ops{k}"] = str(lat[m]["ops_per_call"])
    out["opsFitRsq"] = f"{fit.rvalue**2:.2f}"
    out["opsFitSlope"] = f"{fit.slope:.1f}"
    pilot_speedup = np.mean([PILOT["Sequential"][t][0] / PILOT["Modular"][t][0] for t in TASKS])
    out["pilotSpeedup"] = f"{pilot_speedup:.2f}"
    out["torchVersion"] = latency["meta"]["torch"].split("+")[0]

    #Derived comparisons quoted in Sections V-VI
    seq, hb, ha = "Sequential", "H-B Staged Modular Chain", "H-A Parallel of Sequentials"
    mse = lambda t, m: get(hybrid, t, m, "test_mse")[0]
    floor_chain = 0.25  # sigma^2 of the chained task
    out["chainHBGain"] = f"{100 * (1 - mse('Chain', hb) / mse('Chain', seq)):.0f}"
    out["chainHAGain"] = f"{100 * (1 - mse('Chain', ha) / mse('Chain', seq)):.0f}"
    out["chainExcessSeq"] = f"{mse('Chain', seq) - floor_chain:.3f}"
    out["chainExcessHB"] = f"{mse('Chain', hb) - floor_chain:.3f}"
    out["chainExcessRatio"] = f"{(mse('Chain', seq) - floor_chain) / (mse('Chain', hb) - floor_chain):.1f}"
    out["latOneRatioHB"] = f"{lat[hb]['timings']['1|1']['median_us'] / lat[seq]['timings']['1|1']['median_us']:.1f}"
    out["stepRatioHB"] = f"{get(hybrid, 'Chain', hb, 'ms_per_step')[0] / get(hybrid, 'Chain', seq, 'ms_per_step')[0]:.2f}"
    steps_per_epoch = 4000 // 32 + (4000 % 32 > 0)
    wall = lambda t, m: (get(hybrid, t, m, "converge_epoch")[0] * steps_per_epoch
                         * get(hybrid, t, m, "ms_per_step")[0] / 1000)
    out["curvedConvSeq"] = f"{get(hybrid, 'Curved', seq, 'converge_epoch')[0]:.0f}"
    out["curvedConvHB"] = f"{get(hybrid, 'Curved', hb, 'converge_epoch')[0]:.0f}"
    out["curvedWallSeq"] = f"{wall('Curved', seq):.2f}"
    out["curvedWallHB"] = f"{wall('Curved', hb):.2f}"
    out["curvedWallRatio"] = f"{wall('Curved', seq) / wall('Curved', hb):.1f}"
    t1, t4 = (lambda m, b: lat[m]["timings"][f"1|{b}"]["median_us"]), (lambda m, b: lat[m]["timings"][f"4|{b}"]["median_us"])
    out["threadParallelOne"] = f"{t1('Parallel', 1000):.0f}"
    out["threadParallelFour"] = f"{t4('Parallel', 1000):.0f}"
    out["threadSeqBigFour"] = f"{t4(seq, 4096):.0f}"
    out["threadHABigFour"] = f"{t4(ha, 4096):.0f}"
    #What explains batch-1 latency: kernels, and ops + nn.Module calls (Section V-C)
    from hybrid_study import MODELS as MODEL_CLASSES
    import torch
    def module_calls(cls):
        model, n = cls().eval(), [0]
        for mod in model.modules():
            mod.register_forward_pre_hook(lambda *_: n.__setitem__(0, n[0] + 1))
        with torch.no_grad():
            model(torch.rand(1, 1))
        return n[0]
    mcalls = np.array([module_calls(MODEL_CLASSES[m]) for m in MODELS])
    y1 = np.array([lat[m]["timings"]["1|1"]["median_us"] for m in MODELS])
    ops_arr = np.array([lat[m]["ops_per_call"] for m in MODELS])
    kern = np.array([lat[m]["kernels_per_call"] for m in MODELS])
    def ols(*cols):
        X = np.column_stack([np.ones(len(y1)), *cols])
        coef, *_ = np.linalg.lstsq(X, y1, rcond=None)
        return 1 - (y1 - X @ coef).var() / y1.var(), coef
    out["kernFitRsq"] = f"{ols(kern)[0]:.2f}"
    rsq, coef = ols(ops_arr, mcalls)
    out["opsModFitRsq"] = f"{rsq:.3f}"
    out["perOpUs"] = f"{coef[1]:.1f}"
    out["perModUs"] = f"{coef[2]:.1f}"
    for m, c in zip(MODELS, mcalls):
        out[f"mod{SHORT[m].replace('-', '')}"] = str(c)
    out["ctrlLatGap"] = f"{100 * abs(t1(seq, 1) - t1('Modular', 1)) / t1('Modular', 1):.0f}"
    out["ctrlStepGap"] = f"{100 * abs(get(hybrid, 'Linear', seq, 'ms_per_step')[0] - get(hybrid, 'Linear', 'Modular', 'ms_per_step')[0]) / get(hybrid, 'Linear', 'Modular', 'ms_per_step')[0]:.0f}"
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items()))
    (TAB / "numbers.tex").write_text("% Auto-generated by scripts/make_figures.py. Do not edit.\n" + body + "\n")


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    hybrid, latency = load()
    fig_tasks()
    fig_curves(hybrid)
    fig_tradeoff(hybrid)
    fig_latency_batch(latency)
    fit = fig_ops_latency(latency)
    fig_pilot()
    table_main(hybrid)
    table_arch(latency)
    table_pilot()
    macros(hybrid, latency, fit)
    print("figures and tables written")


if __name__ == "__main__":
    main()
