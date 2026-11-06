from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, _ROOT)

_FIGURES = _Path(_ROOT) / "figures" / "regenerated"
_FIGURES.mkdir(parents=True, exist_ok=True)

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
alphas = [0.5, 0.3, 0.1]
deltas = [0.9, 0.7, 0.5, 0.3]
run_id = "20260205_195651_634076"
STEPS = 32

PALETTE = {
    0.5: "#E69F00",
    0.3: "#0072B2",
    0.1: "#009E73",
    0.05: "#D55E00",
}
MARKERS = {
    0.5: "o",
    0.3: "s",
    0.1: "^",
    0.05: "D",
}
BON_COLOR = "#E41A1C"
BON_MARKER = "X"

base_dir = f"{_ROOT}/data_runs/{run_id}"
ref_results_dir = f"{_ROOT}/data_runs/20251224_233551_699407/results"

plot_data = {alpha: [] for alpha in alphas}
lcb_bo2_data = {alpha: [] for alpha in alphas}
bon_data_points = []
rs_ground_truth = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def mean_ci(values):
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = np.mean(values)
    if n == 1:
        return mean, 0.0
    return mean, 1.96 * (np.std(values, ddof=1) / np.sqrt(n))


def perfect_rate_ci(scores):
    scores = np.asarray(scores)
    n = len(scores)
    if n == 0:
        return np.nan, np.nan
    p = np.mean(scores == 1.0)
    return p * 100, 1.96 * np.sqrt((p * (1 - p)) / n) * 100


# -----------------------------------------------------------------------------
# 1. Load LCB Data
# -----------------------------------------------------------------------------
for alpha in alphas:
    alpha_str = str(alpha).replace(".", "p")

    rs_path = os.path.join(ref_results_dir, f"rejection_sampling_results_temp_{alpha}.json")
    if os.path.exists(rs_path):
        with open(rs_path, "r") as f:
            rs_data = json.load(f)
            stats = rs_data.get("stats", rs_data)
            rs_ground_truth[alpha] = {
                "mean_reward": stats["mean_reward"],
                "perfect_rate": stats.get("success_rate_perfect", stats.get("perfect_rate")) * 100,
            }

    for delta in deltas:
        delta_str = str(delta).replace(".", "p")
        stem = f"lcb_lambda_only_SYM_post_trained_temp_{alpha_str}_del_{delta_str}_inference_results"
        file_path = os.path.join(base_dir, "inference_results", "sweeps", stem)
        if not os.path.exists(file_path):
            file_path = file_path + ".json"
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        if not samples:
            continue
        scores = np.array([s["score"] for s in samples])
        if len(scores) == 0:
            continue

        r_mean, r_ci = mean_ci(scores)
        p_mean, p_ci = perfect_rate_ci(scores)
        n_eff = data.get("avg_nfe", 0) / STEPS
        plot_data[alpha].append((n_eff, r_mean, r_ci, p_mean, p_ci))

        bo2_source = scores.copy()
        if len(bo2_source) % 2 != 0:
            bo2_source = bo2_source[:-1]
        if len(bo2_source) > 0:
            bo2_scores = bo2_source.reshape(-1, 2).max(axis=1)
            bo2_r_mean, bo2_r_ci = mean_ci(bo2_scores)
            bo2_p_mean, bo2_p_ci = perfect_rate_ci(bo2_scores)
            lcb_bo2_data[alpha].append((n_eff * 2, bo2_r_mean, bo2_r_ci, bo2_p_mean, bo2_p_ci))


# -----------------------------------------------------------------------------
# 2. Load Best-of-N Baseline
# -----------------------------------------------------------------------------
bon_files = {
    1: "unguided_results_1000.json",
    2: "best_of_2_results.json",
    3: "best_of_3_results.json",
    5: "best_of_5_results.json",
}
for n, fname in bon_files.items():
    fpath = os.path.join(ref_results_dir, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r") as f:
        data = json.load(f)
    scores = np.array([s["score"] for s in data.get("samples", [])])
    if len(scores) == 0:
        continue
    r_mean, r_ci = mean_ci(scores)
    p_mean, p_ci = perfect_rate_ci(scores)
    bon_data_points.append((n, r_mean, r_ci, p_mean, p_ci))

for alpha in alphas:
    plot_data[alpha].sort(key=lambda x: x[0])
    lcb_bo2_data[alpha].sort(key=lambda x: x[0])
bon_data_points.sort(key=lambda x: x[0])


# -----------------------------------------------------------------------------
# 3. Plotting
# -----------------------------------------------------------------------------
def plot_efficiency(metric="reward", save_path=None):
    if metric == "reward":
        mean_idx, ci_idx = 1, 2
        ylabel = "Mean reward"
    elif metric == "perfect":
        mean_idx, ci_idx = 3, 4
        ylabel = "Perfect success rate (%)"
    else:
        raise ValueError("metric must be 'reward' or 'perfect'")

    fig, ax = plt.subplots(figsize=(5, 3.8))

    for alpha in alphas:
        color  = PALETTE[alpha]
        marker = MARKERS[alpha]

        if plot_data[alpha]:
            n_vals = np.array([x[0]        for x in plot_data[alpha]])
            y_vals = np.array([x[mean_idx] for x in plot_data[alpha]])
            y_errs = np.array([x[ci_idx]   for x in plot_data[alpha]])
            ax.plot(n_vals, y_vals, marker=marker, color=color,
                    label=rf"LCB $\alpha={alpha}$", linewidth=1.5, markersize=5)
            ax.fill_between(n_vals, y_vals - y_errs, y_vals + y_errs,
                            color=color, alpha=0.15)

        if lcb_bo2_data[alpha]:
            b2_n   = np.array([x[0]        for x in lcb_bo2_data[alpha]])
            b2_y   = np.array([x[mean_idx] for x in lcb_bo2_data[alpha]])
            b2_err = np.array([x[ci_idx]   for x in lcb_bo2_data[alpha]])
            ax.plot(b2_n, b2_y, marker="x", linestyle="--", color=color,
                    label=rf"LCB+Bo2 $\alpha={alpha}$", linewidth=1.5, markersize=5)
            ax.fill_between(b2_n, b2_y - b2_err, b2_y + b2_err,
                            color=color, alpha=0.10)

        if alpha in rs_ground_truth:
            oracle_val = rs_ground_truth[alpha]["mean_reward" if metric == "reward" else "perfect_rate"]
            ax.axhline(oracle_val, color=color, linestyle=":", alpha=0.4, linewidth=1,
                       label=rf"RS oracle $\alpha={alpha}$")

    if bon_data_points:
        bn_vals = np.array([x[0]        for x in bon_data_points])
        by_vals = np.array([x[mean_idx] for x in bon_data_points])
        by_errs = np.array([x[ci_idx]   for x in bon_data_points])
        ax.plot(bn_vals, by_vals, marker=BON_MARKER, color=BON_COLOR,
                linewidth=2, markersize=7, zorder=5, label="Best-of-$N$")
        ax.fill_between(bn_vals, by_vals - by_errs, by_vals + by_errs,
                        color=BON_COLOR, alpha=0.15, zorder=4)

    ax.set_xlabel(r"Effective $N$", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=7, loc="lower right",
              framealpha=0.85, edgecolor="0.8", borderpad=0.6,
              labelspacing=0.25, handlelength=1.4, handletextpad=0.4)

    plt.tight_layout()
    fig.suptitle("Setting II: Inference scaling", fontsize=12, y=0.95)
    fig.subplots_adjust(top=0.88)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.close()


# -----------------------------------------------------------------------------
# 4. Save
# -----------------------------------------------------------------------------
plot_efficiency(metric="reward",  save_path=str(_FIGURES / "reward_efficiency.pdf"))
plot_efficiency(metric="perfect", save_path=str(_FIGURES / "success_efficiency.pdf"))
