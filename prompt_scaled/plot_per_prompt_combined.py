"""
plot_per_prompt_combined.py

Two-row figure: top row = per-prompt Z scores, bottom row = per-prompt mean rewards.
One column per selected alpha (descending left-to-right), shared y-axis within each row.

Each panel:
  - Bars: per-stem metric sorted ascending
  - Line: Bo2 metric in same stem order
  - Colored dashed line + shaded band: guided mean ± 95% CI
  - Gray dotted line + shaded band: Bo2 mean ± 95% CI

Usage (Colab):
    !python .../plot_per_prompt_combined.py
    !python .../plot_per_prompt_combined.py --bon 3
"""

from pathlib import Path as _ReleasePath
_LCB_RELEASE_ROOT = str(_ReleasePath(__file__).resolve().parents[1])


import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
BASE_DIR       = f"{_LCB_RELEASE_ROOT}"
RESULTS_DIR    = os.path.join(BASE_DIR, "prompt_scaled", "inference_results")
OUT_DIR        = os.path.join(BASE_DIR, "prompt_scaled", "inference_results", "z_score_plots")
REF_FILENAME   = "lcb_clean_d0p5_a2p0.json"
INCLUDE_ALPHAS = {0.45, 1.0, 2.0}
ALPHA_COLORS   = ["tab:blue", "tab:orange", "tab:green", "tab:brown", "tab:purple",
                  "tab:pink", "tab:cyan", "tab:olive"]
# -------------------------------------------------------------------------


def load_samples(path):
    with open(path) as f:
        d = json.load(f)
    return d, d.get("samples", [])


def per_stem_scores(samples, guided):
    out = defaultdict(list)
    for s in samples:
        if s["guided"] == guided:
            out[s["stem"]].append(s["score"])
    return dict(out)


def bon_mean_per_stem(stem_scores, n):
    result = {}
    for stem, scores in stem_scores.items():
        arr = np.array(scores)
        n_groups = len(arr) // n
        if n_groups == 0:
            continue
        result[stem] = float(arr[:n_groups * n].reshape(n_groups, n).max(axis=1).mean())
    return result


def mean_and_ci(values, z=1.96):
    mu = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mu, z * se


def short_label(stem):
    m = re.search(r"Thank you for your (.+?), but", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def extract_alpha(filename, meta):
    alpha = meta.get("alpha")
    if alpha is None:
        m = re.search(r"_a([\d]+p[\d]+)", filename)
        if m:
            alpha = float(m.group(1).replace("p", "."))
    return alpha


def extract_eff_n(filename, meta, samples):
    guided_eff = [s["eff_n"] for s in samples if s.get("guided") and "eff_n" in s]
    if guided_eff:
        v = float(np.mean(guided_eff))
        if not np.isnan(v):
            return v
    v = meta.get("mean_eff_n")
    if v is not None and not np.isnan(float(v)):
        return float(v)
    m = re.search(r"eff_N_([\d]+p[\d]+)", filename)
    if m:
        return float(m.group(1).replace("p", "."))
    return float("nan")


def draw_panel(ax, stems_sorted, guided_vals, bon_vals, guided_mean, guided_ci,
               bon_mean, bon_ci, color, ylabel=None):
    x       = np.arange(len(stems_sorted))
    xlabels = [short_label(s) for s in stems_sorted]

    bar_colors = [color if v >= 0 else "tab:red" for v in guided_vals]
    ax.bar(x, guided_vals, color=bar_colors, width=0.7, alpha=0.75, zorder=2)
    ax.axhline(0, color="black", linewidth=0.6, zorder=3)

    ax.plot(x, bon_vals, color="gray", linewidth=1.2, alpha=0.85, zorder=4)

    ax.axhline(guided_mean, color=color, linewidth=1.8, linestyle="--", zorder=6)
    ax.axhspan(guided_mean - guided_ci, guided_mean + guided_ci,
               color=color, alpha=0.12, zorder=1)

    ax.axhline(bon_mean, color="gray", linewidth=1.8, linestyle=":", zorder=6)
    ax.axhspan(bon_mean - bon_ci, bon_mean + bon_ci,
               color="gray", alpha=0.12, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
    ax.grid(True, alpha=0.25, axis="y")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=REF_FILENAME)
    parser.add_argument("--bon", type=int, default=2)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    N = args.bon

    # ---- Reference unguided statistics ----
    ref_path = os.path.join(args.results_dir, args.ref)
    _, ref_samples = load_samples(ref_path)
    ref_unguided = per_stem_scores(ref_samples, guided=False)
    mu_ug  = {s: float(np.mean(v)) for s, v in ref_unguided.items()}
    std_ug = {s: max(float(np.std(v)), 1e-6) for s, v in ref_unguided.items()}

    # ---- Bo{N} per stem — Z and reward ----
    bon_by_stem  = bon_mean_per_stem(ref_unguided, N)
    bon_z_stem   = {s: (mu - mu_ug[s]) / std_ug[s] for s, mu in bon_by_stem.items() if s in mu_ug}
    bon_r_stem   = bon_by_stem

    bon_z_arr    = np.array(list(bon_z_stem.values()))
    bon_r_arr    = np.array(list(bon_r_stem.values()))
    bon_z_mean, bon_z_ci = mean_and_ci(bon_z_arr)
    bon_r_mean, bon_r_ci = mean_and_ci(bon_r_arr)

    # ---- Load guided runs ----
    runs = []
    for jpath in sorted(glob(os.path.join(args.results_dir, "*.json"))):
        fname = os.path.basename(jpath)
        meta, samples = load_samples(jpath)
        guided_by_stem = per_stem_scores(samples, guided=True)
        if not guided_by_stem:
            continue
        alpha = extract_alpha(fname, meta)
        eff_n = extract_eff_n(fname, meta, samples)
        if alpha is None or alpha not in INCLUDE_ALPHAS:
            continue

        z_stem = {s: (float(np.mean(g)) - mu_ug[s]) / std_ug[s]
                  for s, g in guided_by_stem.items() if s in mu_ug}
        r_stem = {s: float(np.mean(g))
                  for s, g in guided_by_stem.items() if s in mu_ug}

        z_arr = np.array(list(z_stem.values()))
        r_arr = np.array(list(r_stem.values()))
        z_mean, z_ci = mean_and_ci(z_arr)
        r_mean, r_ci = mean_and_ci(r_arr)

        runs.append(dict(alpha=alpha, eff_n=eff_n,
                         z_stem=z_stem, r_stem=r_stem,
                         z_mean=z_mean, z_ci=z_ci,
                         r_mean=r_mean, r_ci=r_ci))

    runs.sort(key=lambda r: -r["alpha"])   # descending: highest alpha on left
    alpha_vals_asc = sorted(set(r["alpha"] for r in runs))
    color_map = {a: ALPHA_COLORS[i % len(ALPHA_COLORS)] for i, a in enumerate(alpha_vals_asc)}

    # ---- Layout: 2 rows × n_panels cols, shared y per row ----
    n_panels   = len(runs)
    stem_count = len(next(r["z_stem"] for r in runs))
    panel_w    = max(3.5, stem_count * 0.11)

    fig, axes = plt.subplots(2, n_panels,
                             figsize=(panel_w * n_panels, 5.5 * 2),
                             sharey="row")

    for col, run in enumerate(runs):
        alpha = run["alpha"]
        eff_n = run["eff_n"]
        color = color_map[alpha]
        eff_n_str = f"{eff_n:.2f}" if not np.isnan(eff_n) else "?"

        # ---- Z score row ----
        stems_z = sorted(run["z_stem"], key=lambda s: run["z_stem"][s])
        g_z  = np.array([run["z_stem"][s] for s in stems_z])
        bo_z = np.array([bon_z_stem.get(s, np.nan) for s in stems_z])
        draw_panel(axes[0, col], stems_z, g_z, bo_z,
                   run["z_mean"], run["z_ci"], bon_z_mean, bon_z_ci,
                   color, ylabel="Z score" if col == 0 else None)
        axes[0, col].set_title(f"α={alpha} (eff N = {eff_n_str})", fontsize=10)

        # ---- Reward row ----
        stems_r = sorted(run["r_stem"], key=lambda s: run["r_stem"][s])
        g_r  = np.array([run["r_stem"][s] for s in stems_r])
        bo_r = np.array([bon_r_stem.get(s, np.nan) for s in stems_r])
        draw_panel(axes[1, col], stems_r, g_r, bo_r,
                   run["r_mean"], run["r_ci"], bon_r_mean, bon_r_ci,
                   color, ylabel="Mean reward" if col == 0 else None)

    fig.suptitle("Per prompt Z score", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(args.out_dir, f"per_prompt_combined_bo{N}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")

