"""
plot_z_score_histograms.py

Overlapping histograms of per-stem Z scores for each guidance level alpha,
plus Bo2 (unguided) as a gray reference. All on one shared axes.

Usage (Colab):
    !python .../plot_z_score_histograms.py
    !python .../plot_z_score_histograms.py --bins 25
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
BASE_DIR     = f"{_LCB_RELEASE_ROOT}"
RESULTS_DIR  = os.path.join(BASE_DIR, "prompt_scaled", "inference_results")
OUT_DIR      = os.path.join(BASE_DIR, "prompt_scaled", "inference_results", "z_score_plots")
REF_FILENAME = "lcb_clean_d0p5_a2p0.json"
ALPHA_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:brown", "tab:purple",
                "tab:pink", "tab:cyan", "tab:olive"]
# -------------------------------------------------------------------------


def load_samples(path):
    with open(path) as f:
        d = json.load(f)
    return d, d.get("samples", [])


def per_stem_scores(samples, guided: bool):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=REF_FILENAME)
    parser.add_argument("--bins", type=int, default=35)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Reference unguided statistics ----
    ref_path = os.path.join(args.results_dir, args.ref)
    _, ref_samples = load_samples(ref_path)
    ref_unguided = per_stem_scores(ref_samples, guided=False)
    mu_ug  = {s: float(np.mean(v)) for s, v in ref_unguided.items()}
    std_ug = {s: max(float(np.std(v)), 1e-6) for s, v in ref_unguided.items()}

    # ---- Guided Z scores per alpha ----
    json_files = sorted(glob(os.path.join(args.results_dir, "*.json")))
    runs = []
    for jpath in json_files:
        fname = os.path.basename(jpath)
        meta, samples = load_samples(jpath)
        guided_by_stem = per_stem_scores(samples, guided=True)
        if not guided_by_stem:
            continue
        alpha = extract_alpha(fname, meta)
        if alpha == 0.4:  # Omit this exploratory temperature from paper outputs.
            continue
        eff_n = extract_eff_n(fname, meta, samples)
        if alpha is None:
            continue
        z_vals = np.array([(float(np.mean(g)) - mu_ug[s]) / std_ug[s]
                           for s, g in guided_by_stem.items() if s in mu_ug])
        runs.append(dict(alpha=alpha, eff_n=eff_n, z_vals=z_vals))

    runs.sort(key=lambda r: r["alpha"])
    alpha_vals_asc = sorted(set(r["alpha"] for r in runs))
    color_map = {a: ALPHA_COLORS[i % len(ALPHA_COLORS)] for i, a in enumerate(alpha_vals_asc)}

    # ---- Shared bin edges across all distributions ----
    all_z = np.concatenate([r["z_vals"] for r in runs])
    bin_edges = np.linspace(all_z.min() - 0.1, all_z.max() + 0.1, args.bins + 1)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Guided alphas (descending so most-aggressive is drawn last / on top)
    for run in reversed(runs):
        alpha = run["alpha"]
        eff_n = run["eff_n"]
        z     = run["z_vals"]
        color = color_map[alpha]
        eff_n_str = f"{eff_n:.2f}" if not np.isnan(eff_n) else "?"
        weights = np.ones(len(z)) / len(z)   # PMF: bars sum to 1
        ax.hist(z, bins=bin_edges, weights=weights, color=color, alpha=0.30,
                edgecolor=color, linewidth=0.8,
                label=f"α={alpha} (eff N = {eff_n_str})", zorder=3)
        ax.axvline(z.mean(), color=color, linewidth=1.4, linestyle="--", zorder=5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Z score  (guided mean − unguided mean) / unguided std", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Z score distribution", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "z_score_histograms.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")

