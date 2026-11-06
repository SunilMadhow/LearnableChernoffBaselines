"""
plot_survival_scaling_combined_v3.py

Two-panel figure for Setting III (held-out prompts):
  Left:  Survival function — P(Z > z) per alpha  [unchanged from v1/v2]
  Right: Inference scaling — mean Z score vs. effective N (alpha=1.0 only, M labels)

Usage (Colab):
    !python .../plot_survival_scaling_combined_v3.py
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
BASE_DIR        = f"{_LCB_RELEASE_ROOT}"
RESULTS_DIR     = os.path.join(BASE_DIR, "prompt_scaled", "inference_results")
OUT_DIR         = os.path.join(BASE_DIR, "prompt_scaled", "inference_results", "z_score_plots")
REF_FILENAME    = "lcb_clean_d0p5_a2p0.json"
EFF_N_CAP       = 5.5
HIGHLIGHT_ALPHA = 1.0
ALPHA_COLORS    = ["tab:blue", "tab:orange", "tab:green", "tab:brown", "tab:purple",
                   "tab:pink", "tab:cyan", "tab:olive"]
MARKER_CYCLE    = ["o", "s", "^", "D", "P", "v", "*", "X"]
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


def agg_z(stem_means, mu_ug, std_ug):
    zs = [
        (v - mu_ug[s]) / std_ug[s]
        for s, v in stem_means.items() if s in mu_ug
    ]
    return float(np.mean(zs)) if zs else float("nan")


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
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--eff_n_cap", type=float, default=EFF_N_CAP)
    parser.add_argument("--max_bon_unguided", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Reference unguided statistics ----
    ref_path = os.path.join(args.results_dir, args.ref)
    _, ref_samples = load_samples(ref_path)
    ref_unguided = per_stem_scores(ref_samples, guided=False)
    mu_ug  = {s: float(np.mean(v)) for s, v in ref_unguided.items()}
    std_ug = {s: max(float(np.std(v)), 1e-6) for s, v in ref_unguided.items()}

    # ---- Load all guided runs ----
    run_data = []
    for jpath in sorted(glob(os.path.join(args.results_dir, "*.json"))):
        fname = os.path.basename(jpath)
        meta, samples = load_samples(jpath)
        guided_by_stem = per_stem_scores(samples, guided=True)
        if not guided_by_stem:
            continue
        alpha = extract_alpha(fname, meta)
        if alpha == 0.4:  # Omit this exploratory temperature from paper outputs.
            continue
        eff_n = extract_eff_n(fname, meta, samples)
        if alpha is None or np.isnan(eff_n):
            print(f"  [warn] {fname}: skipping")
            continue
        run_data.append(dict(alpha=alpha, eff_n=eff_n, guided_by_stem=guided_by_stem))

    run_data.sort(key=lambda r: r["alpha"])
    alpha_vals = sorted(set(r["alpha"] for r in run_data))
    color_map  = {a: ALPHA_COLORS[i % len(ALPHA_COLORS)] for i, a in enumerate(alpha_vals)}

    # ---- Figure ----
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("Setting III (held out prompts)", fontsize=12, y=1.01)

    # ================================================================
    # Left panel — Survival function  [unchanged]
    # ================================================================
    ax_s.set_title("Survival function", fontsize=10)

    for r in run_data:
        alpha = r["alpha"]
        z_vals = np.array([
            (float(np.mean(g)) - mu_ug[s]) / std_ug[s]
            for s, g in r["guided_by_stem"].items() if s in mu_ug
        ])
        z        = np.sort(z_vals)
        n        = len(z)
        survival = 1.0 - np.arange(1, n + 1) / n
        eff_n_str = f"{r['eff_n']:.2f}"
        ax_s.step(z, survival, where="post",
                  color=color_map[alpha], linewidth=1.8,
                  label=f"α={alpha} (eff N={eff_n_str})")

    ax_s.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_s.set_xlabel("$Z$ score", fontsize=10)
    ax_s.set_ylabel("$P(Z > z)$", fontsize=10)
    ax_s.tick_params(labelsize=9)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8, loc="upper right",
                framealpha=0.85, edgecolor="0.8", borderpad=0.6,
                labelspacing=0.3, handlelength=1.4, handletextpad=0.4)

    # ================================================================
    # Right panel — Inference scaling in Z-score units (alpha=1.0 only)
    # ================================================================
    ax_r.set_title("Inference scaling", fontsize=10)

    # Unguided BoN baseline in Z-score units (Z=0 at N=1 by construction)
    bon_pts = [(1.0, 0.0)]
    for n in range(2, args.max_bon_unguided + 1):
        sm = bon_mean_per_stem(ref_unguided, n)
        bon_pts.append((float(n), agg_z(sm, mu_ug, std_ug)))
    bon_x, bon_z = zip(*bon_pts)
    ax_r.plot(bon_x, bon_z, color="tab:red", linewidth=2, marker="o", markersize=6,
              zorder=2, label="BoN (unguided)")

    # Build guided plot points for alpha=1.0 only: (eff_n_x, mean_z, k)
    guided_pts = []
    for r in run_data:
        if r["alpha"] != HIGHLIGHT_ALPHA:
            continue
        eff_n, by_stem = r["eff_n"], r["guided_by_stem"]
        sm1 = {s: float(np.mean(v)) for s, v in by_stem.items()}
        guided_pts.append((eff_n, agg_z(sm1, mu_ug, std_ug), 1))
        k = 2
        while eff_n * k <= args.eff_n_cap:
            smk = bon_mean_per_stem(by_stem, k)
            guided_pts.append((eff_n * k, agg_z(smk, mu_ug, std_ug), k))
            k += 1

    # Green line: unguided anchor (Z=0, N=1) → all alpha=1.0 points
    highlight_pts = [(eff_n_x, z_val) for (eff_n_x, z_val, _k) in guided_pts]
    green_pts = sorted([(1.0, 0.0)] + highlight_pts, key=lambda t: t[0])

    if green_pts:
        gx, gy = zip(*green_pts)
        x_shared     = np.linspace(min(min(bon_x), min(gx)), max(max(bon_x), max(gx)), 300)
        bon_interp   = np.interp(x_shared, bon_x, bon_z)
        green_interp = np.interp(x_shared, gx, gy)
        ax_r.fill_between(x_shared, bon_interp, green_interp,
                          color="tab:green", alpha=0.15, zorder=1)
        ax_r.plot(gx, gy, color="tab:green", linewidth=1.8, zorder=3)

    # Proxy legend entry
    ax_r.scatter([], [], color="tab:green", marker="o", s=70,
                 label=f"LCB + BoM (α={HIGHLIGHT_ALPHA})")

    # Per-point annotation offsets: (dx, dy, ha, va)
    _annot = {
        1: (0,  12, "center", "bottom"),
        4: (0, -12, "center", "top"),
    }
    _default_annot = (0, 6, "center", "bottom")

    # Scatter dots with M annotations
    for eff_n_x, z_val, k in guided_pts:
        ax_r.scatter(eff_n_x, z_val, color="tab:green", marker="o",
                     s=70, zorder=5)
        dx, dy, ha, va = _annot.get(k, _default_annot)
        ax_r.annotate(f"M={k}", (eff_n_x, z_val),
                      textcoords="offset points", xytext=(dx, dy),
                      ha=ha, va=va, fontsize=7, color="tab:green")

    ax_r.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_r.set_xlabel("Effective $N$", fontsize=10)
    ax_r.set_ylabel("Mean $Z$ score", fontsize=10)
    ax_r.tick_params(labelsize=9)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, loc="lower right",
                framealpha=0.85, edgecolor="0.8", borderpad=0.6,
                labelspacing=0.3, handlelength=1.4, handletextpad=0.4)

    # ================================================================
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    out_path = os.path.join(args.out_dir, "survival_scaling_combined_v3.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")

