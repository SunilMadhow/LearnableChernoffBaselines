"""
print_results_table.py

Prints a results table: Mean Z and Mean Reward with 95% CIs; Eff N point estimates per alpha.
Rows: Unguided, guided alphas (ascending), Bo2.

Usage (Colab):
    !python .../print_results_table.py
"""

from pathlib import Path as _ReleasePath
_LCB_RELEASE_ROOT = str(_ReleasePath(__file__).resolve().parents[1])


import argparse
import csv
import json
import os
import re
from collections import defaultdict
from glob import glob

import numpy as np

# -------------------------------------------------------------------------
BASE_DIR     = f"{_LCB_RELEASE_ROOT}"
RESULTS_DIR  = os.path.join(BASE_DIR, "prompt_scaled", "inference_results")
REF_FILENAME = "lcb_clean_d0p5_a2p0.json"
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


def ci(arr, z=1.96):
    """95% CI half-width (normal approximation)."""
    return z * float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def extract_alpha(filename, meta):
    alpha = meta.get("alpha")
    if alpha is None:
        m = re.search(r"_a([\d]+p[\d]+)", filename)
        if m:
            alpha = float(m.group(1).replace("p", "."))
    return alpha


def extract_eff_n_samples(filename, meta, samples):
    """Return per-sample eff_n array for guided samples, or scalar fallback."""
    guided_eff = [s["eff_n"] for s in samples if s.get("guided") and "eff_n" in s]
    if guided_eff:
        return np.array(guided_eff, dtype=float)
    v = meta.get("mean_eff_n")
    if v is not None and not np.isnan(float(v)):
        return np.array([float(v)])
    m = re.search(r"eff_N_([\d]+p[\d]+)", filename)
    if m:
        return np.array([float(m.group(1).replace("p", "."))])
    return np.array([float("nan")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=REF_FILENAME)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--output_dir", help="Also save LaTeX and full-precision CSV tables here.")
    args = parser.parse_args()

    # ---- Reference unguided statistics ----
    ref_path = os.path.join(args.results_dir, args.ref)
    _, ref_samples = load_samples(ref_path)
    ref_unguided = per_stem_scores(ref_samples, guided=False)
    mu_ug  = {s: float(np.mean(v)) for s, v in ref_unguided.items()}
    std_ug = {s: max(float(np.std(v)), 1e-6) for s, v in ref_unguided.items()}

    # ---- Collect rows ----
    rows = []

    # Unguided baseline (eff_N = 1 exactly, no variance)
    ug_rewards = np.array(list(mu_ug.values()))
    rows.append(dict(
        alpha="Unguided",
        mean_z=0.0,         mean_z_ci=None,
        mean_reward=float(ug_rewards.mean()), reward_ci=ci(ug_rewards),
        eff_n=1.0,
    ))

    # Guided runs
    for jpath in sorted(glob(os.path.join(args.results_dir, "*.json"))):
        fname = os.path.basename(jpath)
        meta, samples = load_samples(jpath)
        guided_by_stem = per_stem_scores(samples, guided=True)
        if not guided_by_stem:
            continue
        alpha = extract_alpha(fname, meta)
        if alpha == 0.4:  # Omit this exploratory temperature from paper outputs.
            continue
        if alpha is None:
            continue

        z_vals      = np.array([(float(np.mean(g)) - mu_ug[s]) / std_ug[s]
                                for s, g in guided_by_stem.items() if s in mu_ug])
        reward_vals = np.array([float(np.mean(g))
                                for s, g in guided_by_stem.items() if s in mu_ug])
        eff_n_vals  = extract_eff_n_samples(fname, meta, samples)
        eff_n_vals  = eff_n_vals[~np.isnan(eff_n_vals)]

        rows.append(dict(
            alpha=alpha,
            mean_z=float(z_vals.mean()),         mean_z_ci=ci(z_vals),
            mean_reward=float(reward_vals.mean()), reward_ci=ci(reward_vals),
            eff_n=float(eff_n_vals.mean()),
        ))

    rows.sort(key=lambda r: (r["alpha"] == "Unguided", r["alpha"]))

    # Bo2 row (eff_N = 2 exactly)
    from collections import defaultdict as dd
    bon2_stems = {}
    for stem, scores in ref_unguided.items():
        arr = np.array(scores)
        n_groups = len(arr) // 2
        if n_groups:
            bon2_stems[stem] = float(arr[:n_groups*2].reshape(n_groups, 2).max(axis=1).mean())
    bon2_z_vals      = np.array([(mu - mu_ug[s]) / std_ug[s] for s, mu in bon2_stems.items()])
    bon2_reward_vals = np.array(list(bon2_stems.values()))
    rows.append(dict(
        alpha="Bo2 (unguided)",
        mean_z=float(bon2_z_vals.mean()),         mean_z_ci=ci(bon2_z_vals),
        mean_reward=float(bon2_reward_vals.mean()), reward_ci=ci(bon2_reward_vals),
        eff_n=2.0,
    ))

    # ---- Print table ----
    def fmt_val_ci(val, ci_val, fmt=".4f"):
        ci_str = f"{ci_val:.2f}" if ci_val is not None else "0.00"
        return f"{val:{fmt}} ±{ci_str}"

    col_w = 28
    header = (f"{'α':<18} {'Mean Z':>{col_w}} {'Mean Reward':>{col_w}} {'Eff N':>{col_w}}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        alpha_str  = str(r["alpha"])
        z_str      = fmt_val_ci(r["mean_z"], r.get("mean_z_ci"))
        reward_str = fmt_val_ci(r["mean_reward"], r.get("reward_ci"))
        eff_n_str  = f'{r["eff_n"]:.4f}'
        print(f"{alpha_str:<18} {z_str:>{col_w}} {reward_str:>{col_w}} {eff_n_str:>{col_w}}")
    print(sep)

    # ---- Export table ----
    if args.output_dir:
        output_dir = _ReleasePath(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "prompt_conditioned_table.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        def tex_value(value, half_width):
            half_width = 0.0 if half_width is None else half_width
            return f"${value:.4f} \\pm {half_width:.2f}$"

        tex = [
            "\\begin{tabular}{lrrr}",
            "\\hline",
            "Method & Mean Z & Mean Reward & Eff N \\\\",
            "\\hline",
        ]
        for row in rows:
            alpha = row["alpha"]
            label = f"LCB ($\\alpha={alpha}$)" if isinstance(alpha, (int, float)) else ("Bo2" if alpha == "Bo2 (unguided)" else alpha)
            values = [tex_value(row[key], row[ci_key]) for key, ci_key in
                      [("mean_z", "mean_z_ci"), ("mean_reward", "reward_ci")]]
            values.append(f'${row["eff_n"]:.4f}$')
            tex.append(" & ".join([label] + values) + " \\\\")
        tex.extend(["\\hline", "\\end{tabular}"])
        tex_path = output_dir / "prompt_conditioned_table.tex"
        tex_path.write_text("\n".join(tex) + "\n")
        print(f"Saved {csv_path}\nSaved {tex_path}")
