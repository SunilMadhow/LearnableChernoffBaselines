from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, _ROOT)

_FIGURES = _Path(_ROOT) / "figures" / "regenerated"
_FIGURES.mkdir(parents=True, exist_ok=True)

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RUN_ID      = "20260205_195651_634076"
REWARD_TEMP = 0.1
DATA_DIR    = f"{_ROOT}/data_runs"

SMC_DIR  = os.path.join(DATA_DIR, RUN_ID, "SMC")
BASE_DIR = os.path.join(DATA_DIR, RUN_ID, "inference_results")
REF_DIR  = os.path.join(DATA_DIR, "20251224_233551_699407", "results")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def distinct_4(texts):
    total, unique = 0, set()
    for text in texts:
        words = text.lower().split()
        if len(words) < 4:
            continue
        for fg in zip(words[:-3], words[1:-2], words[2:-1], words[3:]):
            unique.add(fg)
            total += 1
    return len(unique) / total if total > 0 else 0.0


def within_run_diversity(samples):
    by_run = {}
    for s in samples:
        by_run.setdefault(s["run_idx"], []).append(s["text"])
    return np.mean([distinct_4(texts) for texts in by_run.values()])


def load_json(fpath):
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def bon_on_samples(samples, N):
    trim = len(samples) - (len(samples) % N)
    if trim == 0:
        return None
    kept = []
    for i in range(0, trim, N):
        group = samples[i:i + N]
        kept.append(max(group, key=lambda s: s["score"]))
    return kept


# -----------------------------------------------------------------------------
# Load SMC data by scanning directory
# -----------------------------------------------------------------------------
_pat = re.compile(
    rf"smc_results_temp_{re.escape(str(REWARD_TEMP))}_K_(\d+)_ess_(\d+)\.json"
)

plot_data = {}   # K -> list of (ess/K, d4, perfect_rate_pct)

for fname in sorted(os.listdir(SMC_DIR)):
    m = _pat.fullmatch(fname)
    if not m:
        continue
    K, ess = int(m.group(1)), int(m.group(2))
    data = load_json(os.path.join(SMC_DIR, fname))
    if data is None:
        continue
    samples = data.get("samples", [])
    if not samples:
        continue
    d4 = within_run_diversity(samples)
    pr = data["stats"]["success_rate_perfect"] * 100
    plot_data.setdefault(K, []).append((ess / K, d4, pr))

all_K = sorted(plot_data)
for K in all_K:
    plot_data[K].sort()

PALETTE = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_K)))
K_COLOR = {K: PALETTE[i] for i, K in enumerate(all_K)}

# -----------------------------------------------------------------------------
# Load reference baselines
# -----------------------------------------------------------------------------
tag = str(REWARD_TEMP).replace(".", "p")

lcb_data = load_json(os.path.join(
    BASE_DIR, f"lcb_lambda_only_SYM_post_trained_temp_{tag}_inference_results.json"
))
rs_data = load_json(os.path.join(
    REF_DIR, f"rejection_sampling_results_temp_{REWARD_TEMP}.json"
))

lcb_samples = lcb_data.get("samples", []) if lcb_data else []
lcb_d4 = distinct_4([s["text"] for s in lcb_samples]) if lcb_samples else None
lcb_pr = lcb_data["perfect_rate"] * 100               if lcb_data    else None

lcb_bon = {}
for N in (2, 3):
    kept = bon_on_samples(lcb_samples, N)
    if kept:
        lcb_bon[N] = (
            distinct_4([s["text"] for s in kept]),
            np.mean([s["score"] == 1.0 for s in kept]) * 100,
        )

rs_texts = [s["text"] for s in rs_data.get("samples", [])] if rs_data else None
rs_d4    = distinct_4(rs_texts)                              if rs_texts else None
rs_pr    = rs_data["stats"]["success_rate_perfect"] * 100   if rs_data  else None

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))

for K in all_K:
    pts = plot_data[K]
    xs  = np.array([p[1] for p in pts])   # distinct-4
    ys  = np.array([p[2] for p in pts])   # perfect rate %
    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
            color=K_COLOR[K], label=f"K={K}")

if lcb_d4 is not None:
    ax.scatter(lcb_d4, lcb_pr, marker="*", s=150, color="#E41A1C",
               zorder=5, label="LCB", edgecolors="black", linewidths=0.5)
for N, (d4, pr) in lcb_bon.items():
    ax.scatter(d4, pr, marker="*", s=150, color="#E41A1C",
               zorder=5, label=f"LCB+Bo{N}", edgecolors="black", linewidths=0.5,
               alpha=0.5 + 0.15 * N)
if rs_d4 is not None:
    ax.scatter(rs_d4, rs_pr, marker="*", s=150, color="#666666",
               zorder=5, label="RS oracle", edgecolors="black", linewidths=0.5)

ax.set_xlabel("Mean within-run Distinct-4")
ax.set_ylabel("Perfect rate (%)")
ax.set_title(f"Quality–Diversity frontier  (α={REWARD_TEMP})")
ax.legend(fontsize=7, framealpha=0.85, edgecolor="0.8",
          borderpad=0.5, labelspacing=0.25, handlelength=1.4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(str(_FIGURES), f"diversity_vs_perfrate_temp_{REWARD_TEMP}.pdf")
plt.savefig(save_path, bbox_inches="tight")
print(f"Saved to {save_path}")
