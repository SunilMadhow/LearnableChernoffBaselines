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

# -----------------------------------------------------------------------------
# ICML-style rcParams
# -----------------------------------------------------------------------------
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
STEPS       = 32
MAX_K       = 8
DATA_DIR    = f"{_ROOT}/data_runs"
DELTAS      = [0.9, 0.7, 0.5, 0.3]

SMC_DIR = os.path.join(DATA_DIR, RUN_ID, "SMC")
LCB_DIR = os.path.join(DATA_DIR, RUN_ID, "inference_results", "sweeps")
REF_DIR = os.path.join(DATA_DIR, "20251224_233551_699407", "results")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_json(fpath):
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def bon_mean_reward(scores, N):
    scores = np.asarray(scores)
    trim = len(scores) - (len(scores) % N)
    if trim == 0:
        return None
    return float(scores[:trim].reshape(-1, N).max(axis=1).mean())


# -----------------------------------------------------------------------------
# Load SMC data by scanning directory
# -----------------------------------------------------------------------------
_pat = re.compile(
    rf"smc_results_temp_{re.escape(str(REWARD_TEMP))}_K_(\d+)_ess_(\d+)\.json"
)

smc_points = []   # list of (K, mean_reward)
all_K = set()

if os.path.isdir(SMC_DIR):
    for fname in os.listdir(SMC_DIR):
        m = _pat.fullmatch(fname)
        if not m:
            continue
        K = int(m.group(1))
        if K > MAX_K:
            continue
        data = load_json(os.path.join(SMC_DIR, fname))
        if data is None:
            continue
        smc_points.append((K, data["stats"]["mean_reward"]))
        all_K.add(K)

all_K = sorted(all_K)
PALETTE = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_K)))
K_COLOR = {K: PALETTE[i] for i, K in enumerate(all_K)}

# -----------------------------------------------------------------------------
# Load LCB sweep data
# -----------------------------------------------------------------------------
alpha_str  = str(REWARD_TEMP).replace(".", "p")
lcb_points = []

for delta in DELTAS:
    delta_str = str(delta).replace(".", "p")
    stem  = f"lcb_lambda_only_SYM_post_trained_temp_{alpha_str}_del_{delta_str}_inference_results"
    fpath = os.path.join(LCB_DIR, stem)
    if not os.path.exists(fpath):
        fpath = fpath + ".json"
    data = load_json(fpath)
    if data is None:
        continue
    eff_n  = data.get("avg_nfe", 0) / STEPS
    scores = np.array([s["score"] for s in data.get("samples", [])])
    lcb_points.append((eff_n, data["mean_reward"], scores))

lcb_points.sort(key=lambda x: x[0])

lcb_bo2_points, lcb_bo3_points = [], []
for eff_n, _, scores in lcb_points:
    r2 = bon_mean_reward(scores, 2)
    r3 = bon_mean_reward(scores, 3)
    if r2 is not None:
        lcb_bo2_points.append((eff_n * 2, r2))
    if r3 is not None:
        lcb_bo3_points.append((eff_n * 3, r3))

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))

# SMC dots coloured by K
for K, reward in smc_points:
    ax.scatter(K, reward, color=K_COLOR[K], s=30, zorder=4, alpha=0.85,
               edgecolors="none")

# Legend entries for K
for K in all_K:
    ax.scatter([], [], color=K_COLOR[K], s=30, label=f"SMC K={K}")

# LCB lines
if lcb_points:
    lx, ly = zip(*[(p[0], p[1]) for p in lcb_points])
    ax.plot(lx, ly, color="#E41A1C", marker="o", markersize=4,
            linewidth=1.5, zorder=5, label="LCB")

if lcb_bo2_points:
    bx, by = zip(*lcb_bo2_points)
    ax.plot(bx, by, color="#E41A1C", marker="x", markersize=5,
            linewidth=1.5, linestyle="--", zorder=5, label="LCB+Bo2")

if lcb_bo3_points:
    bx, by = zip(*lcb_bo3_points)
    ax.plot(bx, by, color="#E41A1C", marker="+", markersize=6,
            linewidth=1.5, linestyle=":", zorder=5, label="LCB+Bo3")

ax.set_xlabel("Effective $N$ (compute scaling)")
ax.set_ylabel("Mean reward")
ax.set_title(f"Inference scaling  (α={REWARD_TEMP})")
ax.legend(fontsize=7, framealpha=0.85, edgecolor="0.8",
          borderpad=0.5, labelspacing=0.25, handlelength=1.4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(str(_FIGURES), f"scaling_temp_{REWARD_TEMP}.pdf")
plt.savefig(save_path, bbox_inches="tight")
print(f"Saved to {save_path}")
