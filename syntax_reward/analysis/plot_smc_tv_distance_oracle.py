from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, _ROOT)

_FIGURES = _Path(_ROOT) / "figures" / "regenerated"
_FIGURES.mkdir(parents=True, exist_ok=True)

import os
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--lcb_samples", type=int, default=None,
                    help="Max LCB samples per K (default: M*K matched).")
parser.add_argument("--display", action="store_true",
                    help="Display plot inline in Colab instead of saving to file.")
args = parser.parse_args()

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "figure.dpi":         150,
})

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
RUN_ID      = "20260205_195651_634076"
REWARD_TEMP = 0.1
DATA_DIR    = f"{_ROOT}/data_runs"
K_KEEP      = set(range(2, 9))
M           = 5   # SMC runs per (K, ESS) — used to match LCB sample count

SMC_DIR = os.path.join(DATA_DIR, RUN_ID, "SMC")
LCB_DIR = os.path.join(DATA_DIR, RUN_ID, "inference_results", "sweeps")
REF_DIR = os.path.join(DATA_DIR, "20251224_233551_699407", "results")

BINS = np.array([0.0, round(1/3, 4), round(2/3, 4), 1.0])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_json(fpath):
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        return json.load(f)


def scores_to_pmf(scores):
    counts = np.zeros(len(BINS))
    for s in scores:
        counts[np.argmin(np.abs(BINS - s))] += 1
    n = counts.sum()
    return counts / n if n > 0 else counts


def tv_distance(pmf_a, pmf_b):
    return 0.5 * np.sum(np.abs(pmf_a - pmf_b))


# -----------------------------------------------------------------------------
# RS reference PMF
# -----------------------------------------------------------------------------
rs_data   = load_json(os.path.join(REF_DIR, f"rejection_sampling_results_temp_{REWARD_TEMP}.json"))
rs_scores = [s["score"] for s in rs_data.get("samples", [])]
rs_pmf    = scores_to_pmf(rs_scores)

# -----------------------------------------------------------------------------
# LCB data (loaded once)
# -----------------------------------------------------------------------------
alpha_str = str(REWARD_TEMP).replace(".", "p")
lcb_path  = os.path.join(LCB_DIR, f"lcb_lambda_only_SYM_post_trained_temp_{alpha_str}_del_0p5_inference_results")
if not os.path.exists(lcb_path):
    lcb_path = lcb_path + ".json"
lcb_data   = load_json(lcb_path)
lcb_scores_all = []
lcb_eff_n      = None
if lcb_data:
    lcb_scores_all = [s["score"] for s in lcb_data.get("samples", [])]
    lcb_eff_n      = round(lcb_data.get("avg_nfe", 0) / 32, 1)

# -----------------------------------------------------------------------------
# SMC: best ESS per K (direct PMF over all M*K samples)
# -----------------------------------------------------------------------------
_pat = re.compile(
    rf"smc_results_temp_{re.escape(str(REWARD_TEMP))}_K_(\d+)_ess_(\d+)\.json"
)

best_tv  = {}   # K -> lowest TV across ESS settings
best_ess = {}   # K -> ESS that achieved lowest TV
n_smc    = {}   # K -> number of SMC samples in best file

for fname in sorted(os.listdir(SMC_DIR)):
    m = _pat.fullmatch(fname)
    if not m:
        continue
    K, ess = int(m.group(1)), int(m.group(2))
    if K not in K_KEEP:
        continue
    data = load_json(os.path.join(SMC_DIR, fname))
    if data is None:
        continue
    samples = data.get("samples", [])
    if not samples:
        continue
    smc_pmf = scores_to_pmf([s["score"] for s in samples])
    tv = tv_distance(smc_pmf, rs_pmf)
    if K not in best_tv or tv < best_tv[K]:
        best_tv[K]  = tv
        best_ess[K] = ess
        n_smc[K]    = len(samples)

# -----------------------------------------------------------------------------
# LCB TV per K (matched M*K samples)
# -----------------------------------------------------------------------------
lcb_tv = {}   # K -> tv distance using M*K samples

for K in sorted(best_tv):
    n_match = args.lcb_samples if args.lcb_samples is not None else M * K
    scores  = lcb_scores_all[:n_match]
    if not scores:
        continue
    lcb_tv[K] = tv_distance(scores_to_pmf(scores), rs_pmf)

# -----------------------------------------------------------------------------
# Grouped bar plot
# -----------------------------------------------------------------------------
sorted_Ks = sorted(best_tv)
n_K       = len(sorted_Ks)

PALETTE  = plt.cm.viridis(np.linspace(0.1, 0.9, n_K))
K_COLOR  = {K: PALETTE[i] for i, K in enumerate(sorted_Ks)}
LCB_RED  = "#E41A1C"

bar_width  = 0.35
x_pos      = np.arange(n_K)

fig, ax = plt.subplots(figsize=(7, 3.2))

smc_bars = ax.bar(
    x_pos - bar_width / 2,
    [best_tv[K] for K in sorted_Ks],
    width=bar_width,
    color=[K_COLOR[K] for K in sorted_Ks],
    edgecolor="white", linewidth=0.5,
    label="SMC (oracle ESS)",
)

lcb_bars = ax.bar(
    x_pos + bar_width / 2,
    [lcb_tv.get(K, 0) for K in sorted_Ks],
    width=bar_width,
    color=LCB_RED,
    edgecolor="white", linewidth=0.5,
    label=f"LCB (eff N={lcb_eff_n})",
)

ax.set_xticks(x_pos)
ax.set_xticklabels([f"K={K}" for K in sorted_Ks])
ax.set_ylabel("TV distance to RS")
ax.set_title("TV Distance Between Induced Reward Distributions (Oracle SMC)")
ax.legend(fontsize=8, framealpha=0.85, edgecolor="0.8",
          borderpad=0.5, labelspacing=0.3, handlelength=1.4)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
if args.display:
    from io import BytesIO
    from IPython.display import Image, display as ipy_display
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    ipy_display(Image(buf.read()))
    plt.close()
else:
    save_path = os.path.join(str(_FIGURES), f"tv_distance_oracle_temp_{REWARD_TEMP}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved to {save_path}")
