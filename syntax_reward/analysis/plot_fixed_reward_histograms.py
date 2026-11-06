from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, _ROOT)

_FIGURES = _Path(_ROOT) / "figures" / "regenerated"
_FIGURES.mkdir(parents=True, exist_ok=True)

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os


CUSTOM_TITLES = [
    ["RS Temp 0.5", "LCB Temp 0.5"],
    ["RS Temp 0.3", "LCB Temp 0.3"],
    ["RS Temp 0.1", "LCB Temp 0.1"],
    ["Best of 2", "Unguided"]
]

# --- Configuration ---
root_dir = f"{_ROOT}/data_runs/"  # renamed for clarity

results_dir = os.path.join(root_dir, "20251224_233551_699407",  "results")                 # <<< CHANGED
lcb_inference_dir = os.path.join(root_dir, "20260205_195651_634076", "inference_results") # <<< CHANGED

filename_pairs = [
    ("rejection_sampling_results_temp_0.5.json", "lcb_lambda_only_SYM_post_trained_temp_0p5_inference_results.json"),
    ("rejection_sampling_results_temp_0.3.json", "lcb_lambda_only_SYM_post_trained_temp_0p3_inference_results.json"),
    ("rejection_sampling_results_temp_0.1.json", "lcb_lambda_only_SYM_post_trained_temp_0p1_inference_results.json"),
    ("best_of_2_results.json", "unguided_results_1000.json"),
]

# Reference samples are stored in <run_id>/results
# LCB files come from <run_id>/inference_results
json_pairs = []  # <<< CHANGED
for f1, f2 in filename_pairs:
    path_a = os.path.join(results_dir, f1)  # <<< CHANGED

    is_lcb_inference = f2.startswith("lcb_") and f2.endswith("_inference_results.json")
    path_b = os.path.join(lcb_inference_dir, f2) if is_lcb_inference else os.path.join(results_dir, f2)  # <<< CHANGED

    json_pairs.append((path_a, path_b))


CONFIDENCE = 0.95

# --- Aesthetics ---
EXPECTED_SCORES = [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]
SCORE_LABELS = ["0", "1/3", "2/3", "1"]
BAR_COLORS = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
})

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_gaussian_ci(data, confidence=0.95):
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0.0
    z = stats.norm.ppf((1 + confidence) / 2.)
    margin = z * (std / np.sqrt(n)) if n > 1 else 0.0
    return mean, margin

def get_bernoulli_ci(binary_data, confidence=0.95):
    n = len(binary_data)
    if n == 0:
        return 0.0, 0.0
    p = np.mean(binary_data)
    z = stats.norm.ppf((1 + confidence) / 2.)
    margin = z * np.sqrt(p * (1 - p) / n)
    return p, margin

def get_pmf_counts(scores):
    total = len(scores)
    if total == 0:
        return [0] * len(EXPECTED_SCORES)

    rounded_scores = [round(s, 2) for s in scores]
    rounded_expected = [round(s, 2) for s in EXPECTED_SCORES]

    heights = []
    for ref in rounded_expected:
        count = rounded_scores.count(ref)
        heights.append(count / total)
    return heights

def get_nfe(sample):
    """
    Robustly handle the different formats without changing plotting logic:
    - RS samples may use 'proposals'
    - New LCB inference_results format uses 'num_proposals'
    - Some runs may already use 'num_proposals'
    """
    if "num_proposals" in sample:
        return sample.get("num_proposals", 0)
    if "proposals" in sample:
        return sample.get("proposals", 0)
    # Keep compatibility with your existing special-case for RS/best-of naming
    if "num_proposals" in sample:
        return sample.get("num_proposals", 0)
    return 0

def main():
    row_names = ["Mean Reward", "Avg NFE", "Perfect Rate"]
    table_data = {}

    fig, axes = plt.subplots(
        2, 4,
        figsize=(6.75, 2.65),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace': 0.12, 'hspace': 0.55}
    )

    x_positions = np.arange(len(EXPECTED_SCORES))
    bar_width = 1.0

    for i, (file_a, file_b) in enumerate(json_pairs):
        pair_files = [file_a, file_b]

        for j, filepath in enumerate(pair_files):
            try:
                data = load_json(filepath)
            except Exception:
                print(f"Warning: Could not load {filepath}")
                data = {'samples': []}

            samples = data.get('samples', [])

            scores = [s.get('score', 0) for s in samples]

            # <<< CHANGED: robust NFE extraction (works for new LCB inference_results.json files)
            nfes = [get_nfe(s) for s in samples]

            perfects = [1.0 if s.get('score', 0) == 1.0 else 0.0 for s in samples]

            # --- Stats (unchanged) ---
            rew_mu, rew_ci = get_gaussian_ci(scores, CONFIDENCE)
            rew_str = f"{rew_mu:.3f} ± {rew_ci:.3f}"
            nfe_mu, nfe_ci = get_gaussian_ci(nfes, CONFIDENCE)
            nfe_str = f"{nfe_mu:.1f} ± {nfe_ci:.1f}"
            perf_mu, perf_ci = get_bernoulli_ci(perfects, CONFIDENCE)
            perf_str = f"{perf_mu:.2%} ± {perf_ci:.2%}"

            model_type = 'A' if j == 0 else 'B'
            col_label = f"Pair {i+1} - {model_type}"
            table_data[col_label] = [rew_str, nfe_str, perf_str]

            ax = axes[j, i]
            pmf_heights = get_pmf_counts(scores)

            ax.bar(
                x_positions, pmf_heights,
                width=bar_width,
                color=BAR_COLORS,
                edgecolor='black',
                linewidth=0.45,
                zorder=3
            )

            ax.set_xlim(-0.5, len(EXPECTED_SCORES) - 0.5)

            plot_title = CUSTOM_TITLES[i][j] if (i < len(CUSTOM_TITLES) and j < len(CUSTOM_TITLES[i])) else f"Pair {i+1} - {model_type}"
            ax.set_title(plot_title, fontsize=10, fontname='serif', pad=1.5)

            if i == 0:
                ax.set_ylabel("Prob.", fontsize=9)
            else:
                ax.set_ylabel("")

            if j == 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(SCORE_LABELS, fontsize=8.5)
            else:
                ax.set_xticks([])

            ax.tick_params(axis='y', labelsize=8.5)

    fig.text(0.5, 0.04, "Score", ha='center', va='center', fontsize=9)

    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        top=0.92,
        bottom=0.14
    )

    plt.savefig(_FIGURES / 'comparison_histograms.png', dpi=300, bbox_inches='tight')
    print("Figure saved to comparison_histograms.png")

    # --- Table (unchanged) ---
    df = pd.DataFrame(table_data, index=row_names)
    ordered_columns = []
    for i in range(len(json_pairs)):
        ordered_columns.append(f"Pair {i+1} - A")
    for i in range(len(json_pairs)):
        ordered_columns.append(f"Pair {i+1} - B")
    ordered_columns = [c for c in ordered_columns if c in df.columns]
    df = df[ordered_columns]

    print("\nStatistical Table:")
    print(df)
    df.to_csv(_FIGURES / 'comparison_stats.csv')

if __name__ == "__main__":
    main()
