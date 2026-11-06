from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, _ROOT)

_FIGURES = _Path(_ROOT) / "figures" / "regenerated"
_FIGURES.mkdir(parents=True, exist_ok=True)

import re
import json
import numpy as np
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt
from pathlib import Path
from wordfreq import zipf_frequency


LCB_DIR   = Path(f"{_ROOT}/data_runs/20260205_195651_634076/inference_results")
OTHER_DIR = Path(f"{_ROOT}/data_runs/20251224_233551_699407/results")

# Where to write outputs (choose whichever makes sense for you)
OUT_BASE_DIR = _FIGURES

EXPECTED_CANONICAL = {
    "rs_0.1": "rejection_sampling_results_temp_0.1.json",
    "rs_0.3": "rejection_sampling_results_temp_0.3.json",
    "rs_0.5": "rejection_sampling_results_temp_0.5.json",
    "lcb_0.1": "lcb_lambda_only_SYM_post_trained_temp_0p1_inference_results.json",
    "lcb_0.3": "lcb_lambda_only_SYM_post_trained_temp_0p3_inference_results.json",
    "lcb_0.5": "lcb_lambda_only_SYM_post_trained_temp_0p5_inference_results.json",
    "bo2": "best_of_2_results.json",
    "bo3": "best_of_3_results.json",
}

BAR_ORDER = ["rs_0.5","lcb_0.5","rs_0.3","lcb_0.3","rs_0.1","lcb_0.1","bo2","bo3"]
# Bootstrap
N_BOOT = 2000
CI = 0.95
RNG = np.random.default_rng(0)

OUT_DIR = OUT_BASE_DIR / "quality_barplots_3metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1) Load files (ONLY expected; tolerate Drive "(1)" suffixes)
#     - LCB labels resolved from LCB_DIR
#     - others resolved from OTHER_DIR
# =========================
VERSION_SUFFIX_RE = re.compile(r"\s*\(\d+\)(?=\.json$)", re.IGNORECASE)

def canonicalize(name: str) -> str:
    return VERSION_SUFFIX_RE.sub("", name)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_canon_map(parent: Path) -> dict[str, list[Path]]:
    parent = Path(parent)
    if not parent.exists():
        print(f"WARNING: directory does not exist: {parent}")
        return {}
    canon_map: dict[str, list[Path]] = {}
    for p in parent.glob("*.json"):
        canon_map.setdefault(canonicalize(p.name), []).append(p)
    return canon_map

def is_lcb_label(lbl: str) -> bool:
    return lbl.startswith("lcb_")

# Build separate maps
canon_map_lcb   = build_canon_map(LCB_DIR)
canon_map_other = build_canon_map(OTHER_DIR)

FILES: dict[str, Path] = {}

for label, canon_name in EXPECTED_CANONICAL.items():
    cmap = canon_map_lcb if is_lcb_label(label) else canon_map_other
    matches = cmap.get(canon_name, [])
    if not matches:
        search_dir = LCB_DIR if is_lcb_label(label) else OTHER_DIR
        print(f"WARNING: missing {canon_name} for {label} in {search_dir}")
        continue

    # Choose newest modified (handles duplicates / "(1)" copies)
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    FILES[label] = matches[0]

print("Resolved paths:")
for k in BAR_ORDER:
    print(f"  {k}: {FILES.get(k, '(missing)')}")

datasets = {k: load_json(p) for k, p in FILES.items()}


# =========================
# 2) Text extraction + metric helpers
# =========================
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")

def extract_text(sample) -> str:
    if sample is None:
        return ""
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        for k in ("text", "response", "completion", "output", "decoded", "final"):
            if k in sample and isinstance(sample[k], str):
                return sample[k]
        msg = sample.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""

def has_non_latin_script(text: str) -> int:
    """Binary: any non-Latin Unicode script char present."""
    for ch in text:
        if ch.isspace():
            continue
        if ord(ch) <= 127:
            continue  # ASCII treated as Latin-ish
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            continue
        return 1
    return 0

def sentence_count(text: str) -> int:
    s = (text or "").strip()
    if not s:
        return 0
    parts = [p for p in SENT_SPLIT_RE.split(s) if p.strip()]
    return max(1, len(parts))

# ---- dictionary-based "standard word" metric (wordfreq) ----
DICT_WORD_RE = re.compile(r"\b[\w'-]+\b")

ZIPF_THRESHOLD = 2.0
# 2.0 ≈ appears in ~1 per 100M words; tighten to 3.0 to be stricter

def normalize_word(w: str) -> str:
    return w.lower().strip("-'")

def is_standard_word(w: str) -> bool:
    w = normalize_word(w)
    if not w:
        return True

    if zipf_frequency(w, "en") >= ZIPF_THRESHOLD:
        return True

    # Simple plural
    if w.endswith("s") and zipf_frequency(w[:-1], "en") >= ZIPF_THRESHOLD:
        return True

    # Possessive
    if w.endswith("'s") and zipf_frequency(w[:-2], "en") >= ZIPF_THRESHOLD:
        return True

    return False

def has_nonstandard_word(text: str) -> int:
    """
    Dictionary-based definition.
    Flags sample if ANY alphabetic token is not a standard word.
    """
    for token in DICT_WORD_RE.findall(text or ""):
        if any(c.isalpha() for c in token):
            if not is_standard_word(token):
                return 1
    return 0

def infer_method_temp(label: str, obj: dict):
    if label.startswith(("rs_", "lcb_")):
        m, t = label.split("_")
        return m, float(t)
    return label, float("nan")  # best-of

# =========================
# 3) Per-sample metrics table
# =========================
rows = []
for label, obj in datasets.items():
    if not isinstance(obj, dict):
        raise ValueError(f"{label}: expected dict, got {type(obj)}")
    samples = obj.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{label}: expected top-level 'samples' list, got {type(samples)}")

    method, temp = infer_method_temp(label, obj)

    for i, s in enumerate(samples):
        text = extract_text(s)
        rows.append({
            "label": label,
            "method": method,
            "temp": temp,
            "sample_idx": i,
            "has_non_latin": has_non_latin_script(text),
            "n_sentences": sentence_count(text),
            "has_nonstandard_word": has_nonstandard_word(text),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "per_sample_metrics.csv", index=False)
print(f"Wrote: {OUT_DIR/'per_sample_metrics.csv'}")

# =========================
# 4) Bootstrap summaries with CIs
# =========================
def bootstrap_mean_ci(x: np.ndarray, n_boot=N_BOOT, ci=CI):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo = np.quantile(boots, alpha)
    hi = np.quantile(boots, 1 - alpha)
    return float(x.mean()), float(lo), float(hi)

METRICS = [
    ("Non-Latin rate", "has_non_latin"),
    ("Avg # sentences", "n_sentences"),
    ("Nonstandard-word rate", "has_nonstandard_word"),
]

summary_rows = []
for label, g in df.groupby("label", sort=False):
    for pretty, col in METRICS:
        mean, lo, hi = bootstrap_mean_ci(g[col].values)
        summary_rows.append({
            "label": label,
            "metric": pretty,
            "mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(g),
        })

summ = pd.DataFrame(summary_rows)
summ.to_csv(OUT_DIR / "summary_with_ci.csv", index=False)
print(f"Wrote: {OUT_DIR/'summary_with_ci.csv'}")

# =========================
# 5) Plot helpers
# =========================
def pretty_label(lbl: str) -> str:
    if lbl.startswith(("rs_", "lcb_")):
        m, t = lbl.split("_")
        return f"{m}{t}"
    if lbl in ("bo2", "bo3"):
        return lbl
    return lbl

# Only include labels that exist, in desired order
labels = [l for l in BAR_ORDER if l in df["label"].unique()]
y = np.arange(len(labels))

# Distinct color per bar (no manual colors beyond using a colormap)
# tab10 has 10 distinct colors; we only need up to 8 bars
BAR_COLORS = plt.cm.tab10(np.linspace(0, 1, len(labels)))

# =========================
# 6) Separate HORIZONTAL barplots (one per metric) + one combined figure
# =========================
# --- individual plots (horizontal, each bar different color) ---
for metric_name, _ in METRICS:
    sub = summ[summ["metric"] == metric_name].set_index("label").reindex(labels)
    means = sub["mean"].values
    xerr = np.vstack([means - sub["ci_lo"].values, sub["ci_hi"].values - means])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(y, means, xerr=xerr, capsize=4, color=BAR_COLORS)

    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(l) for l in labels])
    ax.invert_yaxis()  # show rs0.5 at top
    ax.set_title(metric_name)
    ax.set_xlabel("mean ± 95% bootstrap CI")

    plt.tight_layout()
    out = OUT_DIR / f"barh_{metric_name.replace('#','num').replace(' ','_').replace('-','_').lower()}.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"Wrote: {out}")

# --- combined figure with 3 horizontal subplots (one per metric) ---
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 11), sharey=True)

for ax, (metric_name, _) in zip(axes, METRICS):
    sub = summ[summ["metric"] == metric_name].set_index("label").reindex(labels)
    means = sub["mean"].values
    xerr = np.vstack([means - sub["ci_lo"].values, sub["ci_hi"].values - means])

    ax.barh(y, means, xerr=xerr, capsize=4, color=BAR_COLORS)
    ax.set_title(metric_name)
    ax.set_xlabel("mean ± 95% bootstrap CI")
    ax.invert_yaxis()

axes[-1].set_yticks(y)
axes[-1].set_yticklabels([pretty_label(l) for l in labels])

plt.tight_layout()
combined_path = OUT_DIR / "combined_3metric_barh.png"
plt.savefig(combined_path)
plt.close(fig)
print(f"Wrote: {combined_path}")

print("\nDone.")


# =========================
# 6) ONE FIGURE: 3 subplots horizontally (one per metric)
#    - vertical bars
#    - grouped order as before
#    - shared colors by temperature; bo2/bo3 get their own colors
# =========================

def pretty_label(lbl: str) -> str:
    if lbl.startswith(("rs_", "lcb_")):
        m, t = lbl.split("_")
        return f"{m}{t}"
    return lbl

# Only include labels that exist
labels = [l for l in BAR_ORDER if l in df["label"].unique()]
x = np.arange(len(labels))

# ---- Color mapping: same color per temperature across rs/lcb, plus bo2/bo3 colors
# Use tab10 for distinct, readable defaults
palette = list(plt.cm.tab10.colors)

TEMP_COLOR = {
    0.5: palette[0],
    0.3: palette[1],
    0.1: palette[2],
}
BO_COLOR = {
    "bo2": palette[3],
    "bo3": palette[4],
}

def label_color(lbl: str):
    if lbl in BO_COLOR:
        return BO_COLOR[lbl]
    if lbl.startswith(("rs_", "lcb_")):
        _, t = lbl.split("_")
        t = float(t)
        return TEMP_COLOR.get(t, palette[5])
    return palette[6]

bar_colors = [label_color(l) for l in labels]

# ---- Create a 1x3 layout (horizontal lineup)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4.5), sharex=True)

for ax, (metric_name, _) in zip(axes, METRICS):
    sub = summ[summ["metric"] == metric_name].set_index("label").reindex(labels)

    means = sub["mean"].values
    yerr = np.vstack([means - sub["ci_lo"].values, sub["ci_hi"].values - means])

    ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=4,
        color=bar_colors
    )

    ax.set_title(metric_name)
    ax.set_ylabel("mean ± 95% bootstrap CI")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(l) for l in labels], rotation=45, ha="right")

# Optional: a shared legend explaining colors (by temperature + bo2/bo3)
from matplotlib.patches import Patch
legend_handles = [
    Patch(color=TEMP_COLOR[0.5], label="temp 0.5 (rs+lcb)"),
    Patch(color=TEMP_COLOR[0.3], label="temp 0.3 (rs+lcb)"),
    Patch(color=TEMP_COLOR[0.1], label="temp 0.1 (rs+lcb)"),
]
if "bo2" in labels:
    legend_handles.append(Patch(color=BO_COLOR["bo2"], label="bo2"))
if "bo3" in labels:
    legend_handles.append(Patch(color=BO_COLOR["bo3"], label="bo3"))

# Put legend once for the whole figure
fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.88])  # leave space for legend
combined_path = OUT_DIR / "combined_3metric_barplots_horizontal.png"
plt.savefig(combined_path)
plt.close(fig)

print(f"Wrote: {combined_path}")
