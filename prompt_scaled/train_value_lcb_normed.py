"""
train_value_lcb_normed.py

Same as train_value_lcb.py but uses the per-stem-normalised value function
(value_fn_normed.pt) and saves LCB params to value_lcb_normed_params.pt.

Because the normed value function predicts z-scores (mean-0/std-1 within each
stem), the value differences V(x_t) - V(x_{tp1}) are also on the z-score scale.
The Chernoff tau fitting is unchanged in structure; the resulting taus are
interpretable on the same normalised scale.

Usage (Colab):
    !python /content/drive/MyDrive/LCB_cam_ready/prompt_scaled/train_value_lcb_normed.py
"""

from pathlib import Path as _ReleasePath
_LCB_RELEASE_ROOT = str(_ReleasePath(__file__).resolve().parents[1])


import argparse
import json
import math
import os
import sys
from glob import glob

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_value_fn import PromptConditionedValueFn   # same architecture

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CRITIC_MODEL   = "distilroberta-base"
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_MASK_ID  = 126336

NUM_SAVE_STEPS = 32   # t=1..32 saved; tp1 ranges 2..32 for differences
MAX_SEQ_LEN    = 128
DELTA          = 0.1
LAM_MIN        = 0.01
LAM_MAX        = 20.0
GSS_ITERS      = 50   # golden-section search iterations

BASE_DIR   = f"{_LCB_RELEASE_ROOT}"
DATA_DIR   = os.path.join(BASE_DIR, "prompt_scaled", "trajectory_data")
CKPT_PATH  = os.path.join(BASE_DIR, "prompt_scaled", "value_fn_normed.pt")
META_PATH  = os.path.join(BASE_DIR, "prompt_scaled", "value_fn_normed_meta.json")
PARAMS_OUT = os.path.join(BASE_DIR, "prompt_scaled", "value_lcb_normed_params.pt")
EVAL_BATCH = 256  # batch size for value function forward passes
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Chernoff utilities (mirrors RewardTauLCB in single_baseline_sampler.py)
# -------------------------------------------------------------------------

def _lcb_objective(s: torch.Tensor, lam: float, delta: float) -> float:
    """
    tau(lam) = (1/lam) log E[exp(lam * s)] + log(1/delta) / lam
    We minimise over lam to get the tightest bound.
    """
    m1 = lam * s.max()
    log_mgf = m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))
    return (log_mgf / lam + math.log(1.0 / delta) / lam).item()


def _golden_section_search(s: torch.Tensor, delta: float,
                            lam_min: float = LAM_MIN, lam_max: float = LAM_MAX,
                            max_iter: int = GSS_ITERS) -> float:
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lam_min, lam_max
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    for _ in range(max_iter):
        if abs(b - a) < 1e-4:
            break
        if _lcb_objective(s, c, delta) < _lcb_objective(s, d, delta):
            b = d; d = c; c = b - phi * (b - a)
        else:
            a = c; c = d; d = a + phi * (b - a)
    return (a + b) / 2.0


def fit_tau(s: np.ndarray, delta: float) -> dict:
    """
    Fit lambda and tau for a 1-D array of centered advantages s.
    Returns dict with lam, tau, mu.
    """
    mu = float(s.mean())
    s_centered = s - mu
    s_t = torch.tensor(s_centered, dtype=torch.float64)
    lam = _golden_section_search(s_t.float(), delta)
    tau = _lcb_objective(s_t.float(), lam, delta)
    return {"mu": mu, "lam": lam, "tau": tau}


# -------------------------------------------------------------------------
# Value function forward pass helper
# -------------------------------------------------------------------------

def encode_batch(prompt_texts, gen_states, t_scalar, critic_tok, llada_tok,
                 mask_token, max_seq_len, device):
    texts = []
    for pt, gs in zip(prompt_texts, gen_states):
        gen_decoded = llada_tok.decode(gs.tolist(), skip_special_tokens=False)
        gen_decoded = (gen_decoded
                       .replace("<|mdm_mask|>", mask_token)
                       .replace("<|endoftext|>", "")
                       .replace("<|begin_of_text|>", "")
                       .strip())
        texts.append(pt + " " + gen_decoded)
    enc = critic_tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return enc.input_ids.to(device)


@torch.no_grad()
def predict_values(model, input_ids, t_scalar, device):
    t_vec = torch.full((input_ids.shape[0],), t_scalar, dtype=torch.float32, device=device)
    return model(input_ids, t_vec).cpu().numpy()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_steps", nargs="+", type=int, default=None,
        help="tp1 values (1-indexed) to fit taus for. Default: all with sufficient data.",
    )
    parser.add_argument("--delta", type=float, default=DELTA,
                        help="Chernoff confidence level (default: 0.1).")
    parser.add_argument("--base_name", type=str, default="value_lcb_normed_params",
                        help="Base name for output file. Output: {base_name}_d{delta}.pt")
    args = parser.parse_args()
    fit_tp1s = set(args.train_steps) if args.train_steps else set(range(1, NUM_SAVE_STEPS + 1))

    delta     = args.delta
    delta_str = f"{delta:.1f}".replace(".", "p")
    params_out = os.path.join(BASE_DIR, "prompt_scaled", f"{args.base_name}_d{delta_str}.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  delta={delta}  params_out={params_out}")
    print(f"Fitting taus for tp1 in: {sorted(fit_tp1s)}")

    # Load tokenizers
    print("Loading tokenizers...")
    llada_tok  = AutoTokenizer.from_pretrained(LLADA_MODEL_ID, trust_remote_code=True)
    critic_tok = AutoTokenizer.from_pretrained(CRITIC_MODEL)
    mask_token = critic_tok.mask_token or "[MASK]"

    # Load normed value function
    print(f"Loading normed value function from {CKPT_PATH}...")
    with open(META_PATH) as f:
        vf_meta = json.load(f)
    value_fn = PromptConditionedValueFn(vf_meta["critic_model"]).to(device)
    value_fn.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    value_fn.eval()

    # Load all trajectory batches
    meta_files = sorted(glob(os.path.join(DATA_DIR, "*_meta.json")))
    if not meta_files:
        raise FileNotFoundError(f"No trajectory data found in {DATA_DIR}")
    print(f"Found {len(meta_files)} batches in {DATA_DIR}")

    # advantages[tp1] = list of float arrays  (s = V(x_t) - V(x_{tp1}))
    # tp1=1 uses r(x_0) - V(x_1) instead of a value difference.
    advantages = {tp1: [] for tp1 in range(1, NUM_SAVE_STEPS + 1)}

    for batch_num, meta_path in enumerate(meta_files):
        batch_id = os.path.basename(meta_path).replace("_meta.json", "")
        states_path     = os.path.join(DATA_DIR, f"{batch_id}_states.npy")
        prompt_ids_path = os.path.join(DATA_DIR, f"{batch_id}_prompt_ids.npy")
        rewards_path    = os.path.join(DATA_DIR, f"{batch_id}_rewards.npy")

        if not (os.path.exists(states_path) and os.path.exists(rewards_path)):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        states     = np.load(states_path)      # (NUM_SAVE_STEPS, S, L_gen)
        prompt_ids = np.load(prompt_ids_path)  # (1, L_prompt)
        rewards    = np.load(rewards_path)     # (S,)  r(x_0)
        n_steps, S, _ = states.shape
        prompt_text = meta["prompt_text"]

        # Compute V(x_{t_idx}) for all t_idx in 0..n_steps-1.
        # Trajectory ordering:
        #   t_idx=0 → states[0] = t=1 (least masked, t_scalar=1.0)
        #   t_idx=k → t=k+1, t_scalar=(n_steps-k)/NUM_SAVE_STEPS
        values = np.zeros((n_steps, S), dtype=np.float32)
        for t_idx in range(n_steps):
            t_scalar = float(n_steps - t_idx) / n_steps  # robust to any n_steps
            for start in range(0, S, EVAL_BATCH):
                end = min(start + EVAL_BATCH, S)
                batch_states = states[t_idx, start:end]
                ids = encode_batch(
                    [prompt_text] * (end - start),
                    batch_states,
                    t_scalar, critic_tok, llada_tok, mask_token,
                    MAX_SEQ_LEN, device,
                )
                vals = predict_values(value_fn, ids, t_scalar, device)
                values[t_idx, start:end] = vals

        # tp1=1: final step x_1 → x_0.  Advantage = r_normed(x_0) - V(x_1).
        # V(x_1) predicts z-scores; r(x_0) must be normalized the same way.
        mu_r    = float(rewards.mean())
        sigma_r = float(rewards.std())
        sigma_r = max(sigma_r, 0.1)   # same MIN_SIGMA floor as training
        normed_r = (rewards - mu_r) / sigma_r   # (S,)
        adv_1 = normed_r - values[0]   # (S,)  both on z-score scale
        advantages[1].append(adv_1)

        # tp1 in [2..NUM_SAVE_STEPS]: consecutive value differences V(x_t) - V(x_{tp1}).
        # t_idx for t=k is k-1 (t_idx=0 → t=1, t_idx=k → t=k+1).
        #   idx for t=tp1-1 (cleaner)  = tp1-2
        #   idx for t=tp1   (noisier)  = tp1-1
        for tp1 in range(2, NUM_SAVE_STEPS + 1):
            idx_t   = tp1 - 2   # t_idx for t=tp1-1 (less noisy / cleaner)
            idx_tp1 = tp1 - 1   # t_idx for t=tp1   (more noisy)
            if idx_t < 0 or idx_tp1 >= n_steps:
                continue
            adv = values[idx_t] - values[idx_tp1]   # (S,)  V(x_t) - V(x_{tp1})
            advantages[tp1].append(adv)

        if (batch_num + 1) % 20 == 0:
            print(f"  Processed {batch_num + 1}/{len(meta_files)} batches...")

    print("\nFitting Chernoff bounds per timestep...")
    results = {}
    for tp1 in range(1, NUM_SAVE_STEPS + 1):
        if tp1 not in fit_tp1s:
            continue
        adv_all = np.concatenate(advantages[tp1], axis=0) if advantages[tp1] else np.array([])
        if len(adv_all) < 10:
            print(f"  tp1={tp1}: insufficient data ({len(adv_all)} samples), skipping")
            continue
        fit = fit_tau(adv_all, delta)
        results[tp1] = fit
        print(
            f"  tp1={tp1:2d}  n={len(adv_all):5d}  mu={fit['mu']:+.4f}  "
            f"lam={fit['lam']:.4f}  tau={fit['tau']:+.4f}"
        )

    torch.save(results, params_out)
    print(f"\nSaved normed LCB params to {params_out}")
    print("Keys: " + ", ".join(f"tp1={k}" for k in sorted(results)))

