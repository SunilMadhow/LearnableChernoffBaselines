from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import gc
import torch
import sys
import os
import numpy as np
import json
import re
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# Ensure paths are correct
project_path = f'{_ROOT}'
if os.path.exists(project_path) and project_path not in sys.path:
    sys.path.append(project_path)

from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore, ModelConfig
from util import *


# -----------------------------------------------------------------------------
# SMC Guidance Loop (Algorithm 4)
# -----------------------------------------------------------------------------
def run_smc(
    core,
    value_estimator,
    ess_threshold=0.5,
    log_proposal_equals_model=True,
):
    """
    SMC-Based Guidance following Algorithm 4.

    Temperature (alpha) is NOT passed here — set value_estimator.reward_temperature
    before calling this function and the estimator handles it internally.

    Args:
        core:                      LLaDACore instance (wraps the pre-trained model)
        value_estimator:           Trained ValueClassifier with predict_under_log().
                                   reward_temperature must already be set on this object.
        ess_threshold_ratio:       Resample when ESS / K falls below this fraction. (0.5 is standard)
        log_proposal_equals_model: If True, the proposal q IS the pre-trained model p_pre,
                                   so the p_pre / q ratio is exactly 1 and we skip that computation.
                                   Set False if core exposes log-probabilities for a different q.

    Returns:
        core.X:          Final token sequences, shape (K, gen_len)
        log_weights:     Final unnormalised log-weights, shape (K,)
        resample_steps:  List of timestep indices where resampling was triggered
    """
    core.reset()
    K = core.S                          # number of particles

    # --- Initialise particles and weights ---
    # Weights are maintained in log-space for numerical stability.
    log_weights = np.zeros(K, dtype=np.float64)   # w^[i] = 1 for all i at t=T

    # Value at the *current* state x_t, needed for the denominator each step.
    # At t=T the sequence is fully masked, so v̂_T is evaluated once here.
    current_t_idx = core.model_config["index_set"][core.t]
    log_v_current = value_estimator.predict_under_log(core.X, current_t_idx)  # (K,)

    resample_steps = []

    with tqdm(total=core.T, desc="SMC") as pbar:
        while core.t > 0:
            current_t     = core.t
            prev_t_idx    = core.model_config["index_set"][current_t - 1]

            # ------------------------------------------------------------------
            # Step 3 – Importance Sampling: x_{t-1}^[i] ~ q_{t-1}(·| x_t^[i])
            # core.propose() draws N proposals per particle; we take the 0th.
            # ------------------------------------------------------------------
            all_proposals = core.propose()          # (S, N, gen_len)
            x_prev        = all_proposals[:, 0, :]  # (S, gen_len)

            # ------------------------------------------------------------------
            # Step 4 – Weight Update
            #
            #          p_pre(x_{t-1} | x_t)   exp(v̂_{t-1}(x_{t-1}) / α)
            # w ← w · ─────────────────────── · ──────────────────────────
            #          q(x_{t-1}   | x_t)       exp(v̂_t(x_t)       / α)
            #
            # In log-space:
            #   log_w += log_p_pre - log_q + (log_v_{t-1} - log_v_t) / α
            #
            # When q == p_pre the first ratio is 0 and we skip it.
            # ------------------------------------------------------------------
            log_v_next = value_estimator.predict_under_log(x_prev, prev_t_idx)  # (K,)

            # Value ratio in log-space. Temperature is already baked into
            # value_estimator.reward_temperature, so no manual /alpha needed.
            log_value_ratio = log_v_next - log_v_current

            if log_proposal_equals_model:
                # p_pre / q = 1  =>  log ratio = 0
                log_model_ratio = np.zeros(K, dtype=np.float64)
            else:
                # If core exposes per-sample log-probabilities for p_pre and q,
                # compute them here and subtract.  Replace with real calls as needed.
                log_p_pre = core.log_prob_model(x_prev)    # (K,) — implement in LLaDACore
                log_q     = core.log_prob_proposal(x_prev)  # (K,)
                log_model_ratio = log_p_pre - log_q

            log_weights = log_weights + log_model_ratio + log_value_ratio

            # ------------------------------------------------------------------
            # Step 5 – Resampling
            # ESS = (Σ w)² / Σ(w²)  computed in log-space via the log-sum-exp trick.
            # ------------------------------------------------------------------
            log_w_norm = log_weights - logsumexp(log_weights)
            w_norm     = np.exp(log_w_norm)          # normalised weights, sum = 1
            ess        = (np.sum(w_norm)**2) / np.sum(w_norm ** 2)  # ESS when weights sum to 1

            resampled = False
            print("ESS = ", ess)
            if ess < ess_threshold:
                ancestors   = np.random.choice(K, size=K, replace=True, p=w_norm)
                x_prev      = x_prev[ancestors]
                log_weights = np.zeros(K, dtype=np.float64)
                resample_steps.append(current_t)
                resampled   = True

            # Commit x_{t-1} to the core state (advances t by -1)
            core.commit(x_prev)

            # Carry forward the value of the committed state as the new "current".
            # After resampling, index through ancestors to stay aligned.
            log_v_current = log_v_next[ancestors] if resampled else log_v_next

            pbar.set_postfix({
                "ESS": f"{ess:.0f}/{K}",
                "Resamples": len(resample_steps),
                "logW_std": f"{log_weights.std():.2f}",
            })
            pbar.update(1)

    return core.X, log_weights, resample_steps


def logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = log_x.max()
    return c + np.log(np.sum(np.exp(log_x - c)))


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",        required=True)
    parser.add_argument("--reward_temp",   type=float, required=True,
                        help="Value estimator temperature (α). Set on the estimator directly.")
    parser.add_argument("--num_samples",   type=int,   default=100,
                        help="Total number of output samples to collect.")
    parser.add_argument("--num_particles", type=int,   default=250,
                        help="Particles per SMC run (K). Independent runs are repeated until "
                             "num_samples is reached.")
    parser.add_argument("--ess_threshold",     type=int,   default=-1,
                        help="Absolute ESS count; resample when ESS drops below this. Default -1 → K//2.")
    parser.add_argument("--save_name",     type=str,   default=None)
    args = parser.parse_args()

    data_dir      = f"{_ROOT}/data_runs"
    run_dir       = os.path.join(data_dir, str(args.run_id))
    TOTAL_SAMPLES = args.num_samples
    K             = args.num_particles
    ess_threshold = args.ess_threshold if args.ess_threshold > 0 else K // 2
    num_batches   = max(-(-TOTAL_SAMPLES // K), 1)   # ceil division

    smc_dir     = os.path.join(run_dir, "SMC")
    os.makedirs(smc_dir, exist_ok=True)
    output_file = os.path.join(
        smc_dir,
        args.save_name or f"smc_results_temp_{args.reward_temp}_K_{K}_ess_{ess_threshold}.json"
    )

    # 1. Value estimator — set temperature once here, never touch it again
    print("Loading Value Classifier...")
    ve = ValueClassifier.load(os.path.join(data_dir, args.run_id))
    ve.reward_temperature = args.reward_temp

    # 2. LLaDA model (4-bit)
    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"
    print("Loading LLaDA (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    model     = AutoModel.from_pretrained(
        hf_model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # 3. Prompt & Core (built once; core.reset() is called inside run_smc each batch)
    prompt_text = (
        "Write a creative story of exactly 3 sentences. "
        "Stylistically, it is preferred that the first word of each sentence "
        "start with 'A', 'B', and 'C' respectively."
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    cfg = ModelConfig()
    cfg.model_name  = "llada"
    cfg.state_shape = (32,)
    cfg.index_set   = list(range(33))
    cfg.deets = {
        "steps":       32,
        "hf_model_id": hf_model_id,
        "mask_id":     126336,
        "temperature": 1.12,
        "cfg_scale":   0.0,
        "remasking":   "low_confidence",
        "prompt_len":  len(prompt_ids),
        "block_len":   8,
    }

    core = LLaDACore(cfg, tokenizer, model, S=K, N=1)
    core.set_prompt(prompt_ids)

    # 4. Collection loop — repeat independent SMC runs until we have enough samples
    all_texts          = []
    all_scores         = []
    all_log_weights    = []
    all_resample_steps = []
    reward_fn          = LLaDAReward(tokenizer)

    print(f"\n=== Starting SMC Collection: {TOTAL_SAMPLES} samples, K={K}, {num_batches} run(s) ===")

    for i in range(num_batches):
        print(f"\nRun {i+1}/{num_batches}")

        batch_tokens, batch_log_w, resample_steps = run_smc(
            core,
            ve,
            ess_threshold=ess_threshold,
        )

        batch_scores = reward_fn(batch_tokens)
        batch_texts  = tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)

        all_texts.extend(batch_texts)
        all_scores.extend(batch_scores)
        all_log_weights.extend(batch_log_w.tolist())
        all_resample_steps.append(resample_steps)

        print(f"  Run Mean Reward: {np.mean(batch_scores):.4f}")
        print(f"  Run Perfects:    {np.sum(np.array(batch_scores) == 1.0)}/{K}")
        print(f"  Resample steps:  {resample_steps}")

    # Trim to exactly TOTAL_SAMPLES in case K doesn't divide evenly
    all_texts       = all_texts[:TOTAL_SAMPLES]
    all_scores      = all_scores[:TOTAL_SAMPLES]
    all_log_weights = all_log_weights[:TOTAL_SAMPLES]

    # 5. Statistics
    all_scores  = np.array(all_scores)
    n           = len(all_scores)
    all_details = [get_constraint_details(t) for t in all_texts]

    count_perfect = np.sum(all_scores == 1.0)
    count_a       = sum(1 for d in all_details if d['A'])
    count_b       = sum(1 for d in all_details if d['B'])
    count_c       = sum(1 for d in all_details if d['C'])

    a_indices     = [i for i, d in enumerate(all_details) if d['A']]
    b_given_a     = sum(1 for i in a_indices if all_details[i]['B'])
    prob_b_given_a = b_given_a / len(a_indices) if a_indices else 0.0

    ab_indices    = [i for i, d in enumerate(all_details) if d['A'] and d['B']]
    c_given_ab    = sum(1 for i in ab_indices if all_details[i]['C'])
    prob_c_given_ab = c_given_ab / len(ab_indices) if ab_indices else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples":    TOTAL_SAMPLES,
            "num_particles": K,
            "reward_temp":  args.reward_temp,
            "ess_threshold":    ess_threshold,
        },
        "stats": {
            "mean_reward":          float(np.mean(all_scores)),
            "success_rate_perfect": float(count_perfect / n),
            "constraint_A_rate":    float(count_a / n),
            "constraint_B_rate":    float(count_b / n),
            "constraint_C_rate":    float(count_c / n),
            "funnel_stats": {
                "prob_B_given_A":  float(prob_b_given_a),
                "prob_C_given_AB": float(prob_c_given_ab),
            },
        },
        "samples": [
            {"text": txt, "score": float(scr), "log_weight": lw}
            for txt, scr, lw in zip(all_texts, all_scores, all_log_weights)
        ],
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(" SMC RESULTS REPORT ")
    print("=" * 60)
    print(f"Total Samples:   {n}")
    print(f"Mean Reward:     {report['stats']['mean_reward']:.4f}")
    print(f"Perfect Rate:    {report['stats']['success_rate_perfect']:.1%}")
    print("-" * 30)
    print(f"Constraint A:    {report['stats']['constraint_A_rate']:.1%}")
    print(f"Constraint B:    {report['stats']['constraint_B_rate']:.1%}")
    print(f"Constraint C:    {report['stats']['constraint_C_rate']:.1%}")
    print("-" * 30)
    print(f"Retention A->B:  {report['stats']['funnel_stats']['prob_B_given_A']:.1%}")
    print(f"Retention B->C:  {report['stats']['funnel_stats']['prob_C_given_AB']:.1%}")
    print(f"\nFull results saved to: {output_file}")
