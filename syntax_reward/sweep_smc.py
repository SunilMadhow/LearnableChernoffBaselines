from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import argparse
import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

project_path = f'{_ROOT}'
if os.path.exists(project_path) and project_path not in sys.path:
    sys.path.append(project_path)

from SMC import run_smc
from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore, ModelConfig
from util import LLaDAReward, seed_everything, get_constraint_details

K_GRID = [2, 4, 8, 16, 32, 64, 128]

PROMPT_TEXT = (
    "Write a creative story of exactly 3 sentences. "
    "Stylistically, it is preferred that the first word of each sentence "
    "start with 'A', 'B', and 'C' respectively."
)


# -----------------------------------------------------------------------------
# Multi-run SMC
# -----------------------------------------------------------------------------
def run_smc_multi(core, value_estimator, ess_threshold, M, reward_fn, tokenizer):
    """M independent SMC runs → M*K samples, each tagged with run_idx."""
    samples, resample_steps_per_run = [], []
    for run_idx in range(M):
        print(f"  SMC run {run_idx + 1}/{M}")
        tokens, log_weights, resample_steps = run_smc(
            core, value_estimator, ess_threshold=ess_threshold
        )
        scores = reward_fn(tokens)
        texts  = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        resample_steps_per_run.append(resample_steps)
        print(f"    mean reward={np.mean(scores):.4f}  "
              f"perfects={int(np.sum(np.array(scores) == 1.0))}/{core.S}  "
              f"resamples={resample_steps}")
        for txt, scr, lw in zip(texts, scores, log_weights.tolist()):
            samples.append({"text": txt, "score": float(scr),
                            "log_weight": float(lw), "run_idx": run_idx})
    return samples, resample_steps_per_run


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ess_vals_for_k(k):
    """Absolute ESS thresholds from k//2 to k, step 10, always including k."""
    vals = list(range(k // 2, k, 10))
    vals.append(k)
    return vals


def make_core(tokenizer, model, prompt_ids, K):
    cfg = ModelConfig()
    cfg.model_name  = "llada"
    cfg.state_shape = (32,)
    cfg.index_set   = list(range(33))
    cfg.deets = {
        "steps":       32,
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "mask_id":     126336,
        "temperature": 1.12,
        "cfg_scale":   0.0,
        "remasking":   "low_confidence",
        "prompt_len":  len(prompt_ids),
        "block_len":   8,
    }
    core = LLaDACore(cfg, tokenizer, model, S=K, N=1)
    core.set_prompt(prompt_ids)
    return core


def build_report(samples, resample_steps_per_run, reward_temp, K, ess_threshold, M):
    scores_arr  = np.array([s["score"] for s in samples])
    n           = len(scores_arr)
    all_details = [get_constraint_details(s["text"]) for s in samples]

    count_perfect = int(np.sum(scores_arr == 1.0))
    count_a = sum(1 for d in all_details if d['A'])
    count_b = sum(1 for d in all_details if d['B'])
    count_c = sum(1 for d in all_details if d['C'])

    a_idx  = [i for i, d in enumerate(all_details) if d['A']]
    ab_idx = [i for i in a_idx if all_details[i]['B']]
    prob_b_given_a  = sum(1 for i in a_idx  if all_details[i]['B'])  / len(a_idx)  if a_idx  else 0.0
    prob_c_given_ab = sum(1 for i in ab_idx if all_details[i]['C'])  / len(ab_idx) if ab_idx else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "M":             M,
            "num_particles": K,
            "n_samples":     n,
            "reward_temp":   reward_temp,
            "ess_threshold": ess_threshold,
        },
        "stats": {
            "mean_reward":          float(np.mean(scores_arr)),
            "success_rate_perfect": float(count_perfect / n),
            "constraint_A_rate":    float(count_a / n),
            "constraint_B_rate":    float(count_b / n),
            "constraint_C_rate":    float(count_c / n),
            "funnel_stats": {
                "prob_B_given_A":  float(prob_b_given_a),
                "prob_C_given_AB": float(prob_c_given_ab),
            },
        },
        "resample_steps_per_run": resample_steps_per_run,
        "samples": samples,
    }


def run_one(core, ve, tokenizer, reward_fn, K, ess, M, smc_dir, reward_temp):
    fname    = f"smc_results_temp_{reward_temp}_K_{K}_ess_{ess}.json"
    out_path = os.path.join(smc_dir, fname)
    if os.path.exists(out_path):
        print(f"  SKIP  ESS={ess:3d}  (exists)")
        return

    print(f"  RUN   K={K:3d}  ESS={ess:3d}  M={M}")
    samples, resample_steps_per_run = run_smc_multi(
        core, ve, ess, M, reward_fn, tokenizer
    )
    report = build_report(samples, resample_steps_per_run, reward_temp, K, ess, M)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Saved {fname}  (mean={report['stats']['mean_reward']:.4f})")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sweep SMC over num_particles and ESS threshold."
    )
    parser.add_argument("--run_id",      required=True)
    parser.add_argument("--reward_temp", type=float, required=True)
    parser.add_argument("--M",           type=int,   default=5,
                        help="Independent SMC runs per (K, ESS) setting. Total samples = M*K.")
    parser.add_argument("--data_dir",    type=str,
                        default=f"{_ROOT}/data_runs")
    parser.add_argument("--saved_grid", default=None, help="JSON list of saved K/ESS/M settings to rerun")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    seed_everything()
    run_dir = os.path.join(args.data_dir, args.run_id)
    smc_dir = args.output_dir or os.path.join(run_dir, "SMC")
    os.makedirs(smc_dir, exist_ok=True)

    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"

    print("Loading Value Classifier...")
    ve = ValueClassifier.load(run_dir)
    ve.reward_temperature = args.reward_temp

    print("Loading LLaDA (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        hf_model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    prompt_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    reward_fn  = LLaDAReward(tokenizer)

    if args.saved_grid:
        with open(args.saved_grid) as f:
            jobs = json.load(f)
    else:
        jobs = [{"K": k, "ess": e, "M": args.M} for k in K_GRID for e in ess_vals_for_k(k)]
    for job_num, job in enumerate(jobs, 1):
        K, ess = job["K"], job["ess"]
        core = make_core(tokenizer, model, prompt_ids, K)
        print(f"[{job_num}/{len(jobs)}]", end=" ")
        run_one(core, ve, tokenizer, reward_fn, K, ess, job["M"], smc_dir, args.reward_temp)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
