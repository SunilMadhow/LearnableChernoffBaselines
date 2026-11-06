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
# Rejection Sampling Loop
# -----------------------------------------------------------------------------
def run_rejection_sampling(core, value_estimator, max_reward=1, temperature=0.1, max_retries=20):
    B = max_reward / temperature
    core.reset()

    total_proposed = 0
    total_accepted = 0

    num_proposals = [0]*core.S

    with tqdm(total=core.T, desc="RS Batch") as pbar:
        while core.t > 0:
            current_t = core.t
            prev_t_idx = core.model_config["index_set"][current_t - 1]

            accepted_mask = np.zeros(core.S, dtype=bool)
            final_proposals = np.zeros((core.S, core.gen_len), dtype=np.int64)
            step_attempts = 0

            for attempt in range(max_retries):
                active_indices = np.where(~accepted_mask)[0]
                if len(active_indices) == 0: break

                step_attempts += 1

                # Propose (S, N, L) -> Take 0th proposal
                all_proposals = core.propose()
                candidates = all_proposals[:, 0, :]

                # Evaluate (Core hides prompt, so candidates is just generation)
                active_candidates = candidates[active_indices]
                v = value_estimator.predict_under_log(active_candidates, prev_t_idx)

                # Accept
                log_accept_prob = v - B
                log_u = np.log(np.random.rand(len(active_indices)) + 1e-10)

                just_accepted = log_u < log_accept_prob
                newly_accepted_indices = active_indices[just_accepted]

                if len(newly_accepted_indices) > 0:
                    accepted_mask[newly_accepted_indices] = True
                    final_proposals[newly_accepted_indices] = candidates[newly_accepted_indices]
                for i in active_indices:
                    num_proposals[i] = num_proposals[i] + 1
                total_proposed += len(active_indices)
                total_accepted += len(newly_accepted_indices)

            # Fill failures
            failed_indices = np.where(~accepted_mask)[0]
            if len(failed_indices) > 0:
                final_proposals[failed_indices] = candidates[failed_indices]

            step_acc_rate = (total_accepted / (total_proposed + 1e-6)) * 100
            pbar.set_postfix({"AccRate": f"{step_acc_rate:.1f}%", "Retries": step_attempts})

            core.commit(final_proposals)
            pbar.update(1)

    return core.X, num_proposals, total_proposed

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
import argparse
if __name__ == "__main__":
    # --- Config ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--reward_temp", type=float, required = True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--save_name", type=str, default = None)
    args = parser.parse_args()

    run_id = args.run_id
    reward_temp = args.reward_temp 

    data_dir = f"{_ROOT}/data_runs"
    run_dir = os.path.join(data_dir, str(run_id))
    if args.save_name:
      output_file = os.path.join(run_dir, args.save_name)
    else:
      output_file = os.path.join(run_dir, f"rejection_sampling_results_temp_{reward_temp}.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    TOTAL_SAMPLES = args.num_samples
    BATCH_SIZE = 250

    print("Loading Value Classifier...")
    # Note: Use ValueClassifier, NOT ValueEstimate
    ve = ValueClassifier.load(os.path.join(data_dir, run_id))
    ve.reward_temperature = reward_temp

    # 2. Load Model
    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"
    print("Loading LLaDA (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
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

    # 3. Setup Core
    prompt_text = "Write a creative story of exactly 3 sentences. Stylistically, it is preferred that the first word of each sentence start with 'A', 'B', and 'C' respectively."
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    cfg = ModelConfig()
    cfg.model_name = "llada"
    cfg.state_shape = (32,)
    cfg.index_set = list(range(33))
    cfg.deets = {
        "steps": 32,
        "hf_model_id": hf_model_id,
        "mask_id": 126336,
        "temperature": 1.12,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "prompt_len": len(prompt_ids),
        "block_len": 8,
    }

    core = LLaDACore(cfg, tokenizer, model, S=BATCH_SIZE, N=1)
    core.set_prompt(prompt_ids)

    # 4. Collection Loop
    all_texts = []
    all_scores = []
    all_num_proposals = []
    reward_fn = LLaDAReward(tokenizer)

    print(f"\n=== Starting Collection: {TOTAL_SAMPLES} Samples ===")

    num_batches = max(TOTAL_SAMPLES // BATCH_SIZE, 1)
    for i in range(num_batches):
        print(f"\nBatch {i+1}/{num_batches}")

        # Run RS
        batch_tokens, num_proposals_list, total_proposals = run_rejection_sampling(
            core,
            ve,
            max_reward=1,
            temperature=ve.reward_temperature,
            max_retries=12
        )

        # Decode & Score
        batch_scores = reward_fn(batch_tokens)
        batch_texts = tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)

        # Store
        all_texts.extend(batch_texts)
        all_scores.extend(batch_scores)
        all_num_proposals.extend(num_proposals_list)

        print(f"  Batch Mean Reward: {np.mean(batch_scores):.2f}")
        print(f"  Batch Perfects: {np.sum(batch_scores == 1.0)}/{BATCH_SIZE}")

    # 5. Statistics Calculation
    all_scores = np.array(all_scores)
    n = len(all_scores)

    # Generate detailed breakdown for stats (since Reward obj only gives floats)
    all_details = [get_constraint_details(t) for t in all_texts]

    count_perfect = np.sum(all_scores == 1.0)
    count_a = sum(1 for d in all_details if d['A'])
    count_b = sum(1 for d in all_details if d['B'])
    count_c = sum(1 for d in all_details if d['C'])

    # Funnel Analysis
    a_indices = [i for i, d in enumerate(all_details) if d['A']]
    b_given_a = sum(1 for i in a_indices if all_details[i]['B'])
    prob_b_given_a = b_given_a / len(a_indices) if a_indices else 0.0

    ab_indices = [i for i, d in enumerate(all_details) if d['A'] and d['B']]
    c_given_ab = sum(1 for i in ab_indices if all_details[i]['C'])
    prob_c_given_ab = c_given_ab / len(ab_indices) if ab_indices else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": TOTAL_SAMPLES,
            "rs_temperature": ve.reward_temperature,
            "reward_temp": ve.reward_temperature,
        },
        "stats": {
            "mean_reward": float(np.mean(all_scores)),
            "success_rate_perfect": float(count_perfect / n),
            "avg_nfe": np.sum(all_num_proposals)/TOTAL_SAMPLES,
            "constraint_A_rate": float(count_a / n),
            "constraint_B_rate": float(count_b / n),
            "constraint_C_rate": float(count_c / n),
            "funnel_stats": {
                "prob_B_given_A": float(prob_b_given_a),
                "prob_C_given_AB": float(prob_c_given_ab)
            }
        },
        "samples": []
    }

    # Add samples to report
    for txt, scr, prop in zip(all_texts, all_scores, all_num_proposals):
        report["samples"].append({
            "text": txt,
            "score": float(scr),
            "num_proposals": prop
        })

    # Save
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print(" RESULTS REPORT ")
    print("="*60)
    print(f"Total Samples: {n}")
    print(f"Mean Reward:   {report['stats']['mean_reward']:.4f}")
    print(f"Perfect Rate:  {report['stats']['success_rate_perfect']:.1%}")
    print("-" * 30)
    print(f"Constraint A:  {report['stats']['constraint_A_rate']:.1%}")
    print(f"Constraint B:  {report['stats']['constraint_B_rate']:.1%}")
    print(f"Constraint C:  {report['stats']['constraint_C_rate']:.1%}")
    print("-" * 30)
    print(f"Retention A->B: {report['stats']['funnel_stats']['prob_B_given_A']:.1%}")
    print(f"Retention B->C: {report['stats']['funnel_stats']['prob_C_given_AB']:.1%}")
    print(f"\nFull results saved to: {output_file}")
