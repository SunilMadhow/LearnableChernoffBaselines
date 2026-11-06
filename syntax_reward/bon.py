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
from real_data.rewards import *
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from llada_core_hide_prompt import *
from diffusion_core import *

def get_constraint_details(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    targets = ['A', 'B', 'C']
    flags = {'A': False, 'B': False, 'C': False}
    for i in range(min(3, len(sentences))):
        clean_sent = re.sub(r'^[^a-zA-Z]+', '', sentences[i])
        if clean_sent and clean_sent[0].upper() == targets[i]:
            flags[targets[i]] = True
    return flags

def run_standard_generation(core):
    core.reset()
    for _ in range(core.T):
        if core.t <= 0: break
        props = core.propose()
        core.commit(props[:, 0, :])
    return core.X
def get_constraint_details(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    targets = ['A', 'B', 'C']
    flags = {'A': False, 'B': False, 'C': False}

    for i in range(min(3, len(sentences))):
        clean_sent = re.sub(r'^[^a-zA-Z]+', '', sentences[i])
        if clean_sent and clean_sent[0].upper() == targets[i]:
            flags[targets[i]] = True
    return flags
# -----------------------------------------------------------------------------
# Main Execution: Best of N (Oracle)
# -----------------------------------------------------------------------------
import argparse
if __name__ == "__main__":
    # --- Config ---
    parser = argparse.ArgumentParser(description="LCB Robust Training Script")

    # --- 1. Required Arguments (No Defaults) ---
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    TARGET_SAMPLES = args.num_samples
    N_BEST = args.N

    output_file = args.output_file or f"{_ROOT}/reruns/best_of_{N_BEST}_results.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Batch size = Number of winners we generate per loop
    WINNERS_PER_BATCH = 10
    CANDIDATES_PER_BATCH = WINNERS_PER_BATCH * N_BEST

    # 1. Load Model
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
    llada_temp = 1.12
    # 2. Setup Core
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
        "temperature": llada_temp,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "prompt_len": len(prompt_ids),
        "block_len": 8,
    }

    core = LLaDACore(cfg, tokenizer, model, S=CANDIDATES_PER_BATCH, N=1)
    core.set_prompt(prompt_ids)

    reward_fn = LLaDAReward(tokenizer)

    # -------------------------------------------------------------------------
    # Best-of-N Loop
    # -------------------------------------------------------------------------
    print(f"\n=== Running Best-of-{N_BEST} (Oracle) Collection ===")
    print(f"Target Winners: {TARGET_SAMPLES}")
    print(f"Candidates per Step: {CANDIDATES_PER_BATCH} -> Winners per Step: {WINNERS_PER_BATCH}")

    all_winner_tokens = []
    num_batches = TARGET_SAMPLES // WINNERS_PER_BATCH

    for i in tqdm(range(num_batches), desc="BoN Batches"):

        # A. Generate Candidates (e.g. 20 samples)
        candidate_tokens = run_standard_generation(core) # (20, 32)

        # B. Score Candidates with ORACLE (Ground Truth)
        candidate_scores = reward_fn(candidate_tokens) # (20,)

        # C. Select Best of N
        # Reshape to (10, 2)
        scores_grouped = candidate_scores.reshape(WINNERS_PER_BATCH, N_BEST)
        tokens_grouped = candidate_tokens.reshape(WINNERS_PER_BATCH, N_BEST, -1)

        # Find index of max score in each group
        # Note: argmax breaks ties by taking the first one.
        best_indices = np.argmax(scores_grouped, axis=1) # (10,)

        # Gather winners
        winner_tokens_batch = tokens_grouped[np.arange(WINNERS_PER_BATCH), best_indices]

        all_winner_tokens.append(winner_tokens_batch)

    all_winner_tokens = np.concatenate(all_winner_tokens, axis=0)
    print(f"\nCollected {len(all_winner_tokens)} winners.")

    # -------------------------------------------------------------------------
    # Final Stats
    # -------------------------------------------------------------------------
    final_scores = reward_fn(all_winner_tokens)
    final_texts = tokenizer.batch_decode(all_winner_tokens, skip_special_tokens=True)

    all_scores = np.array(final_scores)
    all_details = [get_constraint_details(t) for t in final_texts]
    n = len(all_scores)

    count_perfect = np.sum(all_scores == 1.0)
    count_a = sum(1 for d in all_details if d['A'])
    count_b = sum(1 for d in all_details if d['B'])
    count_c = sum(1 for d in all_details if d['C'])

    a_indices = [i for i, d in enumerate(all_details) if d['A']]
    b_given_a = sum(1 for i in a_indices if all_details[i]['B'])
    prob_b_given_a = b_given_a / len(a_indices) if a_indices else 0.0

    ab_indices = [i for i, d in enumerate(all_details) if d['A'] and d['B']]
    c_given_ab = sum(1 for i in ab_indices if all_details[i]['C'])
    prob_c_given_ab = c_given_ab / len(ab_indices) if ab_indices else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "method": f"Best-of-{N_BEST}-Oracle",
        "config": {
            "n_samples": TARGET_SAMPLES,
            "candidates_per_winner": N_BEST,
            "temperature": llada_temp
        },
        "stats": {
            "mean_reward": float(np.mean(all_scores)),
            "success_rate_perfect": float(count_perfect / n),
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
    for txt, scr in zip(final_texts, all_scores):
        report["samples"].append({
            "text": txt,
            "score": float(scr),
            "num_proposals": int(core.T * N_BEST)
        })
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print(f" BEST-OF-{N_BEST} (ORACLE) RESULTS ")
    print("="*60)
    print(f"Mean Reward:   {report['stats']['mean_reward']:.4f}")
    print(f"Perfect Rate:  {report['stats']['success_rate_perfect']:.1%}")
    print("-" * 30)
    print(f"Constraint A:  {report['stats']['constraint_A_rate']:.1%}")
    print(f"Constraint B:  {report['stats']['constraint_B_rate']:.1%}")
    print(f"Constraint C:  {report['stats']['constraint_C_rate']:.1%}")
    print(f"\nFull results saved to: {output_file}")
