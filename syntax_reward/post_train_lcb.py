from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import argparse
import torch
import numpy as np
import os
import json
import random
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# Assuming these are your local modules
from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore, ModelConfig
from lcb_with_offline import LCB, LCBSmartSampler
from util import seed_everything, LLaDAReward

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LCB Robust Training Script")

    # --- 1. Required Arguments (No Defaults) ---
    parser.add_argument("--run_id", type=str, required=True, help="ID of the run to load (Required)")
    parser.add_argument("--reward_temp", type=float, required=True, help="Temperature for the value classifier (Required)")

    # --- 2. Optional Arguments (With Defaults) ---
    parser.add_argument("--train_batch_size", type=int, default=150, help="Batch size for training")
    parser.add_argument("--accumulation_loops", type=int, default=1, help="Number of accumulation loops")
    parser.add_argument("--baseline_lr", type=float, default=1e-6, help="Learning rate for the baseline")
    parser.add_argument("--pretrained_lcb_name", type=str)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--num_eval_loops", type=int, default=5)
    parser.add_argument("--lambda_max", type=float, default=6)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--sym", type=bool, default=False)
    parser.add_argument("--clip", type=bool, default=False)
    args = parser.parse_args()

    seed_everything()

    # --- 3. Hardcoded Parameters & Configs ---
    data_dir = f"{_ROOT}/data_runs"
    run_specific_dir = os.path.join(data_dir, args.run_id)
    
    # Robust Parameters (Hardcoded)
    OUTER_STEPS = 1          
    LAMBDA_LR = 1e-3          
    LAMBDA_MAX = args.lambda_max          
    LAMBDA_INIT = 1.0  
    delta = args.delta
    
    # Naming logic
    save_name = None
    if args.save_name is None:
      if not args.sym:
        save_name = f"lcb_post_trained_temp_{str(args.reward_temp).replace('.', 'p')}"
      else:
        save_name = f"lcb_SYM_post_trained_temp_{str(args.reward_temp).replace('.', 'p')}"
    else:
      save_name = args.save_name

    output_file = os.path.join(run_specific_dir, f"{save_name}_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Model Config (Hardcoded)
    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"
    prompt_text = "Write a creative story of exactly 3 sentences. Stylistically, it is preferred that the first word of each sentence start with 'A', 'B', and 'C' respectively."

    # -------------------------------------------------------------------------
    # Execution Logic
    # -------------------------------------------------------------------------

    # 1. Resources
    print(f"Loading Value Classifier from {args.run_id}...")
    vc = ValueClassifier.load(run_specific_dir)
    vc.reward_temperature = args.reward_temp 
    
    # ve = ValueEstimate.load(run_specific_dir)

    print("Loading LLM...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    # 2. Setup Core
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    cfg = ModelConfig()
    cfg.model_name = "llada"
    cfg.state_shape = (32,)
    cfg.index_set = list(range(33))
    cfg.deets = {"steps": 32, "hf_model_id": hf_model_id, "mask_id": 126336, "temperature": 1.12, "remasking": "low_confidence", "prompt_len": len(prompt_ids), "block_len": 8}

    # =========================================================================
    # PHASE 1: ROBUST LEARNING
    # =========================================================================
    total_trajectories = args.train_batch_size * args.accumulation_loops
    print(f"\n=== PHASE 1: Robust Learning (total_trajectories={total_trajectories}) ===")
    
    core_train = LLaDACore(cfg, tokenizer, model, S=args.train_batch_size, N=1)
    core_train.set_prompt(prompt_ids)
    
    lcb_name = args.pretrained_lcb_name
    lcb = LCB.load(run_specific_dir, None, vc, core_train, lcb_name)
    lcb.delta = delta
    lcb.lam_max = LAMBDA_MAX
    lcb.lambda_mode = "search"
    lcb.symmetric_objective = args.sym
    for param_group in lcb.net_opt.param_groups:
      param_group['lr'] = args.baseline_lr
    print("Training on symmetrized loss: ", args.sym)
    # Note: baseline_lr isn't explicitly used in the snippet provided, 
    # but likely used inside LCB.load or configured here if the class supports it.
    # If LCB has a learning rate attribute, set it here:
    # lcb.lr = args.baseline_lr 

    # sampler_train = LCBSmartSampler(core_train, lcb, max_retries=100, save_name = save_name, run_specific_dir = run_specific_dir)
    if args.clip:
      sampler_train = LCBSmartSampler(core_train, lcb, upper_bound = 1/args.reward_temp, max_retries=100, save_name = save_name, run_specific_dir = run_specific_dir)
    else:
      sampler_train = LCBSmartSampler(core_train, lcb, max_retries=100, save_name = save_name, run_specific_dir = run_specific_dir)
    sampler_train.train_batches_per_step = 1
    
    stats = sampler_train.train_policy(n_trajectories = args.accumulation_loops)

    # =========================================================================
    # PHASE 2: INFERENCE
    # =========================================================================
    print(f"\n=== PHASE 2: Inference (S=100) ===")
    
    core_eval = LLaDACore(cfg, tokenizer, model, S=args.eval_batch_size, N=1)
    core_eval.set_prompt(prompt_ids)
    
    sampler = LCBSmartSampler(core_eval, lcb, max_retries=20)
    
    all_scores = []
    total_nfe = 0
    reward_fn = LLaDAReward(tokenizer)
    all_samples = []
    all_num_proposals = []
    for i in range(args.num_eval_loops): 
        print(f"Batch {i+1}/5")
        tokens, stats = sampler.run_inference()
        scores = reward_fn(tokens)
        all_scores.extend(scores)
        all_num_proposals.extend(sampler.num_proposals_list)
        batch_texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        for text, score, prop in zip(batch_texts, scores, sampler.num_proposals_list):
          all_samples.append({"text": text, "score": float(score), "proposals":prop})

        total_nfe += stats['inference_nfe']
        print(f"  Reward: {np.mean(scores):.2f} | NFE: {stats['inference_nfe']}")

    # Report
    mean_reward = np.mean(all_scores)
    perfect = np.mean(np.array(all_scores) == 1.0)
    avg_nfe = np.mean(all_num_proposals)
    
    print("\n" + "="*60)
    print(" ROBUST RESULTS (T4 Config) ")
    print("="*60)
    print(f"Mean Reward:  {mean_reward:.4f}")
    print(f"Perfect Rate: {perfect:.1%}")
    print(f"Avg NFE:      {avg_nfe:.1f}")
    
    report = {
        "method": "LCB_Robust",
        "run_id": args.run_id,
        "reward_temp": args.reward_temp,
        "mean_reward": float(mean_reward),
        "perfect_rate": float(perfect),
        "n_trajectories": args.train_batch_size,
        "accumulation_steps": args.accumulation_loops,
        "baseline_post_train_lr": args.baseline_lr,
        "lambda_mode": lcb.lambda_mode,
        "lambda_max": LAMBDA_MAX,
        "delta": delta,
        "avg_nfe": float(avg_nfe),
        "samples": all_samples
    }
    with open(output_file, "w") as f: json.dump(report, f, indent=2)
