from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import os
import argparse
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# Import your modules
from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore, ModelConfig
from lcb_with_offline import LCB, LCBSmartSampler
# Import the user's Collector
from collection import SequentialCollector 

import random

from util import LLaDAReward, seed_everything, load_core_and_tokenizer


def run_inference(run_id, data_dir, lcb_name, temp, num_samples = 100, sample_batch_size = 100, delta = 0.1, no_tau = False, output_file = None):
  assert num_samples % sample_batch_size == 0
  run_specific_dir = os.path.join(data_dir, run_id)
  seed_everything()
  # reward_temp_str = str(reward_temp).replace('.', 'p')
  # lcb_dir = os.path.join(run_specific_dir, "lcb_pretrain_data")
  vc = ValueClassifier.load(run_specific_dir)
  vc.reward_temperature = temp

  hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"
  core, tokenizer = load_core_and_tokenizer(num_samples)
  core.S = sample_batch_size

  upper_bound = 1/temp
  # upper_bound = None <--- uncomment to turn off clipping
  print(upper_bound)

  lcb = LCB.load(run_specific_dir, None, vc, core, lcb_name)
  lcb.delta = delta
  
  sampler = LCBSmartSampler(core, lcb, max_retries=100, upper_bound = upper_bound)
  sampler.no_tau= no_tau
  
  all_scores = []
  total_nfe = 0
  reward_fn = LLaDAReward(tokenizer)
  all_tokens = []
  all_num_proposals = []

  
  for i in range(0, int(num_samples/sample_batch_size)):
    print(f"Batch {i}")
    tokens, stats = sampler.run_inference()
    scores = reward_fn(tokens)
    all_tokens.extend(tokens)
    all_scores.extend(scores)
    all_num_proposals.extend(sampler.num_proposals_list)

  batch_texts = tokenizer.batch_decode(all_tokens, skip_special_tokens=True)

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
      "reward_temp": temp,
      "mean_reward": float(mean_reward),
      "perfect_rate": float(perfect),
      "avg_nfe": float(avg_nfe),
      "samples": []
  }
  for txt, scr, prop in zip(batch_texts, all_scores, all_num_proposals):
        report["samples"].append({
            "text": txt,
            "score": float(scr),
            "num_proposals": prop
        })
  if output_file is None:
    output_file = f"{lcb_name}_inference_results.json"
  output_file = os.path.join(run_specific_dir, "inference_results", output_file)
  os.makedirs(os.path.dirname(output_file), exist_ok=True)
  with open(output_file, "w") as f: json.dump(report, f, indent=2)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_id")
  parser.add_argument("--lcb_name", type=str)
  parser.add_argument("--temp", type=float)
  
  parser.add_argument("--num_samples", type = int, default = 100)
  parser.add_argument("--no_tau", type = bool, default = False)
  parser.add_argument("--delta", type=float, default = 0.1)
  parser.add_argument("--sample_batch_size", type = int, default = 100)
  parser.add_argument("--output_file", type=str, default=None)

  args = parser.parse_args()
  run_id = args.run_id
  policy_name = args.lcb_name
  temp = args.temp
  sample_batch_size = args.sample_batch_size
  seed_everything()

  run_inference(run_id, f"{_ROOT}/data_runs", policy_name, temp, args.num_samples, sample_batch_size, delta = args.delta, no_tau = args.no_tau, output_file = args.output_file)
