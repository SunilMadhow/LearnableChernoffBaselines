"""
collect_data.py

Collect LLaDA trajectories + Skywork rewards for training the prompt-conditioned
value function.

For each training stem, generates S trajectories and saves all 32 intermediate
states (t=1..32, i.e. every denoising step before the final) plus the final
Skywork reward on the fully-decoded generation.

Saved per stem-batch (stems of the same prompt length are grouped together):
  {batch_id}_states.npy      — shape (32, S, L_gen), dtype int64
  {batch_id}_prompt_ids.npy  — shape (1, L_prompt),  dtype int64
  {batch_id}_rewards.npy     — shape (S,),            dtype float32
  {batch_id}_meta.json       — stem text, prompt text, generation params

collect_run() returns shape (T+1, S, L_gen) ordered most-noisy→least-noisy (index 0
is fully masked t=T, index T is the final clean state).  We want:
  states[t=1]  → index T-1  (one step before final, still has 1 masked token)
  states[t=10] → index T-10 (10 steps before final)
i.e., slices [-11:-1][::-1]  → shape (10, S, L_gen) with index 0 = least noisy.

Usage (Colab):
    !python /content/drive/MyDrive/LCB_cam_ready/prompt_scaled/collect_data.py
"""

from pathlib import Path as _ReleasePath
_LCB_RELEASE_ROOT = str(_ReleasePath(__file__).resolve().parents[1])


import json
import os
import sys
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llada_imputation_core import LLaDACore, ModelConfig
from single_baseline_sampler import HFRewardScorer

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
HF_MODEL_ID  = "GSAI-ML/LLaDA-8B-Instruct"
REWARD_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"

GEN_LEN     = 32
STEPS       = 32
BLOCK_LEN   = 8
TEMPERATURE = 1.12
REMASKING   = "low_confidence"
MASK_ID     = 126336

SAMPLES_PER_STEM = 40   # S: trajectories per stem
STEMS_PER_BATCH  = 7   # stems processed in parallel per collect_run() call; 7×40=280 ≈ A100-High-RAM capacity
NUM_SAVE_STEPS   = 32   # how many final-trajectory steps to save (t=1..32)

BASE_PROMPT = 'Complete the sentence in a polite but firm tone. Stem: "{stem}"'

BASE_DIR  = f"{_LCB_RELEASE_ROOT}"
STEMS_FILE = os.path.join(os.path.dirname(__file__), "stems_train.txt")
DATA_OUT   = os.path.join(BASE_DIR, "prompt_scaled", "trajectory_data")

SEED = 42
# -------------------------------------------------------------------------


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bucket_by_prompt_length(stems, tokenizer):
    buckets = defaultdict(list)
    for stem in stems:
        prompt_text = BASE_PROMPT.format(stem=stem)
        prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=False)
        buckets[len(prompt_ids)].append((stem, prompt_text, prompt_ids))
    return buckets


def make_core(model, tokenizer, prompt_ids_batch, S):
    cfg = ModelConfig()
    cfg.model_name  = "llada"
    cfg.state_shape = (GEN_LEN,)
    cfg.index_set   = list(range(STEPS + 1))
    cfg.deets = {
        "steps":      STEPS,
        "hf_model_id": HF_MODEL_ID,
        "mask_id":    MASK_ID,
        "temperature": TEMPERATURE,
        "remasking":  REMASKING,
        "prompt_len": len(prompt_ids_batch[0]),
        "block_len":  BLOCK_LEN,
    }
    core = LLaDACore(cfg, tokenizer, model, S=S, N=1)
    core.set_prompt(np.array(prompt_ids_batch, dtype=np.int64))
    return core


def decode_final_states(core: LLaDACore, final_states: np.ndarray):
    """Decode fully-unmasked token ids to strings."""
    texts = core.tokenizer.batch_decode(
        final_states,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [t.strip() for t in texts]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    seed_everything()
    os.makedirs(DATA_OUT, exist_ok=True)

    # Load stems
    with open(STEMS_FILE, "r", encoding="utf-8") as f:
        stems = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(stems)} stems from {STEMS_FILE}")

    # Load reward model
    print("Loading reward model...")
    reward_scorer = HFRewardScorer(REWARD_MODEL)

    # Load LLM
    print("Loading LLaDA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    # Bucket by prompt length to avoid ragged batches
    buckets = bucket_by_prompt_length(stems, tokenizer)
    print(f"\nPrompt-length buckets: { {k: len(v) for k, v in sorted(buckets.items())} }")

    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_idx = 0
    total_saved = 0

    for prompt_len, items in sorted(buckets.items()):
        print(f"\n=== Bucket prompt_len={prompt_len}  stems={len(items)} ===")

        for start in range(0, len(items), STEMS_PER_BATCH):
            batch_items = items[start : start + STEMS_PER_BATCH]
            B = len(batch_items)
            S = B * SAMPLES_PER_STEM

            seed_everything(SEED + batch_idx)

            # Build prompt_ids: (B*S, L_prompt) — stem i's prompt repeated S times
            prompt_ids_expanded = np.repeat(
                np.array([x[2] for x in batch_items], dtype=np.int64),  # (B, L_prompt)
                SAMPLES_PER_STEM, axis=0,
            )  # (B*S, L_prompt)

            core = make_core(model, tokenizer, prompt_ids_expanded, S=S)

            all_states = core.collect_run()   # (T+1, B*S, L_gen)

            # After np.flip: index 0 = clean (t=0), index T = fully masked (t=T).
            # t=1..NUM_SAVE_STEPS are the least-noisy intermediate states.
            finals = all_states[0]                              # (B*S, L_gen) — clean
            window = all_states[1 : NUM_SAVE_STEPS + 1].copy() # (NUM_SAVE_STEPS, B*S, L_gen)

            # Split and save one file per stem
            for stem_i, (stem, prompt_text, prompt_ids) in enumerate(batch_items):
                sl = slice(stem_i * SAMPLES_PER_STEM, (stem_i + 1) * SAMPLES_PER_STEM)

                stem_window = window[:, sl, :].copy()   # (NUM_SAVE_STEPS, S, L_gen)
                stem_finals = finals[sl]                # (S, L_gen)

                final_texts = decode_final_states(core, stem_finals)
                rewards = reward_scorer.score(
                    [prompt_text] * SAMPLES_PER_STEM,
                    final_texts,
                    batch_size=8,
                ).astype(np.float32)

                batch_id = f"{run_ts}_b{batch_idx:05d}"
                np.save(os.path.join(DATA_OUT, f"{batch_id}_states.npy"),     stem_window)
                np.save(os.path.join(DATA_OUT, f"{batch_id}_prompt_ids.npy"), np.array([prompt_ids], dtype=np.int64))
                np.save(os.path.join(DATA_OUT, f"{batch_id}_rewards.npy"),    rewards)

                meta = {
                    "batch_id":       batch_id,
                    "stem":           stem,
                    "prompt_text":    prompt_text,
                    "prompt_len":     prompt_len,
                    "S":              SAMPLES_PER_STEM,
                    "num_save_steps": NUM_SAVE_STEPS,
                    "gen_len":        GEN_LEN,
                    "steps":          STEPS,
                    "temperature":    TEMPERATURE,
                    "remasking":      REMASKING,
                    "mean_reward":    float(rewards.mean()),
                    "std_reward":     float(rewards.std()),
                }
                with open(os.path.join(DATA_OUT, f"{batch_id}_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                print(
                    f"  [{batch_idx:05d}] stem={stem!r:40s}  "
                    f"mean_r={rewards.mean():.3f}  std_r={rewards.std():.3f}"
                )
                batch_idx   += 1
                total_saved += SAMPLES_PER_STEM

    print(f"\nDone. Saved {batch_idx} batches ({total_saved} trajectories) to {DATA_OUT}")

