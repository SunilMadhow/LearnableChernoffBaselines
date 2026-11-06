"""
train_value_fn_normed.py

Same architecture as train_value_fn.py but normalises rewards per stem before
training.  This prevents the model from achieving low MSE by memorising
E[r | prompt] (the per-stem mean) rather than learning within-stem variation
from the generation tokens.

Normalisation: for each stem batch, compute mu and sigma over the S=40
trajectory rewards and map to z-scores:
    r_norm = (r - mu_stem) / max(sigma_stem, MIN_SIGMA)

The model therefore learns to predict "how much better/worse than average is
this trajectory for its stem", which requires reading the generation tokens at
every timestep rather than just the prompt.

Checkpoints saved to value_fn_normed.pt / value_fn_normed_meta.json.

Usage (Colab):
    !python /content/drive/MyDrive/LCB_cam_ready/prompt_scaled/train_value_fn_normed.py
"""

from pathlib import Path as _ReleasePath
_LCB_RELEASE_ROOT = str(_ReleasePath(__file__).resolve().parents[1])


import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CRITIC_MODEL   = "distilroberta-base"
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

NUM_SAVE_STEPS = 32
MAX_SEQ_LEN    = 128
MIN_SIGMA      = 0.1   # floor on per-stem std to avoid division by near-zero

EPOCHS       = 5
BATCH_SIZE   = 64
LR           = 2e-5
WEIGHT_DECAY = 1e-2

BASE_DIR  = f"{_LCB_RELEASE_ROOT}"
DATA_DIR  = os.path.join(BASE_DIR, "prompt_scaled", "trajectory_data")
CKPT_PATH = os.path.join(BASE_DIR, "prompt_scaled", "value_fn_normed.pt")
META_PATH = os.path.join(BASE_DIR, "prompt_scaled", "value_fn_normed_meta.json")
# -------------------------------------------------------------------------


class NormedValueDataset(Dataset):
    """
    Like ValueDataset but normalises rewards per stem (z-score within each
    stem's S trajectories).  The model target is mean-0 / std-1 within each
    stem, forcing it to learn within-stem ranking from generation tokens.
    """

    def __init__(self, data_dir: str, critic_tokenizer, llada_tokenizer,
                 max_seq_len: int = MAX_SEQ_LEN, train_steps=None):
        self.critic_tok  = critic_tokenizer
        self.llada_tok   = llada_tokenizer
        self.max_seq_len = max_seq_len
        self.mask_token  = critic_tokenizer.mask_token or "[MASK]"
        # train_steps: set of diffusion t values (1-indexed) to include; None = all
        self.train_steps = set(train_steps) if train_steps is not None else None

        meta_files = sorted(glob(os.path.join(data_dir, "*_meta.json")))
        if not meta_files:
            raise FileNotFoundError(f"No *_meta.json files found in {data_dir}")

        self.samples = []  # (prompt_text, gen_tokens_1d, t_scalar, normed_reward)

        skipped_flat = 0
        for meta_path in meta_files:
            batch_id = os.path.basename(meta_path).replace("_meta.json", "")
            states_path  = os.path.join(data_dir, f"{batch_id}_states.npy")
            rewards_path = os.path.join(data_dir, f"{batch_id}_rewards.npy")

            if not (os.path.exists(states_path) and os.path.exists(rewards_path)):
                skipped_flat += 1
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            states  = np.load(states_path)   # (NUM_SAVE_STEPS, S, L_gen)
            rewards = np.load(rewards_path)  # (S,)

            n_steps, S, _ = states.shape
            prompt_text = meta["prompt_text"]

            # Per-stem z-score normalisation
            mu    = float(rewards.mean())
            sigma = float(rewards.std())
            sigma = max(sigma, MIN_SIGMA)
            normed = (rewards - mu) / sigma   # (S,)

            for t_idx in range(n_steps):
                # t_idx=0 → t=1 (least masked); t_idx=k → t=k+1 (more masked)
                t = t_idx + 1
                if self.train_steps is not None and t not in self.train_steps:
                    continue
                t_scalar = float(n_steps - t_idx) / n_steps  # in (0, 1]; robust to any n_steps
                for s in range(S):
                    self.samples.append(
                        (prompt_text, states[t_idx, s], t_scalar, normed[s])
                    )

        steps_desc = f"t={sorted(self.train_steps)}" if self.train_steps is not None else "all t"
        print(f"NormedValueDataset: {len(self.samples)} (t, s) pairs "
              f"from {len(meta_files) - skipped_flat} batches "
              f"({skipped_flat} skipped) [{steps_desc}]")

    def __len__(self):
        return len(self.samples)

    def _transcode(self, prompt_text: str, gen_tokens: np.ndarray) -> str:
        gen_text = self.llada_tok.decode(gen_tokens.tolist(), skip_special_tokens=False)
        gen_text = (gen_text
                    .replace("<|mdm_mask|>", self.mask_token)
                    .replace("<|endoftext|>", "")
                    .replace("<|begin_of_text|>", "")
                    .strip())
        return prompt_text + " " + gen_text

    def __getitem__(self, idx):
        prompt_text, gen_tokens, t_scalar, normed_reward = self.samples[idx]
        joint_text = self._transcode(prompt_text, gen_tokens)
        enc = self.critic_tok(
            joint_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return (
            enc.input_ids.squeeze(0),
            torch.tensor(t_scalar,       dtype=torch.float32),
            torch.tensor(normed_reward,  dtype=torch.float32),
        )


# -------------------------------------------------------------------------
# Model — identical to PromptConditionedValueFn in train_value_fn.py
# -------------------------------------------------------------------------
class PromptConditionedValueFn(nn.Module):
    def __init__(self, model_name: str = CRITIC_MODEL):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        hidden = config.hidden_size

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embeds = self.backbone.embeddings(input_ids=input_ids)
        t_emb  = self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        embeds = embeds + t_emb
        out = self.backbone(inputs_embeds=embeds)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_steps", type=int, nargs="+", default=None,
        help="Diffusion timesteps (1-indexed) to train on. Default: all (1..NUM_SAVE_STEPS).",
    )
    parser.add_argument("--resume",      action="store_true",
                        help="Load weights from CKPT_PATH before training.")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="First epoch to run (use with --resume, e.g. 6).")
    parser.add_argument("--epochs",      type=int, default=EPOCHS,
                        help="Last epoch to run (inclusive). Default: 5.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if args.train_steps:
        print(f"Training on steps: {sorted(args.train_steps)}")
    else:
        print(f"Training on all steps (1..{NUM_SAVE_STEPS})")

    print("Loading tokenizers...")
    llada_tok  = AutoTokenizer.from_pretrained(LLADA_MODEL_ID, trust_remote_code=True)
    critic_tok = AutoTokenizer.from_pretrained(CRITIC_MODEL)

    print("Building dataset...")
    dataset = NormedValueDataset(DATA_DIR, critic_tok, llada_tok, max_seq_len=MAX_SEQ_LEN,
                                 train_steps=args.train_steps)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True)

    print("Building model...")
    model     = PromptConditionedValueFn(CRITIC_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if args.resume:
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"Resumed from {CKPT_PATH} (starting at epoch {args.start_epoch})")

    # Baseline MSE for normalised targets is 1.0 (predicting zero everywhere).
    # Anything meaningfully below 1.0 means real within-stem signal is learned.
    print("Baseline MSE (predict-zero) = 1.00 for normalised targets\n")

    best_loss = float("inf")
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t_sq_err = defaultdict(float)
        t_count  = defaultdict(int)

        for input_ids, t, reward in loader:
            input_ids = input_ids.to(device)
            t         = t.to(device)
            reward    = reward.to(device)

            pred       = model(input_ids, t)
            per_sample = (pred - reward) ** 2
            loss       = per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                for t_val, sq in zip(t.cpu().tolist(), per_sample.detach().cpu().tolist()):
                    key = round(t_val, 1)
                    t_sq_err[key] += sq
                    t_count[key]  += 1

        avg_loss = total_loss / len(loader)
        t_mse_str = "  ".join(
            f"t={k:.1f}:{t_sq_err[k]/t_count[k]:.3f}"
            for k in sorted(t_sq_err, reverse=True)
        )
        print(f"[Epoch {epoch:3d}/{args.epochs}] MSE = {avg_loss:.4f}  |  {t_mse_str}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  -> saved checkpoint (best so far)")

    meta = {
        "critic_model":   CRITIC_MODEL,
        "llada_model_id": LLADA_MODEL_ID,
        "num_save_steps": NUM_SAVE_STEPS,
        "max_seq_len":    MAX_SEQ_LEN,
        "normalised":     True,
        "min_sigma":      MIN_SIGMA,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest MSE: {best_loss:.4f}  (baseline = 1.00)")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Metadata:   {META_PATH}")

