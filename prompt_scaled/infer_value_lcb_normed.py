"""
infer_value_lcb_normed.py

Inference with the per-stem-normalised value-LCB.

Same loop as infer_value_lcb.py but uses value_fn_normed.pt and
value_lcb_normed_params.pt.  GUIDE_STEPS is set to all steps 2..10 so the
normed taus determine which steps actually provide signal (steps where the
normed model learned within-stem variation will have tight taus; steps with no
signal will have tau≈log(1/δ)/λ_max and be effectively uninformative but
harmless).

Usage (Colab):
    !python /content/drive/MyDrive/LCB_cam_ready/prompt_scaled/infer_value_lcb_normed.py
"""

from pathlib import Path as _ReleasePath
_LCB_RELEASE_ROOT = str(_ReleasePath(__file__).resolve().parents[1])


import argparse
import json
import os
import sys
import random
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llada_imputation_core import LLaDACore, ModelConfig
from single_baseline_sampler import HFRewardScorer
from train_value_fn import PromptConditionedValueFn

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
HF_MODEL_ID  = "GSAI-ML/LLaDA-8B-Instruct"
REWARD_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
CRITIC_MODEL = "distilroberta-base"
LLADA_MASK_ID = 126336

GEN_LEN     = 32
STEPS       = 32
BLOCK_LEN   = 8
TEMPERATURE = 1.12
REMASKING   = "low_confidence"

NUM_SAVE_STEPS = 32   # guide steps t=1..NUM_SAVE_STEPS; t_scalar = (32-t+1)/32
NUM_SAMPLES    = 100
MAX_RETRIES    = 20   # rejection sampling retries per trajectory per step
MAX_SEQ_LEN    = 128

BASE_PROMPT = 'Complete the sentence in a polite but firm tone. Stem: "{stem}"'

BASE_DIR    = f"{_LCB_RELEASE_ROOT}"
CKPT_PATH   = os.path.join(BASE_DIR, "prompt_scaled", "value_fn_normed.pt")
META_PATH   = os.path.join(BASE_DIR, "prompt_scaled", "value_fn_normed_meta.json")
PARAMS_PATH = os.path.join(BASE_DIR, "prompt_scaled", "lcb_clean_d0p5.pt")
STEMS_FILE  = os.path.join(BASE_DIR, "prompt_scaled", "stems_eval.txt")
OUT_DIR     = os.path.join(BASE_DIR, "prompt_scaled", "inference_results")

COMPARE_UNGUIDED = True   # run an unguided pass per stem for direct comparison
SEED = 0
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
        "mask_id":    LLADA_MASK_ID,
        "temperature": TEMPERATURE,
        "remasking":  REMASKING,
        "prompt_len": len(prompt_ids_batch[0]),
        "block_len":  BLOCK_LEN,
    }
    core = LLaDACore(cfg, tokenizer, model, S=S, N=1)
    core.set_prompt(np.array(prompt_ids_batch, dtype=np.int64))
    return core


# -------------------------------------------------------------------------
# Value function helpers
# -------------------------------------------------------------------------

def encode_states(states: np.ndarray, prompt_text: str, t_scalar: float,
                  critic_tok, llada_tok, mask_token: str, device: str) -> torch.Tensor:
    texts = []
    for row in states:
        gen_decoded = llada_tok.decode(row.tolist(), skip_special_tokens=False)
        gen_decoded = (gen_decoded
                       .replace("<|mdm_mask|>", mask_token)
                       .replace("<|endoftext|>", "")
                       .replace("<|begin_of_text|>", "")
                       .strip())
        texts.append(prompt_text + " " + gen_decoded)
    enc = critic_tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    return enc.input_ids.to(device)


@torch.no_grad()
def score_value(value_fn: PromptConditionedValueFn,
                states: np.ndarray,
                prompt_text: str,
                t_scalar: float,
                critic_tok, llada_tok, mask_token: str,
                device: str) -> np.ndarray:
    input_ids = encode_states(states, prompt_text, t_scalar, critic_tok, llada_tok, mask_token, device)
    t_vec = torch.full((input_ids.shape[0],), t_scalar, dtype=torch.float32, device=device)
    return value_fn(input_ids, t_vec).cpu().numpy()


# -------------------------------------------------------------------------
# Value-LCB sampler (identical logic to infer_value_lcb.py)
# -------------------------------------------------------------------------

class ValueLCBSampler:
    def __init__(
        self,
        core: LLaDACore,
        value_fn: PromptConditionedValueFn,
        lcb_params: dict,
        prompt_text: str,
        critic_tok, llada_tok, mask_token: str,
        device: str,
        *,
        max_retries: int = MAX_RETRIES,
        guide_steps: Optional[set] = None,
        reward_scorer=None,
        alpha: float = 1.0,
    ):
        self.core          = core
        self.value_fn      = value_fn
        self.lcb_params    = lcb_params
        self.prompt_text   = prompt_text
        self.critic_tok    = critic_tok
        self.llada_tok     = llada_tok
        self.mask_token    = mask_token
        self.device        = device
        self.max_retries   = max_retries
        self.reward_scorer = reward_scorer
        self.alpha         = alpha
        self.guide_steps   = guide_steps if guide_steps is not None else set()

        self.inference_nfe        = 0
        self.n_accepted           = 0
        self.n_total              = 0
        self.n_fallback           = 0
        self.proposals_per_sample = None   # (S,) int32, set in run_inference

    def _t_scalar(self, t: int) -> float:
        # Match training: t_scalar = (n_steps - t_idx) / NUM_SAVE_STEPS
        # where t_idx = t - 1, so t_scalar = (NUM_SAVE_STEPS - t + 1) / NUM_SAVE_STEPS
        # t=1 → 1.0 (least masked), t=10 → 0.1 (most masked)
        return float(NUM_SAVE_STEPS - t + 1) / NUM_SAVE_STEPS

    def _unguided_step(self):
        proposals = self.core.propose()
        self.core.commit(proposals[:, 0, :])
        self.inference_nfe += self.core.S
        self.proposals_per_sample += 1

    def _propose_subset(self, unresolved):
        """Propose for a subset of trajectories, restoring core state after."""
        real_X      = self.core.X
        real_sched  = getattr(self.core, "transfer_schedule", None)
        real_S      = self.core.S
        real_prompt = self.core.prompt_ids

        self.core.X          = real_X[unresolved]
        self.core.S          = len(unresolved)
        self.core.prompt_ids = real_prompt
        if real_sched is not None:
            self.core.transfer_schedule = real_sched[unresolved]

        proposals = self.core.propose()
        self.inference_nfe += len(unresolved)

        self.core.X          = real_X
        self.core.S          = real_S
        self.core.prompt_ids = real_prompt
        if real_sched is not None:
            self.core.transfer_schedule = real_sched

        return proposals[:, 0, :]   # (|unresolved|, L_gen)

    def _decode_cands(self, cands: np.ndarray) -> List[str]:
        texts = self.llada_tok.batch_decode(
            cands, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return [t.strip() for t in texts]

    def _guided_final_step(self):
        """tp1=1: final step x_1 → x_0.  Accepts based on r(x_0) - V(x_1) - tau_1."""
        tau   = self.lcb_params[1]["tau"]
        t_src = self._t_scalar(1)
        S     = self.core.S

        v_src = score_value(
            self.value_fn, self.core.X, self.prompt_text, t_src,
            self.critic_tok, self.llada_tok, self.mask_token, self.device,
        )

        unresolved = np.arange(S)
        next_X = self.core.X.copy()

        for retry in range(self.max_retries + 1):
            if len(unresolved) == 0:
                break

            cands = self._propose_subset(unresolved)
            self.proposals_per_sample[unresolved] += 1

            if retry == self.max_retries:
                next_X[unresolved] = cands
                self.n_fallback += len(unresolved)
                break

            texts  = self._decode_cands(cands)
            r_vals = self.reward_scorer.score(
                [self.prompt_text] * len(cands), texts, batch_size=len(cands),
            )

            advantage   = r_vals - v_src[unresolved] - tau
            u           = np.random.uniform(size=len(unresolved))
            accept_mask = advantage / self.alpha > np.log(u + 1e-12)

            next_X[unresolved[accept_mask]] = cands[accept_mask]
            self.n_accepted += int(accept_mask.sum())
            self.n_total    += len(unresolved)
            unresolved = unresolved[~accept_mask]

        self.core.X  = next_X
        self.core.t -= 1

    def _guided_step(self, tp1: int):
        tau   = self.lcb_params[tp1]["tau"]
        t_src = self._t_scalar(tp1)
        t_dst = self._t_scalar(tp1 - 1)
        S     = self.core.S

        v_src = score_value(
            self.value_fn, self.core.X, self.prompt_text, t_src,
            self.critic_tok, self.llada_tok, self.mask_token, self.device,
        )

        unresolved = np.arange(S)
        next_X = self.core.X.copy()

        for retry in range(self.max_retries + 1):
            if len(unresolved) == 0:
                break

            cands = self._propose_subset(unresolved)
            self.proposals_per_sample[unresolved] += 1

            if retry == self.max_retries:
                next_X[unresolved] = cands
                self.n_fallback += len(unresolved)
                break

            v_dst = score_value(
                self.value_fn, cands, self.prompt_text, t_dst,
                self.critic_tok, self.llada_tok, self.mask_token, self.device,
            )

            advantage   = v_dst - v_src[unresolved] - tau
            u           = np.random.uniform(size=len(unresolved))
            accept_mask = advantage / self.alpha > np.log(u + 1e-12)

            next_X[unresolved[accept_mask]] = cands[accept_mask]
            self.n_accepted += int(accept_mask.sum())
            self.n_total    += len(unresolved)
            unresolved = unresolved[~accept_mask]

        self.core.X  = next_X
        self.core.t -= 1

    def run_inference(self):
        self.core.reset()
        self.inference_nfe        = 0
        self.n_accepted           = 0
        self.n_total              = 0
        self.n_fallback           = 0
        self.proposals_per_sample = np.zeros(self.core.S, dtype=np.int32)

        while self.core.t > 0:
            tp1 = self.core.t
            if tp1 in self.guide_steps and tp1 in self.lcb_params:
                if tp1 == 1:
                    self._guided_final_step()
                else:
                    self._guided_step(tp1)
            else:
                self._unguided_step()

        n_guided_slots = self.n_total + self.n_fallback
        props  = self.proposals_per_sample
        eff_n  = props / STEPS
        stats = {
            "inference_nfe":        self.inference_nfe,
            "accept_rate":          float(self.n_accepted / self.n_total) if self.n_total > 0 else None,
            "fallback_rate":        float(self.n_fallback / n_guided_slots) if n_guided_slots > 0 else None,
            "proposals_per_sample": props,
            "eff_n_per_sample":     eff_n,
            "mean_eff_n":           float(eff_n.mean()),
            "median_eff_n":         float(np.median(eff_n)),
            "max_eff_n":            float(eff_n.max()),
        }
        return self.core.X.copy(), stats


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guide_steps", nargs="+", type=int, default=None,
        help="tp1 values to guide (default: all keys in params file).",
    )
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Temperature: acceptance is (advantage - tau) / alpha > log U.")
    parser.add_argument("--params_path", type=str, default=None,
                        help="Path to LCB params .pt file (overrides PARAMS_PATH constant).")
    parser.add_argument("--base_name",       type=str, default=None,
                        help="Base name for output file. Output: {base_name}_a{alpha}.json")
    parser.add_argument("--no_unguided",     action="store_true",
                        help="Skip the unguided comparison pass.")
    args = parser.parse_args()

    params_path = args.params_path or PARAMS_PATH
    alpha_str   = str(args.alpha).replace(".", "p")
    params_tag  = os.path.splitext(os.path.basename(params_path))[0]

    seed_everything()
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(STEMS_FILE, "r", encoding="utf-8") as f:
        all_stems = [line.strip() for line in f if line.strip()]

    heldout_stems = all_stems
    print(f"Evaluating all {len(heldout_stems)} stems from {STEMS_FILE}, {NUM_SAMPLES} samples each")

    print("Loading value function (normed)...")
    with open(META_PATH) as f:
        vf_meta = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    value_fn = PromptConditionedValueFn(vf_meta["critic_model"]).to(device)
    value_fn.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    value_fn.eval()

    print("Loading normed LCB params...")
    lcb_params = torch.load(params_path, map_location="cpu")
    if args.guide_steps:
        guided_steps = set(args.guide_steps) & set(lcb_params.keys())
    else:
        guided_steps = set(lcb_params.keys())
    print(f"  Guided steps: {sorted(guided_steps)}")
    for tp1 in sorted(guided_steps):
        p = lcb_params[tp1]
        print(f"  tp1={tp1}  tau={p['tau']:+.4f}  lam={p['lam']:.4f}  mu={p['mu']:+.4f}")

    print("Loading tokenizers...")
    llada_tok  = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    critic_tok = AutoTokenizer.from_pretrained(CRITIC_MODEL)
    mask_token = critic_tok.mask_token or "[MASK]"

    print("Loading LLaDA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    llada_model = AutoModel.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True,
        quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16,
    )
    llada_model.eval()

    print("Loading reward model...")
    reward_scorer = HFRewardScorer(REWARD_MODEL)

    all_samples   = []
    all_scores    = []
    all_ug_scores = []

    buckets = bucket_by_prompt_length(heldout_stems, llada_tok)
    print(f"\nBuckets: { {k: len(v) for k, v in sorted(buckets.items())} }")

    for prompt_len, items in sorted(buckets.items()):
        print(f"\n=== Bucket prompt_len={prompt_len}  stems={len(items)} ===")

        for stem, prompt_text, prompt_ids in items:
            seed_everything()

            core = make_core(llada_model, llada_tok, [prompt_ids], S=NUM_SAMPLES)

            sampler = ValueLCBSampler(
                core, value_fn, lcb_params, prompt_text,
                critic_tok, llada_tok, mask_token, device,
                max_retries=MAX_RETRIES,
                guide_steps=guided_steps,
                reward_scorer=reward_scorer if 1 in guided_steps else None,
                alpha=args.alpha,
            )

            final_states, stats = sampler.run_inference()

            final_texts = llada_tok.batch_decode(
                final_states, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            final_texts = [t.strip() for t in final_texts]

            scores = reward_scorer.score(
                [prompt_text] * len(final_texts), final_texts, batch_size=8,
            )

            proposals_arr = stats["proposals_per_sample"]
            all_scores.extend(scores.tolist())
            for i, (text, score) in enumerate(zip(final_texts, scores.tolist())):
                all_samples.append({
                    "stem": stem, "prompt": prompt_text,
                    "text": text, "score": float(score), "guided": True,
                    "n_proposals": int(proposals_arr[i]),
                    "eff_n":       float(proposals_arr[i] / STEPS),
                })

            # Unguided comparison pass (same seed → same initial noise)
            ug_str = ""
            if COMPARE_UNGUIDED and not args.no_unguided:
                seed_everything()
                ug_core = make_core(llada_model, llada_tok, [prompt_ids], S=NUM_SAMPLES)
                ug_core.reset()
                while ug_core.t > 0:
                    proposals = ug_core.propose()
                    ug_core.commit(proposals[:, 0, :])
                ug_texts = llada_tok.batch_decode(
                    ug_core.X, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                ug_texts = [t.strip() for t in ug_texts]
                ug_scores = reward_scorer.score(
                    [prompt_text] * len(ug_texts), ug_texts, batch_size=8,
                )
                all_ug_scores.extend(ug_scores.tolist())
                for text, score in zip(ug_texts, ug_scores.tolist()):
                    all_samples.append({
                        "stem": stem, "prompt": prompt_text,
                        "text": text, "score": float(score), "guided": False,
                    })
                delta = np.mean(scores) - np.mean(ug_scores)
                ug_str = f"  ug={np.mean(ug_scores):.3f}  Δ={delta:+.3f}"

            fb = stats["fallback_rate"]
            print(
                f"  stem={stem!r:45s}  "
                f"guided={np.mean(scores):.3f}{ug_str}  "
                f"accept={stats['accept_rate']:.1%}  fallback={fb:.1%}  "
                f"eff_N={stats['mean_eff_n']:.1f} (max={stats['max_eff_n']})"
            )

    arr = np.array(all_scores, dtype=np.float32)
    all_eff_n = np.array(
        [s["eff_n"] for s in all_samples if s["guided"]], dtype=np.float32
    )
    summary = {
        "guided_mean":    float(arr.mean()),
        "guided_std":     float(arr.std()),
        "guided_perfect": float(np.mean(arr == 1.0)),
        "count":          int(len(arr)),
        "alpha":          args.alpha,
        "mean_eff_n":     float(all_eff_n.mean()),
        "median_eff_n":   float(np.median(all_eff_n)),
        "max_eff_n":      float(all_eff_n.max()),
    }
    if all_ug_scores:
        ug_arr = np.array(all_ug_scores, dtype=np.float32)
        summary["unguided_mean"]    = float(ug_arr.mean())
        summary["unguided_std"]     = float(ug_arr.std())
        summary["unguided_perfect"] = float(np.mean(ug_arr == 1.0))
        summary["delta_mean"]       = float(arr.mean() - ug_arr.mean())

    out_name = f"{args.base_name}_a{alpha_str}.json" if args.base_name else f"value_lcb_normed_results_{params_tag}_a{alpha_str}.json"
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump({**summary, "samples": all_samples}, f, indent=2)

    print(f"\n{'Metric':<22} {'Value':>10}")
    print("-" * 34)
    for k, v in summary.items():
        print(f"  {k:<20} {v:>10.4f}" if isinstance(v, float) else f"  {k:<20} {v:>10}")
    print(f"\nResults → {out_path}")

