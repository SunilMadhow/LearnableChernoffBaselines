import os
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from diffusion_core import ModelConfig



from diffusion_core import Core, ModelConfig  # type: ignore

# ============================================================
# Helpers
# ============================================================

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature-scaled Gumbel noise to token logits.
    """
    if temperature <= 1e-6:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Distribute masked tokens across the remaining diffusion steps.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    if steps <= 0:
        return torch.zeros((mask_index.shape[0], 0), device=mask_index.device, dtype=torch.int64)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


# ============================================================
# Generation and imputation steps
# ============================================================

@torch.no_grad()
def llada_single_step(
    model,
    x_gen: torch.Tensor,
    prompt_ids: torch.Tensor,
    transfer_count: torch.Tensor,
    *,
    active_limit: int,
    mask_id: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """
    Apply one diffusion transition to the generation tokens.
    """
    device = next(model.parameters()).device
    x_gen = x_gen.to(device)
    prompt_ids = prompt_ids.to(device)
    transfer_count = transfer_count.to(device)

    B, L_gen = x_gen.shape
    L_prompt = prompt_ids.shape[1]

    if prompt_ids.shape[0] != B:
        prompt_ids = prompt_ids.repeat(B, 1)

    x_full = torch.cat([prompt_ids, x_gen], dim=1)

    if cfg_scale > 0.0:
        gen_visible_mask = (x_gen != mask_id)
        x_full_uncond = x_full.clone()
        x_full_uncond[:, L_prompt:][gen_visible_mask] = mask_id
        x_cat = torch.cat([x_full, x_full_uncond], dim=0)
        logits_cat = model(x_cat).logits
        logits_full, un_logits_full = torch.chunk(logits_cat, 2, dim=0)
        logits_full = un_logits_full + (cfg_scale + 1.0) * (logits_full - un_logits_full)
    else:
        logits_full = model(x_full).logits

    logits_gen = logits_full[:, L_prompt:, :]
    logits_with_noise = add_gumbel_noise(logits_gen, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits_gen.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, L_gen), device=device)
    else:
        raise NotImplementedError(f"Unknown remasking mode: {remasking}")

    if active_limit < L_gen:
        x0_p[:, active_limit:] = -np.inf

    is_masked = (x_gen == mask_id)
    confidence = torch.where(is_masked, x0_p, torch.full_like(x0_p, -np.inf))

    transfer_index = torch.zeros_like(x_gen, dtype=torch.bool, device=device)
    for i in range(B):
        k = transfer_count[i].item()
        if k <= 0:
            continue
        _, select_indices = torch.topk(confidence[i], k=min(k, int(is_masked[i].sum().item())))
        transfer_index[i, select_indices] = True

    x_next = x_gen.clone()
    x_next[transfer_index] = x0[transfer_index]
    return x_next


@torch.no_grad()
def _llada_forward_impute(
    model,
    x_gen: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    mask_id: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact forward/imputation logic from llada_single_step up to x0 sampling.

    Returns:
        logits_gen: (B, L_gen, vocab)
        x0:         (B, L_gen)
    """
    device = next(model.parameters()).device
    x_gen = x_gen.to(device)
    prompt_ids = prompt_ids.to(device)

    B, _ = x_gen.shape
    L_prompt = prompt_ids.shape[1]

    if prompt_ids.shape[0] != B:
        prompt_ids = prompt_ids.repeat(B, 1)

    x_full = torch.cat([prompt_ids, x_gen], dim=1)

    if cfg_scale > 0.0:
        gen_visible_mask = (x_gen != mask_id)
        x_full_uncond = x_full.clone()
        x_full_uncond[:, L_prompt:][gen_visible_mask] = mask_id
        x_cat = torch.cat([x_full, x_full_uncond], dim=0)
        logits_cat = model(x_cat).logits
        logits_full, un_logits_full = torch.chunk(logits_cat, 2, dim=0)
        logits_full = un_logits_full + (cfg_scale + 1.0) * (logits_full - un_logits_full)
    else:
        logits_full = model(x_full).logits

    logits_gen = logits_full[:, L_prompt:, :]
    logits_with_noise = add_gumbel_noise(logits_gen, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    return logits_gen, x0


@torch.no_grad()
def llada_step_with_imputation(
    model,
    x_gen: torch.Tensor,
    prompt_ids: torch.Tensor,
    transfer_count: torch.Tensor,
    *,
    active_limit: int,
    mask_id: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same transition as llada_single_step, but also returns x0.

    Returns:
        x_next: next partial state
        x0:     full imputation from the source state
    """
    device = next(model.parameters()).device
    x_gen = x_gen.to(device)
    prompt_ids = prompt_ids.to(device)
    transfer_count = transfer_count.to(device)

    B, L_gen = x_gen.shape
    logits_gen, x0 = _llada_forward_impute(
        model,
        x_gen,
        prompt_ids,
        mask_id=mask_id,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )

    if remasking == "low_confidence":
        p = F.softmax(logits_gen.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, L_gen), device=device)
    else:
        raise NotImplementedError(f"Unknown remasking mode: {remasking}")

    if active_limit < L_gen:
        x0_p[:, active_limit:] = -np.inf

    is_masked = (x_gen == mask_id)
    confidence = torch.where(is_masked, x0_p, torch.full_like(x0_p, -np.inf))

    transfer_index = torch.zeros_like(x_gen, dtype=torch.bool, device=device)
    for i in range(B):
        k = transfer_count[i].item()
        if k <= 0:
            continue
        masked_count = int(is_masked[i].sum().item())
        if masked_count == 0:
            continue
        _, select_indices = torch.topk(confidence[i], k=min(k, masked_count))
        transfer_index[i, select_indices] = True

    x_next = x_gen.clone()
    x_next[transfer_index] = x0[transfer_index]
    return x_next, x0


@torch.no_grad()
def llada_impute_only(
    model,
    x_gen: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    mask_id: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
) -> torch.Tensor:
    """
    Pure imputation pass on an arbitrary partial generation state.
    Only masked positions are filled in; already-decoded tokens are preserved.
    """
    _, x0 = _llada_forward_impute(
        model,
        x_gen,
        prompt_ids,
        mask_id=mask_id,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )
    is_masked = (x_gen == mask_id)
    result = x_gen.clone()
    result[is_masked] = x0[is_masked]
    return result


# ============================================================
# Core
# ============================================================

class LLaDACore(Core):
    model_id = "llada"

    def __init__(self, config, tokenizer, llada_model, S: int = 32, N: int = 4, load_from_config: bool = False):
        if isinstance(config, dict):
            self.model_config_obj = ModelConfig.from_dict(config)
        else:
            self.model_config_obj = config

        super().__init__(self.model_config_obj)

        self.tokenizer = tokenizer
        self.model = llada_model
        self.S = S
        self.N = N

        if load_from_config:
            self.model_config = self.model_config_obj.to_dict()
            self.state_shape = tuple(self.model_config["state_shape"])
            self.deets = dict(self.model_config.get("deets", {}))
        else:
            self.model_config = self.model_config_obj.to_dict()
            self.state_shape = tuple(self.model_config_obj.state_shape)
            self.deets = dict(self.model_config_obj.deets)

        if hasattr(llada_model, "is_loaded_in_4bit") and llada_model.is_loaded_in_4bit:
            self.deets["quantization"] = "4bit"
            self.model_config["deets"] = self.deets

        self.dummy = self.deets.get("dummy", False)
        self.gen_len = int(self.state_shape[0])
        self.prompt_len = int(self.deets.get("prompt_len", 0))

        self.total_steps = int(self.deets.get("steps", 64))
        self.T = self.total_steps

        self.block_len = int(self.deets.get("block_len", self.gen_len))
        self.num_blocks = math.ceil(self.gen_len / self.block_len)

        if self.num_blocks > 0 and self.total_steps % self.num_blocks != 0:
            self.steps_per_block = max(1, self.total_steps // self.num_blocks)
            self.total_steps = self.steps_per_block * self.num_blocks
            self.T = self.total_steps
        else:
            self.steps_per_block = self.total_steps // max(1, self.num_blocks)

        default_mask_id = getattr(tokenizer, "mask_token_id", None)
        self.mask_id = int(self.deets.get("mask_id", default_mask_id if default_mask_id is not None else 126336))

        self.temperature = float(self.deets.get("temperature", 0.0))
        # Separate temperature for reward-scoring imputation; defaults to 0 (greedy argmax).
        # Diffusion steps always use self.temperature for diversity.
        self.impute_temperature = float(self.deets.get("impute_temperature", 0.0))
        self.cfg_scale = float(self.deets.get("cfg_scale", 0.0))
        self.remasking = self.deets.get("remasking", "low_confidence")

        self.prompt_ids = None
        prompt_cfg = self.model_config.get("prompt_ids", None)
        if prompt_cfg is not None:
            self.set_prompt(prompt_cfg)

        self.X = None
        self.t = None
        self.transfer_schedule = None
        self.tentative_states = None

    def set_prompt(self, prompt_ids):
        prompt_ids = np.array(prompt_ids, dtype=np.int64)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[None, :]

        self.prompt_ids = prompt_ids
        self.prompt_len = prompt_ids.shape[1]
        self.model_config["prompt_ids"] = prompt_ids.tolist()
        self.deets["prompt_len"] = self.prompt_len
        self.model_config["deets"] = self.deets

    def reset(self):
        self.t = self.total_steps
        self.X = np.full((self.S, self.gen_len), self.mask_id, dtype=np.int64)

        if self.dummy:
            self.tentative_states = None
            return

        device = next(self.model.parameters()).device
        x_torch = torch.from_numpy(self.X).long().to(device)

        schedules = []
        for b in range(self.num_blocks):
            start = b * self.block_len
            end = min((b + 1) * self.block_len, self.gen_len)
            block_masks = (x_torch[:, start:end] == self.mask_id)
            block_sched = get_num_transfer_tokens(block_masks, self.steps_per_block)
            schedules.append(block_sched)

        if schedules:
            self.transfer_schedule = torch.cat(schedules, dim=1)
        else:
            self.transfer_schedule = torch.zeros((self.S, self.total_steps), device=device, dtype=torch.int64)

        self.tentative_states = None

    def _step_index_from_t(self, t: int) -> int:
        return self.total_steps - t

    def _get_active_limit(self, step_index: int) -> int:
        current_block_idx = step_index // max(1, self.steps_per_block)
        current_block_idx = min(current_block_idx, self.num_blocks - 1)
        return min((current_block_idx + 1) * self.block_len, self.gen_len)

    def propose(self, S: int = 1):
        if self.t is None:
            self.reset()
        if self.t <= 0:
            raise ValueError("No remaining timesteps.")
        if self.prompt_ids is None:
            raise ValueError("prompt_ids must be set before calling propose()")

        if self.dummy:
            vocab_limit = getattr(self.tokenizer, "vocab_size", 32000)
            proposals = np.random.randint(0, vocab_limit, size=(self.S, self.N, self.gen_len))
            self.tentative_states = proposals
            return proposals

        device = next(self.model.parameters()).device
        x_gen_torch = torch.from_numpy(self.X).long().to(device)
        prompt_torch = torch.from_numpy(self.prompt_ids).long().to(device)

        step_index = self._step_index_from_t(self.t)
        current_transfer_count = self.transfer_schedule[:, step_index]
        active_limit = self._get_active_limit(step_index)

        proposals = []
        for _ in range(self.N):
            x_next = llada_single_step(
                self.model,
                x_gen=x_gen_torch,
                prompt_ids=prompt_torch,
                transfer_count=current_transfer_count,
                active_limit=active_limit,
                mask_id=self.mask_id,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
            )
            proposals.append(x_next.cpu())

        proposals = torch.stack(proposals, dim=0).permute(1, 0, 2).numpy()
        self.tentative_states = proposals
        return proposals

    def commit(self, state):
        state = np.asarray(state, dtype=np.int64)
        if state.shape[1] != self.gen_len:
            raise ValueError(f"Commit shape mismatch: expected {self.gen_len}, got {state.shape[1]}")
        self.X = state
        self.t -= 1

    def collect_run(self):
        states = []
        self.reset()
        states.append(self.X.copy())
        while self.t > 0:
            proposals = self.propose()
            self.commit(proposals[:, 0, :])
            states.append(self.X.copy())
        return np.flip(np.array(states), axis=0)

    def save(self, states, data_dir="data", run_id=None):
        os.makedirs(data_dir, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cfg_path = os.path.join(data_dir, f"{run_id}_config.json")
        states_path = os.path.join(data_dir, f"{run_id}_states.npy")

        def _to_serializable(obj):
            if hasattr(obj, "to_dict"):
                return _to_serializable(obj.to_dict())
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(self.model_config), f, indent=2)

        np.save(states_path, np.asarray(states))
        return {"id": run_id, "config": cfg_path, "states": states_path}


class LLaDAImputationCore(LLaDACore):
    """
    Generation core with arbitrary-state imputation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_imputation = None  # (S, N, gen_len)

    def propose(self, S: int = 1):
        if self.t is None:
            self.reset()
        if self.t <= 0:
            raise ValueError("No remaining timesteps.")
        if self.prompt_ids is None:
            raise ValueError("prompt_ids must be set before calling propose()")

        if self.dummy:
            vocab_limit = getattr(self.tokenizer, "vocab_size", 32000)
            proposals = np.random.randint(0, vocab_limit, size=(self.S, self.N, self.gen_len))
            self.tentative_states = proposals
            self.last_imputation = proposals.copy()
            return proposals

        device = next(self.model.parameters()).device
        x_gen_torch = torch.from_numpy(self.X).long().to(device)
        prompt_torch = torch.from_numpy(self.prompt_ids).long().to(device)

        step_index = self._step_index_from_t(self.t)
        current_transfer_count = self.transfer_schedule[:, step_index]
        active_limit = self._get_active_limit(step_index)

        proposals = []
        imputations = []
        for _ in range(self.N):
            x_next, x0 = llada_step_with_imputation(
                self.model,
                x_gen=x_gen_torch,
                prompt_ids=prompt_torch,
                transfer_count=current_transfer_count,
                active_limit=active_limit,
                mask_id=self.mask_id,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
            )
            proposals.append(x_next.cpu())
            # Masked-merge: keep revealed tokens, only fill masked positions.
            # Raw x0 re-predicts all positions; we only want model predictions
            # where the input was actually masked.
            is_masked = (x_gen_torch == self.mask_id)
            x0_merged = x_gen_torch.clone()
            x0_merged[is_masked] = x0[is_masked]
            imputations.append(x0_merged.cpu())

        self.tentative_states = torch.stack(proposals, dim=0).permute(1, 0, 2).numpy()
        self.last_imputation = torch.stack(imputations, dim=0).permute(1, 0, 2).numpy()
        return self.tentative_states

    @torch.no_grad()
    def impute_tokens_from_states(
        self,
        states: np.ndarray,
        prompt_ids: Optional[np.ndarray] = None,
        batch_size: int = 16,
    ) -> np.ndarray:
        if self.prompt_ids is None and prompt_ids is None:
            raise ValueError("prompt_ids must be provided either via set_prompt() or call-time argument")

        states = np.asarray(states, dtype=np.int64)
        prompt_ids = np.asarray(self.prompt_ids if prompt_ids is None else prompt_ids, dtype=np.int64)

        if states.ndim != 2:
            raise ValueError(f"Expected states with shape (B, gen_len), got {states.shape}")
        if prompt_ids.ndim != 2:
            raise ValueError(f"Expected prompt_ids with shape (1 or B, prompt_len), got {prompt_ids.shape}")
        if prompt_ids.shape[0] not in (1, states.shape[0]):
            raise ValueError(
                f"Prompt batch size must be 1 or equal to B={states.shape[0]}, got {prompt_ids.shape[0]}"
            )

        device = next(self.model.parameters()).device
        outs = []
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            x_batch = torch.from_numpy(states[start:end]).long().to(device)
            if prompt_ids.shape[0] == 1:
                p_batch = torch.from_numpy(prompt_ids).long().to(device)
            else:
                p_batch = torch.from_numpy(prompt_ids[start:end]).long().to(device)

            x0 = llada_impute_only(
                self.model,
                x_batch,
                p_batch,
                mask_id=self.mask_id,
                temperature=self.impute_temperature,
                cfg_scale=self.cfg_scale,
            )
            outs.append(x0.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def decode_generation_tokens(self, token_ids: np.ndarray, *, strip: bool = True) -> List[str]:
        texts = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if strip:
            texts = [t.strip() for t in texts]
        return texts

    def impute_strings_from_states(
        self,
        states: np.ndarray,
        prompt_ids: Optional[np.ndarray] = None,
        batch_size: int = 16,
        *,
        strip: bool = True,
    ) -> List[str]:
        token_ids = self.impute_tokens_from_states(states, prompt_ids=prompt_ids, batch_size=batch_size)
        return self.decode_generation_tokens(token_ids, strip=strip)

    def get_imputed_strings(self, *, strip: bool = True) -> List[List[str]]:
        if self.last_imputation is None:
            return []
        all_texts = []
        for batch_idx in range(self.last_imputation.shape[0]):
            all_texts.append(self.decode_generation_tokens(self.last_imputation[batch_idx], strip=strip))
        return all_texts


# ============================================================
# __main__ smoke test
# ============================================================

class _DummyTokenizer:
    def __init__(self, vocab_size: int = 32, mask_token_id: int = 31):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.special_ids = {self.mask_token_id, self.bos_token_id, self.eos_token_id, self.pad_token_id}

    def batch_decode(self, batch_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        outs = []
        for row in batch_ids:
            toks = []
            for x in row:
                x = int(x)
                if skip_special_tokens and x in self.special_ids:
                    continue
                toks.append(f"tok{x}")
            outs.append(" ".join(toks))
        return outs


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 24):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        logits = self.proj(h)
        return SimpleNamespace(logits=logits)


def test_full_imputation_trajectory():
    print("--- Starting LLaDA Full Trajectory Imputation Test ---")
    
    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModel.from_pretrained(
        hf_model_id, 
        trust_remote_code=True, 
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 2. Define a Prompt
    prompt_text = "Complete the sentence in a polite but firm tone. Stem: \“I appreciate your message, but\""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # 3. Configure the Core
    gen_len = 32
    steps = 32
    
    cfg = ModelConfig()
    cfg.model_name = "llada"
    cfg.state_shape = (gen_len,)
    cfg.index_set = list(range(steps + 1))
    cfg.deets = {
        "steps": steps,
        "mask_id": 126336,
        "temperature": 0.9, # You can tweak this if it gets too repetitive
        "prompt_len": len(prompt_ids),
        "block_len": 16,
        "dummy": False
    }
    
    # S=1 to keep the printout clean
    core = LLaDAImputationCore(cfg, tokenizer, model, S=1, N=1) 
    core.set_prompt(prompt_ids)
    core.reset()

    print(f"\nPrompt: '{prompt_text}'")
    print("-" * 60)

    # 4. Loop through all 32 diffusion steps
    while core.t > 0:
        step_num = core.total_steps - core.t + 1
        
        # Propose the next state (this also generates the full x0 imputation)
        core.propose()
        
        # Get the full imputation string
        imputed_texts = core.get_imputed_strings()
        
        # Check how many masks are left in the ACTUAL diffusion state
        num_masks_in_state = (core.tentative_states[0, 0] == core.mask_id).sum()
        
        # Print step stats
        print(f"Step {step_num:02d}/{steps} | Masks Remaining: {num_masks_in_state:02d}")
        print(f"Imputation: '{imputed_texts[0][0]}'")
        print("-" * 60)
        
        # COMMIT the state to advance to the next timestep
        # tentative_states is shape (S, N, gen_len), commit expects (S, gen_len)
        core.commit(core.tentative_states[:, 0, :])

    # 5. Print the final fully unmasked state
    print("\n--- Final Generation ---")
    final_text = tokenizer.decode(core.X[0], skip_special_tokens=True)
    print(final_text)

if __name__ == "__main__":
    test_full_imputation_trajectory()
