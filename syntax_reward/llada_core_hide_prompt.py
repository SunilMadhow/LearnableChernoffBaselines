from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import torch
import numpy as np
import os
import json
import math
from datetime import datetime
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

from diffusion_core import Core, ModelConfig
from collection import Collector

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 1e-6:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Computes schedule based on the mask of the GENERATION part only.
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

@torch.no_grad()
def llada_single_step(model, x_gen, prompt_ids, transfer_count, *, active_limit, mask_id, temperature=0.0, cfg_scale=0.0, remasking="low_confidence"):
    """
    Performs one diffusion step ONLY on x_gen, using prompt_ids as context.
    
    Args:
        x_gen: (B, L_gen) - The mutable generation buffer
        prompt_ids: (1, L_prompt) or (B, L_prompt) - Fixed context
    """
    device = next(model.parameters()).device
    x_gen = x_gen.to(device)
    prompt_ids = prompt_ids.to(device)
    transfer_count = transfer_count.to(device)
    
    B, L_gen = x_gen.shape
    L_prompt = prompt_ids.shape[1]
    
    # 1. Construct Model Input (Prompt + Gen)
    if prompt_ids.shape[0] != B:
        prompt_ids = prompt_ids.repeat(B, 1)
        
    x_full = torch.cat([prompt_ids, x_gen], dim=1) # (B, L_total)
    
    # 2. Forward Pass
    if cfg_scale > 0.0:
        # CFG: Unconditional input masks the generation part fully? 
        # Usually in text diffusion, 'unconditional' means replacing text with mask, 
        # but preserving prompt? Or masking everything?
        # Standard implementation: Mask the GENERATION part for the 'uncond' branch.
        
        x_gen_masked = x_gen.clone()
        # We assume standard LLaDA CFG masks the generation tokens that aren't already masks?
        # Actually, standard CFG usually masks the *conditioning*. 
        # If prompt is the condition, we mask the prompt? 
        # Let's stick to the previous logic: "un_x" had prompt_index masked. 
        # If we treat prompt as fixed context, maybe we don't mask it?
        # Let's assume standard "Mask-based" CFG: 
        #   Cond: [Prompt, Gen]
        #   Uncond: [Prompt, Masked_Gen] (This drives model towards structure?)
        #   Actually, previous code masked the *non-mask* tokens.
        
        # Simpler approach matching previous logic:
        # Construct full input, then create un_x_full by masking the visible parts of x_gen
        
        # Identify what is currently visible in generation
        gen_visible_mask = (x_gen != mask_id)
        
        x_full_uncond = x_full.clone()
        # Mask out the visible generation tokens in the uncond branch
        # (Offset by prompt length)
        x_full_uncond[:, L_prompt:][gen_visible_mask] = mask_id
        
        x_cat = torch.cat([x_full, x_full_uncond], dim=0)
        logits_cat = model(x_cat).logits
        logits_full, un_logits_full = torch.chunk(logits_cat, 2, dim=0)
        
        logits_full = un_logits_full + (cfg_scale + 1.0) * (logits_full - un_logits_full)
    else:
        logits_full = model(x_full).logits

    # 3. Slice Logits (We only care about predictions for the Generation part)
    logits_gen = logits_full[:, L_prompt:, :] # (B, L_gen, Vocab)

    # 4. Sampling (On Generation Part Only)
    logits_with_noise = add_gumbel_noise(logits_gen, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) 

    # 5. Confidence / Remasking (On Generation Part Only)
    if remasking == "low_confidence":
        p = F.softmax(logits_gen.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, L_gen), device=device)
    else:
        raise NotImplementedError

    # 6. Apply Barriers (On Generation Part Only)
    # active_limit is now 0..L_gen
    if active_limit < L_gen:
        x0_p[:, active_limit:] = -np.inf

    # Protect already masked tokens (standard diffusion logic)
    is_masked = (x_gen == mask_id)
    confidence = torch.where(is_masked, x0_p, torch.full_like(x0_p, -np.inf))

    # 7. Select Top-K to Transfer
    transfer_index = torch.zeros_like(x_gen, dtype=torch.bool, device=device)
    for i in range(B):
        k = transfer_count[i].item()
        if k <= 0: continue
        _, select_indices = torch.topk(confidence[i], k=k)
        transfer_index[i, select_indices] = True

    # 8. Update
    x_next = x_gen.clone()
    x_next[transfer_index] = x0[transfer_index]
    
    return x_next


# --------------------------------------------------
# LLaDACore (Hides Prompt)
# --------------------------------------------------

class LLaDACore(Core):
    model_id = "llada"

    def __init__(self, config, tokenizer, llada_model, S: int = 32, N: int = 4, load_from_config: bool = False):
        
        # Normalize config
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
            self.state_shape = tuple(self.model_config["state_shape"]) # This should be (gen_len,) now
            self.deets = dict(self.model_config.get("deets", {}))
        else:
            self.model_config = self.model_config_obj.to_dict()
            self.state_shape = tuple(self.model_config_obj.state_shape)
            self.deets = dict(self.model_config_obj.deets)

        # Quantization check
        if hasattr(llada_model, 'is_loaded_in_4bit') and llada_model.is_loaded_in_4bit:
             self.deets['quantization'] = "4bit"
             self.model_config['deets'] = self.deets
        
        self.dummy = self.deets.get("dummy", False)
        
        # --- DIMENSION LOGIC ---
        # The 'state_shape' passed in config is now strictly the GENERATION length.
        self.gen_len = int(self.state_shape[0])
        
        # Prompt length is auxiliary, tracked in deets but not state shape
        self.prompt_len = int(self.deets.get("prompt_len", 0))
        
        self.total_steps = int(self.deets.get("steps", 64)) 
        self.T = self.total_steps
        
        self.block_len = int(self.deets.get("block_len", self.gen_len))
        self.num_blocks = math.ceil(self.gen_len / self.block_len)
        
        if self.total_steps % self.num_blocks != 0:
            self.steps_per_block = self.total_steps // self.num_blocks
            self.total_steps = self.steps_per_block * self.num_blocks
            self.T = self.total_steps
        else:
            self.steps_per_block = self.total_steps // self.num_blocks

        default_mask_id = getattr(tokenizer, "mask_token_id", None)
        self.mask_id = int(self.deets.get("mask_id", default_mask_id if default_mask_id is not None else 126336))
        
        self.temperature = float(self.deets.get("temperature", 0.0))
        self.cfg_scale = float(self.deets.get("cfg_scale", 0.0))
        self.remasking = self.deets.get("remasking", "low_confidence")

        self.prompt_ids = None
        if "prompt_ids" in self.model_config:
            self.set_prompt(self.model_config["prompt_ids"])

        self.X = None
        self.t = None
        self.transfer_schedule = None 

    def set_prompt(self, prompt_ids):
        """
        Sets the prompt. 
        Note: This does NOT affect self.X shape (which is fixed to gen_len).
        """
        prompt_ids = np.array(prompt_ids, dtype=np.int64)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[None, :] # Ensure (1, L)
            
        self.prompt_ids = prompt_ids
        self.prompt_len = prompt_ids.shape[1]
        
        # Update Config
        self.model_config["prompt_ids"] = prompt_ids.tolist()[0] # Save list
        self.deets["prompt_len"] = self.prompt_len
        self.model_config["deets"] = self.deets

    def reset(self):
        self.t = self.total_steps 
        # X contains ONLY generation tokens
        self.X = np.full((self.S, self.gen_len), self.mask_id, dtype=np.int64)
        
        if self.dummy:
            self.tentative_states = None
            return

        device = next(self.model.parameters()).device
        x_torch = torch.from_numpy(self.X).long().to(device)
        
        schedules = []
        for b in range(self.num_blocks):
            # Blocks are now relative to 0
            start = b * self.block_len
            end = min((b + 1) * self.block_len, self.gen_len)
            
            block_masks = (x_torch[:, start:end] == self.mask_id)
            block_sched = get_num_transfer_tokens(block_masks, self.steps_per_block)
            schedules.append(block_sched)
            
        if schedules:
            self.transfer_schedule = torch.cat(schedules, dim=1)
        else:
             self.transfer_schedule = torch.zeros((self.S, self.total_steps), device=device)

        self.tentative_states = None

    def _step_index_from_t(self, t: int) -> int:
        return self.total_steps - t

    def _get_active_limit(self, step_index: int) -> int:
        current_block_idx = step_index // self.steps_per_block
        current_block_idx = min(current_block_idx, self.num_blocks - 1)
        # Barrier is relative to start of generation (0)
        return min((current_block_idx + 1) * self.block_len, self.gen_len)

    def propose(self, S: int = 1):
        if self.t is None: self.reset()
        if self.t <= 0: raise ValueError("No remaining timesteps.")

        if self.dummy:
            vocab_limit = getattr(self.tokenizer, "vocab_size", 32000)
            # Return purely random generation part
            proposals = np.random.randint(0, vocab_limit, size=(self.S, self.N, self.gen_len))
            self.tentative_states = proposals
            return proposals

        device = next(self.model.parameters()).device
        
        # Prepare inputs
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
        # State MUST be (S, gen_len)
        if state.shape[1] != self.gen_len:
            raise ValueError(f"Commit shape mismatch: Expected {self.gen_len}, got {state.shape[1]}")
        self.X = np.asarray(state, dtype=np.int64)
        self.t -= 1

    def collect_run(self):
        states = []
        self.reset()
        states.append(self.X.copy()) # Save initial noise (Generation only)
        
        while self.t > 0: 
            proposals = self.propose()
            self.commit(proposals[:, 0, :])
            states.append(self.X.copy()) # Save generation only
            
        return np.flip(np.array(states), axis=0)

    def save(self, states, data_dir="data", run_id=None):
        # ... (Save logic remains identical) ...
        os.makedirs(data_dir, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cfg_path = os.path.join(data_dir, f"{run_id}_config.json")
        states_path = os.path.join(data_dir, f"{run_id}_states.npy")
        
        def _to_serializable(obj):
            if hasattr(obj, "to_dict"): return _to_serializable(obj.to_dict())
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)): return obj.item()
            if isinstance(obj, dict): return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_to_serializable(v) for v in obj]
            return obj

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(self.model_config), f, indent=2)

        np.save(states_path, np.asarray(states))
        return {"id": run_id, "config": cfg_path, "states": states_path}

from collection import SequentialCollector
# --------------------------------------------------
# Main Execution (Updated for Refactor)
# --------------------------------------------------
if __name__ == "__main__":
    dummy = False 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    except:
        tokenizer = None

    if not dummy and tokenizer:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModel.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",              
            torch_dtype=torch.float16,      
        )
        model.eval()
    else:
        model = None

    prompt_text = "Write a creative story of exactly 3 sentences. Stylistically, it is preferred that the first word of each sentence start with 'A', 'B', and 'C' respectively."
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if tokenizer else [1,2,3]
    prompt_len = len(prompt_ids)
    
    # --- CONFIG CHANGE: state_shape is gen_len only ---
    gen_len = 32
    
    cfg = ModelConfig()
    cfg.model_name = "llada"
    cfg.state_shape = (gen_len,) # Just generation length!
    cfg.index_set = list(range(32 + 1))
    cfg.deets = {
        "steps": 32,
        "hf_model_id": hf_model_id,
        "mask_id": 126336,
        "temperature": 1.12,
        "prompt_len": prompt_len,
        "block_len": 8, 
        "dummy": dummy 
    }
    S = 5000
    N = 1
    
    core = LLaDACore(cfg, tokenizer, model, S=S, N=N)
    core.set_prompt(prompt_ids)
    core.reset()

    from real_data.rewards import LLaDAReward

    reward_calculator = LLaDAReward(tokenizer)
    
    os.makedirs(f"{_ROOT}/data_runs", exist_ok=True)
    collector = SequentialCollector(core, reward_calculator, base_dir=f"{_ROOT}/data_runs", batch_size = 250)
    output = collector.collect()
    
    print("\nSuccess! Data collected at:", output['run_dir'])
