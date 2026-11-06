import json
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import math

from diffusion_core import Core, ModelConfig

# ---------- Helpers ----------

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Applies Gumbel noise to logits for stochastic sampling.
    """
    if temperature <= 1e-6:
        return logits
    
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Computes the static transfer schedule for a specific block.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)
    
    if steps <= 0:
        return torch.zeros((mask_index.shape[0], 0), device=mask_index.device, dtype=torch.int64)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps,
        device=mask_index.device,
        dtype=torch.int64,
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def llada_single_step(
    model,
    x: torch.Tensor,
    transfer_count: torch.Tensor, 
    *,
    prompt_len: int,
    active_limit: int, # <--- NEW: The boundary of the current block
    mask_id: int,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
) -> torch.Tensor:
    """
    Executes a single step, constrained to the 'active_limit'.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    transfer_count = transfer_count.to(device)
    B, L = x.shape

    # 1. Model Forward Pass
    prompt_index = (x != mask_id)
    
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id 
        
        x_cat = torch.cat([x, un_x], dim=0)
        logits_cat = model(x_cat).logits
        logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
        
        logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
    else:
        logits = model(x).logits

    # 2. Sampling
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) 

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, L), device=device)
    else:
        raise NotImplementedError(f"Unknown remasking={remasking}")

    # 3. Apply Constraints
    
    # A. Protect Prompt
    x0_p[:, :prompt_len] = -np.inf
    
    # B. Protect Future Blocks (Semi-Autoregressive Barrier)
    # Any token beyond active_limit is effectively invisible for selection
    if active_limit < L:
        x0_p[:, active_limit:] = -np.inf

    # C. Protect Already Revealed Tokens
    is_masked = (x == mask_id)
    confidence = torch.where(is_masked, x0_p, torch.full_like(x0_p, -np.inf))

    # 4. Selection (Top-k)
    transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
    
    for i in range(B):
        k = transfer_count[i].item()
        if k <= 0:
            continue
            
        _, select_indices = torch.topk(confidence[i], k=k)
        transfer_index[i, select_indices] = True

    # 5. Update
    x_next = x.clone()
    x_next[transfer_index] = x0[transfer_index]

    return x_next



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
def llada_single_step(model, x, transfer_count, *, prompt_len, active_limit, mask_id, temperature=0.0, cfg_scale=0.0, remasking="low_confidence"):
    device = next(model.parameters()).device
    x = x.to(device)
    transfer_count = transfer_count.to(device)
    B, L = x.shape
    prompt_index = (x != mask_id)
    
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id 
        x_cat = torch.cat([x, un_x], dim=0)
        logits_cat = model(x_cat).logits
        logits, un_logits = torch.chunk(logits_cat, 2, dim=0)
        logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
    else:
        logits = model(x).logits

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) 

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, L), device=device)
    else:
        raise NotImplementedError(f"Unknown remasking={remasking}")

    x0_p[:, :prompt_len] = -np.inf
    if active_limit < L:
        x0_p[:, active_limit:] = -np.inf
    is_masked = (x == mask_id)
    confidence = torch.where(is_masked, x0_p, torch.full_like(x0_p, -np.inf))

    transfer_index = torch.zeros_like(x, dtype=torch.bool, device=device)
    for i in range(B):
        k = transfer_count[i].item()
        if k <= 0: continue
        _, select_indices = torch.topk(confidence[i], k=k)
        transfer_index[i, select_indices] = True

    x_next = x.clone()
    x_next[transfer_index] = x0[transfer_index]
    return x_next


# --------------------------------------------------
# LLaDACore
# --------------------------------------------------

class LLaDACore(Core):
    """
    Stateful Diffusion Core with Semi-Autoregressive Blocking.
    Auto-updates config with prompt and quantization info for saving.
    """
    model_id = "llada"

    def __init__(self, config: ModelConfig | dict, tokenizer, llada_model, S: int = 32, N: int = 4, load_from_config: bool = False):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.model = llada_model
        self.S = S
        self.N = N

        if load_from_config:
            self._from_config_dict(config)
        else:
            self.model_config = config.to_dict()
            self.state_shape = tuple(config.state_shape)
            self.deets = dict(config.deets)

        # --- AUTO-DETECT QUANTIZATION ---
        # If the model is loaded in 4-bit, we record it so `load_core_from_config` knows.
        if hasattr(llada_model, 'is_loaded_in_4bit') and llada_model.is_loaded_in_4bit:
             self.deets['quantization'] = "4bit"
             self.model_config['deets'] = self.deets # Update main dict
        
        self.dummy = self.deets.get("dummy", False)
        self.max_length = int(self.state_shape[0])
        self.prompt_len = int(self.deets.get("prompt_len", 0))
        self.total_steps = int(self.deets.get("steps", 64)) 
        self.T = self.total_steps
        
        self.gen_len = self.max_length - self.prompt_len
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

        # --- AUTO-RESTORE PROMPT ---
        self.prompt_ids = None
        if "prompt_ids" in self.model_config:
            # If reloading from config, restore the prompt automatically
            self.set_prompt(self.model_config["prompt_ids"])

        self.X = None
        self.t = None
        self.transfer_schedule = None 

    def _from_config_dict(self, cfg_dict: dict):
        self.model_config = cfg_dict
        self.state_shape = tuple(cfg_dict["state_shape"])
        self.deets = dict(cfg_dict.get("deets", {}))
        self.__init__(self.model_config, self.tokenizer, self.model, self.S, self.N, load_from_config=False)

    def set_prompt(self, prompt_ids):
        """
        Sets the prompt and UPDATES THE CONFIG so it saves correctly.
        """
        prompt_ids = np.array(prompt_ids, dtype=np.int64)
        self.prompt_ids = prompt_ids
        self.prompt_len = len(prompt_ids)
        
        # --- UPDATE CONFIG ---
        self.model_config["prompt_ids"] = prompt_ids.tolist()
        # Update prompt_len in deets too, just in case
        self.deets["prompt_len"] = self.prompt_len
        self.model_config["deets"] = self.deets
        
        # Recalculate Blocks
        self.gen_len = self.max_length - self.prompt_len
        self.num_blocks = math.ceil(self.gen_len / self.block_len)
        self.steps_per_block = self.total_steps // self.num_blocks

    def reset(self):
        self.t = self.total_steps 
        self.X = np.full((self.S, self.max_length), self.mask_id, dtype=np.int64)
        if self.prompt_ids is not None:
            for s in range(self.S):
                self.X[s, :self.prompt_len] = self.prompt_ids
        
        if self.dummy:
            self.tentative_states = None
            return

        device = next(self.model.parameters()).device
        x_torch = torch.from_numpy(self.X).long().to(device)
        
        schedules = []
        for b in range(self.num_blocks):
            start = self.prompt_len + (b * self.block_len)
            end = min(self.prompt_len + ((b + 1) * self.block_len), self.max_length)
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
        gen_end_relative = (current_block_idx + 1) * self.block_len
        active_limit = self.prompt_len + gen_end_relative
        return min(active_limit, self.max_length)

    def propose(self, S: int = 1):
        if self.t is None: self.reset()
        if self.t <= 0: raise ValueError("No remaining timesteps.")

        if self.dummy:
            # Dummy generation logic
            vocab_limit = getattr(self.tokenizer, "vocab_size", 32000)
            proposals = np.random.randint(0, vocab_limit, size=(self.S, self.N, self.max_length))
            if self.prompt_ids is not None:
                proposals[:, :, :self.prompt_len] = self.prompt_ids
            self.tentative_states = proposals
            return proposals

        device = next(self.model.parameters()).device
        x_torch = torch.from_numpy(self.X).long().to(device)
        step_index = self._step_index_from_t(self.t)
        
        current_transfer_count = self.transfer_schedule[:, step_index] 
        active_limit = self._get_active_limit(step_index)

        proposals = []
        for _ in range(self.N):
            x_next = llada_single_step(
                self.model, x_torch,
                transfer_count=current_transfer_count,
                prompt_len=self.prompt_len,
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
        self.X = np.asarray(state, dtype=np.int64)
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


# --------------------------------------------------
# Main Execution (Collection)
# --------------------------------------------------
if __name__ == "__main__":
    dummy = False 
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer load failed ({e}). Using mock if dummy=True.")
        if not dummy: raise e
        tokenizer = None

    if not dummy:
        print("Loading LLaDA Model with 4-bit Quantization...")
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
        print("Model loaded on:", next(model.parameters()).device)
    else:
        print("!!! DUMMY MODE: Skipping Model Load !!!")
        model = None

    prompt_text = "Write a story of exactly 3 sentences. Do not start with cliched openings like 'Once upon a time'"
    
    if tokenizer:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    else:
        prompt_ids = [128000, 500, 501] 
        
    prompt_len = len(prompt_ids)
    print("Prompt length:", prompt_len)

    S = 20                       
    N = 1
    T_steps = 32                 
    gen_len = 32                
    max_length = prompt_len + gen_len

    cfg = ModelConfig()
    cfg.model_name = "llada"
    cfg.state_shape = (max_length,)
    cfg.index_set = list(range(T_steps + 1))
    cfg.deets = {
        "steps": T_steps,
        "hf_model_id": hf_model_id,
        "mask_id": 126336,
        "temperature": 1,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "prompt_len": prompt_len,
        "block_len": 8, 
        "dummy": dummy 
    }
    
    core = LLaDACore(cfg, tokenizer, model, S=S, N=N)
    # The set_prompt call now records the prompt in the config
    core.set_prompt(prompt_ids)
    core.reset()

    # Define Reward for Collection
    class LLaDAReward:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, token_ids):
            token_ids = np.array(token_ids)
            if self.tokenizer:
                if token_ids.ndim == 2:
                    texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                else:
                    texts = [self.tokenizer.decode(token_ids, skip_special_tokens=True)]
            else:
                count = len(token_ids) if token_ids.ndim == 2 else 1
                texts = ["dummy text"] * count
                
            scores = []
            for text in texts:
                if "dummy" in text: 
                    scores.append(0.5)
                else:
                    scores.append(0.1) # Placeholder reward
            
            return np.array(scores) if token_ids.ndim == 2 else scores[0]

    reward_calculator = LLaDAReward(tokenizer)
    
    os.makedirs("data_runs", exist_ok=True)
    collector = Collector(core, reward_calculator, base_dir="data_runs")
    output = collector.collect()
    
    print("\nSuccess! Data collected at:", output['run_dir'])