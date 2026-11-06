from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import torch
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore, ModelConfig
# from nets_language import ValueEstimate


import random

def seed_everything(seed=42):
    # Standard Python randomness
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numerical libraries
    np.random.seed(seed)
    
    # PyTorch randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # CuDNN backend (Crucial for GPU reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Execute
seed_everything(42)
# --- Evaluation Helpers ---
import re
def get_constraint_details(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    targets = ['A', 'B', 'C']
    flags = {'A': False, 'B': False, 'C': False}
    for i in range(min(3, len(sentences))):
        clean_sent = re.sub(r'^[^a-zA-Z]+', '', sentences[i])
        if clean_sent and clean_sent[0].upper() == targets[i]:
            flags[targets[i]] = True
    return flags

def score_abc_text(text):
    d = get_constraint_details(text)
    return (int(d['A']) + int(d['B']) + int(d['C'])) / 3.0


def load_core_and_tokenizer(num_trajectories = 25):
  hf_model_id = "GSAI-ML/LLaDA-8B-Instruct"
  bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
  tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
  reward_fn = LLaDAReward(tokenizer)
  model = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
  model.eval()

  # Config
  prompt_text = "Write a creative story of exactly 3 sentences. Stylistically, it is preferred that the first word of each sentence start with 'A', 'B', and 'C' respectively."
  prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
  
  cfg = ModelConfig()
  cfg.model_name = "llada"
  cfg.state_shape = (32,)
  cfg.index_set = list(range(33))
  cfg.deets = {"steps": 32, "hf_model_id": hf_model_id, "mask_id": 126336, "temperature": 1.12, "remasking": "low_confidence", "prompt_len": len(prompt_ids), "block_len": 8}
  
  core = LLaDACore(cfg, tokenizer, model, S=num_trajectories, N=1)
  core.set_prompt(prompt_ids)

  return core, tokenizer

class LLaDAReward:
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, token_ids):
        token_ids_np = np.array(token_ids)
        if token_ids_np.ndim == 2:
            texts = self.tokenizer.batch_decode(token_ids_np.tolist(), skip_special_tokens=True)
            return np.array([score_abc_text(t) for t in texts])
        return score_abc_text(self.tokenizer.decode(token_ids_np.tolist(), skip_special_tokens=True))
