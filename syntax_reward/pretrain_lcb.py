from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# Import your modules
from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore, ModelConfig
from lcb_with_offline import LCB
# Import the user's Collector
from collection import SequentialCollector 

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

class LLaDAReward:
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, token_ids):
        token_ids_np = np.array(token_ids)
        if token_ids_np.ndim == 2:
            texts = self.tokenizer.batch_decode(token_ids_np.tolist(), skip_special_tokens=True)
            return np.array([score_abc_text(t) for t in texts])
        return score_abc_text(self.tokenizer.decode(token_ids_np.tolist(), skip_special_tokens=True))




def train_offline(run_id, data_dir, pretrain_data_id, num_samples=1000, epochs=5, batch_size=64, reward_temp=1.0, continue_training = False, save_name=None):
    run_specific_dir = os.path.join(data_dir, run_id)
    reward_temp_str = str(reward_temp).replace('.', 'p')
    if save_name is None:
      save_name = f"lcb_baseline_pretrained_temp_{reward_temp_str}"
    print(f"Pretrained LCB to be saved as  = {save_name}")
    # -------------------------------------------------------------------------
    # 1. Collect Data (using SequentialCollector)
    # -------------------------------------------------------------------------
    print(f"\n>>> [1/3] Collecting {num_samples} samples...")
    
    # Check if we already have a collection run in this folder
    # We look for a folder inside run_specific_dir/lcb_pretrain_data/
    # Or we just dump it directly? Let's use a subdir to keep it clean.
    collection_dir = os.path.join(run_specific_dir, "lcb_pretrain_data")
    if pretrain_data_id:
      collection_dir = os.path.join(collection_dir, pretrain_data_id)
    os.makedirs(collection_dir, exist_ok=True)
    
    # Check if states already exist to skip expensive collection
    states_file = None
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
    
    core = LLaDACore(cfg, tokenizer, model, S=num_samples, N=1)
    core.set_prompt(prompt_ids)
        
    for f in os.listdir(collection_dir):
        if f.endswith("_states.npy"):
            states_file = os.path.join(collection_dir, f)
            break
    if states_file:
        print(f"Found existing states: {states_file}")
    else:
        # Load Model
        # Run Collection
        collector = SequentialCollector(core, reward_fn, base_dir=collection_dir, batch_size=150)
        out = collector.collect()
        states_file = out['states']
        # SequentialCollector creates a timestamped subdir. We need to point LCB to *that* subdir.
        collection_dir = out['run_dir'] 

    # -------------------------------------------------------------------------
    # 2. Label Data (Distillation)
    # -------------------------------------------------------------------------
    print(f"\n>>> [2/3] Labeling Data (Distillation)...")
    targets_file = os.path.join(collection_dir, f"lcb_targets_temp_{reward_temp_str}.npy")
    # targets_file_name = f"lcb_targets_temp_{reward_temp_str}"

    vc = ValueClassifier.load(run_specific_dir)
    vc.reward_temperature = reward_temp
    
    if os.path.exists(targets_file):
        print(f"Found existing targets: {targets_file}")
    else:
        states = np.load(states_file) # (T, S, L)
        T, S, L = states.shape
        targets = np.zeros((T, S), dtype=np.float32)
        
        bs = 50
        for t_idx in range(T):
            diffusion_t = vc.index_set[t_idx] 
            step_states = states[t_idx]
            step_preds = []
            for i in range(0, S, bs):
                batch = step_states[i : i+bs]
                preds = vc.predict_under_log(batch, diffusion_t)
                step_preds.append(preds)
            targets[t_idx] = np.concatenate(step_preds)
            if t_idx % 10 == 0: print(f"  Labeled step {t_idx}")
            
        np.save(targets_file, targets)

    # -------------------------------------------------------------------------
    # 3. Train LCB
    # -------------------------------------------------------------------------
    print(f"\n>>> [3/3] Training LCB Baseline...")
    
    # We need a Core instance just to pass config/tokenizer to LCB init (api requirement)
    # We can use a dummy one or reload.
    # if 'core' not in locals():
    #     # Minimal dummy core setup if we skipped collection
    #     tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    #     class DummyCore:
    #         T = 32
    #         model_config = {"index_set": list(range(33))}
    #     core = DummyCore()
    
    if continue_training:
      print("continuing to train")
      lcb = LCB.load(run_specific_dir, collection_dir, vc, core, f"lcb_baseline_pretrained_temp_{str(reward_temp).replace('.', 'p')}", targets_file)
      # logic for loading LCB
    # Initialize LCB pointing to the data directory
    else:
      lcb = LCB(
          vc, core, 
          run_dir=collection_dir, # Loads dataset internally
          lr=1e-5,
          targets_file=targets_file
      )
    
    lcb.regression_fit(epochs=epochs, batch_size=batch_size)
    
    # Save final model back to the main run directory
    # save_path = os.path.join(run_specific_dir, "lcb_baseline_pretrained")
    lcb.save(run_specific_dir, filename= save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--pretrain_data_id", type=str, default=None)
    parser.add_argument("--continue_training", type=bool, default=False)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()
    
    train_offline(args.run_id, f"{_ROOT}/data_runs", pretrain_data_id = args.pretrain_data_id,   
                  num_samples=args.samples, epochs=args.epochs, reward_temp=args.temp, continue_training = args.continue_training, save_name = args.save_name)
