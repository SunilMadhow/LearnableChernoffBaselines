from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
import os
import json
import argparse
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Dataset (PRE-TOKENIZED + LENGTH OPTIMIZED)
# -----------------------------------------------------------------------------
class ClassificationDataset(Dataset):
    def __init__(self, states_path, rewards_path, index_set, critic_model_name):
        print(f"Loading Dataset from {states_path}...")
        self.states = np.load(states_path)   
        raw_rewards = np.load(rewards_path) 
        
        # Bins: [0.15, 0.45, 0.75] maps to Classes 0, 1, 2, 3
        bins = [0.15, 0.45, 0.75]
        self.labels = np.digitize(raw_rewards, bins) 
        
        self.index_set = index_set
        self.T, self.S, self.L = self.states.shape
        self.flat_states = self.states.reshape(-1, self.L)
        
        # Create Flattened Labels
        if self.labels.ndim == 1:
            self.flat_labels = np.tile(self.labels, (self.T, 1)).reshape(-1)
        else:
            self.flat_labels = self.labels.reshape(-1)

        # Create Flattened Time Indices (Critical for diagnostics)
        # index_set[0] corresponds to t=32 (Noise), index_set[-1] to t=0 (Clean)
        # We need to broadcast correctly.
        # self.index_set is usually [32, 31, ..., 0]
        # We need a flat array where the first S entries are index_set[0], next S are index_set[1]...
        self.flat_t = np.repeat(self.index_set, self.S)

        print("Initializing Tokenizers...")
        self.src_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(critic_model_name)
        self.tgt_mask_token = self.tgt_tokenizer.mask_token

        # --- PRE-TOKENIZATION ---
        print(f"Pre-tokenizing {len(self.flat_states)} states...")
        chunk_size = 5000 
        all_texts = []
        for i in tqdm(range(0, len(self.flat_states), chunk_size), desc="Decoding"):
            chunk = self.flat_states[i : i + chunk_size]
            texts = self.src_tokenizer.batch_decode(chunk, skip_special_tokens=False)
            texts = [t.replace("<|mdm_mask|>", self.tgt_mask_token).replace("<|endoftext|>", "").replace("<|begin_of_text|>", "") for t in texts]
            all_texts.extend(texts)

        print("Encoding for Critic...")
        encodings = self.tgt_tokenizer(
            all_texts, 
            padding="max_length", 
            truncation=True, 
            max_length=64,
            return_tensors="pt"
        )
        
        self.input_ids = encodings["input_ids"]
        print("Pre-tokenization Complete.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            torch.tensor(self.flat_t[idx], dtype=torch.float32), 
            torch.tensor(self.flat_labels[idx], dtype=torch.long)
        )


# -----------------------------------------------------------------------------
# 2. Classifier Critic Network
# -----------------------------------------------------------------------------
class ClassifierCritic(nn.Module):
    def __init__(self, model_name="distilroberta-base", num_classes=4):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_classes
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        self.head = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, t):
        inputs_embeds = self.backbone.embeddings(input_ids=input_ids)
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        inputs_embeds = inputs_embeds + t_emb
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)


# -----------------------------------------------------------------------------
# 3. ValueClassifier Wrapper
# -----------------------------------------------------------------------------
class ValueClassifier:
    def __init__(self, run_dir=None, index_set=None, device="cuda", lr=5e-5, batch_size=32, model_name="distilroberta-base"):
        self.device = device
        self.model_name = model_name
        self.index_set = index_set
        self.num_classes = 4 
        self.class_values = torch.tensor([0.0, 0.3333, 0.6667, 1.0], device=self.device)
        self.net = ClassifierCritic(self.model_name, num_classes=self.num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.loader = None
        self.reward_temperature = 1.0 

        if run_dir:
            self._load_data(run_dir, batch_size)
        else:
            self.src_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
            self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tgt_mask_token = self.tgt_tokenizer.mask_token

    def _load_data(self, run_dir, batch_size):
        try:
            s_file = next((f for f in os.listdir(run_dir) if "states.npy" in f), None)
            r_file = next((f for f in os.listdir(run_dir) if "rewards.npy" in f), None)
            
            if not s_file or not r_file: raise FileNotFoundError("Missing npy files")
                
            self.dataset = ClassificationDataset(
                os.path.join(run_dir, s_file), 
                os.path.join(run_dir, r_file), 
                self.index_set, 
                self.model_name
            )
            self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2) # 2 workers safe with pre-tokenized
            self.src_tokenizer = self.dataset.src_tokenizer
            self.tgt_tokenizer = self.dataset.tgt_tokenizer
            self.tgt_mask_token = self.dataset.tgt_mask_token
        except Exception as e:
            print(f"Warning: Data load failed in {run_dir}: {e}")

    def fit(self, epochs=5):
        if not self.loader: return
        self.net.train()
        print(f"Starting CLASSIFIER training on {self.device}...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Diagnostic Counters
            clean_correct = 0; clean_total = 0
            noisy_correct = 0; noisy_total = 0
            
            for tokens, t, labels in tqdm(self.loader, desc=f"Epoch {epoch+1}"):
                tokens, t, labels = tokens.to(self.device), t.to(self.device), labels.to(self.device)
                
                logits = self.net(tokens, t)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # --- Diagnostics ---
                preds = torch.argmax(logits, dim=1)
                matches = (preds == labels)
                
                # Assume t < 5 is Clean (Index > 27 approx), t > 25 is Noisy (Index < 7 approx)
                # Note: t here is the raw scalar (e.g. 0.0 or 32.0). 
                # LLaDA convention: 0 is clean, 32 is noise.
                
                clean_mask = (t < 5.0)
                noisy_mask = (t > 25.0)
                
                clean_correct += matches[clean_mask].sum().item()
                clean_total += clean_mask.sum().item()
                
                noisy_correct += matches[noisy_mask].sum().item()
                noisy_total += noisy_mask.sum().item()
            
            avg_loss = total_loss / len(self.loader)
            clean_acc = clean_correct / (clean_total + 1e-6)
            noisy_acc = noisy_correct / (noisy_total + 1e-6)
            
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Clean Acc: {clean_acc:.1%} | Noisy Acc: {noisy_acc:.1%}")

    def predict_under_log(self, x, t_scalar):
        T = getattr(self, 'reward_temperature', 1.0)
        self.net.eval()
        if isinstance(x, np.ndarray):
             texts = self.src_tokenizer.batch_decode(x, skip_special_tokens=False)
             clean = [t.replace("<|mdm_mask|>", self.tgt_mask_token).replace("<|endoftext|>", "") for t in texts]
             enc = self.tgt_tokenizer(clean, padding="longest", truncation=True, return_tensors="pt")
             x_in = enc.input_ids.to(self.device)
        else:
             x_in = x.to(self.device)

        t_tensor = torch.full((x_in.shape[0],), t_scalar, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.net(x_in, t_tensor)
            probs = torch.softmax(logits, dim=1) 
            exp_rewards = torch.exp(self.class_values / T) 
            expected_exp_value = (probs * exp_rewards.unsqueeze(0)).sum(dim=1)
            out = torch.log(expected_exp_value + 1e-6)
            
        return out.cpu().numpy()

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(save_dir, "value_classifier.pt"))
        meta = {"model_name": self.model_name, "index_set": self.index_set, "type": "classifier"}
        with open(os.path.join(save_dir, "value_classifier_meta.json"), "w") as f: json.dump(meta, f)
            
    @classmethod
    def load(cls, save_dir, device="cuda"):
        with open(os.path.join(save_dir, "value_classifier_meta.json"), "r") as f: meta = json.load(f)
        instance = cls(None, meta["index_set"], device, model_name=meta.get("model_name"))
        instance.net.load_state_dict(torch.load(os.path.join(save_dir, "value_classifier.pt"), map_location=device))
        print(f"Loaded ValueClassifier from {save_dir}")
        return instance

def train_classifier_critic(run_id, data_dir=f"{_ROOT}/data_runs", epochs=5, batch_size=32, lr=5e-5, device=None):
    run_path = os.path.join(data_dir, run_id)
    try:
        cfg_files = [f for f in os.listdir(run_path) if "_config.json" in f]
        if cfg_files:
            with open(os.path.join(run_path, cfg_files[0]), "r") as f: idx_set = json.load(f)["index_set"]
        else: idx_set = list(range(65))
    except: idx_set = list(range(65))

    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vc = ValueClassifier(run_path, idx_set, device=device, lr=lr, batch_size=batch_size)
    vc.fit(epochs)
    vc.save(run_path)
    return vc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--dir", default=f"{_ROOT}/data_runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32)
    args = parser.parse_args()
    
    train_classifier_critic(args.run_id, args.dir, args.epochs, args.bs)
