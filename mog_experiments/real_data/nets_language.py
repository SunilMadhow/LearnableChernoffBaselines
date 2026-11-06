import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer
import os
import json
import argparse

# -----------------------------------------------------------------------------
# 1. Dataset
# -----------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, states_path, rewards_path, index_set, critic_model_name, reward_temperature=1.0):
        # Load pre-calculated files
        self.states = np.load(states_path)   # (T, S, L)
        raw_rewards = np.load(rewards_path)  # (S,)
        
        # --- CRITICAL CHANGE: EXPONENTIATE REWARDS WITH TEMPERATURE ---
        # Theory requires V(x) = E[exp(r(x) / temp)]
        self.reward_temperature = reward_temperature
        self.rewards = np.exp(raw_rewards / reward_temperature) 
        
        self.index_set = index_set
        
        self.T, self.S, self.L = self.states.shape
        self.flat_states = self.states.reshape(-1, self.L)
        
        # Broadcast rewards
        if self.rewards.ndim == 1:
            self.flat_rewards = np.tile(self.rewards, (self.T, 1)).reshape(-1)
        else:
            self.flat_rewards = self.rewards.reshape(-1)

        # Tokenizers
        self.src_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(critic_model_name)
        
        self.llada_mask_id = 126336
        self.tgt_mask_token = self.tgt_tokenizer.mask_token

    def __len__(self):
        return len(self.flat_states)

    def __getitem__(self, idx):
        llada_ids = self.flat_states[idx]
        
        # 1. Translate Tokenizers
        text = self.src_tokenizer.decode(llada_ids, skip_special_tokens=False)
        text = text.replace("<|mdm_mask|>", self.tgt_mask_token)
        text = text.replace("<|endoftext|>", "").replace("<|begin_of_text|>", "")

        enc = self.tgt_tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.L, 
            return_tensors="pt"
        )
        
        return (
            enc.input_ids.squeeze(0),
            torch.tensor(self.index_set[idx // self.S], dtype=torch.float32), 
            torch.tensor(self.flat_rewards[idx], dtype=torch.float32) 
        )


# -----------------------------------------------------------------------------
# 2. Critic Network
# -----------------------------------------------------------------------------
class TransformerCritic(nn.Module):
    def __init__(self, model_name="distilroberta-base"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = 1
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        self.head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, t):
        inputs_embeds = self.backbone.embeddings(input_ids=input_ids)
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        inputs_embeds = inputs_embeds + t_emb
        
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return self.head(cls_embedding).squeeze(-1)


# -----------------------------------------------------------------------------
# 3. ValueEstimate Wrapper
# -----------------------------------------------------------------------------
class ValueEstimate:
    def __init__(self, run_dir=None, index_set=None, device="cuda", lr=1e-4, batch_size=32, model_name="distilroberta-base", reward_temperature=1.0):
        self.device = device
        self.model_name = model_name
        self.index_set = index_set
        self.reward_temperature = reward_temperature
        
        self.net = TransformerCritic(self.model_name).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.loader = None
        
        if run_dir:
            self._load_data(run_dir, batch_size)
        else:
            # Inference init
            self.src_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
            self.tgt_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tgt_mask_token = self.tgt_tokenizer.mask_token

    def _load_data(self, run_dir, batch_size):
        try:
            s_file = next((f for f in os.listdir(run_dir) if "states.npy" in f), None)
            r_file = next((f for f in os.listdir(run_dir) if "rewards.npy" in f), None)
            
            if not s_file or not r_file:
                raise FileNotFoundError("Missing states.npy or rewards.npy")
                
            self.dataset = TrajectoryDataset(
                os.path.join(run_dir, s_file), 
                os.path.join(run_dir, r_file), 
                self.index_set, 
                self.model_name,
                reward_temperature=self.reward_temperature
            )
            self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            
            self.src_tokenizer = self.dataset.src_tokenizer
            self.tgt_tokenizer = self.dataset.tgt_tokenizer
            self.tgt_mask_token = self.dataset.tgt_mask_token
            
        except Exception as e:
            print(f"Warning: Data load failed in {run_dir}: {e}")

    def fit(self, epochs=5):
        if not self.loader: return
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for tokens, t, target in self.loader:
                tokens, t, target = tokens.to(self.device), t.to(self.device), target.to(self.device)
                
                pred = self.net(tokens, t)
                loss = self.criterion(pred, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch+1}] MSE (on exp(r/{self.reward_temperature})): {total_loss / len(self.loader):.4f}")

    def predict_under_log(self, x, t_scalar):
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
            preds = self.net(x_in, t_tensor)
            out = torch.log(torch.nn.functional.softplus(preds) + 1e-6)
            
        return out.cpu().numpy()

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(save_dir, "value_net.pt"))
        meta = {
            "model_name": self.model_name, 
            "index_set": self.index_set,
            "reward_temperature": self.reward_temperature
        }
        with open(os.path.join(save_dir, "value_net_meta.json"), "w") as f:
            json.dump(meta, f)
            
    @classmethod
    def load(cls, save_dir, device="cuda"):
        with open(os.path.join(save_dir, "value_net_meta.json"), "r") as f:
            meta = json.load(f)
        
        # Load temperature from meta, default to 1.0 if missing (backward compatibility)
        temp = meta.get("reward_temperature", 1.0)
        
        instance = cls(
            None, 
            meta["index_set"], 
            device, 
            model_name=meta.get("model_name"),
            reward_temperature=temp
        )
        instance.net.load_state_dict(torch.load(os.path.join(save_dir, "value_net.pt"), map_location=device))
        print(f"Loaded ValueEstimate from {save_dir} (Temp={temp})")
        return instance

def train_critic(run_id, data_dir="data_runs", epochs=10, batch_size=32, lr=1e-4, device=None, temperature=1.0):
    run_path = os.path.join(data_dir, run_id)
    try:
        cfg_files = [f for f in os.listdir(run_path) if "_config.json" in f]
        if cfg_files:
            with open(os.path.join(run_path, cfg_files[0]), "r") as f:
                 idx_set = json.load(f)["index_set"]
        else:
            idx_set = list(range(65))
    except:
        idx_set = list(range(65))

    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ve = ValueEstimate(run_path, idx_set, device=device, lr=lr, batch_size=batch_size, reward_temperature=temperature)
    ve.fit(epochs)
    ve.save(run_path)
    return ve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--dir", default="data_runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()
    
    train_critic(args.run_id, args.dir, args.epochs, args.bs, temperature=args.temp)