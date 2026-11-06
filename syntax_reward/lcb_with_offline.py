from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer

from nets_language_classifier import ValueClassifier

# -----------------------------------------------------------------------------
# 1. LCB Offline Dataset (Analogous to TrajectoryDataset)
# -----------------------------------------------------------------------------
class LCBOfflineDataset(Dataset):
    def __init__(self, run_dir, index_set, targets_file, model_name="distilroberta-base"):
        # Expects standard Collector output structure
        states_path = next((os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.endswith("_states.npy")), None)
        targets_path = targets_file # os.path.join(run_dir, "{targets_file_name}.npy") # We will generate this specific file
        print()
        
        if not states_path or not os.path.exists(targets_path):
            raise FileNotFoundError(f"Missing states.npy or lcb_targets.npy in {run_dir}")

        self.states = np.load(states_path)   # (T, S, L)
        self.targets = np.load(targets_path) # (T, S)
        
        # Flatten
        self.T, self.S, self.L = self.states.shape
        self.flat_states = self.states.reshape(-1, self.L)
        self.flat_targets = self.targets.reshape(-1)
        
        # Time Mapping: Collector saves [t=32, t=31... t=0].
        # We construct a flat time array matching this order.
        self.index_set = index_set
        # Create array of diffusion times corresponding to generation steps
        # Assuming index_set maps generation_step -> diffusion_t
        times_per_step = [self.index_set[t] for t in range(self.T)]
        self.flat_times = np.repeat(times_per_step, self.S) # Repeat for batch S

        # Tokenizers
        self.src_tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tgt_mask_token = self.tgt_tokenizer.mask_token

    def __len__(self):
        return len(self.flat_states)

    def __getitem__(self, idx):
        llada_ids = self.flat_states[idx]
        target_v = self.flat_targets[idx]
        t_val = self.flat_times[idx]
        
        # Transcode tokens (LLaDA -> RoBERTa)
        text = self.src_tokenizer.decode(llada_ids, skip_special_tokens=False)
        text = text.replace("<|mdm_mask|>", self.tgt_mask_token).replace("<|endoftext|>", "")
        
        enc = self.tgt_tokenizer(
            text, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        
        return (
            enc.input_ids.squeeze(0),
            torch.tensor(t_val, dtype=torch.float32),
            torch.tensor(target_v, dtype=torch.float32)
        )

class LCBNetwork(nn.Module):
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
        if t.ndim == 1: t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t).unsqueeze(1)
        inputs_embeds = inputs_embeds + t_emb
        outputs = self.backbone(inputs_embeds=inputs_embeds)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_embedding).squeeze(-1)

class LCB(nn.Module): 
# learns to minimize log E e^(lambda(v_t(xt) - u_tp1(xtp1)))/lambda + log E e^(utp1(xtp1 - v_t(xt))) + log(1/delta)/lambda by directly optimizing lambda
    def __init__(
        self,
        value_classifier: ValueClassifier,
        core,
        delta: float = 0.1,
        run_dir = None,
        fix_lambda: bool = False,
        lambda_value: float = 2.0,
        lam_min: float = 1.0,
        lam_max: float = 20.0,
        erm_error: float = 0.0,
        lr: float = 1e-5,
        lambda_lr: float = 1e-3, 
        lambda_mode: str = "gradient",
        targets_file = None
    ):
        super().__init__()
        self.value_estimator = value_classifier
        self.device = value_classifier.device
        self.T = core.T
        self.index_set = core.model_config["index_set"]
        
        self.net = LCBNetwork("distilroberta-base").to(self.device)
        self.net_opt = optim.AdamW(self.net.parameters(), lr=lr)
        
        self.lambda_mode = lambda_mode
        self.fix_lambda = bool(fix_lambda)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.delta = float(delta)
        self.erm_error = float(erm_error)
        self.lambda_value = float(lambda_value)

        self.accumulated_gradients = False

        self.outer_steps = 1
        
        init_val = float(lambda_value)
        if self.lambda_mode == "gradient":
            self.lambdas = nn.Parameter(torch.full((self.T + 1,), init_val, device=self.device))
            self.lambda_opt = optim.Adam([self.lambdas], lr=lambda_lr)
        else:
            self.register_buffer("lambdas", torch.full((self.T + 1,), init_val, device=self.device))
            self.lambda_opt = None

        self.taus = [None] * (self.T + 1)
        self.trained_t = [False] * (self.T + 1)
        self.tokenizer = value_classifier.tgt_tokenizer
        self.src_tokenizer = value_classifier.src_tokenizer 

        self.targets_file_name = targets_file
        self.symmetric_objective = False

        self.offline_dataset = None
        if run_dir:
            self._load_data(run_dir)

    def _load_data(self, run_dir):
        """Initializes the offline dataset from the run directory."""
        try:
            self.offline_dataset = LCBOfflineDataset(run_dir, self.index_set, self.targets_file_name)
            print(f"LCB: Loaded offline dataset from {run_dir} ({len(self.offline_dataset)} samples)")
        except Exception as e:
            print(f"LCB Warning: Could not load offline data from {run_dir}: {e}")

    def _get_lambda_val(self, t):
        return self.lambdas[t].clamp(min=self.lam_min, max=self.lam_max)

    def _prepare_inputs(self, x_ids, t_idx):
        S = x_ids.shape[0]
        texts = self.src_tokenizer.batch_decode(x_ids, skip_special_tokens=False)
        clean = [t.replace("<|mdm_mask|>", self.value_estimator.tgt_mask_token).replace("<|endoftext|>", "") for t in texts]
        enc = self.tokenizer(clean, padding="longest", truncation=True, max_length=64, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        t_val = self.index_set[t_idx]
        t_tensor = torch.full((S,), t_val, dtype=torch.float32, device=self.device)
        return input_ids, t_tensor
    
    def regression_fit(self, epochs=5, batch_size=32):
        """Fits the baseline u(x) to the offline targets V(x)."""
        if not self.offline_dataset:
            print("LCB Error: No offline dataset loaded. Call with run_dir or check paths.")
            return

        print(f"\n>>> LCB Regression Fit ({epochs} epochs)...")
        loader = DataLoader(self.offline_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        criterion = nn.MSELoss()
        
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_ids, batch_t, batch_y in loader:
                batch_ids = batch_ids.to(self.device)
                batch_t = batch_t.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred = self.net(batch_ids, batch_t)
                loss = criterion(pred, batch_y)
                
                self.net_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.net_opt.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"  [Epoch {epoch+1}] MSE Loss: {avg_loss:.4f}")

        self.trained_t = [True] * (self.T + 1)

    def predict_under_log(self, x_ids, t_idx, requires_grad=False): # Forward pass in log space. 
        orig_shape = x_ids.shape
        if x_ids.ndim == 3: x_ids = x_ids.reshape(-1, x_ids.shape[-1])
            
        input_ids, t_tensor = self._prepare_inputs(x_ids, t_idx)
        
        if requires_grad:
            self.net.train()
            out = self.net(input_ids, t_tensor)
            # Linear output in log space.
            val = out 
        else:
            self.net.eval()
            with torch.no_grad():
                out = self.net(input_ids, t_tensor)
                val = out
            val = val.cpu().numpy()
            
        if len(orig_shape) == 3:
            if requires_grad: val = val.view(orig_shape[0], orig_shape[1])
            else: val = val.reshape(orig_shape[0], orig_shape[1])
        return val

    def _lcb_objective(self, logvt, logu, lam):
        # logu is the raw output (already in log space)
        s = logvt - logu 
        m1 = torch.max(lam * s)
        term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam
        term2 = None
        term3 = None
        if not self.symmetric_objective:
          m2 = torch.max(-1 * s)
          term2 = (m2 + torch.log(torch.mean(torch.exp(-1 * s - m2))))
          term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam
        else:
          m2 = torch.max(-lam * s)
          term2 = (m2 + torch.log(torch.mean(torch.exp(-lam * s - m2)))) / lam 
          term3 = 2*torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam
        loss = term1 + term2 + term3 + self.erm_error
        print("lcb loss = ", loss)
        return loss

    def train_on_step(self, Xt, Xtp1, tp1):
        t = tp1 - 1
        if Xt.ndim == 3: Xt = Xt.reshape(-1, Xt.shape[-1])
        if Xtp1.shape[0] != Xt.shape[0]:
            ratio = Xt.shape[0] // Xtp1.shape[0]
            Xtp1 = np.repeat(Xtp1, ratio, axis=0)
        outer_steps = self.outer_steps
        vt_np = self.value_estimator.predict_under_log(Xt, self.index_set[t])
        vt = torch.tensor(vt_np, device=self.device, dtype=torch.float32)
        
        for step in range(self.outer_steps):
            lam_curr = self._get_lambda_val(t).detach()
            
            input_ids, t_tensor = self._prepare_inputs(Xtp1, tp1)
            pred_raw = self.net(input_ids, t_tensor)
            
            # The network output is logu.
            u = pred_raw 
            
            loss_net = self._lcb_objective(vt, u, lam_curr)
            
            if not self.accumulated_gradients:
              self.net_opt.zero_grad()
              loss_net.backward()
              torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
              self.net_opt.step()
            
            if not self.fix_lambda and self.lambda_mode == "gradient":
                with torch.no_grad():
                    pred_new = self.net(input_ids, t_tensor)
                    logu_detached = pred_new.detach() 
                
                lam_param = self.lambdas[t]
                loss_lam = self._lcb_objective(vt, logu_detached, lam_param)
                
                self.lambda_opt.zero_grad()
                loss_lam.backward()
                self.lambda_opt.step()
                
                with torch.no_grad():
                    self.lambdas[t].clamp_(min=self.lam_min, max=self.lam_max)
                        
            elif not self.fix_lambda and self.lambda_mode == "search":
                with torch.no_grad():
                    u_np = self.predict_under_log(Xtp1, tp1, requires_grad=False)
                    u_t = torch.tensor(u_np, device=self.device)
                    self._golden_section_search_lambda(vt, u_t, t)
                    self.lambdas[t].clamp_(min=self.lam_min, max=self.lam_max)

    def _golden_section_search_lambda(self, logvt, logu, t, max_iter=30):
        a, b = self.lam_min, self.lam_max
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        for _ in range(max_iter):
            if abs(b - a) < 1e-3: break
            fc = self._lcb_objective(logvt, logu, c).item()
            fd = self._lcb_objective(logvt, logu, d).item()
            if fc < fd: b = d; d = c; c = b - phi * (b - a)
            else: a = c; c = d; d = a + phi * (b - a)
        self.lambdas[t] = (a + b) / 2.0

    def calibrate_tau(self, Xt, Xtp1, tp1):
        t = tp1 - 1
        if Xt.ndim == 3: Xt = Xt.reshape(-1, Xt.shape[-1])
        if Xtp1.shape[0] != Xt.shape[0]:
            ratio = Xt.shape[0] // Xtp1.shape[0]
            Xtp1 = np.repeat(Xtp1, ratio, axis=0)
            
        lam = self._get_lambda_val(t).item()
        logvt = self.value_estimator.predict_under_log(Xt, self.index_set[t])
        
        # Raw log output
        logu = self.predict_under_log(Xtp1, tp1, requires_grad=False)
        
        s = logvt - logu
        m1 = np.max(lam * s)
        term1 = (m1 + np.log(np.mean(np.exp(lam * s - m1)))) / lam
        term2 = math.log(1.0 / self.delta) / lam
        self.taus[tp1] = float(term1 + term2)

    def save(self, run_dir, filename="lcb_baseline"):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"{filename}.pt")
        lambdas_data = self.lambdas.data if isinstance(self.lambdas, nn.Parameter) else self.lambdas
        state = {
            "net": self.net.state_dict(),
            "lambdas": lambdas_data,
            "taus": self.taus,
            "trained_t": self.trained_t,
            "config": {
                "delta": self.delta, "fix_lambda": self.fix_lambda,
                "lambda_value": self.lambda_value, "lam_min": self.lam_min,
                "lam_max": self.lam_max, "erm_error": self.erm_error,
                "lambda_mode": self.lambda_mode
            }
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Saved LCB to {path}")

    @classmethod
    def load(cls, run_dir, collection_dir,  value_classifier, core, filename="lcb_baseline", targets_file=None):
        path = os.path.join(run_dir, f"{filename}.pt")
        meta_path = os.path.join(run_dir, f"{filename}_meta.json")
        if not os.path.exists(path): raise FileNotFoundError(f"No checkpoint at {path}")
        
        state = torch.load(path, map_location=value_classifier.device)
        cfg = state["config"]
        instance = cls(
            value_classifier, core,
            delta=cfg["delta"], fix_lambda=cfg["fix_lambda"],
            lambda_value=cfg["lambda_value"], lam_min=cfg["lam_min"],
            lam_max=cfg["lam_max"], erm_error=cfg["erm_error"],
            lambda_mode=cfg.get("lambda_mode", "gradient"),
            targets_file=targets_file
        )
        instance.net.load_state_dict(state["net"])
        if isinstance(instance.lambdas, nn.Parameter):
            instance.lambdas.data.copy_(state["lambdas"])
        else:
            instance.lambdas.copy_(state["lambdas"])
        instance.taus = state["taus"]
        instance.trained_t = state["trained_t"]
        if collection_dir:
            instance._load_data(collection_dir)
        return instance

class LCBSmartSampler:
    def __init__(self, core, lcb, max_retries=100, save_name = "lcb", run_specific_dir = "./", upper_bound = None):
        self.core = core
        self.lcb = lcb
        self.max_retries = max_retries
        self.train_batches_per_step = 1
        self.inference_nfe = 0
        self.training_nfe = 0
        self.num_proposals_list = [0]*self.core.S
        self.save_name = save_name
        self.run_specific_dir = run_specific_dir

        self.no_tau = False

        self.upper_bound = upper_bound

        self.n_trajectories = 50

    def train_policy(self, max_retries=None, n_trajectories=None):
        self.training_nfe = 0
        self.inference_nfe = 0
        current_retries = max_retries if max_retries is not None else self.max_retries
        n_traj = n_trajectories if n_trajectories is not None else self.n_trajectories

        for traj in range(n_traj):
            print(f"\n================ Trajectory {traj+1}/{n_traj} ================\n")
            self.core.reset()
            self.lcb.net_opt.zero_grad()
            while self.core.t > 0:
                tp1 = self.core.t
                print(f"--- Training Step {tp1} -> {tp1-1} ---")
                Xtp1 = self.core.X.copy()

                # Train baseline
                for _ in range(self.train_batches_per_step):
                    train_proposals = self._propose_smart(Xtp1, count_as_training=True)
                    self.lcb.train_on_step(train_proposals, Xtp1, tp1)

                # Calibrate tau + mark trained + monitor
                cal_proposals = self._propose_smart(Xtp1, count_as_training=True)
                self.lcb.calibrate_tau(cal_proposals, Xtp1, tp1)
                self.lcb.trained_t[tp1] = True

                if tp1 % 5 == 0:
                    self._print_vital_signs(cal_proposals, Xtp1, tp1)
                else:
                    lam_val = self.lcb._get_lambda_val(tp1 - 1).item()
                    print(f"  Lambda: {lam_val:.2f} | Tau: {self.lcb.taus[tp1]:.4f}")

                # Rejection step in blended phase
                self._lcb_step_rejection(tp1, max_retries=current_retries)
            if (traj + 1) % 2 == 0:
                print(f"  Saving checkpoint {traj+1}...")
                self.lcb.save(self.run_specific_dir, filename=f"{self.save_name}_ckpt_{traj+1}")
            if self.lcb.accumulated_gradients:
              torch.nn.utils.clip_grad_norm_(self.lcb.net.parameters(), 1)
              self.lcb.net_opt.step()
              self.lcb.net_opt.zero_grad()
        print(f"\nPolicy Learned. Saving FINAL to {self.save_name}...")
        self.lcb.save(self.run_specific_dir, filename=f"{self.save_name}")
        return self._get_stats()

        
    def _print_vital_signs(self, proposals, source_X, tp1):
            flat_props = proposals.reshape(-1, proposals.shape[-1])
            log_v = self.lcb.value_estimator.predict_under_log(flat_props, self.lcb.index_set[tp1-1])
            log_u = self.lcb.predict_under_log(source_X, tp1, requires_grad=False)
            if proposals.shape[1] > 1: log_u = np.repeat(log_u, proposals.shape[1])
            
            diff = log_v - log_u
            tau = self.lcb.taus[tp1]
            lam = self.lcb._get_lambda_val(tp1-1).item()
            if not tau:
              print("No tau, no vital signs")
              return
            print(f"  [Vital Signs t={tp1}] Lam: {lam:.2f} | Tau: {tau:.2f}")
            print(f"    -> Adv (V-u): {np.mean(diff):.2f} +/- {np.std(diff):.2f}")
            print(f".   ->  avg v = {np.mean(log_v):.2f}, avg_u = {np.mean(log_u):.2f}")
            print(f".   -> value variance {np.var(log_v):.2f}")
            print(f"    -> Baseline Var(u): {np.var(log_u):.3f}")
            print(f"    -> Pass Rate (>Tau): {np.mean(diff > tau):.1%}")

    def run_inference(self):
        self.core.reset()
        self.num_proposals_list = [0]*self.core.S
        self.inference_nfe = 0
        self.training_nfe = 0
        while self.core.t > 0:
            tp1 = self.core.t
            if not self.lcb.trained_t[tp1]:
                print(f"⚠️ Warning: LCB not trained at t={tp1}. Using Tau=0.")
                self.lcb.taus[tp1] = 0.0
            self._lcb_step_rejection(tp1, max_retries=self.max_retries)
        return self.core.X, self._get_stats()

    def _get_stats(self):
        return {"training_nfe": self.training_nfe, "inference_nfe": self.inference_nfe, "total_nfe": self.training_nfe + self.inference_nfe}

    def _propose_smart(self, source_X, active_indices=None, count_as_training=False):
        real_X = self.core.X
        real_schedule = getattr(self.core, "transfer_schedule", None)
        real_S = self.core.S
        if active_indices is not None:
            self.core.X = source_X[active_indices] 
            if real_schedule is not None: self.core.transfer_schedule = real_schedule[active_indices]
            self.core.S = len(active_indices)
        else:
            self.core.X = source_X
            self.core.S = len(source_X)
        proposals = self.core.propose() 
        count = self.core.S * self.core.N
        if count_as_training: self.training_nfe += count
        else: self.inference_nfe += count
        self.core.X = real_X
        if real_schedule is not None: self.core.transfer_schedule = real_schedule
        self.core.S = real_S
        return proposals

    def _lcb_step_rejection(self, tp1, max_retries=20):
        Xtp1 = self.core.X.copy()
        S = self.core.S
        N = self.core.N
        
        if max_retries == 0:
            candidates = self._propose_smart(Xtp1, active_indices=None, count_as_training=False)
            self.core.commit(candidates[:, 0, :])
            return

        active_mask = np.ones(S, dtype=bool)
        next_Xt = np.zeros_like(Xtp1)
        
        log_u_all = self.lcb.predict_under_log(Xtp1, tp1, requires_grad=False)
        tau = self.lcb.taus[tp1]
        # if self.no_tau:
        #   tau = 0
        if tau is None:
          tau = 0
        
        for attempt in range(max_retries):
            active_indices = np.where(active_mask)[0]
            for i in active_indices:
                self.num_proposals_list[i] = self.num_proposals_list[i] + 1
            if len(active_indices) == 0: break

            
            candidates = self._propose_smart(Xtp1, active_indices=active_indices, count_as_training=False)
            
            flat_cands = candidates.reshape(-1, candidates.shape[-1])
            log_v_flat = self.lcb.value_estimator.predict_under_log(flat_cands, self.lcb.index_set[tp1-1])
            log_v = log_v_flat.reshape(len(active_indices), N)
            
            log_u_subset = log_u_all[active_indices]
            
            Bs = log_u_subset[:, None] + tau
            if self.upper_bound:
              Bs = np.clip(Bs, None, self.upper_bound)
            log_acc = log_v - Bs
            
            log_rand = np.log(np.random.rand(len(active_indices), N) + 1e-10)
            is_accepted = log_rand < log_acc
            
            for i, global_idx in enumerate(active_indices):
                accepted_locs = np.where(is_accepted[i])[0]
                if len(accepted_locs) > 0: # this logic is not used as core.N = 1
                    first = accepted_locs[0]
                    next_Xt[global_idx] = candidates[i, first]
                    active_mask[global_idx] = False
                    unused = N - (first + 1)
                    self.inference_nfe -= unused

        if np.any(active_mask):
            active_indices = np.where(active_mask)[0]
            last_resort = self._propose_smart(Xtp1, active_indices=active_indices, count_as_training=False)
            for i, global_idx in enumerate(active_indices):
                next_Xt[global_idx] = last_resort[i, 0]
                self.inference_nfe -= (N - 1) # = 0
            
        self.core.commit(next_Xt)
