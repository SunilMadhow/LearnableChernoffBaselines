import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import json
import random


from collection import Collector
from diffusion_core import Core
from mixture_model import MixtureModel
from nets import ValueEstimate, train_val_estimate, load_val_estimate

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import json
import random


from collection import Collector
from diffusion_core import Core
from mixture_model import MixtureModel
from nets import ValueEstimate, train_val_estimate, load_val_estimate

# --- Helper for single-step network ---
class SingleStepNet(nn.Module):
    def __init__(self, dim, hidden=(64, 64)):
        super().__init__()
        layers = []
        # Input is just dimension d (time is handled by selecting the network from the list)
        in_dim = dim 
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LCB(nn.Module):
    def __init__(
        self,
        dim, 
        value_estimator: ValueEstimate,
        core: Core,
        delta: float = 1e-3,
        symmetric: bool = True,
        fix_lambda: bool = False,
        lambda_value: float = 2.0,
        lam_min: float = 1.0,
        lam_max: float = 20.0,
        erm_error: float = 0.0,
        baseline_lr = 1e-3,
        grad_clipping = False,
        train_baseline = True,
    ):
        super().__init__()

        self.in_dim = dim
        self.value_estimator = value_estimator
        self.delta = float(delta)
        self.T = core.T

        self.trained_t = [False]*(self.T + 1) # +1 to be safe for T
        self.grad_clipping = grad_clipping
        self.index_set = core.model_config["index_set"] 
        self.device = getattr(value_estimator, "device", torch.device("cpu"))
        self.erm_error = float(erm_error)

        # per-timestep lambda*
        self.lambdas = [None] * (self.T + 1)
        self.taus = [None] * (self.T + 1)

        self.baseline_lr = baseline_lr
        # lambda search bracket
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)

        # fixed vs learnable lambda
        self.fix_lambda = bool(fix_lambda)
        self.lambda_value = float(lambda_value)

        self.train_baseline = train_baseline

        self.symmetric = symmetric

        # --- FIX: List of independent networks ---
        # We create one network for each timestep 0..T
        # Crucial: These nets take 'dim' (state only), NOT 'dim+1' (state+time)
        self.nets = nn.ModuleList([
            SingleStepNet(dim).to(self.device) for _ in range(self.T + 1)
        ])
        
        self.to(self.device)

    def _current_lambda_for_t(self, t: int) -> torch.Tensor:
        if self.fix_lambda:
            lam_val = self.lambda_value
        else:
            if self.lambdas[t] is None:
                lam_val = self.lambda_value
            else:
                lam_val = self.lambdas[t]
        return torch.tensor(lam_val, dtype=torch.float32, device=self.device)

    def calibrate_tau(self, Xt, Xtp1, tp1):
        if not self.trained_t[tp1]:
            raise RuntimeError(f"Cannot calibrate tau at t={tp1} before training LCB baseline there.")
        
        lam = self._current_lambda_for_t(tp1 - 1).cpu().numpy()
        
        with torch.no_grad():
            logvt = (self.value_estimator.predict_under_log(
                    Xt, self.index_set[tp1 - 1]
                ))
            # FIX: Pass integer tp1
            logu = self.predict_under_log(Xtp1, tp1, requires_grad=False)
            
        s = logvt - logu 
        m1 = np.max(lam * s)
        term1 = (m1 + np.log(np.mean(np.exp(lam * s - m1)))) / lam
        term2 = math.log(1/self.delta)/lam
        self.taus[tp1] = float((term1 + term2).item())

    def _lcb_objective_with_lambda(
        self,
        logvt: torch.Tensor,
        logu: torch.Tensor,
        lam_val: float,
    ) -> float:
        lam = torch.tensor(lam_val, dtype=torch.float32, device=self.device)
        s = logvt - logu 

        m1 = torch.max(lam * s)
        term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam
        term2 = None
        
        if not self.symmetric:
            m2 = torch.max(-s)
            term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))
        else:
            m2 = torch.max(-lam*s)
            term2 = (m2 + torch.log(torch.mean(torch.exp(-lam * s - m2)))) / lam

        term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam
        term4 = torch.tensor(self.erm_error, dtype=torch.float32, device=self.device)

        loss = term1 + term2 + term3 + term4
        return float(loss.item())

    def _golden_section_search_lambda(
        self,
        logvt: torch.Tensor,
        logu: torch.Tensor,
        t: int,
        max_iter: int = 25,
        tol: float = 1e-3,
    ) -> float:
        a = self.lam_min
        b = self.lam_max
        phi = (math.sqrt(5.0) - 1.0) / 2.0 

        c = b - phi * (b - a)
        d = a + phi * (b - a)

        f_c = self._lcb_objective_with_lambda(logvt, logu, c)
        f_d = self._lcb_objective_with_lambda(logvt, logu, d)

        for _ in range(max_iter):
            if abs(b - a) < tol:
                break

            if f_c < f_d:
                b, d, f_d = d, c, f_c
                c = b - phi * (b - a)
                f_c = self._lcb_objective_with_lambda(logvt, logu, c)
            else:
                a, c, f_c = c, d, f_d
                d = a + phi * (b - a)
                f_d = self._lcb_objective_with_lambda(logvt, logu, d)

        lam_star = 0.5 * (a + b)
        self.lambdas[t] = lam_star
        return lam_star

    def train_on_step(self, Xt, Xtp1, tp1: int, baseline_steps: int = 1):
        # FIX: Ensure inputs are 2D (S, d). If 1D, unsqueeze to (1, d)
        if Xt.ndim == 1: Xt = np.expand_dims(Xt, 0)
        if Xtp1.ndim == 1: Xtp1 = np.expand_dims(Xtp1, 0)

        Xt = torch.as_tensor(Xt, dtype=torch.float32, device=self.device)
        Xtp1 = torch.as_tensor(Xtp1, dtype=torch.float32, device=self.device)

        t = tp1 - 1
        
        # Create a fresh optimizer for this specific timestep's network
        optimizer = optim.Adam(self.nets[tp1].parameters(), lr=self.baseline_lr)
        
        lam_fixed = self._current_lambda_for_t(t)

        if self.train_baseline:
            for _ in range(baseline_steps):
                with torch.no_grad():
                    logvt = torch.tensor(self.value_estimator.predict_under_log(
                        Xt, self.index_set[tp1 - 1]
                    )).to(self.device)
                
                # FIX: Pass integer tp1 to select the correct network
                logu = self.predict_under_log(Xtp1, tp1, requires_grad=True).to(self.device)

                s = logvt - logu 

                m1 = torch.max(lam_fixed * s)
                term1 = (m1 + torch.log(torch.mean(torch.exp(lam_fixed * s - m1)))) / lam_fixed

                term2 = None
                if self.symmetric:
                    m2 = torch.max(-lam_fixed*s)
                    term2 = m2 + torch.log(torch.mean(torch.exp(-lam_fixed*s - m2)))/ lam_fixed
                else:
                    m2 = torch.max(-s)
                    term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))

                term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam_fixed
                term4 = torch.tensor(self.erm_error, dtype=torch.float32, device=self.device)

                loss = term1 + term2 + term3 + term4

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.nets[tp1].parameters(), max_norm=1.0)
                optimizer.step()

        if not self.fix_lambda:
            with torch.no_grad():
                logvt = torch.as_tensor(
                    self.value_estimator.predict_under_log(Xt, self.index_set[tp1 - 1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                logu = torch.as_tensor(
                    self.predict_under_log(Xtp1, tp1),
                    device=self.device,
                    dtype=torch.float32,
                )
            self._golden_section_search_lambda(logvt, logu, t)

    def predict_under_log(self, x, t_int, requires_grad: bool = False):
        """
        x: tensor/array (S, d)
        t_int: integer timestep to select the specific network
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        # FIX: Select specific network for timestep t_int
        net = self.nets[t_int]
        
        if requires_grad:
            net.train()
            out = net(x_flat)
            preds_pos = torch.nn.functional.softplus(out) + 1e-6
            # FIX: Use reshape(orig_shape) instead of reshape(*orig_shape)
            # This robustly handles scalar batches where orig_shape is ()
            return torch.log(preds_pos).reshape(orig_shape)
        else:
            net.eval()
            with torch.no_grad():
                out = net(x_flat)
                preds_pos = torch.nn.functional.softplus(out) + 1e-6
            return np.log(preds_pos.cpu().numpy()).reshape(orig_shape)

    def save(self, run_dir, run_id, alpha, name = ""):
        os.makedirs(run_dir, exist_ok=True)
        alpha_str = str(alpha).replace('.', 'p')
        sym_str = "SYM" if self.symmetric else "ASYM"
        ckpt_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_temp_{alpha_str}_{sym_str}.pt")
        
        metadata = {
            "dim": self.in_dim,
            "delta": self.delta,
            "symmetric": self.symmetric,
            "trained_t": self.trained_t,
            "lambda_values": self.lambdas,
            "taus": self.taus,
            "fix_lambda": self.fix_lambda,
            "lambda_value": self.lambda_value,
            "lam_min": self.lam_min,
            "lam_max": self.lam_max,
            "erm_error": self.erm_error,
        }
        metadata_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_metadata_temp_{alpha_str}_{sym_str}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        torch.save(self.state_dict(), ckpt_path)
        return {"ckpt": ckpt_path}
    
    @staticmethod
    def load_lcb(run_dir, run_id, value_estimator, core, alpha, sym_str="", device="cpu", name = ""):
        alpha_str = str(alpha).replace('.', 'p')
        metadata_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_metadata_temp_{alpha_str}_{sym_str}.json")
        ckpt_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_temp_{alpha_str}_{sym_str}.pt")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        lcb = LCB(
            dim=metadata["dim"],
            value_estimator=value_estimator,
            core=core,
            delta=metadata["delta"],
            symmetric=metadata["symmetric"],
            fix_lambda=metadata["fix_lambda"],
            lambda_value=metadata["lambda_value"],
            lam_min=metadata["lam_min"],
            lam_max=metadata["lam_max"],
            erm_error=metadata["erm_error"],
        )

        lcb.trained_t = metadata["trained_t"]
        lcb.lambdas = metadata["lambda_values"]
        lcb.taus = metadata["taus"]
        
        lcb.load_state_dict(torch.load(ckpt_path, map_location=device))
        return lcb


class LCBModel():
    def __init__(self, core: Core, lcb: LCB):
        self.core = core
        self.core.reset()
        self.lcb = lcb
        self.index_set = self.core.model_config["index_set"]
        self.value_estimator = lcb.value_estimator
        self.num_train_steps_per_t = 400
        self.num_steps = 0
        self.delta = lcb.delta
        self.S = self.core.S
        self.n_baseline_epochs_per_step = 1
        self.erm_slack = 0
        print("LCB Model has S = ", self.S)

    def train(self):
        self.core.reset()
        X = self.core.propose()
        self.core.commit(X[:, 0, :])
        self.lcb.trained_t[0] = True
        self.lcb.taus[0] = True
        Xtp1 = self.core.X
        while self.core.t > 0:
            # print(f"Training LCB at t={self.core.t}...")
            # --- FIX: Loop to generate fresh noise for each training step ---
            for k in range(self.num_train_steps_per_t):
                # propose_external returns (N, S, d). 
                Xt_fresh = self.core.propose_external(Xtp1, self.core.t, N=1)
                
                # FIX: Check dims to avoid squeezing the batch dim if N=1 was already squeezed by core
                if Xt_fresh.ndim == 3:
                     Xt_fresh = Xt_fresh[0] # Take first N sample, keeping (S, d)
                
                self.lcb.train_on_step(Xt_fresh, Xtp1, self.core.t, self.n_baseline_epochs_per_step)
            
            self.lcb.trained_t[self.core.t] = True
            
            # Calibrate on fresh noise
            Xt_calib = self.core.propose_external(Xtp1, self.core.t, N=1)
            if Xt_calib.ndim == 3: Xt_calib = Xt_calib[0]
            
            self.lcb.calibrate_tau(Xt_calib, Xtp1, self.core.t)
            
            Xtp1, _ = self.lcb_step() 
        return Xtp1

    def lcb_step(self):
        tp1 = self.core.t
        if not self.lcb.trained_t[tp1]:
            raise ValueError(f"Attempted LCB step at t+1={tp1} before training on that timestep.")

        Xtp1 = self.core.X 
        self.core.N = 5
        S = Xtp1.shape[0]
        active = list(range(S))
        accepted_states = [None]*S
        num_proposals = [0]*S
    
        while active:
            Xt = self.core.propose_external(Xtp1[active], tp1, N = self.core.N) 
            vt = self.value_estimator.predict_under_log(Xt, self.index_set[tp1-1])
            
            btp1 = self.lcb.predict_under_log(Xtp1[active], tp1)
            
            tau_t = self.lcb.taus[tp1]*np.ones(vt.shape, dtype=np.float32)
            acc = np.exp(vt - btp1[:, None] - tau_t)
            
            u = np.random.rand(*acc.shape)
            new_active = []
            for j, orig_idx in enumerate(active):
                idx = np.where(acc[j, :] > u[j, :])[0]
                if idx.size:
                    first = idx[0]
                    accepted_states[orig_idx] = Xt[j, first]
                    num_proposals[orig_idx] += first + 1
                else:
                    new_active.append(orig_idx)
                    num_proposals[orig_idx] += self.core.N
            active = new_active

        accepted_states = np.stack(accepted_states, axis=0)
        self.core.commit(accepted_states)
        self.num_steps = self.num_steps + sum(num_proposals)
        return accepted_states, num_proposals
    
    def set_S(self, S):
        self.core.S = S
        self.S = S
        self.core.reset()

    def calc_samples(self):
        self.core.reset()
        self.num_steps = 0

        init_props = self.core.propose()
        if init_props.ndim == 3:
            x_init = init_props[:, 0, :]
        else:
            x_init = init_props
        self.core.commit(x_init)

        T_plus_1 = self.core.T + 1
        d = x_init.shape[-1]
        states = np.empty((T_plus_1, self.S, d))
        states[self.core.t + 1] = x_init

        while self.core.t >= 0:
            states_t, _ = self.lcb_step()
            states[self.core.t] = states_t
        return states
    
    def infer(self):
        self.core.reset()
        X = self.core.propose()
        self.core.commit(X[:, 0, :])
        self.lcb.trained_t[0] = True
        self.lcb.taus[0] = True
        Xtp1 = self.core.X
        self.num_steps = 0
        while self.core.t > 0:
            Xtp1, _ = self.lcb_step() # t decremented when lcb calls core.commit 
        return Xtp1

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    import argparse
    parser = argparse.ArgumentParser(description="Train LCB baseline on a collected run.")
    parser.add_argument("run_id", help="Run ID under toy_data_runs to load.")
    parser.add_argument("alpha", type=float)
    args = parser.parse_args()

    run_dir = os.path.join("toy_data_runs", args.run_id)
    alpha = args.alpha
    alpha_str = str(alpha).replace('.', 'p')

    value_estimator, states, index_set, _ = load_val_estimate(args.run_id, alpha, device="mps")
    collector = Collector(None, None)
    core, states, expr = collector.load_collected(run_dir)

    core.S = 5000
    print("states shape:", states.shape)
    in_dim = value_estimator.in_dim

    lcb = LCB(
        in_dim, 
        value_estimator,
        core,
        delta = 1e-3,
        fix_lambda=False,
        symmetric= False,
        lambda_value=10,
        lam_min=1,
        lam_max=20.0,
        erm_error=0.0,
    )

    model = LCBModel(core, lcb)
    model.num_train_steps_per_t = 400
    model.num_baseline_epochs_per_step = 1
    
    # Train
    X = model.train()
    
    # Save
    lcb.save(run_dir, args.run_id, alpha)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
    plt.show()

    print("num proposals = ", model.num_steps)