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


from collection import Collector
from diffusion_core import Core
from mixture_model import MixtureModel
from nets import ValueEstimate, LogUBaselineNet, train_val_estimate, load_val_estimate

# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

class LCB(LogUBaselineNet):
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
    ):
        super().__init__(dim)

        self.in_dim = dim

        self.value_estimator = value_estimator
        self.delta = float(delta)
        self.T = core.T

        self.trained_t = [False]*(self.T)

        self.index_set = core.model_config["index_set"]  # length T+1 or T
        self.device = getattr(value_estimator, "device", torch.device("cpu"))
        self.erm_error = float(erm_error)

        # per-timestep lambda*; initialized to None (will be filled by search)
        # we associate lambda[t] with the transition x_{t+1} -> x_t
        self.lambdas = [None] * self.T
        self.taus = [None] * self.T

        # lambda search bracket (global for all t; you can make this per-t if you want)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)

        # fixed vs learnable lambda
        self.fix_lambda = bool(fix_lambda)
        self.lambda_value = float(lambda_value)

        self.symmetric = symmetric

        # optimizer for the LCB baseline network (self.lcb or self.net)
        # assumes LogUBaselineNet defines self.lcb; if it's self.net, change below
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3)
        
        # colocate on same device as value_estimator
        self.to(self.device)

    def _current_lambda_for_t(self, t: int) -> torch.Tensor:
        """
        Returns a torch scalar (on self.device) with the current λ for timestep t.
        If self.fix_lambda, always returns self.lambda_value.
        Otherwise uses self.lambdas[t] if known, else self.lambda_value as init.
        """
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
            raise RuntimeError(f"Cannot calibrate tau at t={tp1-1} before training LCB baseline there.")
        
        lam = self._current_lambda_for_t(tp1 - 1).cpu().numpy()
        
        
        logvt = (self.value_estimator.predict_under_log(
                Xt, self.index_set[tp1 - 1]
            ))
        logu = self.predict_under_log(Xtp1, self.index_set[tp1], requires_grad=False)
        s = logvt - logu  # (S,)
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
        """
        Monte Carlo estimate of the step-t LCB objective for a given λ.

        logvt: log v_t(Xt) or similar, TENSOR WITHOUT GRAD shape (S,)
        logu:  log baseline u_{tp1}(X_tp1), TENSOR WITH GRAD of shape shape (S,) 
        lam_val: Python float (λ >= 1)

        Returns a Python float.
        """
        lam = torch.tensor(lam_val, dtype=torch.float32, device=self.device)

        # s = logvt - logu
        s = logvt - logu  # (S,)

        # term1 = (1/λ) * log mean exp(λ * s)
        m1 = torch.max(lam * s)
        term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam
        term2 = None
        # term2 = log mean exp(-s)
        if not self.symmetric:
            m2 = torch.max(-s)
            term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))
        else:
            m2 = torch.max(-lam*s)
            term2 = (m2 + torch.log(torch.mean(torch.exp(-lam * s - m2)))) / lam
        # term3 = (1/λ) * log(1/δ)
        term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam

        # term4 = ERM slack (constant)
        term4 = torch.tensor(self.erm_error, dtype=torch.float32, device=self.device)

        loss = term1 + term2 + term3 + term4
        return float(loss.item())

    # ---------------------------------------------------------
    # 1-D minimization for λ via golden-section search
    # ---------------------------------------------------------
    def _golden_section_search_lambda(
        self,
        logvt: torch.Tensor,
        logu: torch.Tensor,
        t: int,
        max_iter: int = 25,
        tol: float = 1e-3,
    ) -> float:
        """
        Minimize J_hat(λ) over [lam_min, lam_max] for a fixed batch and timestep t,
        assuming J is unimodal in λ.

        Returns lam_star as a Python float and stores it in self.lambdas[t].
        """
        a = self.lam_min
        b = self.lam_max
        phi = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618

        # interior points
        c = b - phi * (b - a)
        d = a + phi * (b - a)

        f_c = self._lcb_objective_with_lambda(logvt, logu, c)
        f_d = self._lcb_objective_with_lambda(logvt, logu, d)

        for _ in range(max_iter):
            if abs(b - a) < tol:
                break

            if f_c < f_d:
                # minimum in [a, d]
                b, d, f_d = d, c, f_c
                c = b - phi * (b - a)
                f_c = self._lcb_objective_with_lambda(logvt, logu, c)
            else:
                # minimum in [c, b]
                a, c, f_c = c, d, f_d
                d = a + phi * (b - a)
                f_d = self._lcb_objective_with_lambda(logvt, logu, d)

        lam_star = 0.5 * (a + b)
        self.lambdas[t] = lam_star
        return lam_star

    # ---------------------------------------------------------
    # Main training step: alternating baseline + λ search
    # ---------------------------------------------------------
    def train_on_step(self, Xt, Xtp1, tp1: int, baseline_steps: int = 1):
        """
        Alternating minimization for the LCB objective at timestep t = tp1 - 1.

        Xt:   states at time t   (S, d) as np.ndarray or torch.Tensor
        Xtp1: states at time t+1 (S, d) as np.ndarray or torch.Tensor
        tp1:  integer index for t+1 (so t = tp1 - 1)
        baseline_steps: how many GD steps on the baseline to do for this batch
                        before performing the 1-D λ search.
        """
        Xt = torch.as_tensor(Xt, dtype=torch.float32, device=self.device)
        Xtp1 = torch.as_tensor(Xtp1, dtype=torch.float32, device=self.device)

        t = tp1 - 1
        if t < 0 or t >= self.T:
            raise ValueError(f"train_on_step: got tp1={tp1}, so t={t} is out of range.")

        # --------- (A) baseline update with λ fixed ---------
        lam_fixed = self._current_lambda_for_t(t)  # scalar tensor

        for _ in range(baseline_steps):
            logvt = torch.tensor(self.value_estimator.predict_under_log(
                Xt, self.index_set[tp1 - 1]
            )).to(self.device)
            logu = self.predict_under_log(Xtp1, self.index_set[tp1], requires_grad=True).to(self.device)
            # print(logu)
            # print("shapes logvt, logu:", logvt.shape, logu.shape)

            s = logvt - logu  # (S,)

            # term1 = (1/λ) * log mean exp(λ * s)
            m1 = torch.max(lam_fixed * s)
            term1 = (m1 + torch.log(torch.mean(torch.exp(lam_fixed * s - m1)))) / lam_fixed

            # term2 = log mean exp(-s)
            term2 = None
            if self.symmetric:
                m2 = torch.max(-lam_fixed*s)
                term2 = m2 + torch.log(torch.mean(torch.exp(-lam_fixed*s - m2)))/lam_fixed
            else:
                m2 = torch.max(-s)
                term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))

            # term3 = (1/λ) * log(1/δ)
            term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam_fixed

            # term4 = ERM slack
            term4 = torch.tensor(self.erm_error, dtype=torch.float32, device=self.device)

            loss = term1 + term2 + term3 + term4

            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimizer.step()

        # --------- (B) 1-D minimization over λ (if not fixed) ---------
        if not self.fix_lambda:
            # with the baseline now updated, we freeze it and search λ
            with torch.no_grad():
                logvt = torch.as_tensor(
                    self.value_estimator.predict_under_log(Xt, self.index_set[tp1 - 1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                logu = torch.as_tensor(
                    self.predict_under_log(Xtp1, self.index_set[tp1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                # print("logu = ", logu)
            self._golden_section_search_lambda(logvt, logu, t)


    # ---------------------------------------------------------
    # Forward / predict_under_log for the baseline network
    # ---------------------------------------------------------
    def forward(self, inp):
        return super().forward(inp).squeeze(-1)

    def predict_under_log(self, x, index, requires_grad: bool = False):
        """
        x: tensor/array with trailing dim = d (e.g., (S, d) or (T, S, d)).
        index: scalar time index or array broadcastable to the leading dims of x.

        Returns tensor shaped like x[..., 0] (leading dims only) with log baseline values.
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        if isinstance(index, (int, np.integer)):
            idx_val = float(self.index_set[index])
            idx_feat = torch.full((x_flat.shape[0], 1), idx_val, dtype=torch.float32, device=self.device, requires_grad=requires_grad)
        else:
            index_t = torch.as_tensor(index, dtype=torch.float32, device=self.device)
            if index_t.dim() == 0:
                idx_feat = index_t.expand(x_flat.shape[0]).unsqueeze(-1)
            elif index_t.shape == orig_shape:
                idx_feat = index_t.reshape(-1, 1)
            elif index_t.dim() == 1 and index_t.shape[0] == x_flat.shape[0]:
                idx_feat = index_t.unsqueeze(-1)
            else:
                raise ValueError(f"Incompatible index shape {tuple(index_t.shape)} for x shape {tuple(x.shape)}")

        
        inp = torch.cat([x_flat, idx_feat], dim=-1)

        # print("inp shape in predict_under_log:", inp.shape)
        # out = self.forward(inp)  # (batch,)

        self.net.eval()
        if requires_grad:
            predictions = self.forward(inp)
            preds_pos = torch.nn.functional.softplus(predictions) + 1e-6
            # print(preds_pos)
            return torch.log(preds_pos).reshape(*orig_shape)
        with torch.no_grad():
            predictions = self.forward(inp)
            
            preds_pos = torch.nn.functional.softplus(predictions) + 1e-6
        # print(preds_pos)
        return np.log(preds_pos.cpu().numpy()).reshape(*orig_shape)

        # map to positive baseline then log it, mirroring ValueEstimate.predict_under_log
        # you can tweak this if your baseline is already in log-space.
        eps = 1e-6
        return torch.log(F.softplus(out) + eps).reshape(*orig_shape)
    
    def save(self, run_dir, run_id, alpha, name = ""):
        """
        Save state_dict and a small metadata JSON describing how to reload.
        """
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
        metadata_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_metadata_temp_{alpha_str}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        torch.save(self.state_dict(), ckpt_path)
        return {"ckpt": ckpt_path}
    
    @staticmethod
    def load_lcb(run_dir, run_id, value_estimator, core, alpha, sym_str="", device="cpu", name = ""):
        """
        Load the LCB model from a saved state_dict and metadata.
        """
        alpha_str = str(alpha).replace('.', 'p')
        metadata_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_metadata_temp_{alpha_str}.json")
        ckpt_path = os.path.join(run_dir, f"{name}_{run_id}_lcb_temp_{alpha_str}{sym_str}.pt")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        # Reconstruct LCB with saved hyperparameters
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
        
        # Load state dict
        lcb.load_state_dict(torch.load(ckpt_path, map_location=device))
        # lcb.to(device)
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
        self.acceptance_probabilities = []
        print("LCB Model has S = ", self.S)

    def train(self):
        self.core.reset()
        X = self.core.propose()
        self.core.commit(X[:, 0, :])
        self.lcb.trained_t[0] = True
        self.lcb.taus[0] = True
        Xtp1 = self.core.X
        while self.core.t > 0:
            print(f"Training LCB at t={self.core.t}...")
            for k in range(self.num_train_steps_per_t):
                Xt = self.core.propose()[:, 0, :]
                self.lcb.train_on_step(Xt, Xtp1, self.core.t, self.n_baseline_epochs_per_step)
            self.lcb.trained_t[self.core.t] = True
            Xt = self.core.propose()[:, 0, :]
            self.lcb.calibrate_tau(Xt, Xtp1, self.core.t)
            Xtp1, _ = self.lcb_step() # t decremented when lcb calls core.commit 
        return Xtp1

    def lcb_step(self):
        tp1 = self.core.t
        if self.lcb.trained_t[tp1] == False:
            raise ValueError(f"Attempted LCB step at t+1={tp1} before training on that timestep.")

        Xtp1 = self.core.X # (S, state_shape)
        self.core.N = 5
        S = Xtp1.shape[0]
        active = list(range(S))
        accepted_states = [None]*S
        num_proposals = [0]*S
    
        while active:
            Xt = self.core.propose_external(Xtp1[active], tp1, N = self.core.N)  # shape (N, len(active), d) (N must be set >= 2)
            vt = self.value_estimator.predict_under_log(Xt, self.index_set[tp1-1])                    # (N, len(active))
            # print("shapes Xt, vt:", Xt.shape, vt.shape)
            btp1 = self.lcb.predict_under_log(Xtp1[active], self.index_set[tp1])          # (len(active),)
            # print("shape btp1:", btp1.shape)
            # print("btp1 =", btp1)
            tau_t = self.lcb.taus[tp1]*np.ones(vt.shape, dtype=np.float32)          # (N, len(active))
            acc = np.exp(vt - btp1[:, None] - tau_t)                          # (N, len(active))
            print(acc)
            # print("shape acc:", acc.shape)
            # print("acc:", acc)
            u = np.random.rand(*acc.shape)
            new_active = []
            for j, orig_idx in enumerate(active):
                # print("j, orig_idx:", j, orig_idx)
                idx = np.where(acc[j, :] > u[j, :])[0]
                if idx.size:
                    first = idx[0]
                    accepted_states[orig_idx] = Xt[j, first]  # Xt is already np
                    num_proposals[orig_idx] += first + 1
                else:
                    new_active.append(orig_idx)
                    num_proposals[orig_idx] += self.core.N
                    # print("REJECTED!")
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
        """
        Roll out samples using the trained LCB, returning an array of shape (T+1, S, d).
        """
        self.core.reset()
        self.num_steps = 0

        # initial noisy state at time T
        init_props = self.core.propose()  # shape (S, N, d) or (S, d)
        if init_props.ndim == 3:
            x_init = init_props[:, 0, :]
        else:
            x_init = init_props
        self.core.commit(x_init)


        T_plus_1 = self.core.T + 1
        d = x_init.shape[-1]
        states = np.empty((T_plus_1, self.S, d))
        states[self.core.t + 1] = x_init  # time index T

        while self.core.t > 0:
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
        
        while self.core.t > 0:
            Xtp1, _ = self.lcb_step() # t decremented when lcb calls core.commit 
        return Xtp1


from animate_2d import animate_val_predictions


def plot_val_grid(model, t_index, xlims, ylims, steps=50, index_set=None, outfile=None):
    """
    Plot a single frame of value predictions on a 2D grid for timestep t_index.
    """
    xs = np.linspace(xlims[0], xlims[1], steps)
    ys = np.linspace(ylims[0], ylims[1], steps)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)

    if index_set is None:
        raise ValueError("index_set is required to build the index feature.")
    t_feat = np.full(len(pts), index_set[t_index], dtype=np.float32)

    preds = model.predict(pts, t_feat)  # shape (N,)
    Z = preds.reshape(Xg.shape)

    plt.figure(figsize=(6, 5))
    cp = plt.contourf(Xg, Yg, Z, levels=30, cmap="viridis")
    plt.colorbar(cp, label="Predicted value")
    plt.title(f"Value predictions at t={t_index}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

from animate_2d import animate_val_predictions
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

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
    # Load core, trajectories, and trained value estimator

    value_estimator, states, index_set, _ = load_val_estimate(args.run_id, alpha, device="mps")
    collector = Collector(None, None)
    core, states, expr = collector.load_collected(run_dir)


    core.S = 5000
    print("states shape:", states.shape)
    in_dim = value_estimator.in_dim

    # Train LCB ################
     

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
    X = model.train() # X = (S, d)
    lcb.save(run_dir, args.run_id, alpha)

    # lcb = lcb.load_lcb(run_dir, args.run_id, value_estimator, core, alpha)
    # model = LCBModel(core, lcb)
    # X = model.calc_samples()  # X = (T, S, d)


    # print("lambdas = ", lcb.lambdas)
    # print("taus = ", lcb.taus)
    # print("index_set = ", index_set)
    # print('yooo')
    # print(lcb.predict_under_log(states[0, :, :], index_set[0]))
    # print("-----")
    # print(lcb.predict_under_log(states[1, :, :], index_set[1]))
    # lcb.predict_under_log(states[0, 10, :], index_set[0])

    animate_val_predictions(states, index_set, lcb, outfile=f"{run_dir}/lcb_{alpha_str}.gif")
    # animate_val_predictions(states, index_set, value_estimator, outfile="value.gif")
    # print(X.shape)

    # plt.scatter(X[-1,:, 0], X[-1, :, 1], c='blue', alpha=0.5)
    # bar plot of the first coordinate of each X
    # plt.hist(X[-1, :, 0], bins=100, alpha=0.7)
    # plt.show()
    ###############################

    # Load trained LCB ################

    # lcb = LCB.load_lcb(run_dir, args.run_id, value_estimator, core, device="mps")
    # lcb.trained_t[0] = True # Need this hack because the loaded lcb was trained before I remembered to set the first trained_t to true 
    # lcb.taus[0] = 1
    # print(lcb.lambdas)

    # animate_val_predictions(states, index_set, lcb, outfile="lcb.gif")
    # animate_val_predictions(states, index_set, lcb, outfile="value.gif")

    # model = LCBModel(core, lcb)
    
    # X = model.calc_samples()  # X = (T, S, d)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
    plt.show()

    #################################
    
   

    

    print("num proposals = ", model.num_steps)


    # print("Trained lambdas per timestep:", lambdas)
