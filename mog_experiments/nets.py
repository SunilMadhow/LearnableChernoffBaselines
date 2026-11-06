import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collection import Collector

class LogUBaselineNet(nn.Module):
    """
    Simple MLP returning a scalar per input (log upper baseline / value).
    """

    def __init__(self, in_dim, hidden_sizes=(64, 64), device=None):
        super().__init__()
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        layers = []
        dim = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(dim, h), nn.ReLU(inplace=True)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, inp):
        return self.net(inp).squeeze(-1)


class TrajectoryDataset(Dataset):
    """
    Trajectory dataset for value estimation.

    X is expected to have shape (T, S, d); we iterate over all (i, t) pairs.
    index_set is a length-T array of index features.
    expr is shape (S,).
    """

    def __init__(self, X, index_set, expr):
        self.X = X
        self.expr = expr
        self.index_set = index_set
        self.T, self.S, self.d = X.shape

    def __len__(self):
        return self.S * self.T

    def __getitem__(self, idx):
        i = idx // self.T
        t = idx % self.T
        x_t = self.X[t, i]  # (d,)
        index_feat = torch.tensor([self.index_set[t]], dtype=x_t.dtype, device=x_t.device)
        inp = torch.cat([x_t, index_feat], dim=0)  # (d+1,)
        return inp, self.expr[i]


class ValueEstimate(LogUBaselineNet):
    """
    Value network for flattened d-dimensional data, built on LogUBaselineNet.
    """

    def __init__(
        self,
        X,
        index_set,
        expr,
        hidden_sizes=(64, 64),
        batch_size=128,
        lr=1e-3,
        device="mps",
    ):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.copy()).float()
        if isinstance(expr, np.ndarray):
            expr = torch.from_numpy(expr.copy()).float()
        if isinstance(index_set, np.ndarray):
            index_set = torch.from_numpy(index_set.copy()).float()

        self.T, self.S, self.d = X.shape
        self.in_dim  = self.d + 1
        super().__init__(in_dim=self.in_dim, hidden_sizes=hidden_sizes, device=device)

        # data & training
        self.dataset = TrajectoryDataset(X, index_set, expr)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def fit(self, epochs: int = 20, verbose: bool = True):
        for epoch in range(1, epochs + 1):
            self.net.train(True)
            total_loss = 0.0
            for inp, target in self.loader:
                inp, target = inp.to(self.device), target.to(self.device)
                pred = self.forward(inp)
                loss = self.criterion(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inp.size(0)
            avg_mse = total_loss / len(self.dataset)
            if verbose:
                print(f"[Epoch {epoch:02d}] avg MSE: {avg_mse:.4f}")
        return avg_mse

    def predict(self, x, index, requires_grad: bool = False):
        # x: (..., d), index: scalar or broadcastable to leading dims of x; returns (...,)
        # returns tensor if requires_grad else numpy array
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        orig_shape = x.shape[:-1]  # keep trailing dim as d
        x_flat = x.reshape(-1, x.shape[-1])

        index = torch.as_tensor(index, dtype=torch.float32, device=self.device)
        if index.dim() == 0:
            index_feat = index.expand(x_flat.shape[0]).unsqueeze(-1)
        elif index.shape == orig_shape:
            index_feat = index.reshape(-1, 1)
        elif index.dim() == 1 and index.shape[0] == x_flat.shape[0]:
            index_feat = index.unsqueeze(-1)
        else:
            raise ValueError(f"Incompatible index shape {tuple(index.shape)} for x shape {tuple(x.shape)}")

        inp = torch.cat([x_flat, index_feat], dim=-1)

        self.net.eval()
        if requires_grad:
            preds = self.forward(inp)
            return preds.reshape(*orig_shape)
        with torch.no_grad():
            preds = self.forward(inp)
        return preds.cpu().numpy().reshape(*orig_shape)

    def predict_under_log(self, x, index, requires_grad: bool = False):
        # x: (..., d), index: scalar or broadcastable to leading dims of x; returns log values shaped (...,)
        # returns tensor if requires_grad else numpy array
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        index = torch.as_tensor(index, dtype=torch.float32, device=self.device)
        if index.dim() == 0:
            index_feat = index.expand(x_flat.shape[0]).unsqueeze(-1)
        elif index.shape == orig_shape:
            index_feat = index.reshape(-1, 1)
        elif index.dim() == 1 and index.shape[0] == x_flat.shape[0]:
            index_feat = index.unsqueeze(-1)
        else:
            raise ValueError(f"Incompatible index shape {tuple(index.shape)} for x shape {tuple(x.shape)}")

        inp = torch.cat([x_flat, index_feat], dim=-1)

        self.net.eval()
        if requires_grad:
            predictions = self.forward(inp)
            preds_pos = torch.nn.functional.softplus(predictions) + 1e-6
            return torch.log(preds_pos).reshape(*orig_shape)
        with torch.no_grad():
            predictions = self.forward(inp)
            preds_pos = torch.nn.functional.softplus(predictions) + 1e-6
        return np.log(preds_pos.cpu().numpy()).reshape(*orig_shape)

    def save(self, run_dir, run_id, reward_transform="exp"):
        """
        Save state_dict and a small metadata JSON describing how to reload.
        """
        os.makedirs(run_dir, exist_ok=True)
        ckpt_path = os.path.join(run_dir, f"{run_id}_value_estimator.pt")
        torch.save(self.state_dict(), ckpt_path)

        meta = {
            "hidden_sizes": getattr(self, "hidden_sizes", None),
            "T": self.T,
            "S": self.S,
            "d": self.d,
            "reward_transform": reward_transform,
        }
        meta_path = os.path.join(run_dir, f"{run_id}_value_estimator_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {"ckpt": ckpt_path, "meta": meta_path}


import os

def train_val_estimate(run_id, alpha, epochs = 20, save_name = None):
    dir = "toy_data_runs/" + run_id
    alpha_str = str(alpha).replace('.', 'p')

    # data = np.load(dir + "/states.npy")  # (n, T, d)
    # rewards = np.load(dir + "/rewards.npy")  # (n,)

    collector = Collector(None, None)

    core, states, rewards = collector.load_collected(dir)
    print("States shape:", states.shape)
    index_set = np.array(core.model_config["index_set"])
    print(index_set)
    X_tensor = torch.from_numpy(states).float()
    expr_tensor = torch.from_numpy(np.exp(rewards/alpha)).float()
    print(expr_tensor)

    value_estimator = ValueEstimate(X_tensor, index_set, expr_tensor, [128, 500, 500], device="mps")
    value_estimator.fit(verbose=True, epochs = epochs)
    save_path = os.path.join(dir, run_id + f"_value_estimator_temp_{alpha_str}.pt")
    if save_path:
        save_path = os.path.join(dir, f"{save_name}.pt")
    torch.save(value_estimator.state_dict(), save_path)
    print(f"Saved value estimator to {save_path}")
    return value_estimator, states, index_set


# def train_val_estimate_single_timestep(run_id, t_index=0, epochs=20):
#     """
#     Train ValueEstimate on a single timestep slice to sanity-check fit.
#     Uses states of shape (T, S, d); extracts t_index to shape (1, S, d).
#     """
#     dir = "toy_data_runs/" + run_id
#     collector = Collector(None, None)
#     core, states, rewards = collector.load_collected(dir)
#     index_set = np.array(core.model_config["index_set"])

#     if t_index < 0 or t_index >= states.shape[0]:
#         raise ValueError(f"t_index {t_index} out of range for T={states.shape[0]}")

#     states_slice = states[t_index][None, ...]  # (1, S, d)
#     index_slice = np.array([index_set[t_index]])

#     X_tensor = torch.from_numpy(states_slice).float()
#     expr_tensor = torch.from_numpy(np.exp(rewards)).float()

#     value_estimator = ValueEstimate(X_tensor, index_slice, expr_tensor, [128, 500, 500], device="cpu")
#     value_estimator.fit(verbose=True, epochs=epochs)
#     return value_estimator, states_slice, index_slice


def load_val_estimate(run_id, alpha, hidden_sizes=(128, 500, 500), device="mps", file_name = None):
    """
    Load a trained ValueEstimate and its data for a given run_id.
    Returns (value_estimator, states, index_set, rewards).
    """
    alpha_str = str(alpha).replace('.', 'p')
    dir = os.path.join("toy_data_runs", run_id)
    collector = Collector(None, None)
    core, states, rewards = collector.load_collected(dir)
    index_set = np.array(core.model_config["index_set"])

    X_tensor = torch.from_numpy(states).float()
    expr_tensor = torch.from_numpy(np.exp(rewards)).float()
    value_estimator = ValueEstimate(X_tensor, index_set, expr_tensor,
                                    hidden_sizes=hidden_sizes,
                                    device=device)
    ckpt_path = os.path.join(dir, run_id + f"_value_estimator_temp_{alpha_str}.pt")
    if file_name:
        ckpt_path = os.path.join(dir, f"{file_name}.pt")
    value_estimator.load_state_dict(torch.load(ckpt_path, map_location=device))
    value_estimator.eval()
    return value_estimator, states, index_set, expr_tensor


from animate_2d import animate_val_predictions
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train value estimator on a collected run and animate predictions.")
    parser.add_argument("run_id", help="Run ID under toy_data_runs to load (e.g., 20251201_113803_147987)")
    parser.add_argument("alpha", type = float, help="Alignment temperature")
    parser.add_argument("--epochs", type = int, default=5, help="Number of training epochs for value estimator.")
    parser.add_argument("--save_name", type = str, default=None, help="Number of training epochs for value estimator.")
    parser.add_argument("--single_t", type=int, default=None, help="If set, train on a single timestep index for debugging.")
    args = parser.parse_args()

    run_id = args.run_id
    alpha = args.alpha

    print(f"Training ValueEstimate on run {run_id} with alpha={alpha}...")
    # if args.single_t is not None:
    #     ve, states, index_set = train_val_estimate_single_timestep(run_id, t_index=args.single_t, epochs=10)
    # else:
    ve, states, index_set = train_val_estimate(run_id, alpha, epochs = args.epochs, save_name = args.save_name)
    # ve, states, index_set, rewards = load_val_estimate(run_id)
    print("Loaded trained ValueEstimate.")
    animate_val_predictions(states, index_set, ve, outfile = f"value_net_temp_{str(alpha).replace('.', 'p')}.gif",)
    print("Animation complete.")
