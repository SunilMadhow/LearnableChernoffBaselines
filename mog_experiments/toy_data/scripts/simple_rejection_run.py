import os
import sys

# BE AWARE: These classes use a different indexing standard than the diffusion cores. They output datasets (S, T, state_shape) instead of (T, S, state_shape).


# add repo root to sys.path so imports work when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import numpy as np
import torch

from simple_rejection import SimpleDiffusionRejection
from nets import ValueEstimate
from collection import Collector


def load_value_estimator(run_dir, device="cpu", hidden_sizes=(128, 500, 500)):
    collector = Collector(None, None)
    core, states, rewards = collector.load_collected(run_dir)
    index_set = core.model_config["index_set"]
    X = torch.from_numpy(states).float()
    expr = torch.from_numpy(np.exp(rewards)).float()

    ckpt = os.path.join(run_dir, os.path.basename(run_dir) + "_value_estimator.pt")
    ve = ValueEstimate(X, index_set, expr, hidden_sizes=hidden_sizes, device=device)
    ve.load_state_dict(torch.load(ckpt, map_location=device))
    ve.eval()
    return core, ve, index_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simple rejection using a saved run_id.")
    parser.add_argument("run_id", help="Run ID under toy_data_runs")
    parser.add_argument("--device", default="cpu", help="Device for value estimator (cpu|cuda|mps)")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 500, 500],
                        help="Hidden layer sizes for the value network (must match training).")
    args = parser.parse_args()

    run_dir = os.path.join(REPO_ROOT, "toy_data_runs", args.run_id)
    core, ve, index_set = load_value_estimator(run_dir, device=args.device, hidden_sizes=tuple(args.hidden_sizes))
    dist0 = core.dist0
    alpha_t = core.alpha_t
    upper_bound = B = 3

    guided = SimpleDiffusionRejection(
        dist0,    alpha_t, lambda x, t: np.log(np.clip(ve.predict(x, t), np.e**(-B), np.e**B)), index_set, upper_bound=upper_bound)
    # samples = guided.sample()
    print(f"Loaded core and value estimator from {run_dir}")

    X = guided.calc_samples()
    print("num steps:", guided.num_steps)
    print("X samples shape:", X.shape)

    import matplotlib.pyplot as plt

    x = X[:, 0, 0]
    y = X[:, 0, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.7)
    plt.xlabel("X[:, 0, 0]")
    plt.ylabel("X[:, 0, 1]")
    plt.title("Scatter plot of samples")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    # show and also save a copy next to the run directory if available
    plt.show()
    # try:
    #     out_path = os.path.join(run_dir, "scatter_X_0_0_vs_0_1.png")
    #     plt.savefig(out_path)
    #     print(f"Saved scatter plot to {out_path}")
    # except Exception:
    #     pass

