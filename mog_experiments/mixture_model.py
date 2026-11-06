import numpy as np
from scipy.stats import norm, multivariate_normal, gaussian_kde
from scipy.integrate import nquad
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import tqdm

import sys

from diffusion_core import Core, ModelConfig
from animate_2d import animate_traj  # if needed
from collection import Collector

import os


import json


class MixtureModel(Core):
    model_id = "mixture_model"
    def __init__(self, dist0, alpha_t, mixture_config, S = 10000, N = 4, load_from_config = False):
        super().__init__(mixture_config)
        if load_from_config:
            self.load_from_config(mixture_config)
            return
        self.mixture_config = mixture_config
        self.state_shape = (2)
        self.tentative_states = None
        self.dist0 = dist0
        self.alpha_t = alpha_t
        self.d = dist0.mu.shape[1]
        self.N = N
        self.S = S
        self.alpha_bar = np.cumprod(self.alpha_t)
        self.T = len(self.alpha_t) - 1
        self.reset()

    def load_from_config(self, cfg):
        # cfg = config.to_dict()
        if cfg["model_name"] != self.model_id:
            raise ValueError(f"Config model_name {cfg['model_name']} does not match {self.model_id}")
        self.mixture_config = cfg
        
        self.state_shape = cfg["state_shape"]
        deets = cfg["deets"]
        
        self.dist0 = Mixture(None, None, None, deets["dist0"])
        self.alpha_t = np.array(deets["alpha_t"])
        self.alpha_bar = np.cumprod(self.alpha_t)
        self.T = len(self.alpha_t) - 1
        self.d = self.dist0.mu.shape[1]
        self.N = 2
        self.S = 10_000

    def reset(self):
        self.t = self.T
        self.tentative_states = None

    def propose(self):
        # Generate N proposals per state; output shape (S, N, d)
        if self.t < 0:
            raise ValueError("No remaining timesteps to propose.")
        t = self.t
        if t == self.T:
            return np.transpose(np.random.normal(size=(self.N, self.S, self.d)), (1, 0, 2))
        else:
            X = self.X  # (S, d)
            drift = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
            noise = np.random.normal(size=(self.N, self.S, self.d))
            scaled_noise = np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * noise
            proposals = drift[None, ...] + scaled_noise  # (N, S, d)
            proposals = np.transpose(proposals, (1, 0, 2))  # (S, N, d)
        self.tentative_states = proposals
        return proposals
    
    def propose_external(self, init_state, time, N = 1):
        # Generate single proposal from init_state at given time index
        if N == 1:
            t = time
            X = init_state  # (S, d)
            drift = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
            noise = np.random.normal(size=X.shape)
            scaled_noise = np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * noise
            proposal = drift + scaled_noise  # (S, d)
            return proposal
        else:
            t = time
            X = init_state  # (S, d)
            drift = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
            noise = np.random.normal(size=(N, X.shape[0], X.shape[1]))
            scaled_noise = np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * noise
            proposals = drift[None, ...] + scaled_noise  # (N, S, d)
            proposals = np.transpose(proposals, (1, 0, 2))  # (S, N, d)
            return proposals

    def commit(self, X):
        # Expect X of shape (S, d); advance to next timestep
        X = np.asarray(X)
        if X.shape != (self.S, self.d):
            raise ValueError(f"commit expected shape {(self.S, self.d)}, got {X.shape}")
        self.X = X
        self.t -= 1
    
    def single_sample(self, t, x, size = 1):
        x_next = (x + (1 - self.alpha_t[t]) * self.dist0.score(x, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        x_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(self.d)
        return x_next
    
    def collect_run(self):
        states = []
        self.reset()
        while self.t >= 0:
            proposals = self.propose()
            self.commit(proposals[:, 0, :])
            states.append(self.X.copy())
        self.last_run = states
        return np.flip(np.array(states), axis = 0)
    
    def save(self, states, data_dir="data", run_id = None):
        """Persist model details and states to disk with a timestamp-based id."""
        os.makedirs(data_dir, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cfg_path = os.path.join(data_dir, f"{run_id}_config.json")
        states_path = os.path.join(data_dir, f"{run_id}_states.npy")

        def _to_serializable(obj):
            if hasattr(obj, "to_dict"):
                return _to_serializable(obj.to_dict())
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(self.model_config), f, indent=2)

        np.save(states_path, np.asarray(states))
        return {"id": run_id, "config": cfg_path, "states": states_path}
    
class Mixture(object):
    """
    Represents an N-component Gaussian mixture model.

    Attributes:
        mu (ndarray): shape (N, d) component means
        var (ndarray): covariances, shape depends on input (see below)
        pi (ndarray): mixture weights, shape (N,)
        dist (list): list of scipy.stats multivariate_normal objects
    """
    def __init__(self, mu_0, var_0, pi_0, source_dict = None):
        super(Mixture, self).__init__()
        if source_dict is None:
            self.mu = mu_0
            self.pi = pi_0
            self.N = mu_0.shape[0]
            self.d = mu_0.shape[1]
        else:
            self.from_dict(source_dict)
            return

        # Handle three cases for var_0 input shape:
        if len(var_0.shape) == 1:
            # Diagonal covariances provided as vector
            self.dist = [multivariate_normal(mu_0[i], var_0[i] * I)
                         for i in range(self.N)]
            self.var = np.array([v * I for v in var_0])
        elif len(var_0.shape) == 2:
            # Single shared covariance
            self.dist = [multivariate_normal(mu_0[i], var_0)
                         for i in range(self.N)]
            # Tile to shape (N, d, d)
            self.var = np.tile(var_0, (self.N, 1, 1))
        elif len(var_0.shape) == 3:
        # Each component has its own covariance matrix
            self.dist = [multivariate_normal(mu_0[i], var_0[i]) for i in range(self.N)]
            self.var = var_0

    def to_dict(self):
        return {
            "mu": self.mu.tolist(),
            "var": self.var.tolist(),
            "pi": self.pi.tolist(),
            "N": self.N,
            "d": self.d,
        }

    def from_dict(self, dict):
        self.mu = np.array(dict["mu"])
        self.var = np.array(dict["var"])
        self.pi = np.array(dict["pi"])
        self.N = dict["N"]
        self.d = dict["d"]
        # Rebuild dist list
        self.dist = [multivariate_normal(self.mu[i], self.var[i]) for i in range(self.N)]

    def pdf(self, x):
        pdf_each = np.array([dd.pdf(x) for dd in self.dist])
        # Weighted average by mixture weights
        return np.average(pdf_each, axis=0, weights=self.pi)

    def score(self, x, alpha_bar):
    # --- ensure x is a 2‐D array of shape (n_pts, d) ----
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]    # now (1, d)

        n_pts, d = x.shape
        mu_t  = np.sqrt(alpha_bar) * self.mu      # (N, d)
        var_t = alpha_bar * self.var + (1 - alpha_bar) * np.eye(d) # (N, d, d)
        inv_var_t = np.linalg.inv(var_t)          # (N, d, d)
        N = len(self.pi)

        # --- build frozen distributions (optional) ----
        dist_t = [multivariate_normal(mu_t[i], var_t[i]) for i in range(N)]

        # --- compute pdf_each by looping over samples ----
        # pdf_each[i, j] = p(component i at sample j)
        pdf_each = np.empty((N, n_pts))
        for i, dist in enumerate(dist_t):
            for j in range(n_pts):
                pdf_each[i, j] = dist.pdf(x[j])

        # --- compute the “expected gradient” per component/sample ----
        # deriv_exp[i, j, :] = ∇ log p_i(x_j)
        deriv_exp = np.empty((N, n_pts, d))
        for i in range(N):
            for j in range(n_pts):
                diff = x[j] - mu_t[i]
                deriv_exp[i, j] = - diff.dot(inv_var_t[i])

        # --- weighted average over components ----
        # numerator: E_pi[ deriv * p ]    shape => (n_pts, d)
        num = np.tensordot(self.pi, deriv_exp * pdf_each[..., None], axes=(0, 0))
        # denominator: E_pi[ p ]          shape => (n_pts,)
        den = np.dot(self.pi, pdf_each)           # (n_pts,)

        # final score: (num/den) per sample
        score = num / den[:, None]                # (n_pts, d)

        # if user originally passed in a 1‑D x, you might want to return a (d,) array:
        return score[0] if score.shape[0] == 1 else score


    def hess(self, x, alpha_bar):
        """
        Compute the Hessian (second derivative) of log-density at x.

        Args:
            x: array (num_points, d)
            alpha_bar: noise schedule cumulative product
        Returns:
            hess: shape (num_points, d, d)
        """
        npts = x.shape[0]
        mu_t = np.sqrt(alpha_bar) * self.mu
        var_t = alpha_bar * self.var + (1 - alpha_bar) * np.eye(self.d)
        # Precompute inverses
        var_inv = np.array([np.linalg.inv(v) for v in var_t])

        # Component pdfs
        pdf_each = np.array([
            multivariate_normal(mu_t[i], var_t[i]).pdf(x)
            for i in range(self.N)
        ])

        # First derivative term (score)
        deriv_exp = np.array([
            -(x - mu_t[i]).dot(var_inv[i])
            for i in range(self.N)
        ])  # (N, npts, d)
        score_t = np.average(deriv_exp * pdf_each[:, :, None], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, None]

        # Second derivative term (expected outer product minus inv covariance)
        sec_deriv = np.empty((N, npts, d, d))
        for i in range(self.N):
            delta = (x - mu_t[i]).dot(var_inv[i])
            # Outer(delta, delta) - inv(cov)
            sec_deriv[i] = delta[:, None, :] * delta[:, :, None] - var_inv[i]

        # Weighted average and subtract score outer product
        term = np.average(sec_deriv * pdf_each[:, :, None, None], axis=0, weights=self.pi)
        term /= np.average(pdf_each, axis=0, weights=self.pi)[:, None, None]
        return term - (score_t[:, :, None] * score_t[:, None, :])

    def rvs(self, size):
        """
        Sample from the mixture model.

        Args:
            size: number of samples to draw
        Returns:
            y: array (size, d)
        """
        # Draw from each component independently
        palette = np.zeros((self.N, size, d))
        for i in range(self.N):
            palette[i] = self.dist[i].rvs(size=size)
        # Choose components according to mixture weights
        idx = np.random.choice(self.N, size=size, p=self.pi)
        return palette[idx, np.arange(size)]

from rewards import reward_func_0
if __name__ == "__main__":
    MixtureConfig = ModelConfig()
    MixtureConfig.model_name = "mixture_model"
    MixtureConfig.state_shape = 2

    n = T = 20

    MixtureConfig.index_set = np.array(list(range(T)))/(T - 1)

    t_vals = np.arange(1, n+1)
    c, delta = 4, 0.02
    inner = delta * (1 + c * np.log(n) / n)**t_vals
    alpha_t = 1 - (c * np.log(n) / n) * np.minimum(inner, 1)
    alpha_t[0] = 1 - delta

    mu_0 = np.array([[5, 0], [-5, 0]])
    d = 2
    var_0 = np.eye(d)
    pi_0  = np.array([0.95, 0.05])
    pi_0 /= np.sum(pi_0)
    dist0 = Mixture(mu_0, var_0, pi_0)


    mixture_deets = {"dist0" : dist0, "alpha_t": alpha_t, "alpha_bar": np.cumprod(alpha_t)}
    MixtureConfig.deets = mixture_deets

    MixtureModel = MixtureModel(dist0, alpha_t, MixtureConfig, N = 1, S = 70_000)
    collector = Collector(MixtureModel, reward_fn = reward_func_0, base_dir="toy_data_runs")
    collected_run = collector.collect()
    run_id = collected_run["run_id"]
    dir = "toy_data_runs/" + run_id
    core, states, rewards = collector.load_collected(dir)

    animate_traj(states, rewards = rewards, outfile= dir + "/animation.gif", interval=100)

    print("Ran")
