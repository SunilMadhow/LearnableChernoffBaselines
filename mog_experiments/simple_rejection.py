import numpy as np
from tqdm import tqdm
from toy_data.mixture_model import MixtureModel

# BE AWARE: These classes use a different indexing standard than the diffusion cores. They output datasets (S, T, state_shape) instead of (T, S, state_shape).

class SimpleDiffusion:
    def __init__(self, dist0, alpha_t, S = 10000):
        self.dist0 = dist0
        self.alpha_t = alpha_t
        self.d = dist0.mu.shape[1]
        self.S = S
        self.alpha_bar = np.cumprod(self.alpha_t)
    
    def calc_samples(self):
        # Initialize a matrix to store samples at each time step
        X = np.zeros((self.S, len(self.alpha_t), self.d))
        # Start from standard Gaussian noise at time T
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))
        # Reverse diffusion from t=T-1 down to t=0
        for t in tqdm(range(len(self.alpha_t) - 2, -1, -1), desc="Reverse Diffusion Progress"):
            X[:, t, :] = self.single_step(t, X[:, t + 1, :])
        return X

    def single_step(self, t, X):
        # print("Diffusion.single_step")
        # print("X.shape", X.shape)
        X_next = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        X_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(size=(self.S, self.d))
        return X_next
    
    def single_step_iid_copies(self, t, X, size = 1):
        X_next_arr = []
        for _ in range(size):
            X_next = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
            X_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(size=(X.shape[0], self.d))
            X_next_arr.append(X_next)
        return np.array(X_next_arr)    
    
    def single_sample(self, t, x, size = 1):
        x_next = (x + (1 - self.alpha_t[t]) * self.dist0.score(x, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        x_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(self.d)
        return x_next

class SimpleDiffusionRejection(SimpleDiffusion):
    def __init__(self, dist0, alpha_t, value_function, index_set, 
                 upper_bound: float = 1.0, temperature: float = 1.0, S: int = 10_000, lower_bound = None):
        super().__init__(dist0, alpha_t, S)
        self.value_function = value_function
        self.upper_bound = upper_bound
        self.temperature = temperature
        self.index_set = index_set
        self.T = len(self.alpha_t)
        self.B = self.upper_bound / self.temperature
        if lower_bound is not None:
            self.B = 0
        self.num_steps = 0

    def evaluate(self, x, t):
        x = np.squeeze(x, axis=0)
        print("in evaluate, t = ", t)
        print("index argument:", self.index_set[t])
        t_arr = [self.index_set[t]] * x.shape[0]
        t_arr = np.array(t_arr)
        v = self.value_function(x, t_arr)
        # print(v)
        return v
    
    def single_step(self, t, X):
        N = X.shape[0]
        proposal_fixed = lambda size, src: super(SimpleDiffusionRejection, self).single_step_iid_copies(t, src, size)
        eval_fixed = lambda x: self.evaluate(x, t)
        Z =  vec_rejection_val(X, proposal_fixed, eval_fixed, self.B)
        self.num_steps += Z[1]
        return Z[0]

import numpy as np

def vec_rejection_val(
    init_states,          # shape (N,)
    proposal_rvs,         # fn size=(K,P), src→ draws of shape (K,P)
    val,                  # fn X, S → log‑weight array, shape (K,P)
    B,                    # either scalar or fn S→ array shape (K,)
    batch_size=1
):
    """
    For each starting state s in init_states, draw one sample x ~ q(·|s) by rejection:
      accept x when u < exp(val(x,s) - B(s)).
    Returns array of accepted x's, one per init state.
    """
    N = init_states.shape[0]
    d = init_states.shape[1]
    out  = np.empty((N, d))
    done = np.zeros(N, bool)
    total_proposals = 0 


    # # if no batch_size, pick something so you get ~1 acceptance per iteration
    # if batch_size is None:
    #     # a rough α ≈ mean[exp(val(s_sample,s) - B(s))]
    #     # we just draw one proposal per state to estimate α
    #     test_x = proposal_rvs(size=N, src=init_states)
    #     # print("shape of test_x", test_x.shape)
    #     # print("shape of vals", np.exp(val(test_x)).shape)
    #     est_R  = np.exp(val(test_x))/np.exp(B)  # shape (N,)
    #     # print("shape of est_R", est_R.shape)
    #     alpha  = np.mean(est_R)
    #     batch_size = max(1, int(np.ceil(1/alpha)))
    # # print("batch_size", batch_size)
    

    while not done.all():
        active = np.where(~done)[0]       # indices still needing a sample
        K      = active.size

        # 1) draw K×batch_size proposals in one go
        X_prop = proposal_rvs(size=batch_size, src=init_states[active])
        # print("X_prop", X_prop.shape)
        total_proposals += K * batch_size 
        U      = np.random.rand(batch_size, K)

        # 2) compute bound and log‑weights
        Bvals = B          # shape (K,) or scalar
        V = val(X_prop)  # shape (K, batch_size)
        # 3) acceptance ratios
        R = np.exp(V)/(np.ones_like(V)*np.exp(B))                          # ≤1 by construction
        print("R", R.shape)
        # print("Val(X_prop)/B", R.shape)
        accept = (U < R)                              # boolean mask
        # print("accept", accept.shape)
        # print("active", active.shape)
        # 4) for each chain, find first accepted proposal
        got_any = accept.any(axis=0)                  # which chains accepted
        # print("got_any", got_any.shape)
        if got_any.any():
            first_idx = np.argmax(accept, axis=0)     # first True per chain
            sel       = active[got_any]               # original indices
            picks     = first_idx[got_any]            # column of first accept
            out[sel]  = X_prop[picks, got_any, :]
            done[sel] = True

    return out, total_proposals