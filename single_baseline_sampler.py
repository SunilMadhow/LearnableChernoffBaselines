import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llada_imputation_core import LLaDAImputationCore


class HFRewardScorer:
    """Generic prompt-conditioned reward scorer. Larger score means better."""

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        torch_dtype=None,
        max_length: int = 4096,
    ):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def _render_pair(self, prompt: str, response: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            txt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            if self.tokenizer.bos_token is not None and txt.startswith(self.tokenizer.bos_token):
                txt = txt[len(self.tokenizer.bos_token):]
            return txt
        return f"User: {prompt}\nAssistant: {response}"

    @torch.no_grad()
    def score(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> np.ndarray:
        if len(prompts) != len(responses):
            raise ValueError(f"Length mismatch: {len(prompts)} prompts vs {len(responses)} responses")

        device = next(self.model.parameters()).device
        all_scores = []
        for start in range(0, len(prompts), batch_size):
            p_batch = prompts[start:start + batch_size]
            r_batch = responses[start:start + batch_size]
            rendered = [self._render_pair(p, r) for p, r in zip(p_batch, r_batch)]
            toks = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(device)
            scores = self.model(**toks).logits.squeeze(-1).float().cpu().numpy()
            all_scores.append(scores)
        return np.concatenate(all_scores, axis=0)


class RewardTauLCB(nn.Module):
    """
    Mean-Centered Absolute Reward LCB.

    Works with centered absolute candidate rewards:
        s = r_t - mu_t
    where mu_t is the running mean of all observed rewards at step t.
    """

    def __init__(
        self,
        core: LLaDAImputationCore,
        *,
        delta: float = 0.1,
        fix_lambda: bool = False,
        lambda_value: float = 2.0,
        lam_min: float = 0.0,
        lam_max: float = 20.0,
        erm_error: float = 0.0,
        lambda_lr: float = 1e-3,
        lambda_mode: str = "gradient",
        symmetric_objective: bool = False,
        alignment_temperature: float = 1.0,
    ):
        super().__init__()
        self.T = core.T
        self.delta = float(delta)
        self.fix_lambda = bool(fix_lambda)
        self.lambda_value = float(lambda_value)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.erm_error = float(erm_error)
        self.lambda_mode = str(lambda_mode)
        self.symmetric_objective = bool(symmetric_objective)
        self.alignment_temperature = float(alignment_temperature)

        self.outer_steps = 1

        init_val = float(lambda_value)
        if self.lambda_mode == "gradient":
            self.lambdas = nn.Parameter(torch.full((self.T + 1,), init_val, dtype=torch.float32))
            self.lambda_opt = torch.optim.Adam([self.lambdas], lr=lambda_lr)
        else:
            self.register_buffer("lambdas", torch.full((self.T + 1,), init_val, dtype=torch.float32))
            self.lambda_opt = None

        self.taus = [None] * (self.T + 1)
        self.mu = [0.0] * (self.T + 1)
        self.trained_t = [False] * (self.T + 1)
        self.raw_reward_history = {t: [] for t in range(self.T + 1)}

    def _get_lambda_val(self, t: int) -> torch.Tensor:
        return self.lambdas[t].clamp(min=self.lam_min, max=self.lam_max)

    def _reward_stats_dict(self, candidate_rewards: np.ndarray, source_rewards: np.ndarray):
        if candidate_rewards.ndim == 1:
            candidate_rewards = candidate_rewards[:, None]

        return {
            "source_reward_mean": float(np.mean(source_rewards)),
            "source_reward_std": float(np.std(source_rewards)),
            "candidate_reward_mean": float(np.mean(candidate_rewards)),
            "candidate_reward_std": float(np.std(candidate_rewards)),
            "candidate_reward_min": float(np.min(candidate_rewards)),
            "candidate_reward_max": float(np.max(candidate_rewards)),
        }

    def _lcb_objective(self, s: torch.Tensor, lam) -> torch.Tensor:
        m1 = torch.max(lam * s)
        term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam

        if not self.symmetric_objective:
            m2 = torch.max(-1.0 * s)
            term2 = m2 + torch.log(torch.mean(torch.exp(-1.0 * s - m2)))
            term3 = torch.log(torch.tensor(1.0 / self.delta, device=s.device)) / lam
        else:
            m2 = torch.max(-lam * s)
            term2 = (m2 + torch.log(torch.mean(torch.exp(-lam * s - m2)))) / lam
            term3 = 2.0 * torch.log(torch.tensor(1.0 / self.delta, device=s.device)) / lam

        return term1 + term2 + term3 + self.erm_error

    def _golden_section_search_lambda(self, s: torch.Tensor, t: int, max_iter: int = 10):
        a, b = self.lam_min, self.lam_max
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        for _ in range(max_iter):
            if abs(b - a) < 1e-3:
                break
            fc = self._lcb_objective(s, c).item()
            fd = self._lcb_objective(s, d).item()
            if fc < fd:
                b = d
                d = c
                c = b - phi * (b - a)
            else:
                a = c
                c = d
                d = a + phi * (b - a)
        self.lambdas[t] = (a + b) / 2.0

    def train_on_step(self, candidate_rewards: np.ndarray, source_rewards: np.ndarray, tp1: int):
        t = tp1 - 1
        if candidate_rewards.ndim == 1:
            candidate_rewards = candidate_rewards[:, None]
        
        # Track raw rewards and compute running mean
        new_raw = candidate_rewards.reshape(-1)
        self.raw_reward_history[tp1].append(new_raw)
        
        raw_all = np.concatenate(self.raw_reward_history[tp1], axis=0)
        mu_t = float(np.mean(raw_all))
        self.mu[tp1] = mu_t
        
        # Center all rewards using the updated mean
        s_all = raw_all - mu_t
        s_new = new_raw - mu_t
        s_tensor = torch.tensor(s_all, dtype=torch.float32, device=self.lambdas.device)

        lam_before = self._get_lambda_val(t).item()
        objective_before = float(self._lcb_objective(s_tensor, self._get_lambda_val(t)).item())

        for _ in range(self.outer_steps):
            if not self.fix_lambda and self.lambda_mode == "gradient":
                loss = self._lcb_objective(s_tensor, self.lambdas[t])
                self.lambda_opt.zero_grad()
                loss.backward()
                self.lambda_opt.step()
                with torch.no_grad():
                    self.lambdas[t].clamp_(min=self.lam_min, max=self.lam_max)
            elif not self.fix_lambda and self.lambda_mode == "search":
                self._golden_section_search_lambda(s_tensor, t)
                self.lambdas[t].clamp_(min=self.lam_min, max=self.lam_max)

        lam_after = self._get_lambda_val(t).item()
        objective_after = float(self._lcb_objective(s_tensor, self._get_lambda_val(t)).item())

        out = {
            "tp1": int(tp1),
            "mu_t": mu_t,
            "lambda_before": float(lam_before),
            "lambda_after": float(lam_after),
            "objective_before": float(objective_before),
            "objective_after": float(objective_after),
            "num_new_samples": int(len(s_new)),
            "num_total_samples": int(len(s_all)),
            "history_s_mean": float(np.mean(s_all)),
            "history_s_std": float(np.std(s_all)),
            "history_s_min": float(np.min(s_all)),
            "history_s_max": float(np.max(s_all)),
        }
        out.update(self._reward_stats_dict(candidate_rewards, source_rewards))
        return out

    def calibrate_tau(self, candidate_rewards: np.ndarray, source_rewards: np.ndarray, tp1: int):
        t = tp1 - 1
        if candidate_rewards.ndim == 1:
            candidate_rewards = candidate_rewards[:, None]
            
        new_raw = candidate_rewards.reshape(-1)

        mu_t = self.mu[tp1]  # already set by train_on_step; don't mutate it

        raw_all = np.concatenate(self.raw_reward_history[tp1], axis=0) if self.raw_reward_history[tp1] else new_raw
        s_all = raw_all - mu_t
        s_new = new_raw - mu_t
        lam = self._get_lambda_val(t).item()

        m1 = np.max(lam * s_all)
        term1 = (m1 + np.log(np.mean(np.exp(lam * s_all - m1)))) / lam
        term2 = math.log(1.0 / self.delta) / lam
        tau = float(term1 + term2 + self.erm_error)
        self.taus[tp1] = tau
        self.trained_t[tp1] = True

        out = {
            "tp1": int(tp1),
            "mu_t": float(mu_t),
            "lambda": float(lam),
            "tau": float(tau),
            "tau_term1_mgf": float(term1),
            "tau_term2_delta": float(term2),
            "erm_error": float(self.erm_error),
            "num_new_samples": int(len(s_new)),
            "num_total_samples": int(len(s_all)),
            "history_s_mean": float(np.mean(s_all)),
            "history_s_std": float(np.std(s_all)),
            "history_s_min": float(np.min(s_all)),
            "history_s_max": float(np.max(s_all)),
            "objective_at_lambda": float(
                self._lcb_objective(
                    torch.tensor(s_all, dtype=torch.float32, device=self.lambdas.device),
                    self._get_lambda_val(t),
                ).item()
            ),
        }
        out.update(self._reward_stats_dict(candidate_rewards, source_rewards))
        return out

    def recalculate_taus(self, new_delta: float):
        old_delta = self.delta
        self.delta = float(new_delta)
        
        recalculated_count = 0
        for t in range(1, self.T + 1):
            if not self.trained_t[t]:
                continue
            
            s_all = None
            if len(self.raw_reward_history.get(t, [])) > 0:
                s_all = np.concatenate(self.raw_reward_history[t], axis=0) - self.mu[t]
            elif hasattr(self, 's_history') and len(self.s_history.get(t, [])) > 0:
                s_all = np.concatenate(self.s_history[t], axis=0)
            
            if s_all is None:
                print(f"  [!] WARNING: No saved history for step {t}. Cannot recalculate tau. Keeping old tau: {self.taus[t]}")
                continue

            lam = self._get_lambda_val(t - 1).item()
            m1 = np.max(lam * s_all)
            term1 = (m1 + np.log(np.mean(np.exp(lam * s_all - m1)))) / lam
            term2 = math.log(1.0 / self.delta) / lam
            
            old_tau = self.taus[t]
            self.taus[t] = float(term1 + term2 + self.erm_error)
            print(f"  [+] Step {t} tau updated: {old_tau:.4f} -> {self.taus[t]:.4f}")
            recalculated_count += 1
            
        print(f"Recalculated {recalculated_count} tau thresholds: delta shifted from {old_delta} to {self.delta}")

    def save(self, run_dir: str, filename: str = "reward_tau_lcb_absolute"):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"{filename}.pt")
        lambdas_data = self.lambdas.data if isinstance(self.lambdas, nn.Parameter) else self.lambdas
        torch.save(
            {
                "lambdas": lambdas_data.cpu(),
                "taus": self.taus,
                "mu": self.mu,
                "trained_t": self.trained_t,
                "raw_reward_history": self.raw_reward_history,
                "config": {
                    "delta": self.delta,
                    "fix_lambda": self.fix_lambda,
                    "lambda_value": self.lambda_value,
                    "lam_min": self.lam_min,
                    "lam_max": self.lam_max,
                    "erm_error": self.erm_error,
                    "lambda_mode": self.lambda_mode,
                    "symmetric_objective": self.symmetric_objective,
                    "alignment_temperature": self.alignment_temperature,
                },
            },
            path,
        )
        print(f"Saved Mean-Centered RewardTauLCB to {path}")

    @classmethod
    def load(cls, run_dir: str, core: LLaDAImputationCore, filename: str = "reward_tau_lcb_absolute"):
        path = os.path.join(run_dir, f"{filename}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        inst = cls(
            core,
            delta=cfg["delta"],
            fix_lambda=cfg["fix_lambda"],
            lambda_value=cfg["lambda_value"],
            lam_min=cfg["lam_min"],
            lam_max=cfg["lam_max"],
            erm_error=cfg["erm_error"],
            lambda_mode=cfg["lambda_mode"],
            symmetric_objective=cfg.get("symmetric_objective", False),
            alignment_temperature=cfg.get("alignment_temperature", 1.0),
        )
        if isinstance(inst.lambdas, nn.Parameter):
            inst.lambdas.data.copy_(state["lambdas"])
        else:
            inst.lambdas.copy_(state["lambdas"])
        
        inst.taus = state.get("taus", [None] * (inst.T + 1))
        
        # Backwards compatibility: defaults missing mu to 0.0 for old baselines
        inst.mu = state.get("mu", [0.0] * (inst.T + 1))
        inst.trained_t = state.get("trained_t", [False] * (inst.T + 1))
        
        # Backwards compatibility: falls back to s_history if raw_history is missing
        inst.raw_reward_history = state.get("raw_reward_history", state.get("s_history", {t: [] for t in range(inst.T + 1)}))
        if "s_history" in state and "raw_reward_history" not in state:
            inst.s_history = state["s_history"]
            
        return inst


class RewardTauSampler:
    def __init__(
        self,
        core: LLaDAImputationCore,
        lcb: RewardTauLCB,
        reward_scorer,
        prompt_texts: Sequence[str],
        *,
        max_retries: int = 100,
        run_specific_dir: str = "./",
        save_name: str = "reward_tau_lcb_absolute",
        upper_bound: Optional[float] = None,
        n_trajectories: int = 50,
        train_batches_per_step: int = 1,
        impute_batch_size: int = 16,
        reward_batch_size: int = 8,
        guide_steps: Optional[Sequence[int]] = None
    ):
        self.core = core
        self.lcb = lcb
        self.reward_scorer = reward_scorer
        self.prompt_texts = list(prompt_texts)
        self.base_prompt_ids = np.array(core.prompt_ids, copy=True)

        self.max_retries = max_retries
        self.run_specific_dir = run_specific_dir
        self.save_name = save_name
        self.upper_bound = upper_bound
        self.n_trajectories = n_trajectories
        self.train_batches_per_step = train_batches_per_step
        self.impute_batch_size = impute_batch_size
        self.reward_batch_size = reward_batch_size
        if guide_steps is None:
            self.guide_steps = set(range(1, self.core.T + 1))
        else:
            self.guide_steps = set(guide_steps)
        self.no_tau = False

        self.training_nfe = 0
        self.inference_nfe = 0
        self.training_imputations = 0
        self.inference_imputations = 0
        self.training_reward_evals = 0
        self.inference_reward_evals = 0
        self.num_proposals_list = [0] * self.core.S
        self.max_retries_hits = 0           # (sample, step) pairs that fell through to last_resort
        self.total_guided_step_applications = 0  # total (sample, step) pairs attempted

        if len(self.prompt_texts) not in (1, self.core.S):
            raise ValueError(f"prompt_texts length must be 1 or core.S={self.core.S}, got {len(self.prompt_texts)}")
        if self.base_prompt_ids.shape[0] not in (1, self.core.S):
            raise ValueError(
                f"core.prompt_ids must have batch size 1 or core.S={self.core.S}, got {self.base_prompt_ids.shape[0]}"
            )

    def _should_guide(self, tp1: int) -> bool:
        return tp1 in self.guide_steps

    def _prompt_texts_for_indices(self, indices: np.ndarray) -> List[str]:
        if len(self.prompt_texts) == 1:
            return [self.prompt_texts[0]] * len(indices)
        return [self.prompt_texts[i] for i in indices]

    def _prompt_ids_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.base_prompt_ids.shape[0] == 1:
            return self.base_prompt_ids
        return self.base_prompt_ids[indices]

    def _get_stats(self):
        n = self.total_guided_step_applications
        return {
            "training_nfe": self.training_nfe,
            "inference_nfe": self.inference_nfe,
            "training_imputations": self.training_imputations,
            "inference_imputations": self.inference_imputations,
            "training_reward_evals": self.training_reward_evals,
            "inference_reward_evals": self.inference_reward_evals,
            "total_nfe": self.training_nfe + self.inference_nfe,
            "max_retries_hits": self.max_retries_hits,
            "total_guided_step_applications": n,
            "max_retries_hit_rate": self.max_retries_hits / n if n > 0 else None,
        }

    def _propose_smart(self, source_X: np.ndarray, *, active_indices: Optional[np.ndarray] = None, count_as_training: bool = False) -> np.ndarray:
        real_X = self.core.X
        real_schedule = getattr(self.core, "transfer_schedule", None)
        real_S = self.core.S
        real_prompt_ids = self.core.prompt_ids

        if active_indices is not None:
            self.core.X = source_X[active_indices]
            if real_schedule is not None:
                self.core.transfer_schedule = real_schedule[active_indices]
            self.core.prompt_ids = self._prompt_ids_for_indices(active_indices)
            self.core.S = len(active_indices)
        else:
            self.core.X = source_X
            if real_schedule is not None:
                self.core.transfer_schedule = real_schedule
            self.core.prompt_ids = self.base_prompt_ids
            self.core.S = len(source_X)

        proposals = self.core.propose()
        count = self.core.S * self.core.N
        if count_as_training:
            self.training_nfe += count
        else:
            self.inference_nfe += count

        self.core.X = real_X
        if real_schedule is not None:
            self.core.transfer_schedule = real_schedule
        self.core.S = real_S
        self.core.prompt_ids = real_prompt_ids
        return proposals

    def _score_states(self, states_2d: np.ndarray, prompt_indices: np.ndarray, *, count_as_training: bool = False) -> Tuple[np.ndarray, List[str]]:
        prompt_ids = self._prompt_ids_for_indices(prompt_indices)
        prompt_texts = self._prompt_texts_for_indices(prompt_indices)
        texts = self.core.impute_strings_from_states(
            states=states_2d,
            prompt_ids=prompt_ids,
            batch_size=self.impute_batch_size,
            strip=True,
        )
        scores = self.reward_scorer.score(prompt_texts, texts, batch_size=self.reward_batch_size)
        count = len(texts)
        if count_as_training:
            self.training_imputations += count
            self.training_reward_evals += count
        else:
            self.inference_imputations += count
            self.inference_reward_evals += count
        return scores, texts

    def _score_candidate_states(self, candidate_states: np.ndarray, prompt_indices: np.ndarray, *, count_as_training: bool = False) -> Tuple[np.ndarray, List[List[str]]]:
        B, N, L = candidate_states.shape
        flat_states = candidate_states.reshape(B * N, L)
        flat_indices = np.repeat(prompt_indices, N)
        flat_scores, flat_texts = self._score_states(flat_states, flat_indices, count_as_training=count_as_training)
        scores = flat_scores.reshape(B, N)
        texts = [flat_texts[i * N:(i + 1) * N] for i in range(B)]
        return scores, texts

    def _print_reward_vitals(
        self,
        candidate_rewards: np.ndarray,
        source_rewards: np.ndarray,
        tp1: int,
        *,
        train_info: Optional[dict] = None,
        cal_info: Optional[dict] = None,
    ):
        tau = self.lcb.taus[tp1]
        mu_t = self.lcb.mu[tp1]
        lam = self.lcb._get_lambda_val(tp1 - 1).item()

        print(f"  [Reward Vitals t={tp1}] Lam: {lam:.4f} | Centered Tau: {0.0 if tau is None else tau:.6f} | Mu: {mu_t:.4f}")
        print(
            f"    rewards: r_t={candidate_rewards.mean():.4f} +/- {candidate_rewards.std():.4f} | "
            f"r_tp1={source_rewards.mean():.4f} +/- {source_rewards.std():.4f}"
        )
        
        if tau is not None:
            print(f"    pass rate (>tau): {np.mean((candidate_rewards - mu_t) > tau):.1%}")

        if train_info is not None:
            print(
                f"    objective(train): before={train_info['objective_before']:.6f} "
                f"after={train_info['objective_after']:.6f} | "
                f"lambda: {train_info['lambda_before']:.4f}->{train_info['lambda_after']:.4f}"
            )
            print(
                f"    history(train): n_total={train_info['num_total_samples']} | "
                f"centered s mean={train_info['history_s_mean']:.4f} "
                f"std={train_info['history_s_std']:.4f}"
            )

        if cal_info is not None:
            print(
                f"    objective(cal): {cal_info['objective_at_lambda']:.6f} | "
                f"tau parts: mgf={cal_info['tau_term1_mgf']:.6f}, "
                f"delta={cal_info['tau_term2_delta']:.6f}, erm={cal_info['erm_error']:.6f}"
            )
            print(
                f"    history(cal): n_total={cal_info['num_total_samples']} | "
                f"centered s mean={cal_info['history_s_mean']:.4f} "
                f"std={cal_info['history_s_std']:.4f}"
            )

    def _unguided_step(self, *, count_as_training: bool):
        proposals = self._propose_smart(self.core.X.copy(), active_indices=None, count_as_training=count_as_training)
        self.core.commit(proposals[:, 0, :])
        unused = self.core.S * (self.core.N - 1)
        if count_as_training:
            self.training_nfe -= unused
        else:
            self.inference_nfe -= unused

    def train_policy(self, *, max_retries: Optional[int] = None, n_trajectories: Optional[int] = None):
        self.training_nfe = 0
        self.inference_nfe = 0
        self.training_imputations = 0
        self.inference_imputations = 0
        self.training_reward_evals = 0
        self.inference_reward_evals = 0

        current_retries = max_retries if max_retries is not None else self.max_retries
        n_traj = n_trajectories if n_trajectories is not None else self.n_trajectories

        for traj in range(n_traj):
            print(f"\n================ Trajectory {traj + 1}/{n_traj} ================\n")
            self.core.reset()
            while self.core.t > 0:
                tp1 = self.core.t
                print(f"--- Training Step {tp1} -> {tp1 - 1} ---")

                if not self._should_guide(tp1):
                    print("  (unguided step)")
                    self._unguided_step(count_as_training=True)
                    continue

                Xtp1 = self.core.X.copy()
                idx_all = np.arange(self.core.S)
                source_rewards, _ = self._score_states(Xtp1, idx_all, count_as_training=True)

                last_train_info = None
                for _ in range(self.train_batches_per_step):
                    train_proposals = self._propose_smart(Xtp1, active_indices=None, count_as_training=True)
                    train_rewards, _ = self._score_candidate_states(train_proposals, idx_all, count_as_training=True)
                    last_train_info = self.lcb.train_on_step(train_rewards, source_rewards, tp1)

                cal_proposals = self._propose_smart(Xtp1, active_indices=None, count_as_training=True)
                cal_rewards, _ = self._score_candidate_states(cal_proposals, idx_all, count_as_training=True)
                cal_info = self.lcb.calibrate_tau(cal_rewards, source_rewards, tp1)

                self._print_reward_vitals(
                    cal_rewards,
                    source_rewards,
                    tp1,
                    train_info=last_train_info,
                    cal_info=cal_info,
                )

                self._lcb_step_rejection(tp1, max_retries=current_retries)

            if (traj + 1) % 2 == 0:
                print(f"  Saving checkpoint {traj + 1}...")
                self.lcb.save(self.run_specific_dir, filename=f"{self.save_name}_ckpt_{traj + 1}")

        print(f"\nPolicy learned. Saving FINAL to {self.save_name}...")
        self.lcb.save(self.run_specific_dir, filename=self.save_name)
        return self._get_stats()

    def run_inference(self):
        self.core.reset()
        self.num_proposals_list = [0] * self.core.S
        self.inference_nfe = 0
        self.inference_imputations = 0
        self.inference_reward_evals = 0
        self.max_retries_hits = 0
        self.total_guided_step_applications = 0

        while self.core.t > 0:
            tp1 = self.core.t
            print(f"Inference Step: {tp1}")
            if not self._should_guide(tp1):
                self._unguided_step(count_as_training=False)
                continue
            if not self.lcb.trained_t[tp1]:
                print(f"\nWarning: tau not trained at t={tp1}; using tau=0.")
                self.lcb.taus[tp1] = 0.0
            self._lcb_step_rejection(tp1, max_retries=self.max_retries)
        print() # new line after progress loop
        return self.core.X, self._get_stats()

    def _lcb_step_rejection(self, tp1: int, *, max_retries: int = 20):
        Xtp1 = self.core.X.copy()
        S = self.core.S
        N = self.core.N

        if max_retries == 0:
            candidates = self._propose_smart(Xtp1, active_indices=None, count_as_training=False)
            self.core.commit(candidates[:, 0, :])
            self.inference_nfe -= S * (N - 1)
            return

        active_mask = np.ones(S, dtype=bool)
        next_Xt = np.zeros_like(Xtp1)
        tau = 0.0 if self.no_tau else (self.lcb.taus[tp1] if self.lcb.taus[tp1] is not None else 0.0)
        mu_t = self.lcb.mu[tp1] if self.lcb.mu[tp1] is not None else 0.0

        for _attempt in range(max_retries):
            active_indices = np.where(active_mask)[0]
            if len(active_indices) == 0:
                break

            for i in active_indices:
                self.num_proposals_list[i] += 1

            candidates = self._propose_smart(Xtp1, active_indices=active_indices, count_as_training=False)
            candidate_rewards, _ = self._score_candidate_states(candidates, active_indices, count_as_training=False)
            
            # Bound the centered tau using upper bound (if passed)
            baseline_tau = tau
            if self.upper_bound is not None:
                # Assuming upper bound is absolute, we need to center it as well
                baseline_tau = min(baseline_tau, self.upper_bound - mu_t)
            
            # Center candidate and scale by alpha strictly during inference
            centered_candidate = candidate_rewards - mu_t
            log_acc = (centered_candidate - baseline_tau) / self.lcb.alignment_temperature
            
            log_rand = np.log(np.random.rand(len(active_indices), N) + 1e-10)
            is_accepted = log_rand < log_acc

            for i, global_idx in enumerate(active_indices):
                accepted = np.where(is_accepted[i])[0]
                if len(accepted) == 0:
                    continue
                first = int(accepted[0])
                next_Xt[global_idx] = candidates[i, first]
                active_mask[global_idx] = False
                unused = N - (first + 1)
                self.inference_nfe -= unused

        self.total_guided_step_applications += S
        if np.any(active_mask):
            active_indices = np.where(active_mask)[0]
            self.max_retries_hits += len(active_indices)
            last_resort = self._propose_smart(
                Xtp1,
                active_indices=active_indices,
                count_as_training=False,
            )
            for i, global_idx in enumerate(active_indices):
                next_Xt[global_idx] = last_resort[i, 0]
                self.inference_nfe -= (N - 1)

        self.core.commit(next_Xt)
