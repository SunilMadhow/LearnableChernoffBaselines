from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import glob
import inspect
import json
import os
from datetime import datetime
from typing import Callable, Any, Dict

import numpy as np

from diffusion_core import Core

class Collector:
    """
    Collects trajectories from a diffusion core, evaluates rewards, and saves outputs.
    """

    def __init__(self, core, reward_fn: Callable[[np.ndarray], np.ndarray], base_dir: str = "data"):
        """
        Args:
            core: a diffusion core implementing collect_run() and save(states, data_dir)
            reward_fn: function mapping an array of states (shape (S, n_1, n_2, ...)) to rewards (shape (S,) or scalar)
            base_dir: root directory where run folders will be created
        """
        self.core = core
        self.reward_fn = reward_fn
        self.base_dir = base_dir

    def label_rewards(self, final_states):
        """Compute rewards for final states; allow scalar or per-state outputs."""
        final_states = np.asarray(final_states)
        if final_states.ndim == 0:
            raise ValueError(f"final_states must have at least one dimension, got shape {final_states.shape}")

        rewards = np.asarray(self.reward_fn(final_states))
        if rewards.shape == ():
            rewards = np.full(final_states.shape[0], rewards.item())
        elif rewards.shape != (final_states.shape[0],):
            raise ValueError(f"Tried to label rewards but got shape {rewards.shape}, expected {(final_states.shape[0],)} or scalar")
        return rewards

    def collect(self) -> Dict[str, Any]:
        """
        Run the diffusion core, compute rewards on the final states, and save outputs.

        Returns:
            dict with paths to config, states, rewards, and metadata.
        """
        traj = self.core.collect_run()
        final_states = traj[0]  # by convention index 0 is the most denoised / target side
        rewards = self.label_rewards(final_states)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        save_info = self.core.save(traj, data_dir=run_dir, run_id = run_id)

        rewards_path = os.path.join(run_dir, f"{run_id}_rewards.npy")
        np.save(rewards_path, rewards)

        reward_meta = {
            "name": getattr(self.reward_fn, "__name__", None),
            "source": None,
        }
        try:
            reward_meta["source"] = inspect.getsource(self.reward_fn)
        except (OSError, TypeError):
            reward_meta["source"] = None

        meta = {
            "run_id": run_id,
            "traj_shape": traj.shape,
            "reward_shape": rewards.shape,
            "core_save": save_info,
            "reward_fn": reward_meta,
        }
        meta_path = os.path.join(run_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("Run_id :", run_id)
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "config": save_info.get("config"),
            "states": save_info.get("states"),
            "rewards": rewards_path,
            "meta": meta_path,
        }
    

import torch
import numpy as np
from typing import Dict, Any

class SequentialCollector(Collector):
    def __init__(self, core, reward_fn, base_dir="data", batch_size=10):
        """
        Args:
            batch_size: How many sequences to generate at once on the GPU. 
                        For an 8B model on T4, 5-10 is safe.
        """
        super().__init__(core, reward_fn, base_dir)
        self.batch_size = batch_size

    def collect(self) -> Dict[str, Any]:
        target_S = self.core.S
        # Calculate how many loops we need
        num_batches = (target_S + self.batch_size - 1) // self.batch_size
        
        all_trajs = []
        print(f"Sequential Collection: {target_S} sequences in {num_batches} batches of {self.batch_size}...")

        # --- 1. Generation Loop ---
        for i in range(num_batches):
            # Determine size of this specific batch (handles the last partial batch)
            current_s = min(self.batch_size, target_S - (i * self.batch_size))
            
            # Configure core for this mini-batch
            self.core.S = current_s
            
            # Clear GPU memory from previous run to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"  > Batch {i+1}/{num_batches} (S={current_s})...")
            
            # Run generation (Returns shape: [T, current_s, L])
            traj_batch = self.core.collect_run() 
            all_trajs.append(traj_batch)

        # --- 2. Merge Trajectories ---
        # Concatenate along axis 1 (Batch dimension) to reconstruct full [T, 100, L]
        full_traj = np.concatenate(all_trajs, axis=1)
        
        print("  > Merging complete. Calculating rewards...")

        # --- 3. Patch & Delegate ---
        # We temporarily patch collect_run so the parent class uses our pre-computed data
        # instead of trying to run S=100 all at once.
        
        self.core.S = target_S # Restore correct S for metadata
        original_collect_run = self.core.collect_run
        
        # Override to return our merged result
        self.core.collect_run = lambda: full_traj
        
        try:
            # Call original logic (handles rewards, saving, metadata)
            return super().collect()
        finally:
            # Restore the core to its original state
            self.core.collect_run = original_collect_run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
