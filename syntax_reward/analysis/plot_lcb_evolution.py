from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
import sys
sys.path.insert(0, _ROOT)

_FIGURES = _Path(_ROOT) / "figures" / "regenerated"
_FIGURES.mkdir(parents=True, exist_ok=True)

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import seaborn as sns
from matplotlib.lines import Line2D

# Import project modules
from nets_language_classifier import ValueClassifier
from llada_core_hide_prompt import LLaDACore
from lcb_with_offline import LCB, LCBSmartSampler
from util import seed_everything, load_core_and_tokenizer

# Use seaborn for high-contrast plotting
sns.set_theme(style="whitegrid", context="talk")

def get_abc_reward(text):
    """Calculates the 1/3, 2/3, 3/3 score based on ABC constraint."""
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    targets = ['A', 'B', 'C']
    score = 0
    for i in range(min(3, len(sentences))):
        clean_sent = re.sub(r'^[^a-zA-Z]+', '', sentences[i])
        if clean_sent and clean_sent[0].upper() == targets[i]:
            score += 1
    return score / 3.0

def collect_trajectories_via_api(lcb, core, num_samples=100, delta=0.1):
    """
    Generates trajectories by calling the actual sampler API step-by-step.
    """
    # 1. Initialize Sampler (User order: Core, LCB)
    sampler = LCBSmartSampler(core, lcb)
    lcb.delta = delta 
    
    # 2. Reset and Propose Initial Noise (x_T)
    core.reset()
    init_props = core.propose() 
    if init_props.ndim == 3: init_props = init_props[:, 0, :]
    core.commit(init_props)
    
    # Storage
    logs = [ {"t": [], "v": [], "b": []} for _ in range(num_samples) ]
    
    # 3. Step Loop (T-1 down to 0)
    for t in tqdm(range(core.T - 1, 0, -1), desc="Stepping"):
        
        # tp1 is the time index of the CURRENT state X_{t+1}
        tp1 = t + 1
        
        # --- A. Measure BEFORE the step (State is X_{tp1}) ---
        Xtp1 = core.X 
        
        # Calculate Baseline B_{tp1} = u_{tp1} + tau
        log_u = lcb.predict_under_log(Xtp1, tp1) 
        if isinstance(log_u, torch.Tensor): log_u = log_u.cpu().numpy()
        
        # Handle Tau safely
        tau = lcb.taus[tp1] if tp1 < len(lcb.taus) else 0.0
        if tau is None: tau = 0.0
            
        B_tp1 = log_u + tau
        
        # --- B. Execute the Step ---
        # We pass tp1 because the sampler needs to look up u_{tp1} and tau_{tp1}
        sampler._lcb_step_rejection(tp1) 
        
        # --- C. Measure AFTER the step (State is now X_t) ---
        Xt = core.X
        
        # Calculate Value v_t(x_t)
        log_v = lcb.value_estimator.predict_under_log(Xt, t) 
        if isinstance(log_v, torch.Tensor): log_v = log_v.cpu().numpy()
        
        # --- D. Record ---
        for i in range(num_samples):
            logs[i]["t"].append(t)
            logs[i]["b"].append(B_tp1[i])
            logs[i]["v"].append(log_v[i])

    # Decode
    final_tokens = core.X.astype(np.int64)
    texts = core.tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    
    return texts, logs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--lcb_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=f"{_ROOT}/data_runs")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    run_dir = os.path.join(args.data_dir, args.run_id)
    
    # Load Components
    seed_everything(42)
    vc = ValueClassifier.load(run_dir)
    vc.reward_temperature = args.temp
    core, tokenizer = load_core_and_tokenizer(args.num_samples)
    core.S = args.num_samples
    lcb = LCB.load(run_dir, None, vc, core, args.lcb_name)
    
    # Run Collection
    print("Collecting trajectories...")
    texts, logs = collect_trajectories_via_api(lcb, core, num_samples=args.num_samples)
    
    import json
    trace_path = _FIGURES / f"lcb_traces_{args.run_id}_alpha_{args.temp}.json"
    trace_path.write_text(json.dumps({"texts": texts, "logs": logs}, default=lambda x: np.asarray(x).tolist()))

    # Select Representatives
    representatives = {}
    target_scores = [1.0/3.0, 2.0/3.0, 1.0]
    
    # Visual Styles
    styles = {
        1.0/3.0: {"color": "#d62728", "label": "Reward: 1/3 (Fail)"},    # Red
        2.0/3.0: {"color": "#ff7f0e", "label": "Reward: 2/3 (Partial)"}, # Orange
        1.0:     {"color": "#2ca02c", "label": "Reward: 3/3 (Success)"}  # Green
    }
    
    print("Analyzing results...")
    for i, text in enumerate(texts):
        score = get_abc_reward(text)
        for tgt in target_scores:
            if abs(score - tgt) < 0.01 and tgt not in representatives:
                representatives[tgt] = logs[i]
                print(f"  Found {styles[tgt]['label']}: {text[:40]}...")
        if len(representatives) == 3:
            break
            
    # --- PLOTTING ---
    plt.figure(figsize=(6, 6))
    
    plotted_any = False
    for tgt in sorted(target_scores, reverse=True): # Plot Success first (z-order) or last?
        if tgt in representatives:
            plotted_any = True
            data = representatives[tgt]
            style = styles[tgt]
            
            steps = data["t"]
            vals = data["v"]
            baselines = data["b"]
            
            # Plot Baseline (Dashed)
            plt.plot(steps, baselines, 
                     color=style["color"], linestyle="-", linewidth=3,
                     label=None) # Don't duplicate legend entries
            
            # Plot Value (Solid)
            plt.plot(steps, vals, 
                     color=style["color"], linestyle="--", linewidth=2,
                     label=style['label'])
    traj_legend_handles = [
        Line2D([0], [0], color=styles[tgt]["color"], lw=3,
               label=styles[tgt]["label"])
        for tgt in sorted(representatives, reverse=True)
    ]

    if not plotted_any:
        print("No samples found matching targets.")
        return

    # Dummy legend for line styles
    plt.plot([], [], color='gray', linestyle='-', label='Value Estimate $v_t$')
    plt.plot([], [], color='gray', linestyle='--', label='Baseline $B_{t+1}$')

    plt.title(f"LCB Trajectory Evolution", fontsize=16, pad=15)
    plt.xlabel("Diffusion Timestep $t$ (Generation $\\rightarrow$)", fontsize=14)
    plt.ylabel("Soft Value", fontsize=14)
    # plt.text(
    # 0.02, 0.02,
    #   "dotted: soft value\nsolid: baseline",
    #   fontsize=8,
    #   va="bottom", ha="left",
    #   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    # )

    plt.legend(
    handles=traj_legend_handles,
    loc="lower left",
    fontsize=10,
    frameon=True
)

    plt.gca().invert_xaxis() # T -> 0
    # plt.legend(frameon=True, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_file = _FIGURES / f"lcb_combined_{args.run_id}_alpha_{args.temp}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()
