"""Estimate conditional one-step coverage using fresh model proposals.

Source trajectories use the saved LCB sampler. Probes use the base model
transition, with the source state, timestep and transfer schedule restored.
Values are the critic's log soft values, including at the terminal timestep.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def copy_schedule(schedule):
    return schedule.clone() if hasattr(schedule, 'clone') else schedule.copy()


def collect_source_states(sampler, num_samples):
    """Return states indexed by diffusion time and their original schedule."""
    core = sampler.core
    core.S = num_samples
    core.N = 1
    core.reset()
    schedule = copy_schedule(core.transfer_schedule)
    sampler.num_proposals_list = [0] * num_samples
    sampler.inference_nfe = sampler.training_nfe = 0
    states = np.empty((core.T + 1, *core.X.shape), dtype=core.X.dtype)
    states[core.T] = core.X.copy()
    while core.t > 0:
        tp1 = core.t
        sampler._lcb_step_rejection(tp1, max_retries=sampler.max_retries)
        states[core.t] = core.X.copy()
    return states, schedule


def estimate_coverage(lcb, core, states, schedule, num_probes):
    """Estimate P[V_t(X_t) <= u_{t+1}(x)+tau_{t+1} | x] at each saved x.

The core must use the same prompt/configuration that generated `states`.
Every probe at a given step is drawn afresh from that fixed source; probes
are not committed into the source trajectories.
"""
    source_steps = np.arange(core.T, 0, -1)  # includes the final 1 -> 0 step
    successor_steps = source_steps - 1
    num_samples = states.shape[1]
    thresholds = np.empty((core.T, num_samples))
    values = np.empty((core.T, num_samples, num_probes))
    proposals = np.empty((core.T, num_samples, num_probes, states.shape[-1]), dtype=states.dtype)
    core.S = num_samples
    core.N = 1

    for row, tp1 in enumerate(source_steps):
        t = int(tp1) - 1
        core.X = states[tp1].copy()
        core.t = int(tp1)
        core.transfer_schedule = copy_schedule(schedule)
        core.tentative_states = None
        tau = lcb.taus[tp1]
        if tau is None:
            raise ValueError(f'Missing calibrated tau for source timestep {tp1}.')
        thresholds[row] = np.asarray(lcb.predict_under_log(core.X, int(tp1))).reshape(-1) + float(tau)
        for probe in range(num_probes):
            successor = core.propose()[:, 0, :]
            proposals[row, :, probe] = successor
            values[row, :, probe] = np.asarray(
                lcb.value_estimator.predict_under_log(successor, core.model_config['index_set'][t])
            ).reshape(-1)
        fraction = (values[row] <= thresholds[row, :, None]).mean(axis=1)
        print(f'{tp1:2d} -> {t:2d}: mean coverage {fraction.mean():.4f}', flush=True)

    covered = values <= thresholds[:, :, None]
    return {
        'source_steps': source_steps,
        'successor_steps': successor_steps,
        'critic_times': np.asarray(core.model_config['index_set'])[successor_steps],
        'thresholds': thresholds,
        'log_values': values,
        'covered': covered,
        'coverage': covered.mean(axis=2),
        'source_states': states,
        'proposal_tokens': proposals,
        'transfer_schedule': schedule.detach().cpu().numpy() if hasattr(schedule, 'detach') else np.asarray(schedule),
    }


def plot_results(result, delta, output_dir):
    """Plot coverage by trajectory and successor timestep."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    coverage = result['coverage']
    steps = result['successor_steps']
    with sns.axes_style('whitegrid'), sns.plotting_context('talk'):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            coverage.T, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
            cbar_kws={'label': 'Coverage Probability'},
            linewidths=0.5, linecolor='white',
            xticklabels=steps, yticklabels=2,
        )
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        ax.set_xlabel('Diffusion Step ($t$)')
        ax.set_ylabel(f'Trajectory index (of M={coverage.shape[1]})')
        fig.subplots_adjust(left=0.085, right=0.98, bottom=0.13, top=0.90)
        fig.savefig(output_dir/'coverage_heatmap.png', dpi=300)
        fig.savefig(output_dir/'coverage_heatmap.pdf')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, coverage.mean(axis=1), marker='o', label='Mean across source states')
    ax.axhline(1-delta, color='green', linestyle=':', label=f'Target 1-delta = {1-delta:g}')
    ax.invert_xaxis()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Successor timestep t')
    ax.set_ylabel('Fraction covered')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/'coverage_mean.png', dpi=180)
    fig.savefig(output_dir/'coverage_mean.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--lcb_name', required=True)
    parser.add_argument('--data_dir', default=str(ROOT/'data_runs'))
    parser.add_argument('--temp', type=float, required=True)
    parser.add_argument('--num_samples', type=int, default=50, help='Number of source trajectories.')
    parser.add_argument('--num_probes', type=int, default=20, help='Fresh successors per source state.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default=None, help='A new directory for arrays, metadata and plots.')
    args = parser.parse_args()
    if args.num_samples < 1 or args.num_probes < 1 or args.temp <= 0:
        parser.error('num_samples, num_probes and temp must be positive.')
    output_dir = Path(args.output_dir) if args.output_dir else ROOT/'reruns'/('coverage_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output_dir.mkdir(parents=True, exist_ok=False)

    # Keep the pure measurement/plotting functions usable without GPU packages.
    import torch
    from nets_language_classifier import ValueClassifier
    from lcb_with_offline import LCB, LCBSmartSampler
    from util import seed_everything, load_core_and_tokenizer
    seed_everything(args.seed)
    run_dir = Path(args.data_dir)/args.run_id
    vc = ValueClassifier.load(str(run_dir))
    vc.reward_temperature = args.temp
    core, tokenizer = load_core_and_tokenizer(args.num_samples)
    lcb = LCB.load(str(run_dir), None, vc, core, args.lcb_name)
    missing = [t for t in range(1, core.T+1) if not lcb.trained_t[t] or lcb.taus[t] is None]
    if missing:
        raise ValueError(f'Coverage requires a calibrated checkpoint; missing timesteps: {missing}')
    sampler = LCBSmartSampler(core, lcb, max_retries=100)
    print(f'Collecting {args.num_samples} LCB-guided source trajectories...', flush=True)
    states, schedule = collect_source_states(sampler, args.num_samples)
    print(f'Probing {core.T} transitions with {args.num_probes} fresh proposals per source...', flush=True)
    result = estimate_coverage(lcb, core, states, schedule, args.num_probes)
    np.savez_compressed(output_dir/'coverage.npz', **result)
    summary = {
        'implementation': 'fixed_conditional_coverage_v1',
        'run_id': args.run_id, 'lcb_name': args.lcb_name,
        'alpha': args.temp, 'delta': float(lcb.delta), 'seed': args.seed,
        'num_sources': args.num_samples, 'num_probes': args.num_probes,
        'num_transitions': core.T, 'includes_terminal_transition': True,
        'source_distribution': 'LCB-guided trajectories, max_retries=100, no clipping',
        'probe_distribution': 'unguided model transition conditional on each saved source state',
        'value_quantity': 'critic log soft value, including at timestep 0; not exact reward',
        'mean_coverage': float(result['coverage'].mean()),
        'min_state_coverage': float(result['coverage'].min()),
        'mean_coverage_per_transition': result['coverage'].mean(axis=1).tolist(),
        'target_coverage': 1-float(lcb.delta),
        'inference_parameters': core.model_config,
        'gpu': torch.cuda.get_device_name(0),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'llada_revision': getattr(core.model.config, '_commit_hash', None),
    }
    (output_dir/'coverage_summary.json').write_text(json.dumps(summary, indent=2))
    plot_results(result, lcb.delta, output_dir)
    print(f'Saved arrays, settings and figures to {output_dir}', flush=True)
    print(f'Mean coverage: {summary["mean_coverage"]:.4f}; target: {summary["target_coverage"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
