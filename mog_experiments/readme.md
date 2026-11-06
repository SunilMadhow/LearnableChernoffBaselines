LCB Experiments
================

This repo contains small experiments around diffusion-style sampling with learned value functions and lower confidence bounds (LCB) on a toy Gaussian mixture model.

Contents
--------
- `toy_data/mixture_model.py`: toy diffusion core over a 2D Gaussian mixture; supports collecting trajectories.
- `collection.py`: utilities to collect runs (states + rewards) and reload them.
- `nets.py`: MLP value estimator (`ValueEstimate`) and helpers to train/load it.
- `simple_rejection.py`: reverse-diffusion sampler with rejection using a value function upper bound.
- `toy_data/scripts/simple_rejection_run.py`: script to run the rejection sampler on a saved run.
- `lcb.py`: LCB baseline training and sampling; includes a CLI to train lambdas/baseline on a collected run.

Quick start
-----------
1) Collect a run (creates `toy_data_runs/<run_id>` with states/rewards):
```bash
python toy_data/mixture_model.py
```

2) Train the value estimator on that run:
```bash
python nets.py <run_id>
```
This saves `<run_id>_value_estimator.pt` under `toy_data_runs/<run_id>/`.

3) Code for training LCBs and drawing samples is in main_sym_20_epochs.ipynb
3) Code for creating plots from the paper is in  main_sym_plots.ipynb



Notes
-----
- Default devices are CPU; pass `--device` where supported if you want GPU/MPS.
- Shapes: states are typically `(T+1, S, d)`; value/LCB nets take inputs shaped `(..., d)` with a scalar or broadcastable index feature.
- The code mixes NumPy and torch; keep tensors/arrays on consistent devices when modifying. The current scripts avoid autograd during sampling.
