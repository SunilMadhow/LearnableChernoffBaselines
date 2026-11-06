"""Regenerate coverage PNG/PDF figures from saved arrays, without inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from coverage_experiment import plot_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.run_dir/'coverage.npz') as saved:
        result = {key: saved[key] for key in ('coverage', 'successor_steps')}
    summary = json.loads((args.run_dir/'coverage_summary.json').read_text())
    plot_results(result, summary['delta'], args.run_dir)
    print(f'Regenerated coverage PNG/PDF figures in {args.run_dir}')
