"""Regression checks for source conditioning, time labels and saved outputs."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coverage_experiment', ROOT/'analysis/coverage_experiment.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class ProbeCore:
    T = 4
    model_config = {'index_set': [0, 10, 20, 30, 40]}

    def __init__(self):
        self.S, self.N = 7, 3  # deliberately different from the source batch
        self.calls = []
        self.t, self.X = 99, np.full((7, 2), -999)
        self.transfer_schedule = np.zeros((7, 4))

    def propose(self):
        self.calls.append((self.t, self.X.copy(), self.S, self.N, self.transfer_schedule.copy()))
        # Independent probe number is encoded to give a known fraction covered.
        probe = (len(self.calls)-1) % 3
        return np.full((self.S, 1, 2), probe, dtype=np.int64)


class CoverageTests(unittest.TestCase):
    def test_sources_times_probes_and_empirical_fraction(self):
        core = ProbeCore()
        states = np.array([[[100*t+1, 100*t+2], [100*t+3, 100*t+4]] for t in range(5)])
        original_states = states.copy()
        schedule = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        critic_times, baseline_calls = [], []

        def critic(x, t):
            critic_times.append(t)
            return x[:, 0].astype(float)

        def baseline(x, t):
            baseline_calls.append((t, x.copy()))
            return np.ones(len(x))

        lcb = SimpleNamespace(taus=[0]*5, predict_under_log=baseline,
                              value_estimator=SimpleNamespace(predict_under_log=critic))
        result = coverage.estimate_coverage(lcb, core, states, schedule, num_probes=3)
        self.assertEqual(len(core.calls), 12)
        for row, source_t in enumerate([4, 3, 2, 1]):
            for call in core.calls[3*row:3*row+3]:
                self.assertEqual(call[0], source_t)
                np.testing.assert_array_equal(call[1], states[source_t])
                self.assertEqual(call[2:4], (2, 1))
                np.testing.assert_array_equal(call[4], schedule)
            self.assertEqual(baseline_calls[row][0], source_t)
            np.testing.assert_array_equal(baseline_calls[row][1], states[source_t])
        self.assertEqual(critic_times, [30]*3+[20]*3+[10]*3+[0]*3)
        np.testing.assert_allclose(result['coverage'], np.full((4, 2), 2/3))
        np.testing.assert_array_equal(states, original_states)
        np.testing.assert_array_equal(result['successor_steps'], [3, 2, 1, 0])
        self.assertEqual(result['proposal_tokens'].shape, (4, 2, 3, 2))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            np.savez_compressed(folder/'coverage.npz', **result)
            with np.load(folder/'coverage.npz') as saved:
                recalculated = (saved['log_values'] <= saved['thresholds'][:, :, None]).mean(axis=2)
                np.testing.assert_array_equal(recalculated, saved['coverage'])
            coverage.plot_results(result, 0.1, folder)
            self.assertGreater((folder/'coverage_heatmap.png').stat().st_size, 0)
            self.assertGreater((folder/'coverage_mean.png').stat().st_size, 0)

    def test_source_collection_indexes_states_by_diffusion_time(self):
        class SourceCore:
            T = 3
            S, N = 5, 9
            def reset(self):
                self.t = self.T
                self.X = np.full((self.S, 2), self.T)
                self.transfer_schedule = np.tile([3, 2, 1], (self.S, 1))
        core = SourceCore()
        def step(t, max_retries):
            self.assertEqual(t, core.t)
            core.t -= 1
            core.X[:] = core.t
        sampler = SimpleNamespace(core=core, max_retries=100, _lcb_step_rejection=step)
        states, schedule = coverage.collect_source_states(sampler, 2)
        self.assertEqual((core.S, core.N), (2, 1))
        self.assertEqual(states.shape, (4, 2, 2))
        for t in range(4):
            np.testing.assert_array_equal(states[t], np.full((2, 2), t))
        np.testing.assert_array_equal(schedule, [[3, 2, 1], [3, 2, 1]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
