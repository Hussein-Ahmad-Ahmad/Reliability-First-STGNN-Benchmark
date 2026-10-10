import math
import unittest

import numpy as np

from scripts.run_block_bootstrap_reconciled import bootstrap_flat_pooled_diff


class BootstrapSamplingTests(unittest.TestCase):
    def test_prefix_sum_sampling_matches_explicit_origin_sampling(self):
        for n, block_len in [(31, 7), (32, 8), (7, 7), (19, 1)]:
            with self.subTest(n=n, block_len=block_len):
                rng = np.random.default_rng(123)
                values = [rng.random(n), rng.integers(1, 9, n).astype(float),
                          rng.random(n), rng.integers(1, 9, n).astype(float)]
                expected = []
                reference_rng = np.random.default_rng(456)
                for _ in range(1100):
                    starts = reference_rng.integers(0, n - block_len + 1, math.ceil(n / block_len))
                    indices = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
                    expected.append(values[0][indices].sum() / values[1][indices].sum()
                                    - values[2][indices].sum() / values[3][indices].sum())
                actual = bootstrap_flat_pooled_diff(*values, block_len, 1100, np.random.default_rng(456))
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_invalid_sampling_inputs(self):
        values = [np.ones(10)] * 4
        for block in [0, 11]:
            with self.assertRaises(ValueError):
                bootstrap_flat_pooled_diff(*values, block, 10, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            bootstrap_flat_pooled_diff(*values, 5, 0, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            bootstrap_flat_pooled_diff(np.ones(9), *values[1:], 5, 10, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            bootstrap_flat_pooled_diff(values[0], np.zeros(10), *values[2:], 5, 10, np.random.default_rng(1))


if __name__ == '__main__':
    unittest.main()
