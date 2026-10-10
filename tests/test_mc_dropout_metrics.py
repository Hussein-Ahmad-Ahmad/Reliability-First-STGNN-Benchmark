import unittest

import numpy as np

from scripts.mc_dropout_metrics import summarize_moments


class MCDropoutMetricsTests(unittest.TestCase):
    def test_one_mask_is_used_for_all_metrics(self):
        result = summarize_moments(np.array([0., 1., 4.]), np.array([100., 1., 2.]),
                                   np.array([0., 1., 8.]), 1., ["a", "b"])
        self.assertEqual(result["valid_target_count"], 2)
        self.assertEqual(result["picp_uncalibrated"], .5)
        self.assertEqual(result["mpiw_uncalibrated"], 3.)
        self.assertEqual(result["variance_mean"], 2.5)

    def test_identical_passes_are_inactive_despite_roundoff(self):
        result = summarize_moments(np.ones(2), np.full(2, 2e-5), np.ones(2), 1.645, ["a"] * 50)
        self.assertEqual(result["status"], "inactive")
        self.assertGreater(result["variance_mean"], 1e-12)

    def test_near_null_and_nonfinite_targets_are_excluded(self):
        result = summarize_moments(np.array([0., 0., 1.]), np.ones(3),
                                   np.array([1e-6, np.nan, 1.]), 1., ["a", "b"])
        self.assertEqual(result["valid_target_count"], 1)
        self.assertEqual(result["excluded_target_count"], 2)

    def test_invalid_predictions_are_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            summarize_moments(np.array([np.nan]), np.ones(1), np.ones(1), 1., ["a", "b"])


if __name__ == "__main__":
    unittest.main()
