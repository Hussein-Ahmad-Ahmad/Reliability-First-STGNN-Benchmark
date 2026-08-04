from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pipelines.run_sensor_dropout import make_masks
from scripts.generate_conformal_intervals import (
    compute_conformal_metrics,
    higher_quantile,
)

ROOT = Path(__file__).resolve().parent.parent


class ConformalTests(unittest.TestCase):
    def test_higher_quantile_uses_finite_sample_correction(self) -> None:
        scores = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        threshold, count, level = higher_quantile(scores, alpha=0.25)

        self.assertEqual(count, 4)
        self.assertAlmostEqual(level, 0.9375)
        self.assertEqual(threshold, 4.0)

    def test_chronological_split_and_zero_target_mask(self) -> None:
        targets = np.array(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[1.0], [0.0]],
                [[0.0], [2.0]],
            ],
            dtype=np.float32,
        )
        mean = targets.copy()
        std = np.ones_like(targets)

        result = compute_conformal_metrics(
            mean,
            std,
            targets,
            dataset="TEST",
            alpha=0.1,
            member_count=3,
        )

        fixed = result["fixed"]
        self.assertEqual(
            fixed["chronological_split"],
            {"calibration_origins": 2, "evaluation_origins": 2},
        )
        self.assertEqual(fixed["evaluation_set"]["valid_target_count"], 2)
        self.assertEqual(fixed["evaluation_set"]["PICP"], 1.0)
        self.assertEqual(fixed["evaluation_set"]["MPIW"], 0.0)


class SensorMaskTests(unittest.TestCase):
    def test_masks_are_deterministic_nested_and_floor_sized(self) -> None:
        masks = make_masks(num_sensors=207, seed=42)
        repeated = make_masks(num_sensors=207, seed=42)

        self.assertEqual(masks, repeated)
        self.assertEqual(len(masks[0.10]), 20)
        self.assertEqual(len(masks[0.30]), 62)
        self.assertTrue(set(masks[0.10]).issubset(masks[0.30]))
        self.assertEqual(len(masks[0.30]), len(set(masks[0.30])))


class ArchivedArtifactTests(unittest.TestCase):
    def test_conformal_summary_values(self) -> None:
        expected = {
            ("METR-LA", "fixed"): (0.9056418267, 23.308811),
            ("METR-LA", "per_horizon"): (0.9056021454, 23.111981),
            ("PEMS-BAY", "fixed"): (0.9062850082, 11.690732),
            ("PEMS-BAY", "per_horizon"): (0.9063804840, 11.414535),
            ("PEMS04", "fixed"): (0.9009819565, 91.751328),
            ("PEMS04", "per_horizon"): (0.9006797835, 89.802675),
        }
        root = ROOT / "results" / "task2_uncertainty" / "conformal"
        for (dataset, variant), (picp, mpiw) in expected.items():
            path = root / f"{dataset}_conformal_{variant}_metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["evaluation_set"]["PICP"], picp)
            self.assertAlmostEqual(
                result["evaluation_set"]["MPIW"], mpiw, places=5
            )

    def test_sensor_changes_and_clean_passes_are_consistent(self) -> None:
        path = (
            ROOT
            / "results"
            / "robustness"
            / "sensor_dropout_fixed_masks_seed42.json"
        )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["protocol"]["generation_script"],
            "pipelines/run_sensor_dropout.py",
        )
        for dataset in artifact["datasets"].values():
            ten = set(
                dataset["masks"]["10%"][
                    "dropped_sensor_indices_zero_based"
                ]
            )
            thirty = set(
                dataset["masks"]["30%"][
                    "dropped_sensor_indices_zero_based"
                ]
            )
            self.assertTrue(ten.issubset(thirty))
            for model in dataset["models"].values():
                self.assertTrue(
                    model["clean_pass_verification"]["within_tolerance"]
                )
                baseline = model["rates"]["0%"]["mae"]
                for result in model["rates"].values():
                    expected_change = (
                        (result["mae"] - baseline) / baseline * 100.0
                    )
                    self.assertAlmostEqual(
                        result["relative_mae_change_percent"],
                        expected_change,
                    )


if __name__ == "__main__":
    unittest.main()
