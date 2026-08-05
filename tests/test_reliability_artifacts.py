from __future__ import annotations

import json
import statistics
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

    def test_conformal_manifests_disclose_selection_and_availability(self) -> None:
        root = ROOT / "results" / "task2_uncertainty" / "conformal"
        metr = json.loads(
            (root / "METR-LA_ensemble_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(
            metr["selection_rule"]["coverage_used_for_member_selection"]
        )
        self.assertEqual(
            [member["checkpoint_epoch"] for member in metr["member_order"]],
            [83, 42, 21, 92, 56, 16],
        )
        self.assertTrue(
            all(member["seed"] is None for member in metr["member_order"])
        )
        self.assertIsNone(
            metr["artifact_availability"]["public_download_url"]
        )

        for dataset in ("PEMS-BAY", "PEMS04"):
            manifest = json.loads(
                (root / f"{dataset}_ensemble_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(
                manifest["coverage_used_for_member_selection"]
            )
            self.assertTrue(
                all(
                    isinstance(member["checkpoint_epoch"], int)
                    for member in manifest["member_order"]
                )
            )
            self.assertIsNone(
                manifest["artifact_availability"]["public_download_url"]
            )

    def test_chickenpox_protocol_manifest_is_complete(self) -> None:
        path = (
            ROOT
            / "results"
            / "nontraffic_graph_sanity"
            / "chickenpox_protocol_manifest.json"
        )
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["forecast_task"]["split_windows"],
            {"train": 349, "val": 50, "test": 99},
        )
        self.assertEqual(
            protocol["interval_diagnostic"]["calibration_targets"], 12000
        )
        self.assertEqual(
            protocol["interval_diagnostic"]["evaluation_targets"], 23760
        )
        self.assertEqual(
            sorted(protocol["optimization"]["seeds"]), [43, 44, 45]
        )
        script = (
            ROOT / "scripts" / "run_chickenpox_all_baselines.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("D:/Hussein-Files", script)

    def test_runtime_provenance_matches_declared_aggregation(self) -> None:
        path = (
            ROOT
            / "results"
            / "compute"
            / "METR-LA_runtime_provenance.json"
        )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        for aggregate in artifact["aggregate"]:
            rows = [
                row
                for row in artifact["per_seed"]
                if row["model"] == aggregate["model"]
            ]
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(
                aggregate["train_seconds_per_epoch"],
                statistics.median(
                    row["median_final_ten_train_seconds"] for row in rows
                ),
            )
            self.assertAlmostEqual(
                aggregate["recorded_log_span_minutes"],
                statistics.median(
                    row["recorded_log_span_minutes"] for row in rows
                ),
            )


if __name__ == "__main__":
    unittest.main()
