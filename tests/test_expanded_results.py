"""Consistency checks for the expanded compact result set."""

import hashlib
import csv
import json
import unittest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class ExpandedResultsTests(unittest.TestCase):
    def test_figure_inventory_and_hashes(self):
        index = read("results/release/figure_index.json")
        self.assertEqual(index["source_pages"], 35)
        self.assertEqual({e["figure"] for e in index["figures"]},
                         {str(i) for i in range(1, 11)} | {f"A.{i}" for i in range(1, 14)})
        for entry in index["figures"]:
            self.assertEqual(hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])

    def test_sample_sd_and_reported_means(self):
        data = read("results/task1_point_forecasting/multiseed_aggregation_clean.json")
        seeds = read("results/task1_point_forecasting/seed_metrics.json")
        self.assertEqual(sum(map(len, data.values())), 21)
        for dataset, models in data.items():
            for model, metrics in models.items():
                self.assertEqual(metrics["standard_deviation_ddof"], 1)
                self.assertEqual(metrics["n_seeds"], 3)
                for metric in ["MAE", "RMSE", "MAPE"]:
                    values = [seeds[dataset][model][str(seed)]["overall"][metric] for seed in [43,44,45]]
                    self.assertAlmostEqual(metrics[f"{metric}_mean"], np.mean(values), delta=.00000051)
                    self.assertAlmostEqual(metrics[f"{metric}_std"], np.std(values, ddof=1), delta=.00000051)
        self.assertAlmostEqual(data["METR-LA"]["D2STGNN"]["MAE_mean"], 2.878, delta=.0005)
        self.assertAlmostEqual(data["PEMS04"]["STAEformer"]["MAE_mean"], 18.222, delta=.0005)

    def test_bootstrap_pair_aggregation(self):
        for dataset in ["metr-la", "pems-bay", "pems04"]:
            data = read(f"results/task1_point_forecasting/block_bootstrap_reconciled_{dataset}_seeds43-44-45.json")
            for pair in data["pairs"]:
                self.assertAlmostEqual(pair["mean_loss_diff_a_minus_b"],
                    pair["mae_a_flat_pooled_seeds_used"] - pair["mae_b_flat_pooled_seeds_used"], places=8)
                self.assertLess(pair["ci_low"], pair["ci_high"])
                self.assertTrue(set(pair["seeds_used"]) <= {43, 44, 45})
        # Table 10 reverses the sign used in the JSON artifacts.
        examples = [("metr-la", "D2STGNN", "STAEformer", .064, .045, .084),
                    ("pems-bay", "D2STGNN", "STAEformer", .060, .052, .067),
                    ("pems04", "STAEformer", "D2STGNN", .169, .098, .313)]
        for dataset, a, b, mean, low, high in examples:
            records = read(f"results/task1_point_forecasting/block_bootstrap_reconciled_{dataset}_seeds43-44-45.json")["pairs"]
            pair = next(p for p in records if {p["model_a"],p["model_b"]} == {a,b})
            values = (pair["mean_loss_diff_a_minus_b"],pair["ci_low"],pair["ci_high"])
            if pair["model_a"] == a:
                values = (-values[0],-values[2],-values[1])
            for actual, reported in zip(values, (mean,low,high)):
                self.assertAlmostEqual(actual, reported, delta=.0005)

    def test_block_length_sensitivity(self):
        for dataset, expected in [("metr-la", (17,21)), ("pems-bay", (9,10)), ("pems04", (19,21))]:
            data = read(f"results/task1_point_forecasting/block_bootstrap_sensitivity_reconciled_{dataset}_seeds43-44-45.json")
            self.assertEqual(data["block_lengths_tested"], [72,144,288])
            for result in data["results_by_block_len"].values():
                self.assertEqual((result["n_ci_excluding_zero"], result["n_pairs"]), expected)

    def test_plain_control_and_mc_dropout(self):
        control = read("results/task2_uncertainty/conformal_sigma_control/METR-LA_conformal_sigma_control.json")
        self.assertEqual(control["normalized_split_conformal"]["member_count"], 6)
        for key, picp, width in [("normalized_split_conformal", .9056, 23.309), ("plain_sigma1_control", .9021, 13.790)]:
            metrics = control[key]["evaluation_set"]
            self.assertAlmostEqual(metrics["PICP"], picp, delta=.00005)
            self.assertAlmostEqual(metrics["MPIW"], width, delta=.0005)
        paths = list((ROOT / "results/task2_uncertainty/mc_dropout_generated").glob("*/METR-LA/seed43/summary.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            data = json.loads(path.read_text())
            self.assertEqual(data["n_passes"], 50)
            self.assertLess(data["picp_uncalibrated"], .22)
            self.assertEqual(len(data["pass_prediction_sha256"]), 50)

    def test_crossed_masks_and_degree_matching(self):
        expected = {"METR-LA": {"D2STGNN", "STID", "MegaCRN"},
                    "PEMS-BAY": {"D2STGNN", "STID"}, "PEMS04": {"D2STGNN", "STID"}}
        for seed, suffix in [(43,""),(44,"_ckpt44"),(45,"_ckpt45")]:
            data = read(f"results/robustness/sensor_dropout_additional_mask_seeds{suffix}.json")
            self.assertEqual(data["checkpoint_seed"], seed)
            self.assertEqual(data["new_mask_seeds"], [7,11,19,23,31])
            self.assertEqual({ds:set(models) for ds,models in data["results"].items()}, expected)
            for models in data["results"].values():
                for record in models.values():
                    self.assertEqual(len(record["mask_runs"]), 5)
        paths = list((ROOT / "results/task3_explainability/degree_matched_control").glob("*.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            data = json.loads(path.read_text())
            self.assertEqual(data["degree_matched_random"]["n_draws"], 30)
            for draw in data["degree_matched_random"]["draws"]:
                self.assertEqual(sorted(draw["degrees"]), sorted(data["important_sensor_degrees"]))

    def test_naive_anchors_and_drift(self):
        data = read("results/task1_point_forecasting/naive_baselines.json")
        for dataset, persistence, seasonal in [("METR-LA",5.145,10.335),("PEMS-BAY",2.173,3.368),("PEMS04",31.616,40.656)]:
            self.assertAlmostEqual(data[dataset]["persistence"]["mae"], persistence, delta=.0005)
            self.assertAlmostEqual(data[dataset]["seasonal_naive"]["mae"], seasonal, delta=.0005)
        drift = read("results/release/chickenpox_drift.json")
        self.assertEqual(drift["runs"], 21)
        for key, percent in [("all",87.98),("first_third",82.24),("middle_third",92.29),("final_third",89.41)]:
            self.assertAlmostEqual(drift["coverage"][key]*100, percent, delta=.005)

    def test_reported_coding_agreement(self):
        data = read("results/release/evidence_coding_summary.json")
        self.assertEqual(sum(data["agreements_by_requirement"]), data["agreements"])
        total = data["decisions"]
        expected = sum(data["primary_marginals"][k] * data["independent_marginals"][k]
                       for k in ["M","P","NR"]) / total**2
        kappa = (data["agreements"]/total - expected) / (1-expected)
        self.assertAlmostEqual(kappa, data["reported_kappa"], delta=.0005)
        self.assertEqual(len(data["reconciled_changes"]), 5)
        with (ROOT / "results/release/literature_evidence_map.csv").open() as handle:
            rows = {row["study"]: row for row in csv.DictReader(handle)}
        self.assertEqual(len(rows), 13)
        for change in data["reconciled_changes"]:
            self.assertEqual(rows[change["study"]][change["requirement"]], change["to"])


if __name__ == "__main__":
    unittest.main()
