"""Consistency checks for the expanded compact result set."""

import hashlib
import json
import unittest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class ExpandedResultsTests(unittest.TestCase):
    def test_plot_inventory_and_hashes(self):
        index = read("results/release/plot_index.json")
        self.assertEqual(len(index["plots"]), 24)
        paths = {entry["path"] for entry in index["plots"]}
        self.assertEqual(paths, {p.relative_to(ROOT).as_posix()
                               for p in (ROOT / "figures/results").glob("*.png")})
        for entry in index["plots"]:
            self.assertNotIn("figure_", entry["name"])
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
        # Check both orientations of the pairwise comparison.
        examples = [("metr-la", "D2STGNN", "STAEformer", .064, .046, .084),
                    ("pems-bay", "D2STGNN", "STAEformer", .060, .052, .067),
                    ("pems04", "STAEformer", "D2STGNN", .171, .101, .314)]
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
                n = {"metr-la": 6831, "pems-bay": 10400, "pems04": 3375}[dataset]
                self.assertEqual(result["origin_counts"], [n])
                for pair in result["pairs"]:
                    self.assertEqual(pair["n_forecast_origins"], n)
                    self.assertEqual(pair["blocks_drawn_per_replicate"], -(-n // result["block_len"]))

    def test_plain_control_and_mc_dropout(self):
        control = read("results/task2_uncertainty/conformal_sigma_control/METR-LA_conformal_sigma_control.json")
        self.assertEqual(control["normalized_split_conformal"]["member_count"], 6)
        for key, picp, width in [("normalized_split_conformal", .9057, 23.378), ("plain_sigma1_control", .9014, 13.790)]:
            metrics = control[key]["evaluation_set"]
            self.assertAlmostEqual(metrics["PICP"], picp, delta=.00005)
            self.assertAlmostEqual(metrics["MPIW"], width, delta=.0005)
        paths = list((ROOT / "results/task2_uncertainty/mc_dropout_generated").glob("*/METR-LA/seed43/summary.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            data = json.loads(path.read_text())
            self.assertEqual(data["n_passes"], 50)
            self.assertLess(data["picp_uncalibrated"], .25)
            self.assertEqual(data["valid_target_count"], 14908092)
            self.assertEqual(data["target_mask"]["atol"], 5e-5)
            self.assertAlmostEqual(data["mpiw_uncalibrated"],
                                   2 * data["interval_z_value"] * data["predictive_std_mean"])
            if data["model"] in {"MegaCRN", "STNorm"}:
                self.assertEqual(data["status"], "inactive")
                self.assertEqual(data["distinct_pass_predictions"], 1)
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
        self.assertEqual(len(paths), 10)
        for path in paths:
            data = json.loads(path.read_text())
            self.assertEqual(data["degree_matched_random"]["n_draws"], 30)
            for draw in data["degree_matched_random"]["draws"]:
                if data["degree_matching"]["exact"]:
                    self.assertEqual(sorted(draw["degrees"]), sorted(data["important_sensor_degrees"]))
                else:
                    self.assertEqual((data["model"], data["dataset"]), ("MTGNN", "METR-LA"))
                    self.assertEqual(sum(abs(a-b) for a,b in zip(draw["degrees"],data["important_sensor_degrees"])), 3)

    def test_naive_anchors_and_drift(self):
        data = read("results/task1_point_forecasting/naive_baselines.json")
        for dataset, persistence, seasonal in [("METR-LA",5.145,10.335),("PEMS-BAY",2.173,3.368),("PEMS04",31.616,40.656)]:
            self.assertAlmostEqual(data[dataset]["persistence"]["mae"], persistence, delta=.0005)
            self.assertAlmostEqual(data[dataset]["seasonal_naive"]["mae"], seasonal, delta=.0005)
        drift = read("results/release/chickenpox_drift.json")
        self.assertEqual(drift["runs"], 21)
        for key, percent in [("all",87.98),("first_third",82.24),("middle_third",92.29),("final_third",89.41)]:
            self.assertAlmostEqual(drift["coverage"][key]*100, percent, delta=.005)

    def test_retained_input_mask_provenance(self):
        data = read("results/task3_explainability/ranking_provenance.json")
        self.assertFalse(data["edge_masking_established"])
        self.assertEqual(len(data["records"]), 7)
        for record in data["records"]:
            self.assertTrue(record["top10_matches_retained_array"])
            self.assertTrue(record["fidelity_values_match_original"])
            self.assertFalse(record["same_horizon_perturbation_stability_established"])
            self.assertIn(record["ranking_horizon"], (1, 12))
            self.assertEqual(record["fidelity_horizon"], 12)

    def test_public_overview_links(self):
        import re
        import subprocess

        for name in ["README.md", "RESULTS.md", "REPRODUCIBILITY_CHECK.md"]:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotRegex(text.lower(), r"manuscript|supplied.*pdf|numbered figures")
            targets = re.findall(r"\]\(([^)]+)\)", text)
            targets += re.findall(r'src="([^"]+)"', text)
            for target in targets:
                if not target.startswith(("https://", "http://", "#")):
                    self.assertTrue((ROOT / target).exists(), target)
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        for name in tracked:
            path = ROOT / name
            if path.is_file() and name.startswith(("scripts/", "results/")) and path.suffix in {".py", ".json", ".md"}:
                self.assertNotRegex(path.read_text(encoding="utf-8").lower(),
                                    r"manuscript|reviewer|resubmission", name)


if __name__ == "__main__":
    unittest.main()
