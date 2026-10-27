from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "late_vintage_evaluation", ROOT / "14_exploratory_late_vintage_evaluation.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load late-vintage evaluation script.")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
robustness_spec = importlib.util.spec_from_file_location(
    "late_vintage_robustness", ROOT / "11_robustness.py"
)
if robustness_spec is None or robustness_spec.loader is None:
    raise RuntimeError("Cannot load robustness script.")
robustness = importlib.util.module_from_spec(robustness_spec)
robustness_spec.loader.exec_module(robustness)


class LateVintageEvaluationTests(unittest.TestCase):
    def test_common_cohort_validation_rejects_label_mismatch(self) -> None:
        predictions = pd.DataFrame(
            {
                "split": ["exploratory_late_test"] * 4,
                "horizon_months": [12] * 4,
                "model": ["logistic_regression", "logistic_regression", "xgboost", "xgboost"],
                "strategy": ["naive_full"] * 4,
                "vintage": [2020, 2021, 2020, 2021],
                "loan_identifier": ["a", "b", "a", "b"],
                "y_true": [0, 1, 0, 0],
                "probability": [0.1, 0.8, 0.2, 0.7],
            }
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            evaluation._validate_common_cohorts(predictions)

    def test_decision_cost_reuses_supplied_validation_threshold(self) -> None:
        predictions = pd.DataFrame(
            {
                "horizon_months": [12, 12],
                "model": ["logistic_regression"] * 2,
                "strategy": ["naive_full"] * 2,
                "y_true": [1, 0],
                "probability": [0.8, 0.2],
            }
        )
        thresholds = pd.DataFrame(
            {
                "model": ["logistic_regression"],
                "strategy": ["naive_full"],
                "horizon_months": [12],
                "cost_scenario": ["scenario"],
                "threshold": [0.5],
            }
        )
        settings = {
            "decision_cost_scenarios": {
                "scenario": {
                    "missed_default_cost": 1.0,
                    "review_cost_per_flag": 0.1,
                }
            }
        }
        result = evaluation._calculate_costs(predictions, thresholds, settings)
        self.assertEqual(result.loc[0, "validation_selected_threshold"], 0.5)
        self.assertAlmostEqual(result.loc[0, "test_mean_cost"], 0.05)

    def test_paired_bootstrap_accepts_named_exploratory_split(self) -> None:
        labels = [0, 1] * 10
        vintages = [vintage for vintage in (2020, 2021) for _ in labels]
        identifiers = [f"{vintage}-{index}" for vintage in (2020, 2021) for index in range(20)]
        probabilities = [0.1 if label == 0 else 0.8 for label in labels] * 2
        predictions = pd.DataFrame(
            [
                {
                    "split": "exploratory_late_test",
                    "horizon_months": 12,
                    "model": model,
                    "strategy": "naive_full",
                    "vintage": vintage,
                    "loan_identifier": identifier,
                    "y_true": label,
                    "probability": probability if model == "logistic_regression" else 1 - probability,
                }
                for model in ("logistic_regression", "xgboost")
                for vintage, identifier, label, probability in zip(
                    vintages, identifiers, labels * 2, probabilities
                )
            ]
        )
        scenarios = ["review_to_missed_cost_0.05", "review_to_missed_cost_1"]
        thresholds = pd.DataFrame(
            [
                {
                    "model": model,
                    "strategy": "naive_full",
                    "horizon_months": 12,
                    "cost_scenario": scenario,
                    "threshold": 0.5,
                }
                for model in ("logistic_regression", "xgboost")
                for scenario in scenarios
            ]
        )
        result = robustness.paired_bootstrap(
            predictions,
            thresholds,
            split_name="exploratory_late_test",
        )
        self.assertEqual(len(result), 5)
        self.assertTrue(result["bootstrap_replicates"].eq(1000).all())

    def test_segment_analysis_uses_supplied_years_and_split(self) -> None:
        features = pd.DataFrame(
            {
                "vintage": [2020],
                "loan_identifier": ["a"],
                "classic_fico": [700],
                "occupancy_status": ["P"],
            }
        )
        predictions = pd.DataFrame(
            {
                "split": ["exploratory_late_test"],
                "horizon_months": [12],
                "model": ["logistic_regression"],
                "strategy": ["naive_full"],
                "vintage": [2020],
                "loan_identifier": ["a"],
                "y_true": [1],
                "probability": [0.8],
            }
        )
        with patch.object(robustness.utils, "load_features", return_value=features) as loader:
            segments = robustness.segment_analysis(
                predictions,
                years=[2020],
                split_name="exploratory_late_test",
            )
        loader.assert_called_once_with([2020])
        self.assertEqual(segments.loc[0, "vintage"], 2020)
        self.assertEqual(segments.loc[0, "n"], 1)


if __name__ == "__main__":
    unittest.main()
