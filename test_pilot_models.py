from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
import pilot_model_utils as utils


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


costs = load_script("decision_costs", "10_decision_costs.py")
robustness = load_script("pilot_robustness", "11_robustness.py")
survival = load_script("pilot_survival", "08_train_survival_models.py")


class PilotCohortTests(unittest.TestCase):
    def test_feature_whitelist_excludes_identifiers_and_post_origination_fields(self) -> None:
        self.assertNotIn("loan_identifier", utils.FEATURE_COLUMNS)
        self.assertNotIn("vintage", utils.FEATURE_COLUMNS)
        self.assertNotIn("model_eligible", utils.FEATURE_COLUMNS)
        self.assertNotIn("outcome_type", utils.FEATURE_COLUMNS)

    def test_categorical_missingness_is_not_imputed_to_a_rare_observed_level(self) -> None:
        preprocessor = utils.make_preprocessor(
            feature_columns=["special_eligibility_program"]
        )
        frame = pd.DataFrame(
            {"special_eligibility_program": [np.nan, "H", np.nan]}
        )
        encoded = preprocessor.fit_transform(frame)
        names = preprocessor.get_feature_names_out().tolist()
        self.assertIn(
            "categorical__special_eligibility_program___MISSING__",
            names,
        )
        missing_column = names.index(
            "categorical__special_eligibility_program___MISSING__"
        )
        self.assertEqual(encoded[:, missing_column].toarray().ravel().tolist(), [1.0, 0.0, 1.0])

    def test_complete_horizon_label_boundaries_and_payoff(self) -> None:
        self.assertEqual(
            utils._complete_horizon_label(
                {"entry_month": 1, "outcome_type": "default", "outcome_month": 36}, 36
            ),
            (True, 1),
        )
        self.assertEqual(
            utils._complete_horizon_label(
                {"entry_month": 1, "outcome_type": "default", "outcome_month": 37}, 36
            ),
            (True, 0),
        )
        self.assertEqual(
            utils._complete_horizon_label(
                {"entry_month": 1, "outcome_type": "voluntary_payoff", "outcome_month": 12}, 36
            ),
            (True, 0),
        )
        self.assertEqual(
            utils._complete_horizon_label(
                {"entry_month": 1, "outcome_type": "other_termination", "outcome_month": 12}, 36
            ),
            (False, None),
        )

    def test_delayed_entry_time_conversion_preserves_month_one_boundary(self) -> None:
        entry, exit_time = utils.cox_time_bounds(
            pd.Series([1, 4, 6]), pd.Series([1, 9, 12])
        )
        self.assertEqual(entry.tolist(), [0.0, 3.0, 5.0])
        self.assertEqual(exit_time.tolist(), [1.0, 9.0, 12.0])
        with self.assertRaisesRegex(ValueError, "strictly after"):
            utils.cox_time_bounds(pd.Series([6]), pd.Series([5]))

    def test_matched_naive_cohort_reproduces_complete_strata_counts(self) -> None:
        full = pd.DataFrame(
            {
                "vintage": [2005] * 6 + [2006] * 4,
                "classic_fico": [600, 610, 650, 655, 720, 725, 620, 630, 760, 770],
                "original_ltv": [70, 75, 85, 88, 92, 93, 55, 58, 65, 66],
                "occupancy_status": ["P"] * 10,
                "loan_identifier": [f"loan-{i}" for i in range(10)],
            }
        )
        eligible = full.iloc[[0, 2, 4, 6, 8]].copy()
        matched, balance = utils.matched_naive_sample(full, eligible, seed=42)
        self.assertEqual(len(matched), len(eligible))
        eligible = utils.add_matching_strata(eligible)
        self.assertEqual(
            matched.groupby(utils.MATCH_STRATA, dropna=False, observed=True).size().sort_index().tolist(),
            eligible.groupby(utils.MATCH_STRATA, dropna=False, observed=True).size().sort_index().tolist(),
        )
        self.assertTrue(balance["matched_count"].eq(balance["complete_count"]).all())


class PilotDecisionCostTests(unittest.TestCase):
    def test_cost_includes_missed_defaults_and_reviewed_accounts(self) -> None:
        y = np.array([1, 0, 1, 0], dtype="int8")
        probabilities = np.array([0.9, 0.8, 0.3, 0.1])
        value = utils.cost_value(y, probabilities, 0.5, missed_default=2.0, review=0.25)
        self.assertAlmostEqual(value, (2.0 + 2 * 0.25) / 4)

    def test_validation_threshold_is_minimum_and_ties_prefer_fewer_reviews(self) -> None:
        validation = pd.DataFrame(
            {
                "y_true": [1, 0, 1, 0],
                "probability": [0.9, 0.8, 0.3, 0.1],
            }
        )
        threshold, mean_cost = costs._best_threshold(validation, missed_default=2.0, review=0.25)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertAlmostEqual(
            mean_cost,
            utils.cost_value(
                validation["y_true"].to_numpy(),
                validation["probability"].to_numpy(),
                threshold,
                missed_default=2.0,
                review=0.25,
            ),
        )

    def test_validation_threshold_includes_no_review_action(self) -> None:
        validation = pd.DataFrame(
            {"y_true": [0, 0, 0], "probability": [0.1, 0.2, 0.3]}
        )
        threshold, mean_cost = costs._best_threshold(
            validation, missed_default=1.0, review=0.25
        )
        self.assertGreater(threshold, validation["probability"].max())
        self.assertEqual(mean_cost, 0.0)

    def test_exact_score_boundary_uses_greater_than_or_equal_decision(self) -> None:
        validation = pd.DataFrame(
            {"y_true": [1, 0], "probability": [0.5, 0.1]}
        )
        threshold, mean_cost = costs._best_threshold(
            validation, missed_default=1.0, review=0.1
        )
        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(mean_cost, 0.05)

    def test_validation_threshold_rejects_invalid_probabilities(self) -> None:
        validation = pd.DataFrame({"y_true": [0], "probability": [1.2]})
        with self.assertRaisesRegex(ValueError, "finite and in"):
            costs._best_threshold(validation, missed_default=1.0, review=0.25)

    def test_threshold_tie_break_does_not_trade_away_small_real_cost_difference(self) -> None:
        validation = pd.DataFrame({"y_true": [1], "probability": [0.5]})
        threshold, mean_cost = costs._best_threshold(
            validation, missed_default=1e-10, review=0.0
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(mean_cost, 0.0)


class PilotBootstrapTests(unittest.TestCase):
    def test_bootstrap_delta_is_paired_and_reports_observed_estimate(self) -> None:
        y = np.tile(np.array([0, 1, 0, 1, 0], dtype="int8"), 3)
        reference = np.tile(np.array([0.1, 0.6, 0.2, 0.7, 0.3]), 3)
        candidate = np.tile(np.array([0.2, 0.7, 0.3, 0.8, 0.1]), 3)
        vintage = np.repeat([2017, 2018, 2019], 5)
        estimate, lower, upper = robustness._paired_bootstrap_delta(
            y,
            candidate,
            reference,
            "roc_auc",
            replicates=30,
            seed=12,
            vintage=vintage,
        )
        expected = roc_auc_score(y, candidate) - roc_auc_score(y, reference)
        self.assertAlmostEqual(estimate, expected)
        self.assertLessEqual(lower, upper)

    def test_pr_auc_bootstrap_matches_average_precision_with_tied_scores(self) -> None:
        y = np.tile(np.array([0, 1, 0, 1, 0, 1], dtype="int8"), 4)
        candidate = np.tile(np.array([0.1, 0.2, 0.2, 0.8, 0.8, 0.9]), 4)
        reference = np.full(len(y), 0.5)
        estimate, lower, upper = robustness._paired_bootstrap_delta(
            y,
            candidate,
            reference,
            "pr_auc",
            replicates=30,
            seed=24,
        )
        expected = average_precision_score(y, candidate) - average_precision_score(y, reference)
        self.assertAlmostEqual(estimate, expected)
        self.assertTrue(np.isfinite([estimate, lower, upper]).all())
        self.assertLessEqual(lower, upper)


class PilotSurvivalPredictionTests(unittest.TestCase):
    def test_baseline_hazard_preserves_monthly_values(self) -> None:
        model = SimpleNamespace(
            baseline_cumulative_hazard_=pd.DataFrame(
                {"baseline": [0.1, 0.3, 0.4]}, index=[1.0, 2.0, 3.0]
            )
        )
        actual = survival._baseline_cumulative_hazard(model, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(actual, [0.1, 0.3, 0.4])

    def test_baseline_hazard_rejects_nonmonotone_values(self) -> None:
        model = SimpleNamespace(
            baseline_cumulative_hazard_=pd.DataFrame(
                {"baseline": [0.3, 0.2]}, index=[1.0, 2.0]
            )
        )
        with self.assertRaisesRegex(ValueError, "finite, nonnegative, and monotone"):
            survival._baseline_cumulative_hazard(model, np.array([1.0, 2.0]))

    def test_cumulative_incidence_requires_both_causes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected cause-specific Cox models"):
            survival.predict_default_cumulative_incidence(
                pd.DataFrame(), object(), {}, max_month=12
            )


if __name__ == "__main__":
    unittest.main()
