from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import pilot_model_utils as utils
from research_extensions import (
    MACRO_FEATURES,
    aggregate_shap_by_family,
    choose_explanation_clusters,
    horizon_importance_table,
    merge_macro_asof,
    prepare_macro_series,
    require_existing_inputs,
    transformed_feature_families,
)


ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MacroAlignmentTests(unittest.TestCase):
    @staticmethod
    def raw_series():
        return {
            "UNRATE": pd.DataFrame(
                {
                    "observation_date": [
                        "2020-01-01",
                        "2020-02-01",
                        "2020-03-01",
                        "2020-04-01",
                        "2020-05-01",
                    ],
                    "UNRATE": [3.5, 3.6, 3.7, 3.8, 3.9],
                }
            ),
            "GDPC1": pd.DataFrame(
                {"observation_date": ["2020-01-01", "2020-04-01"], "GDPC1": [100.0, 101.0]}
            ),
            "USSTHPI": pd.DataFrame(
                {
                    "observation_date": ["2020-01-01", "2020-04-01"],
                    "USSTHPI": [200.0, 201.0],
                }
            ),
        }

    def test_prepare_and_asof_merge_never_uses_unavailable_future_observations(self):
        macro = prepare_macro_series(self.raw_series())
        loans = pd.DataFrame(
            {
                "vintage": [2020, 2020],
                "loan_identifier": ["available", "too_early"],
                "prediction_date_proxy": pd.to_datetime(["2020-07-01", "2020-01-15"]),
            }
        )
        merged = merge_macro_asof(loans, macro)
        available = merged.set_index("loan_identifier").loc["available"]
        self.assertTrue(all(pd.notna(available[feature]) for feature in MACRO_FEATURES))
        for feature in MACRO_FEATURES:
            self.assertLessEqual(
                available[f"{feature}_available_date"], available["prediction_date_proxy"]
            )
        early = merged.set_index("loan_identifier").loc["too_early"]
        self.assertTrue(all(pd.isna(early[feature]) for feature in MACRO_FEATURES))

    def test_missing_series_duplicate_loan_keys_and_many_to_many_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing required FRED series"):
            prepare_macro_series({"UNRATE": self.raw_series()["UNRATE"]})
        macro = prepare_macro_series(self.raw_series())
        duplicated = pd.DataFrame(
            {
                "vintage": [2020, 2020],
                "loan_identifier": ["same", "same"],
                "prediction_date_proxy": pd.to_datetime(["2020-07-01", "2020-07-01"]),
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate vintage-loan"):
            merge_macro_asof(duplicated, macro)
        duplicated_macro = pd.concat([macro, macro.iloc[[0]]], ignore_index=True)
        one_loan = duplicated.iloc[:1].copy()
        one_loan["loan_identifier"] = "one"
        with self.assertRaisesRegex(ValueError, "duplicate feature-availability"):
            merge_macro_asof(one_loan, duplicated_macro)

    def test_macro_merge_preserves_input_rows_and_order(self):
        macro = prepare_macro_series(self.raw_series())
        loans = pd.DataFrame(
            {
                "vintage": [2020, 2020],
                "loan_identifier": ["later", "earlier"],
                "prediction_date_proxy": pd.to_datetime(["2020-07-01", "2020-06-01"]),
            }
        )
        merged = merge_macro_asof(loans, macro)
        self.assertEqual(merged["loan_identifier"].tolist(), ["later", "earlier"])
        self.assertEqual(len(merged), len(loans))


class ExplainabilityUtilityTests(unittest.TestCase):
    def test_transformed_feature_mapping_and_family_aggregation(self):
        features = ["original_ltv", "occupancy_status"]
        preprocessor = utils.make_preprocessor(
            scale_numeric=False, feature_columns=features
        )
        sample = pd.DataFrame(
            {
                "original_ltv": [65.0, 80.0, 95.0],
                "occupancy_status": ["P", "I", None],
            }
        )
        preprocessor.fit(sample)
        transformed_names, families = transformed_feature_families(
            preprocessor, features
        )
        values = np.ones((2, len(transformed_names)), dtype=float)
        aggregated = aggregate_shap_by_family(values, families, features)
        self.assertEqual(aggregated.shape, (2, 2))
        self.assertTrue(np.allclose(aggregated.sum(axis=1), values.sum(axis=1)))
        self.assertEqual(set(families), set(features))

    def test_shap_shape_and_feature_integrity_are_validated(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            aggregate_shap_by_family(np.zeros((3, 2)), ["original_ltv"], ["original_ltv"])
        with self.assertRaisesRegex(ValueError, "whitelist"):
            aggregate_shap_by_family(np.zeros((2, 1)), ["unexpected"], ["original_ltv"])

    def test_cluster_selection_is_reproducible_and_rejects_invalid_candidates(self):
        values = np.vstack(
            [
                np.random.default_rng(1).normal(-3, 0.1, size=(20, 2)),
                np.random.default_rng(2).normal(3, 0.1, size=(20, 2)),
            ]
        )
        first_diagnostics, first_model, first_stability = choose_explanation_clusters(
            values, 42, candidates=[2, 3]
        )
        second_diagnostics, second_model, second_stability = choose_explanation_clusters(
            values, 42, candidates=[2, 3]
        )
        self.assertEqual(first_model.n_clusters, second_model.n_clusters)
        self.assertTrue(
            np.array_equal(first_model.labels_, second_model.labels_)
        )
        self.assertTrue(np.allclose(first_diagnostics["silhouette_score"], second_diagnostics["silhouette_score"]))
        self.assertEqual(first_stability, second_stability)
        with self.assertRaisesRegex(ValueError, "Invalid candidate"):
            choose_explanation_clusters(values, 42, candidates=[1])

    def test_cross_horizon_ranks_and_spearman_are_computed(self):
        frames = []
        for horizon, scores in ((12, [3.0, 1.0]), (24, [2.0, 1.0]), (36, [1.0, 3.0])):
            frames.append(
                pd.DataFrame(
                    {
                        "horizon_months": horizon,
                        "feature": ["a", "b"],
                        "mean_abs_shap": scores,
                        "mean_signed_shap": [score / 2 for score in scores],
                    }
                )
            )
        rankings, consistency = horizon_importance_table(frames)
        self.assertEqual(len(rankings), 2)
        self.assertEqual(len(consistency), 3)
        self.assertIn("rank_change_36m_vs_12m", rankings.columns)
        self.assertTrue(consistency["spearman_rank_correlation"].between(-1, 1).all())

    def test_missing_required_input_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "Required extension inputs"):
            require_existing_inputs([ROOT / "does_not_exist_macro_fixture.csv"])


class IsolatedMacroBranchTests(unittest.TestCase):
    def test_macro_branch_uses_dedicated_model_and_result_directories(self):
        macro_script = import_script(
            "macro_experiment_test", "18_macro_augmented_comparison.py"
        )
        explanation_script = import_script(
            "explainability_paths_test", "16_explainability_and_profiles.py"
        )
        self.assertEqual(macro_script.MODEL_DIR.name, "macro_experiment")
        self.assertEqual(macro_script.OUTPUT_DIR.name, "macro_experiment")
        self.assertNotEqual(macro_script.MODEL_DIR, utils.MODEL_DIR)
        self.assertNotEqual(macro_script.OUTPUT_DIR, utils.TABLE_DIR)
        self.assertEqual(explanation_script.OUTPUT_DIR.name, "explainability")
        self.assertEqual(macro_script.OBSERVATIONS_PATH.name, "macro_observations_asof_snapshot.csv")

    def test_weighted_bootstrap_metrics_match_sklearn_on_tied_scores(self):
        macro_script = import_script(
            "weighted_metrics_test", "18_macro_augmented_comparison.py"
        )
        y = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
        probability = np.asarray([0.2, 0.2, 0.5, 0.5, 0.9, 0.9])
        unique_scores, group_inverse = np.unique(probability, return_inverse=True)
        result = macro_script._weighted_binary_metrics(
            y,
            probability,
            group_inverse,
            len(unique_scores),
            np.ones(len(y), dtype=np.int32),
        )
        self.assertAlmostEqual(result["roc_auc"], roc_auc_score(y, probability))
        self.assertAlmostEqual(result["pr_auc"], average_precision_score(y, probability))
        self.assertAlmostEqual(result["brier_score"], brier_score_loss(y, probability))
        weights = np.asarray([2, 0, 1, 3, 1, 2], dtype=np.int32)
        weighted = macro_script._weighted_binary_metrics(
            y, probability, group_inverse, len(unique_scores), weights
        )
        self.assertAlmostEqual(
            weighted["roc_auc"], roc_auc_score(y, probability, sample_weight=weights)
        )
        self.assertAlmostEqual(
            weighted["pr_auc"],
            average_precision_score(y, probability, sample_weight=weights),
        )
        self.assertAlmostEqual(
            weighted["brier_score"],
            brier_score_loss(y, probability, sample_weight=weights),
        )

    def test_macro_cohort_comparison_rejects_different_test_membership(self):
        macro_script = import_script(
            "macro_bootstrap_test", "18_macro_augmented_comparison.py"
        )
        predictions = pd.DataFrame(
            [
                {
                    "model": model,
                    "horizon_months": 12,
                    "vintage": 2017,
                    "loan_identifier": identifier,
                    "y_true": outcome,
                    "probability": probability,
                }
                for model, identifier, outcome, probability in (
                    ("baseline", "a", 0, 0.1),
                    ("baseline", "b", 1, 0.8),
                    ("macro_augmented", "a", 0, 0.2),
                    ("macro_augmented", "different", 1, 0.7),
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "exactly the same test loans"):
            macro_script.paired_bootstrap(predictions, 12, 2, 42)


if __name__ == "__main__":
    unittest.main()
