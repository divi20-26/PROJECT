from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


outcomes = load_script("outcome_pipeline", "04_create_outcomes.py")
features = load_script("feature_pipeline", "05_build_features.py")
labels = load_script("label_pipeline", "06_create_label_strategies.py")


class OutcomeDefinitionTests(unittest.TestCase):
    def test_documented_default_zero_balance_codes(self) -> None:
        self.assertEqual(outcomes.PRIMARY_DEFAULT_ZBC, ("02", "03", "09"))
        self.assertEqual(outcomes.PAYOFF_ZBC, ("01",))
        self.assertNotIn("15", outcomes.PRIMARY_DEFAULT_ZBC)
        self.assertNotIn("16", outcomes.PRIMARY_DEFAULT_ZBC)
        self.assertNotIn("96", outcomes.PRIMARY_DEFAULT_ZBC)

    def test_delinquency_and_reo_default_precede_later_payoff(self) -> None:
        for status in ("03", "3", "99", "RA"):
            with self.subTest(status=status):
                row = {
                    "default_event_month": 10,
                    "default_event_date": "202001",
                    "payoff_month": 12,
                    "payoff_date": "202003",
                    "other_termination_month": None,
                }
                self.assertEqual(outcomes.choose_observed_outcome(row)[0], "default")

    def test_earlier_payoff_is_competing_event_not_default(self) -> None:
        row = {
            "default_event_month": 12,
            "default_event_date": "202003",
            "payoff_month": 11,
            "payoff_date": "202002",
            "other_termination_month": None,
        }
        self.assertEqual(outcomes.choose_observed_outcome(row)[0], "voluntary_payoff")

    def test_same_month_tie_prioritizes_observed_credit_event(self) -> None:
        row = {
            "default_event_month": 12,
            "default_event_date": "202003",
            "payoff_month": 12,
            "payoff_date": "202003",
            "other_termination_month": None,
        }
        self.assertEqual(outcomes.choose_observed_outcome(row)[0], "default")

    def test_relative_month_boundary_across_year(self) -> None:
        self.assertEqual(outcomes.relative_month("202501", "202412"), 2)
        self.assertEqual(outcomes.relative_month("202412", "202412"), 1)
        self.assertEqual(outcomes.relative_month("202411", "202412"), 0)
        with self.assertRaisesRegex(ValueError, "Invalid YYYYMM"):
            outcomes.relative_month("202513", "202412")

    def test_finalized_event_keeps_documented_event_date(self) -> None:
        row = {
            "first_payment_date": "201501",
            "first_observed_period": "201412",
            "last_observed_period": "201512",
            "default_event_date": "201506",
            "payoff_date": None,
            "other_termination_date": None,
        }
        result = outcomes.finalize_outcome_row(row)
        self.assertEqual(result["outcome_type"], "default")
        self.assertEqual(result["outcome_date"], "201506")
        self.assertEqual(result["outcome_month"], 6)
        self.assertEqual(result["followup_months"], 6)

    def test_event_before_first_payment_is_recorded_at_month_zero(self) -> None:
        result = outcomes.finalize_outcome_row(
            {
                "first_payment_date": "201204",
                "first_observed_period": "201203",
                "last_observed_period": "201505",
                "default_event_date": "201203",
                "payoff_date": None,
                "other_termination_date": None,
            }
        )
        self.assertEqual(result["entry_month"], 1)
        self.assertEqual(result["outcome_type"], "default")
        self.assertEqual(result["outcome_month"], 0)

    def test_real_2005_histories_match_manual_event_review(self) -> None:
        outcomes_path = ROOT / "data" / "outcomes" / "outcomes_2005_asof_20260331.parquet"
        performance_path = ROOT / "data" / "extracted" / "vintage_2005" / "performance.parquet"
        if not outcomes_path.exists() or not performance_path.exists():
            self.skipTest("Run 04_create_outcomes.py for the 2005 sample before integration tests.")
        connection = duckdb.connect()
        try:
            observed = connection.execute(
                f"""
                SELECT loan_identifier, outcome_type, outcome_date
                FROM read_parquet('{outcomes_path}')
                WHERE loan_identifier IN (
                    'F05Q10000006', 'F05Q10004106', 'F05Q10000744',
                    'F05Q10016875', 'F05Q10080169'
                )
                """
            ).fetchall()
            by_loan = {loan_id: (outcome_type, outcome_date) for loan_id, outcome_type, outcome_date in observed}
            self.assertEqual(by_loan["F05Q10000006"], ("voluntary_payoff", "201305"))
            self.assertEqual(by_loan["F05Q10004106"], ("default", "201107"))
            self.assertEqual(by_loan["F05Q10000744"], ("default", "201008"))
            self.assertEqual(by_loan["F05Q10016875"], ("other_termination", "202204"))
            self.assertEqual(by_loan["F05Q10080169"], ("default", "201306"))

            source_rows = connection.execute(
                f"""
                SELECT "0", "1", "3", "8", "9"
                FROM read_parquet('{performance_path}')
                WHERE ("0" = 'F05Q10000006' AND "8" = '01')
                   OR ("0" = 'F05Q10004106' AND "1" = '201107')
                   OR ("0" = 'F05Q10000744' AND "1" = '201008')
                   OR ("0" = 'F05Q10016875' AND "8" = '16')
                   OR ("0" = 'F05Q10080169' AND "1" = '201306')
                """
            ).fetchall()
            self.assertEqual(len(source_rows), 5)

            early_default = connection.execute(
                """
                SELECT outcome_type, outcome_month, entry_month
                FROM read_parquet(?)
                WHERE loan_identifier = 'F06Q10088508'
                """,
                [str(ROOT / "data" / "outcomes" / "outcomes_2006_asof_20141231.parquet")],
            ).fetchone()
            self.assertEqual(early_default, ("default", 0.0, 1))
            early_history = connection.execute(
                """
                SELECT "1", "3"
                FROM read_parquet(?)
                WHERE "0" = 'F06Q10088508'
                ORDER BY "1"
                LIMIT 1
                """,
                [str(ROOT / "data" / "extracted" / "vintage_2006" / "performance.parquet")],
            ).fetchone()
            self.assertEqual(early_history, ("201203", "07"))
        finally:
            connection.close()


class LabelStrategyTests(unittest.TestCase):
    def test_complete_horizon_default_at_boundary_is_positive(self) -> None:
        eligible, value = labels.complete_horizon_label(
            {"entry_month": 1, "outcome_type": "default", "outcome_month": 36}, 36
        )
        self.assertTrue(eligible)
        self.assertEqual(value, 1)

    def test_event_after_horizon_is_mature_negative(self) -> None:
        eligible, value = labels.complete_horizon_label(
            {
                "entry_month": 1,
                "outcome_type": "default",
                "outcome_month": 37,
                "last_observed_month": 37,
            },
            36,
        )
        self.assertTrue(eligible)
        self.assertEqual(value, 0)

    def test_payoff_before_horizon_establishes_no_later_default(self) -> None:
        eligible, value = labels.complete_horizon_label(
            {"entry_month": 1, "outcome_type": "voluntary_payoff", "outcome_month": 8}, 36
        )
        self.assertTrue(eligible)
        self.assertEqual(value, 0)

    def test_other_termination_before_horizon_is_unresolved(self) -> None:
        eligible, value = labels.complete_horizon_label(
            {"entry_month": 1, "outcome_type": "other_termination", "outcome_month": 8}, 36
        )
        self.assertFalse(eligible)
        self.assertIsNone(value)

    def test_default_before_risk_entry_is_excluded(self) -> None:
        row = {
            "entry_month": 1,
            "outcome_type": "default",
            "outcome_month": 0,
            "last_observed_month": 38,
        }
        eligible, value = labels.complete_horizon_label(row, 36)
        self.assertFalse(eligible)
        self.assertIsNone(value)
        survival = labels.censor_aware_input(
            {
                **row,
                "followup_months": 0,
                "survival_event_code": 1,
                "loan_identifier": "F06Q10088508",
            },
            36,
        )
        self.assertFalse(survival["survival_eligible"])
        self.assertEqual(survival["survival_exclusion_reason"], "default_before_risk_entry")
        self.assertEqual(labels.mature_exclusion_reason(row, 36), "default_before_entry")

    def test_mature_exclusion_partition_prioritizes_delayed_entry(self) -> None:
        self.assertEqual(
            labels.mature_exclusion_reason(
                {
                    "entry_month": 4,
                    "outcome_type": "other_termination",
                    "outcome_month": 8,
                    "last_observed_month": 8,
                },
                12,
            ),
            "delayed_entry",
        )
        self.assertEqual(
            labels.mature_exclusion_reason(
                {
                    "entry_month": 1,
                    "outcome_type": "right_censored",
                    "outcome_month": None,
                    "last_observed_month": 11,
                },
                12,
            ),
            "right_censored_before_horizon",
        )

    def test_label_sensitivity_reports_immature_denominators_and_common_population(self) -> None:
        rows = []
        for vintage, eligible, naive, label, eval_eligible, eval_label, false_negative in (
            (2005, True, 0, 0, True, 0, False),
            (2005, False, 0, None, True, 1, True),
            (2005, False, 1, None, False, None, False),
        ):
            row = {
                "vintage": vintage,
                "training_cutoff": "2014-12-31",
                "complete_horizon_eligible_12m": eligible,
                "naive_label_12m": naive,
                "complete_horizon_label_12m": label,
                "evaluation_mature_eligible_12m": eval_eligible,
                "evaluation_mature_label_12m": eval_label,
                "naive_false_negative_12m": false_negative,
            }
            for horizon in (24, 36):
                row[f"complete_horizon_eligible_{horizon}m"] = eligible
                row[f"naive_label_{horizon}m"] = naive
                row[f"complete_horizon_label_{horizon}m"] = label
                row[f"evaluation_mature_eligible_{horizon}m"] = eval_eligible
                row[f"evaluation_mature_label_{horizon}m"] = eval_label
                row[f"naive_false_negative_{horizon}m"] = false_negative
            rows.append(row)

        summary = labels.summarize_label_sensitivity(pd.DataFrame(rows))
        overall = summary.loc[
            (summary["vintage"] == "ALL") & (summary["horizon_months"] == 12)
        ].iloc[0]
        self.assertEqual(overall["training_cohort_count"], 3)
        self.assertEqual(overall["training_immature_or_ineligible_count"], 2)
        self.assertEqual(overall["immature_group_mature_at_evaluation_count"], 1)
        self.assertEqual(overall["retrospective_default_count_within_immature_group"], 1)
        self.assertEqual(overall["retrospective_default_rate_of_all_cutoff_immature"], 0.5)
        self.assertEqual(
            overall["retrospective_default_rate_among_immature_matured_at_evaluation"],
            1.0,
        )
        self.assertEqual(overall["common_population_training_count"], 1)
        self.assertEqual(
            overall["naive_vs_complete_disagreement_on_common_population"], 0
        )
        self.assertTrue(overall["same_population_labels_identical"])

    def test_incomplete_followup_excluded_from_complete_horizon(self) -> None:
        eligible, value = labels.complete_horizon_label(
            {"entry_month": 1, "outcome_type": "right_censored", "last_observed_month": 20}, 36
        )
        self.assertFalse(eligible)
        self.assertIsNone(value)

    def test_delayed_entry_is_excluded_from_binary_complete_horizon(self) -> None:
        eligible, value = labels.complete_horizon_label(
            {
                "entry_month": 5,
                "outcome_type": "default",
                "outcome_month": 20,
                "last_observed_month": 30,
            },
            36,
        )
        self.assertFalse(eligible)
        self.assertIsNone(value)

    def test_payoff_before_risk_entry_has_no_survival_exposure(self) -> None:
        result = labels.censor_aware_input(
            {
                "loan_identifier": "F05Q10028988",
                "entry_month": 1,
                "followup_months": 0,
                "survival_event_code": 2,
                "outcome_type": "voluntary_payoff",
            },
            36,
        )
        self.assertFalse(result["survival_eligible"])
        self.assertEqual(result["survival_exclusion_reason"], "terminal_event_before_risk_entry")

    def test_censor_aware_input_preserves_competing_risk_and_entry(self) -> None:
        result = labels.censor_aware_input(
            {
                "loan_identifier": "F05Q10000006",
                "entry_month": 6,
                "followup_months": 15,
                "survival_event_code": 2,
            },
            36,
        )
        self.assertTrue(result["survival_eligible"])
        self.assertEqual(result["survival_entry_month"], 6)
        self.assertEqual(result["survival_exit_month"], 15)
        self.assertEqual(result["survival_outcome_by_horizon"], 2)

    def test_cutoff_label_does_not_use_retrospective_event_for_training(self) -> None:
        training = pd.DataFrame(
            [{
                "vintage": 2013,
                "loan_identifier": "F13Q10000001",
                "first_payment_date": "201301",
                "outcome_type": "right_censored",
                "outcome_month": None,
                "entry_month": 1,
                "last_observed_month": 24,
                "followup_months": 24,
                "survival_event_code": 0,
            }]
        )
        evaluation = pd.DataFrame(
            [{
                "vintage": 2013,
                "loan_identifier": "F13Q10000001",
                "first_payment_date": "201301",
                "outcome_type": "default",
                "outcome_month": 30,
                "entry_month": 1,
                "last_observed_month": 30,
                "followup_months": 30,
                "survival_event_code": 1,
            }]
        )
        result = labels.generate_strategy_rows(training, evaluation, "2014-12-31").iloc[0]
        self.assertEqual(result["naive_label_36m"], 0)
        self.assertTrue(result["naive_false_negative_36m"])
        self.assertEqual(result["evaluation_mature_label_36m"], 1)


class FeatureLeakageTests(unittest.TestCase):
    def test_feature_builder_applies_field_specific_missing_codes(self) -> None:
        raw = [""] * 31
        raw[0] = "9999"
        raw[1] = "201501"
        raw[5] = "0"
        raw[6] = "99"
        raw[19] = "F15Q10000001"
        raw[20] = "P"
        raw[30] = "9999"
        result = features.build_feature_frame(pd.DataFrame([raw]))
        self.assertTrue(pd.isna(result.loc[0, "classic_fico"]))
        self.assertTrue(pd.isna(result.loc[0, "vantagescore_4"]))
        self.assertTrue(pd.isna(result.loc[0, "number_of_units"]))
        self.assertEqual(result.loc[0, "mortgage_insurance_percentage"], 0)
        self.assertNotIn("loan_identifier", [item[0] for item in features.FEATURE_SPEC])
        self.assertNotIn("current_actual_upb", [item[0] for item in features.FEATURE_SPEC])


if __name__ == "__main__":
    unittest.main()
