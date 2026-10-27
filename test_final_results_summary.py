from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "final_results_summary", ROOT / "19_build_final_results_summary.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load final results summary builder.")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class FinalResultsSummaryTests(unittest.TestCase):
    def test_summary_metrics_match_source_rows(self):
        final = summary.build_consolidated_results(ROOT)
        pilot = pd.read_csv(ROOT / "results" / "tables" / "pilot_metrics.csv")
        pilot = pilot.loc[pilot["split"].eq("test")]
        for _, source in pilot.iterrows():
            for metric in (
                "n",
                "defaults",
                "default_rate",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "expected_calibration_error",
                "calibration_intercept",
                "calibration_slope",
            ):
                row = final.loc[
                    final["analysis"].eq("binary_model_test_performance")
                    & final["horizon_months"].eq(source["horizon_months"])
                    & final["model"].eq(source["model"])
                    & final["strategy"].eq(source["strategy"])
                    & final["metric"].eq(metric)
                ]
                self.assertEqual(len(row), 1)
                self.assertAlmostEqual(float(row.iloc[0]["value"]), float(source[metric]))
                self.assertEqual(
                    row.iloc[0]["source_file"], summary.SOURCE_FILES["pilot_metrics"]
                )

        macro = pd.read_csv(ROOT / "results" / "macro_experiment" / "baseline_vs_macro_metrics.csv")
        for _, source in macro.loc[macro["split"].eq("test")].iterrows():
            for metric in (
                "n",
                "defaults",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "expected_calibration_error",
                "calibration_intercept",
                "calibration_slope",
            ):
                row = final.loc[
                    final["analysis"].eq("macro_model_test_performance")
                    & final["horizon_months"].eq(source["horizon_months"])
                    & final["model"].eq(source["model"])
                    & final["metric"].eq(metric)
                ].iloc[0]
                self.assertAlmostEqual(float(row["value"]), float(source[metric]))

    def test_rank_cluster_and_macro_values_match_source_files(self):
        final = summary.build_consolidated_results(ROOT)
        correlations = pd.read_csv(
            ROOT / "results" / "explainability" / "horizon_rank_consistency.csv"
        )
        for _, source in correlations.iterrows():
            strategy = f"{int(source['horizon_a_months'])}m_vs_{int(source['horizon_b_months'])}m"
            row = final.loc[
                final["analysis"].eq("cross_horizon_rank_consistency")
                & final["strategy"].eq(strategy)
                & final["metric"].eq("spearman_rank_correlation")
            ].iloc[0]
            self.assertAlmostEqual(float(row["value"]), float(source["spearman_rank_correlation"]))

        diagnostics = pd.read_csv(
            ROOT / "results" / "explainability" / "cluster_quality_diagnostics.csv"
        )
        for _, source in diagnostics.iterrows():
            row = final.loc[
                final["analysis"].eq("shap_cluster_diagnostic")
                & final["horizon_months"].eq(source["horizon_months"])
                & final["strategy"].eq(f"kmeans_k{int(source['n_clusters'])}")
            ].iloc[0]
            self.assertAlmostEqual(float(row["value"]), float(source["silhouette_score"]))

        differences = pd.read_csv(
            ROOT / "results" / "macro_experiment" / "paired_performance_differences.csv"
        )
        for _, source in differences.iterrows():
            row = final.loc[
                final["analysis"].eq("macro_paired_difference")
                & final["horizon_months"].eq(source["horizon_months"])
                & final["metric"].eq(source["metric"])
            ].iloc[0]
            self.assertAlmostEqual(
                float(row["value"]), float(source["difference_macro_minus_baseline"])
            )
            self.assertAlmostEqual(float(row["ci_lower"]), float(source["ci_95_lower"]))
            self.assertAlmostEqual(float(row["ci_upper"]), float(source["ci_95_upper"]))
        self.assertTrue(final["source_file"].notna().all())

    def test_generator_creates_three_separate_summary_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = summary.generate(Path(temporary), ROOT)
            self.assertEqual(set(outputs), {"consolidated", "markdown", "ppt"})
            for path in outputs.values():
                self.assertTrue(path.exists())
            consolidated = pd.read_csv(outputs["consolidated"])
            ppt = pd.read_csv(outputs["ppt"])
            markdown = outputs["markdown"].read_text(encoding="utf-8")
            self.assertTrue({"analysis", "value", "source_file"}.issubset(consolidated.columns))
            self.assertEqual(len(ppt), 3)
            self.assertIn("FRED", markdown)
            self.assertIn("Spearman rank correlation", markdown)
            self.assertIn("SHAP explanation clusters", markdown)
            self.assertIn("macro_paired_difference", set(consolidated["analysis"]))
            self.assertFalse(np.isinf(pd.to_numeric(consolidated["value"], errors="coerce")).any())
            self.assertTrue(ppt["macro_delta_brier_score_ci95"].astype(str).str.len().gt(0).all())
            self.assertEqual(
                ppt.loc[ppt["horizon_months"].eq(36), "cause_specific_concordance_36m"].iloc[0],
                "0.6964",
            )


if __name__ == "__main__":
    unittest.main()
