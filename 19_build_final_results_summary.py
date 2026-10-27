"""Build a source-traceable summary of existing baseline and extension results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "FINAL"

SOURCE_FILES = {
    "label_strategy_comparison": "results/tables/pilot_label_strategy_comparison.csv",
    "training_cohorts": "results/tables/binary_training_cohort_summary.csv",
    "pilot_metrics": "results/tables/pilot_metrics.csv",
    "survival_ipcw": "results/survival_censor_aware/ipcw_time_dependent_metrics.csv",
    "survival_concordance": "results/survival_censor_aware/cause_specific_concordance.csv",
    "shap_importance": "results/explainability/global_shap_feature_importance.csv",
    "horizon_consistency": "results/explainability/horizon_rank_consistency.csv",
    "cluster_diagnostics": "results/explainability/cluster_quality_diagnostics.csv",
    "cluster_profiles": "results/explainability/cluster_risk_profile_summary.csv",
    "macro_metrics": "results/macro_experiment/baseline_vs_macro_metrics.csv",
    "macro_differences": "results/macro_experiment/paired_performance_differences.csv",
    "macro_manifest": "data/macro/macro_data_manifest.json",
    "macro_metadata": "results/macro_experiment/macro_experiment_metadata.json",
}

SUMMARY_COLUMNS = [
    "analysis",
    "cohort",
    "horizon_months",
    "model",
    "strategy",
    "metric",
    "value",
    "ci_lower",
    "ci_upper",
    "n",
    "defaults",
    "source_file",
    "interpretation",
]


def _read_sources(root: Path) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    missing = [
        path for path in SOURCE_FILES.values() if not (root / Path(path.replace("/", "\\"))).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing final-summary source files: {missing}")
    for key, relative_path in SOURCE_FILES.items():
        if relative_path.endswith(".csv"):
            sources[key] = pd.read_csv(root / Path(relative_path.replace("/", "\\")))
    return sources


def _record(
    *,
    analysis: str,
    cohort: str,
    source: str,
    metric: str,
    value,
    horizon=None,
    model=None,
    strategy=None,
    ci_lower=None,
    ci_upper=None,
    n=None,
    defaults=None,
    interpretation="",
) -> dict:
    return {
        "analysis": analysis,
        "cohort": cohort,
        "horizon_months": horizon,
        "model": model,
        "strategy": strategy,
        "metric": metric,
        "value": value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "defaults": defaults,
        "source_file": source,
        "interpretation": interpretation,
    }


def build_consolidated_results(root: Path = ROOT) -> pd.DataFrame:
    """Read each source table and emit a normalized, source-traceable results CSV."""
    data = _read_sources(root)
    rows = []

    labels = data["label_strategy_comparison"].set_index("horizon_months")
    cohorts = data["training_cohorts"]
    for _, cohort in cohorts.iterrows():
        horizon = int(cohort["horizon_months"])
        strategy = str(cohort["strategy"])
        denominator = int(cohort["training_count"])
        defaults = int(cohort["defaults"])
        rows.append(
            _record(
                analysis="training_label_strategy",
                cohort="2005-2014 training",
                horizon=horizon,
                model="not applicable",
                strategy=strategy,
                metric="training_loans",
                value=denominator,
                n=denominator,
                defaults=defaults,
                source=SOURCE_FILES["training_cohorts"],
                interpretation="Cutoff-qualified pilot training cohort.",
            )
        )
        rows.append(
            _record(
                analysis="training_label_strategy",
                cohort="2005-2014 training",
                horizon=horizon,
                model="not applicable",
                strategy=strategy,
                metric="training_default_prevalence",
                value=defaults / denominator if denominator else np.nan,
                n=denominator,
                defaults=defaults,
                source=SOURCE_FILES["training_cohorts"],
                interpretation="Default count divided by this strategy's training rows.",
            )
        )
    for horizon, label in labels.iterrows():
        for field, metric in (
            ("retrospective_naive_false_negatives", "retrospective_naive_false_negatives"),
            ("common_population_label_disagreements", "common_population_label_disagreements"),
        ):
            rows.append(
                _record(
                    analysis="label_sensitivity",
                    cohort="2005-2014 training labels",
                    horizon=int(horizon),
                    model="not applicable",
                    strategy="naive_vs_complete",
                    metric=metric,
                    value=int(label[field]),
                    source=SOURCE_FILES["label_strategy_comparison"],
                    interpretation="As reported by the frozen label-strategy comparison.",
                )
            )

    pilot = data["pilot_metrics"]
    pilot = pilot.loc[pilot["split"].eq("test")]
    for _, item in pilot.iterrows():
        cohort = "2017-2019 common mature test (exploratory)"
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
            value = item[metric]
            rows.append(
                _record(
                    analysis="binary_model_test_performance",
                    cohort=cohort,
                    horizon=int(item["horizon_months"]),
                    model=str(item["model"]),
                    strategy=str(item["strategy"]),
                    metric=metric,
                    value=value,
                    n=int(item["n"]),
                    defaults=int(item["defaults"]),
                    source=SOURCE_FILES["pilot_metrics"],
                    interpretation=(
                        "Common horizon-mature cohort; test period was previously inspected."
                        if item["model"] != "cause_specific_cox"
                        else "Complete-outcome survival cumulative incidence scoring, not IPCW."
                    ),
                )
            )

    survival_metrics = data["survival_ipcw"]
    survival_metric_names = {
        "roc_auc_ipcw": "ipcw_roc_auc",
        "pr_auc_ipcw": "ipcw_pr_auc",
        "brier_score_ipcw": "ipcw_brier_score",
        "n_observed_for_horizon": "observed_for_horizon",
        "defaults": "defaults",
        "observed_censored_before_horizon": "censored_before_horizon",
    }
    for _, item in survival_metrics.iterrows():
        for source_metric, metric in survival_metric_names.items():
            value = item[source_metric]
            rows.append(
                _record(
                    analysis="survival_censor_aware",
                    cohort=str(item["evaluation_period"]),
                    horizon=int(item["horizon_months"]),
                    model="cause_specific_cox",
                    strategy="marginal_IPCW_cumulative_dynamic",
                    metric=metric,
                    value=value,
                    n=int(item["n_observed_for_horizon"]),
                    defaults=int(item["defaults"]),
                    source=SOURCE_FILES["survival_ipcw"],
                    interpretation=(
                        "Exploratory; marginal independent-censoring assumption; "
                        "voluntary payoff is competing non-default."
                    ),
                )
            )
    for _, item in data["survival_concordance"].iterrows():
        rows.append(
            _record(
                analysis="survival_cause_specific_concordance",
                cohort=str(item["evaluation_period"]),
                horizon=36,
                model="cause_specific_cox",
                strategy="Harrell_concordance_payoff_censored",
                metric="concordance_index",
                value=float(item["concordance_index"]),
                n=int(item["n"]),
                defaults=int(item["defaults"]),
                source=SOURCE_FILES["survival_concordance"],
                interpretation="Exploratory cause-specific measure; not competing-risk concordance.",
            )
        )

    shap = data["shap_importance"]
    for _, item in shap.iterrows():
        for metric in ("mean_abs_shap", "mean_signed_shap"):
            rows.append(
                _record(
                    analysis="shap_global_importance",
                    cohort="2017-2019 common mature sampled test",
                    horizon=int(item["horizon_months"]),
                    model="xgboost_complete_eligible",
                    strategy="original_feature_family",
                    metric=f"{metric}:{item['feature']}",
                    value=float(item[metric]),
                    n=int(item["sample_size"]),
                    source=SOURCE_FILES["shap_importance"],
                    interpretation=(
                        "Raw XGBoost margin (log-odds); model attribution, not causality."
                    ),
                )
            )
    for _, item in data["horizon_consistency"].iterrows():
        rows.append(
            _record(
                analysis="cross_horizon_rank_consistency",
                cohort="same 500-loan common mature test sample",
                model="xgboost_complete_eligible",
                strategy=f"{int(item['horizon_a_months'])}m_vs_{int(item['horizon_b_months'])}m",
                metric="spearman_rank_correlation",
                value=float(item["spearman_rank_correlation"]),
                n=int(item["feature_count"]),
                source=SOURCE_FILES["horizon_consistency"],
                interpretation="Spearman correlation of original-feature-family mean-absolute-SHAP ranks.",
            )
        )

    profiles = data["cluster_profiles"].loc[
        data["cluster_profiles"]["split"].eq("test")
    ]
    diagnostics = data["cluster_diagnostics"]
    for _, item in diagnostics.iterrows():
        horizon = int(item["horizon_months"])
        chosen_k = bool(item["selected"])
        test_profile = profiles.loc[profiles["horizon_months"].eq(horizon)]
        sizes = test_profile.set_index("cluster")["cluster_size"].to_dict()
        rows.append(
            _record(
                analysis="shap_cluster_diagnostic",
                cohort="2017-2019 common mature test; k selected on validation SHAP",
                horizon=horizon,
                model="xgboost_complete_eligible",
                strategy=f"kmeans_k{int(item['n_clusters'])}",
                metric="validation_silhouette_score",
                value=float(item["silhouette_score"]),
                n=int(item["validation_sample_size"]),
                source=SOURCE_FILES["cluster_diagnostics"],
                interpretation=(
                    f"selected={chosen_k}; test cluster sizes={sizes}; "
                    f"mean pairwise validation seed ARI="
                    f"{_format(item['stability_mean_pairwise_ari'], 3)}."
                ),
            )
        )
    for _, item in profiles.iterrows():
        for metric in (
            "cluster_size",
            "mean_predicted_default_probability",
            "median_predicted_default_probability",
            "observed_default_rate",
            "observed_defaults",
            "mean_classic_fico",
            "mean_original_ltv",
            "mean_original_cltv",
            "mean_original_dti",
            "mean_original_interest_rate",
        ):
            rows.append(
                _record(
                    analysis="shap_cluster_test_profile",
                    cohort="2017-2019 sampled exploratory test",
                    horizon=int(item["horizon_months"]),
                    model="xgboost_complete_eligible",
                    strategy=f"validation_fitted_cluster_{int(item['cluster'])}",
                    metric=metric,
                    value=float(item[metric]),
                    n=int(item["cluster_size"]),
                    defaults=int(item["observed_defaults"]),
                    source=SOURCE_FILES["cluster_profiles"],
                    interpretation="Exploratory cluster description; cluster IDs are horizon-specific.",
                )
            )

    macro = data["macro_metrics"]
    macro = macro.loc[macro["split"].eq("test")]
    macro_metric_names = (
        "n",
        "defaults",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
        "calibration_intercept",
        "calibration_slope",
    )
    for _, item in macro.iterrows():
        for metric in macro_metric_names:
            rows.append(
                _record(
                    analysis="macro_model_test_performance",
                    cohort="2017-2019 common macro-available mature test (exploratory)",
                    horizon=int(item["horizon_months"]),
                    model=str(item["model"]),
                    strategy="complete_eligible",
                    metric=metric,
                    value=item[metric],
                    n=int(item["n"]),
                    defaults=int(item["defaults"]),
                    source=SOURCE_FILES["macro_metrics"],
                    interpretation="Baseline and augmented models use identical loans by horizon.",
                )
            )
    differences = data["macro_differences"]
    for _, item in differences.iterrows():
        rows.append(
            _record(
                analysis="macro_paired_difference",
                cohort="2017-2019 common macro-available mature test (exploratory)",
                horizon=int(item["horizon_months"]),
                model="macro_augmented_minus_baseline",
                strategy="complete_eligible; paired vintage-stratified bootstrap",
                metric=str(item["metric"]),
                value=float(item["difference_macro_minus_baseline"]),
                ci_lower=float(item["ci_95_lower"]),
                ci_upper=float(item["ci_95_upper"]),
                n=int(
                    data["macro_metrics"].loc[
                        data["macro_metrics"]["split"].eq("test")
                        & data["macro_metrics"]["horizon_months"].eq(item["horizon_months"])
                        & data["macro_metrics"]["model"].eq("baseline"),
                        "n",
                    ].iloc[0]
                ),
                defaults=int(
                    data["macro_metrics"].loc[
                        data["macro_metrics"]["split"].eq("test")
                        & data["macro_metrics"]["horizon_months"].eq(item["horizon_months"])
                        & data["macro_metrics"]["model"].eq("baseline"),
                        "defaults",
                    ].iloc[0]
                ),
                source=(
                    f"{SOURCE_FILES['macro_differences']}; "
                    f"{SOURCE_FILES['macro_metrics']}"
                ),
                interpretation=(
                    f"{int(item['valid_bootstrap_replicates'])} paired bootstrap replicates; "
                    "interval is descriptive, not a confirmatory p-value."
                ),
            )
        )

    result = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if result.empty or result["source_file"].isna().any():
        raise ValueError("Final summary is empty or contains rows without source provenance.")
    return result


def _format(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    value = float(value)
    if not np.isfinite(value):
        return "Unavailable"
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def build_markdown(root: Path, data: dict[str, pd.DataFrame]) -> str:
    label_rows = []
    cohorts = data["training_cohorts"]
    label_comparison = data["label_strategy_comparison"].set_index("horizon_months")
    for horizon in sorted(cohorts["horizon_months"].unique()):
        group = cohorts.loc[cohorts["horizon_months"].eq(horizon)].set_index("strategy")
        for strategy in ("naive_full", "complete_eligible", "matched_naive"):
            item = group.loc[strategy]
            n = int(item["training_count"])
            defaults = int(item["defaults"])
            label_rows.append(
                [
                    f"{int(horizon)}m",
                    strategy,
                    _format(n, 0),
                    _format(defaults, 0),
                    _format(defaults / n, 4),
                    _format(label_comparison.loc[horizon, "retrospective_naive_false_negatives"], 0)
                    if strategy == "naive_full"
                    else "—",
                ]
            )

    binary_rows = []
    metrics = data["pilot_metrics"]
    metrics = metrics.loc[metrics["split"].eq("test")]
    for horizon in (12, 24, 36):
        for model in ("logistic_regression", "xgboost"):
            for strategy in ("naive_full", "complete_eligible", "matched_naive"):
                item = metrics.loc[
                    metrics["horizon_months"].eq(horizon)
                    & metrics["model"].eq(model)
                    & metrics["strategy"].eq(strategy)
                ]
                if item.empty:
                    continue
                item = item.iloc[0]
                binary_rows.append(
                    [
                        f"{horizon}m",
                        "Logistic Regression" if model == "logistic_regression" else "XGBoost",
                        strategy,
                        _format(item["roc_auc"]),
                        _format(item["pr_auc"]),
                        _format(item["brier_score"]),
                        _format(item["expected_calibration_error"]),
                        _format(item["calibration_slope"]),
                    ]
                )

    survival_rows = []
    for _, item in data["survival_ipcw"].iterrows():
        survival_rows.append(
            [
                str(item["evaluation_period"]).replace("_exploratory", ""),
                f"{int(item['horizon_months'])}m",
                _format(item["n_observed_for_horizon"], 0),
                _format(item["defaults"], 0),
                _format(item["roc_auc_ipcw"]),
                _format(item["pr_auc_ipcw"]),
                _format(item["brier_score_ipcw"]),
            ]
        )
    concordance_rows = [
        [
            str(item["evaluation_period"]).replace("_exploratory", ""),
            _format(item["concordance_index"]),
            _format(item["n"], 0),
            _format(item["defaults"], 0),
            "Payoff treated as censored",
        ]
        for _, item in data["survival_concordance"].iterrows()
    ]

    shap_rows = []
    shap = data["shap_importance"]
    for horizon in (12, 24, 36):
        subset = shap.loc[shap["horizon_months"].eq(horizon)].nlargest(5, "mean_abs_shap")
        shap_rows.extend(
            [
                [
                    f"{horizon}m",
                    str(item["feature"]),
                    _format(item["mean_abs_shap"]),
                    _format(item["mean_signed_shap"]),
                    "raw log-odds margin",
                ]
                for _, item in subset.iterrows()
            ]
        )

    correlation_rows = [
        [
            f"{int(item['horizon_a_months'])}m vs {int(item['horizon_b_months'])}m",
            _format(item["spearman_rank_correlation"], 3),
            _format(item["feature_count"], 0),
        ]
        for _, item in data["horizon_consistency"].iterrows()
    ]
    cluster_rows = []
    diagnostics = data["cluster_diagnostics"]
    profiles = data["cluster_profiles"]
    profiles = profiles.loc[profiles["split"].eq("test")]
    for horizon in (12, 24, 36):
        diagnostic = diagnostics.loc[
            diagnostics["horizon_months"].eq(horizon)
            & diagnostics["selected"].astype(bool)
        ].iloc[0]
        profile = profiles.loc[profiles["horizon_months"].eq(horizon)].sort_values("cluster")
        sizes = ", ".join(
            f"C{int(item['cluster']) + 1}: {_format(item['cluster_size'], 0)} "
            f"(mean risk {_format(item['mean_predicted_default_probability'], 4)})"
            for _, item in profile.iterrows()
        )
        cluster_rows.append(
            [
                f"{horizon}m",
                _format(diagnostic["n_clusters"], 0),
                _format(diagnostic["silhouette_score"]),
                sizes,
                _format(diagnostic["stability_mean_pairwise_ari"], 3),
            ]
        )

    macro_rows = []
    macro = data["macro_metrics"]
    macro = macro.loc[macro["split"].eq("test")]
    differences = data["macro_differences"]
    for horizon in (12, 24, 36):
        base = macro.loc[macro["horizon_months"].eq(horizon) & macro["model"].eq("baseline")].iloc[0]
        aug = macro.loc[
            macro["horizon_months"].eq(horizon) & macro["model"].eq("macro_augmented")
        ].iloc[0]
        metric_differences = differences.loc[differences["horizon_months"].eq(horizon)].set_index("metric")
        for metric in ("roc_auc", "pr_auc", "brier_score"):
            item = metric_differences.loc[metric]
            macro_rows.append(
                [
                    f"{horizon}m",
                    metric,
                    _format(base[metric]),
                    _format(aug[metric]),
                    _format(item["difference_macro_minus_baseline"]),
                    f"[{_format(item['ci_95_lower'])}, {_format(item['ci_95_upper'])}]",
                ]
            )

    manifest = (root / Path(SOURCE_FILES["macro_manifest"].replace("/", "\\"))).read_text(
        encoding="utf-8"
    )
    metadata = (root / Path(SOURCE_FILES["macro_metadata"].replace("/", "\\"))).read_text(
        encoding="utf-8"
    )
    import json

    macro_manifest = json.loads(manifest)
    macro_metadata = json.loads(metadata)
    macro_timing = macro_manifest.get("loan_date_limitation", "Unavailable")
    macro_revisions = macro_manifest.get("revision_limitation", "Unavailable")
    macro_alignment = macro_metadata.get("date_alignment", "Unavailable")
    macro_sources = ", ".join(
        f"{series_id} ({definition['source']})"
        for series_id, definition in macro_manifest["sources"].items()
    )

    return "\n".join(
        [
            "# FINAL — Consolidated research results",
            "",
            "This summary is generated from existing result files. Every row in "
            "`consolidated_results.csv` includes its source path. `ppt_results.csv` "
            "is a compact presentation table. Existing results, scripts, models, and "
            "predictions were not changed.",
            "",
            "> **Interpretation:** 2017–2019 and 2020–2022 have previously been examined. "
            "All reported performance results from those vintages are exploratory, not "
            "untouched confirmatory evidence.",
            "",
            "## 1. Training label strategies",
            "",
            _markdown_table(
                ["Horizon", "Strategy", "Training loans", "Defaults", "Training default prevalence", "Naive false negatives"],
                label_rows,
            ),
            "",
            "Naive false negatives are reported from the retrospective label comparison; "
            "they are not the test-set model false negatives. The common mature test "
            "population has zero label disagreements by the comparison file.",
            "",
            "## 2. Binary models — common mature 2017–2019 test cohort",
            "",
            _markdown_table(
                ["Horizon", "Model", "Training-label strategy", "ROC-AUC", "PR-AUC", "Brier", "ECE", "Calibration slope"],
                binary_rows,
            ),
            "",
            "All model/strategy rows at a horizon share the same test cohort and labels. "
            "The target is default by horizon; voluntary payoff is a non-default competing "
            "outcome and early other termination is excluded. Calibration intercept and "
            "slope are in the consolidated CSV. Lower Brier/ECE is better; higher AUC is better.",
            "",
            "## 3. Survival analysis",
            "",
            "### Marginal-IPCW cumulative/dynamic metrics",
            "",
            _markdown_table(
                ["Exploratory cohort", "Horizon", "Observed N", "Defaults", "IPCW ROC-AUC", "IPCW PR-AUC", "IPCW Brier"],
                survival_rows,
            ),
            "",
            "These censor-aware metrics use a marginal independent-censoring assumption; "
            "voluntary payoff is treated as competing non-default. They are separate from "
            "the complete-outcome model comparison above.",
            "",
            "### Cause-specific concordance",
            "",
            _markdown_table(
                ["Exploratory cohort", "36m concordance", "N", "Defaults", "Payoff handling"],
                concordance_rows,
            ),
            "",
            "This cause-specific Harrell concordance treats payoff as censored and is not "
            "a competing-risk concordance statistic.",
            "",
            "## 4. SHAP explainability and cross-horizon stability",
            "",
            _markdown_table(
                ["Horizon", "Top feature family", "Mean |SHAP|", "Mean signed SHAP", "Scale"],
                shap_rows,
            ),
            "",
            "The SHAP values explain model outputs, not causal effects. Signed values are "
            "raw-margin (log-odds) contributions. Results are based on a fixed common sample "
            "of 500 mature loans per evaluation split.",
            "",
            _markdown_table(
                ["Horizon pair", "Spearman rank correlation", "Feature families"],
                correlation_rows,
            ),
            "",
            "Observed rank correlation range: **0.868–0.955**.",
            "",
            "## 5. SHAP explanation clusters",
            "",
            _markdown_table(
                ["Horizon", "Selected k", "Validation silhouette", "Test cluster sizes and mean risk", "Validation seed ARI"],
                cluster_rows,
            ),
            "",
            "Candidate k values 2–6 were compared using validation explanations only. "
            "Cluster sizes/risk are descriptive of the sampled exploratory test cohort. "
            "Cluster IDs are horizon-specific explanation patterns, not borrower types.",
            "",
            "## 6. Macroeconomic augmentation",
            "",
            _markdown_table(
                ["Horizon", "Metric", "Baseline", "Augmented", "Augmented − baseline", "Paired 95% interval"],
                macro_rows,
            ),
            "",
            f"Indicators: {macro_sources}.",
            "",
            f"Origination-date proxy: {macro_timing}",
            "",
            f"FRED revision limitation: {macro_revisions}",
            "",
            f"Date alignment rule: {macro_alignment}",
            "",
            "On this experiment and these inspected vintages, augmentation did not consistently "
            "improve the reported metrics. That is not evidence that macroeconomic features are "
            "universally ineffective: it is specific to these national series, this date proxy, "
            "the revised FRED snapshot, model settings, and exploratory cohorts.",
            "",
            "## Interpretation and limitations",
            "",
            "- Baseline and macro experiment results are distinct; macro metrics use the "
            "common macro-available cohort and must not be mixed with other score tables "
            "without noting their cohort definitions.",
            "- The macro comparison uses national indicators and an approximate first-payment-"
            "month-minus-two-month decision-date proxy. Fixed availability lags are not exact "
            "historical release timestamps; FRED's current-vintage observations may include "
            "later revisions.",
            "- SHAP reports model attribution, not cause or effect. Cluster quality scores "
            "are validation diagnostics and do not establish naturally occurring borrower groups.",
            "- None of the previously inspected test vintages can support untouched "
            "confirmatory claims.",
            "- Values marked **Unavailable** in the consolidated CSV were not provided by "
            "the corresponding source artifact; unavailable entries are not imputed.",
            "",
            "## Source files",
            "",
            *[f"- `{path}`" for path in SOURCE_FILES.values()],
            "",
        ]
    )


def build_ppt_table(root: Path, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pilot = data["pilot_metrics"]
    pilot = pilot.loc[pilot["split"].eq("test")]
    macro = data["macro_metrics"]
    macro = macro.loc[macro["split"].eq("test")]
    differences = data["macro_differences"]
    shap = data["shap_importance"]
    rank = data["horizon_consistency"]
    diagnostics = data["cluster_diagnostics"]
    profiles = data["cluster_profiles"]
    profiles = profiles.loc[profiles["split"].eq("test")]
    training_cohorts = data["training_cohorts"]
    survival_ipcw = data["survival_ipcw"].loc[
        data["survival_ipcw"]["evaluation_period"].eq("2017-2019_exploratory")
    ]
    result = []
    for horizon in (12, 24, 36):
        row = {
            "horizon_months": horizon,
            "cohort": "2017-2019 exploratory mature test",
            "n": int(
                pilot.loc[
                    pilot["horizon_months"].eq(horizon)
                    & pilot["model"].eq("xgboost")
                    & pilot["strategy"].eq("complete_eligible"),
                    "n",
                ].iloc[0]
            ),
            "label_strategy_training_counts_and_prevalence": "; ".join(
                f"{strategy}={int(item['training_count']):,} "
                f"({100 * int(item['defaults']) / int(item['training_count']):.3f}%)"
                for strategy, item in training_cohorts.loc[
                    training_cohorts["horizon_months"].eq(horizon)
                ].set_index("strategy").iterrows()
            ),
            "defaults": int(
                pilot.loc[
                    pilot["horizon_months"].eq(horizon)
                    & pilot["model"].eq("xgboost")
                    & pilot["strategy"].eq("complete_eligible"),
                    "defaults",
                ].iloc[0]
            ),
            "logistic_regression_complete_roc_auc": np.nan,
            "logistic_regression_complete_pr_auc": np.nan,
            "logistic_regression_complete_brier": np.nan,
            "xgboost_complete_roc_auc": np.nan,
            "xgboost_complete_pr_auc": np.nan,
            "xgboost_complete_brier": np.nan,
            "cox_complete_outcome_roc_auc": np.nan,
            "cox_complete_outcome_pr_auc": np.nan,
            "cox_complete_outcome_brier": np.nan,
            "survival_ipcw_roc_auc": np.nan,
            "survival_ipcw_pr_auc": np.nan,
            "survival_ipcw_brier": np.nan,
            "cause_specific_concordance_36m": "Unavailable (36m statistic only)",
            "macro_delta_roc_auc_ci95": "",
            "macro_delta_pr_auc_ci95": "",
            "macro_delta_brier_score_ci95": "",
            "shap_top_features": "",
            "selected_cluster_count": np.nan,
            "validation_silhouette": np.nan,
            "test_cluster_sizes": "",
            "rank_rho_12_24": np.nan,
            "rank_rho_12_36": np.nan,
            "rank_rho_24_36": np.nan,
            "sources": (
                "pilot_metrics.csv; survival_evaluation.csv; "
                "survival_censor_aware/ipcw_time_dependent_metrics.csv; "
                "survival_censor_aware/cause_specific_concordance.csv; "
                "macro_experiment/baseline_vs_macro_metrics.csv; "
                "macro_experiment/paired_performance_differences.csv; "
                "explainability/global_shap_feature_importance.csv; "
                "explainability/horizon_rank_consistency.csv; "
                "explainability/cluster_quality_diagnostics.csv; "
                "explainability/cluster_risk_profile_summary.csv"
            ),
        }
        for model, prefix in (
            ("logistic_regression", "logistic_regression"),
            ("xgboost", "xgboost"),
        ):
            item = pilot.loc[
                pilot["horizon_months"].eq(horizon)
                & pilot["model"].eq(model)
                & pilot["strategy"].eq("complete_eligible")
            ].iloc[0]
            row[f"{prefix}_complete_roc_auc"] = float(item["roc_auc"])
            row[f"{prefix}_complete_pr_auc"] = float(item["pr_auc"])
            row[f"{prefix}_complete_brier"] = float(item["brier_score"])
        cox = pilot.loc[
            pilot["horizon_months"].eq(horizon)
            & pilot["model"].eq("cause_specific_cox")
        ].iloc[0]
        row["cox_complete_outcome_roc_auc"] = float(cox["roc_auc"])
        row["cox_complete_outcome_pr_auc"] = float(cox["pr_auc"])
        row["cox_complete_outcome_brier"] = float(cox["brier_score"])
        survival = survival_ipcw.loc[survival_ipcw["horizon_months"].eq(horizon)].iloc[0]
        row["survival_ipcw_roc_auc"] = float(survival["roc_auc_ipcw"])
        row["survival_ipcw_pr_auc"] = float(survival["pr_auc_ipcw"])
        row["survival_ipcw_brier"] = float(survival["brier_score_ipcw"])
        if horizon == 36:
            concordance = data["survival_concordance"].loc[
                data["survival_concordance"]["evaluation_period"].eq(
                    "2017-2019_exploratory"
                ),
                "concordance_index",
            ].iloc[0]
            row["cause_specific_concordance_36m"] = f"{float(concordance):.4f}"
        for metric in ("roc_auc", "pr_auc", "brier_score"):
            item = differences.loc[
                differences["horizon_months"].eq(horizon)
                & differences["metric"].eq(metric)
            ].iloc[0]
            row[f"macro_delta_{metric}_ci95"] = (
                f"{float(item['difference_macro_minus_baseline']):+.4f} "
                f"[{float(item['ci_95_lower']):+.4f}, {float(item['ci_95_upper']):+.4f}]"
            )
        top = shap.loc[shap["horizon_months"].eq(horizon)].nlargest(3, "mean_abs_shap")
        row["shap_top_features"] = ", ".join(top["feature"].astype(str))
        diagnostic = diagnostics.loc[
            diagnostics["horizon_months"].eq(horizon)
            & diagnostics["selected"].astype(bool)
        ].iloc[0]
        row["selected_cluster_count"] = int(diagnostic["n_clusters"])
        row["validation_silhouette"] = float(diagnostic["silhouette_score"])
        cluster_profile = profiles.loc[
            profiles["horizon_months"].eq(horizon)
        ].sort_values("cluster")
        row["test_cluster_sizes"] = ", ".join(
            str(int(size)) for size in cluster_profile["cluster_size"]
        )
        for left, right, key in (
            (12, 24, "rank_rho_12_24"),
            (12, 36, "rank_rho_12_36"),
            (24, 36, "rank_rho_24_36"),
        ):
            value = rank.loc[
                rank["horizon_a_months"].eq(left)
                & rank["horizon_b_months"].eq(right),
                "spearman_rank_correlation",
            ].iloc[0]
            row[key] = float(value)
        result.append(row)
    return pd.DataFrame(result)


def generate(output_dir: Path = OUTPUT_DIR, root: Path = ROOT) -> dict[str, Path]:
    data = _read_sources(root)
    consolidated = build_consolidated_results(root)
    ppt = build_ppt_table(root, data)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "consolidated": output_dir / "consolidated_results.csv",
        "markdown": output_dir / "results_tables.md",
        "ppt": output_dir / "ppt_results.csv",
    }
    consolidated.to_csv(outputs["consolidated"], index=False)
    outputs["markdown"].write_text(build_markdown(root, data), encoding="utf-8")
    ppt.to_csv(outputs["ppt"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    outputs = generate(args.output_dir)
    for label, path in outputs.items():
        print(f"Saved {label} summary: {path}")


if __name__ == "__main__":
    main()
