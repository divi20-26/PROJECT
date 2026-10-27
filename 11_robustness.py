"""Run paired vintage-stratified test bootstrap comparisons and segment analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import pilot_model_utils as utils


BOOTSTRAP_REPLICATES = int(utils.pilot_settings()["bootstrap"]["replicates"])
REFERENCE_MODEL = "logistic_regression"
REFERENCE_STRATEGY = "naive_full"


def _paired_bootstrap_delta(
    y_true: np.ndarray,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
    metric: str,
    replicates: int,
    seed: int,
    candidate_threshold: float | None = None,
    reference_threshold: float | None = None,
    candidate_cost: dict | None = None,
    reference_cost: dict | None = None,
    vintage: np.ndarray | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)

    def metric_value(
        y: np.ndarray,
        probability: np.ndarray,
        weights: np.ndarray,
        threshold: float | None,
        cost: dict | None,
        score_codes: np.ndarray,
        score_group_count: int,
    ) -> float:
        if metric == "brier_score":
            return float(np.dot(weights, np.square(y - probability)) / weights.sum())
        if metric in {"roc_auc", "pr_auc"}:
            positive_by_score = np.bincount(
                score_codes, weights=weights * y, minlength=score_group_count
            )
            negative_by_score = np.bincount(
                score_codes, weights=weights * (1 - y), minlength=score_group_count
            )
            positive_total = float(positive_by_score.sum())
            negative_total = float(negative_by_score.sum())
            if positive_total == 0 or negative_total == 0:
                raise ValueError("Bootstrap replicate must contain both outcome classes.")
            if metric == "roc_auc":
                lower_score_negatives = np.cumsum(negative_by_score) - negative_by_score
                concordant = np.sum(
                    positive_by_score * (lower_score_negatives + 0.5 * negative_by_score)
                )
                return float(concordant / (positive_total * negative_total))
            positive_descending = positive_by_score[::-1]
            total_descending = (
                positive_by_score + negative_by_score
            )[::-1]
            cumulative_positive = np.cumsum(positive_descending)
            cumulative_total = np.cumsum(total_descending)
            precision = np.divide(
                cumulative_positive,
                cumulative_total,
                out=np.zeros_like(cumulative_positive, dtype=float),
                where=cumulative_total > 0,
            )
            return float(
                np.sum(positive_descending * precision) / positive_total
            )
        if metric == "decision_cost":
            if threshold is None or cost is None:
                raise ValueError("Decision-cost bootstrap requires a threshold and scenario costs.")
            flagged = probability >= threshold
            false_negative = (y == 1) & ~flagged
            return float(
                (
                    np.dot(weights, false_negative) * cost["missed_default"]
                    + np.dot(weights, flagged) * cost["review"]
                )
                / weights.sum()
            )
        raise ValueError(f"Unsupported paired bootstrap metric: {metric}")

    if (
        len(y_true) == 0
        or len(y_true) != len(candidate_probability)
        or len(y_true) != len(reference_probability)
    ):
        raise ValueError("Paired bootstrap inputs must have equal, nonzero lengths.")
    if vintage is not None and len(vintage) != len(y_true):
        raise ValueError("Vintage labels must align with paired bootstrap observations.")
    candidate_groups, candidate_codes = np.unique(
        candidate_probability, return_inverse=True
    )
    reference_groups, reference_codes = np.unique(
        reference_probability, return_inverse=True
    )
    strata = (
        [np.arange(len(y_true))]
        if vintage is None
        else [np.flatnonzero(vintage == value) for value in np.unique(vintage)]
    )
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("Vintage-stratified bootstrap contains an empty stratum.")

    for iteration in range(replicates):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        weights = np.bincount(sample, minlength=len(y_true))
        differences[iteration] = metric_value(
            y_true,
            candidate_probability,
            weights,
            candidate_threshold,
            candidate_cost,
            candidate_codes,
            len(candidate_groups),
        ) - metric_value(
            y_true,
            reference_probability,
            weights,
            reference_threshold,
            reference_cost,
            reference_codes,
            len(reference_groups),
        )
    original_weights = np.ones(len(y_true), dtype=np.int64)
    estimate = metric_value(
        y_true,
        candidate_probability,
        original_weights,
        candidate_threshold,
        candidate_cost,
        candidate_codes,
        len(candidate_groups),
    ) - metric_value(
        y_true,
        reference_probability,
        original_weights,
        reference_threshold,
        reference_cost,
        reference_codes,
        len(reference_groups),
    )
    return (
        float(estimate),
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
    )


def paired_bootstrap(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    split_name: str = "test",
) -> pd.DataFrame:
    test = predictions.loc[predictions["split"].eq(split_name)]
    threshold_lookup = thresholds.set_index(
        ["model", "strategy", "horizon_months", "cost_scenario"]
    )["threshold"]
    rows = []
    for horizon, horizon_frame in test.groupby("horizon_months"):
        reference = horizon_frame.loc[
            horizon_frame["model"].eq(REFERENCE_MODEL)
            & horizon_frame["strategy"].eq(REFERENCE_STRATEGY)
        ].set_index(["vintage", "loan_identifier"]).sort_index()
        if reference.empty:
            raise ValueError(f"Bootstrap reference predictions missing for {horizon}m.")
        reference_y = reference["y_true"].to_numpy(dtype="int8")
        reference_probability = reference["probability"].to_numpy(dtype=float)
        reference_vintage = reference.index.get_level_values("vintage").to_numpy()
        for (model, strategy), candidate_raw in horizon_frame.groupby(["model", "strategy"]):
            candidate = candidate_raw.set_index(["vintage", "loan_identifier"]).sort_index()
            if not candidate.index.equals(reference.index) or not candidate["y_true"].equals(
                reference["y_true"]
            ):
                raise ValueError(
                    f"Cannot pair bootstrap samples: cohort mismatch for {model}/{strategy}/{horizon}m."
                )
            candidate_probability = candidate["probability"].to_numpy(dtype=float)
            if (model, strategy) != (REFERENCE_MODEL, REFERENCE_STRATEGY):
                estimate, lower, upper = _paired_bootstrap_delta(
                    reference_y,
                    candidate_probability,
                    reference_probability,
                    "brier_score",
                    BOOTSTRAP_REPLICATES,
                    utils.SEED + int(horizon),
                    vintage=reference_vintage,
                )
                rows.append(
                    {
                        "horizon_months": int(horizon),
                        "candidate_model": model,
                        "candidate_strategy": strategy,
                        "reference_model": REFERENCE_MODEL,
                        "reference_strategy": REFERENCE_STRATEGY,
                        "metric": "brier_score",
                        "scenario": None,
                        "delta_candidate_minus_reference": estimate,
                        "ci_2_5": lower,
                        "ci_97_5": upper,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "test_n": len(reference),
                        "bootstrap_sampling": "paired within vintage",
                    }
                )
                for metric, seed_offset in (("roc_auc", 200), ("pr_auc", 300)):
                    estimate, lower, upper = _paired_bootstrap_delta(
                        reference_y,
                        candidate_probability,
                        reference_probability,
                        metric,
                        BOOTSTRAP_REPLICATES,
                        utils.SEED + int(horizon) + seed_offset,
                        vintage=reference_vintage,
                    )
                    rows.append(
                        {
                            "horizon_months": int(horizon),
                            "candidate_model": model,
                            "candidate_strategy": strategy,
                            "reference_model": REFERENCE_MODEL,
                            "reference_strategy": REFERENCE_STRATEGY,
                            "metric": metric,
                            "scenario": None,
                            "delta_candidate_minus_reference": estimate,
                            "ci_2_5": lower,
                            "ci_97_5": upper,
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                            "test_n": len(reference),
                            "bootstrap_sampling": "paired within vintage",
                        }
                    )
            if (model, strategy) == (REFERENCE_MODEL, REFERENCE_STRATEGY):
                continue
            settings = utils.pilot_settings()
            for scenario in settings["bootstrap_decision_cost_scenarios"]:
                parameters = settings["decision_cost_scenarios"][scenario]
                candidate_threshold = float(
                    threshold_lookup.loc[(model, strategy, int(horizon), scenario)]
                )
                reference_threshold = float(
                    threshold_lookup.loc[
                        (REFERENCE_MODEL, REFERENCE_STRATEGY, int(horizon), scenario)
                    ]
                )
                estimate, lower, upper = _paired_bootstrap_delta(
                    reference_y,
                    candidate_probability,
                    reference_probability,
                    "decision_cost",
                    BOOTSTRAP_REPLICATES,
                    utils.SEED + int(horizon) + 100,
                    candidate_threshold=candidate_threshold,
                    reference_threshold=reference_threshold,
                    candidate_cost={
                        "missed_default": parameters["missed_default_cost"],
                        "review": parameters["review_cost_per_flag"],
                    },
                    reference_cost={
                        "missed_default": parameters["missed_default_cost"],
                        "review": parameters["review_cost_per_flag"],
                    },
                    vintage=reference_vintage,
                )
                rows.append(
                    {
                        "horizon_months": int(horizon),
                        "candidate_model": model,
                        "candidate_strategy": strategy,
                        "reference_model": REFERENCE_MODEL,
                        "reference_strategy": REFERENCE_STRATEGY,
                        "metric": "decision_cost",
                        "scenario": scenario,
                        "delta_candidate_minus_reference": estimate,
                        "ci_2_5": lower,
                        "ci_97_5": upper,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "test_n": len(reference),
                        "bootstrap_sampling": "paired within vintage",
                    }
                )
    return pd.DataFrame(rows)


def segment_analysis(
    predictions: pd.DataFrame,
    years: list[int] | None = None,
    split_name: str = "test",
) -> pd.DataFrame:
    selected_years = utils.TEST_YEARS if years is None else years
    features = utils.load_features(selected_years)
    segments = features[
        ["vintage", "loan_identifier", "classic_fico", "occupancy_status"]
    ].copy()
    segments["fico_band"] = utils.fico_band(segments["classic_fico"])
    segments["occupancy_status"] = segments["occupancy_status"].astype("string").fillna("missing")
    segment_columns = ["vintage", "fico_band", "occupancy_status"]
    joined = predictions.loc[predictions["split"].eq(split_name)].merge(
        segments,
        on=["vintage", "loan_identifier"],
        how="left",
        validate="many_to_one",
    )
    if joined["fico_band"].isna().any():
        raise ValueError("Missing test segment fields for one or more prediction rows.")
    rows = []
    for keys, group in joined.groupby(
        ["horizon_months", "model", "strategy", *segment_columns],
        dropna=False,
        observed=True,
    ):
        horizon, model, strategy, vintage, fico, occupancy = keys
        y_true = group["y_true"].to_numpy(dtype="int8")
        probability = group["probability"].to_numpy(dtype=float)
        supported = len(group) >= 1000 and int(y_true.sum()) >= 20 and int((y_true == 0).sum()) >= 20
        rows.append(
            {
                "horizon_months": int(horizon),
                "model": model,
                "strategy": strategy,
                "vintage": int(vintage),
                "fico_band": str(fico),
                "occupancy_status": str(occupancy),
                "n": len(group),
                "defaults": int(y_true.sum()),
                "default_rate": float(y_true.mean()) if len(group) else None,
                "supported_for_discrimination": supported,
                "roc_auc": float(roc_auc_score(y_true, probability)) if supported else None,
                "pr_auc": float(average_precision_score(y_true, probability)) if supported else None,
                "brier_score": float(brier_score_loss(y_true, probability)) if len(group) else None,
            }
        )
    return pd.DataFrame(rows)


def create_results_report(metrics: pd.DataFrame, costs: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    test = metrics.loc[metrics["split"].eq("test")]
    training_summary_path = utils.TABLE_DIR / "binary_training_cohort_summary.csv"
    label_comparison_path = utils.TABLE_DIR / "pilot_label_strategy_comparison.csv"
    if not training_summary_path.exists() or not label_comparison_path.exists():
        raise FileNotFoundError(
            "Binary training cohort and label comparison tables must exist before reporting."
        )
    training_summary = pd.read_csv(training_summary_path)
    label_comparison_frame = pd.read_csv(label_comparison_path)
    threshold_path = utils.TABLE_DIR / "validation_selected_thresholds.csv"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing selected validation thresholds: {threshold_path}")
    thresholds = pd.read_csv(threshold_path)
    no_review_validation = int(thresholds["validation_no_review_action"].sum())
    no_review_test = int(costs["test_flagged"].eq(0).sum())

    lines = [
        "# Controlled Pilot Results",
        "",
        "These are frozen comparative research fits, not a final lending-model selection or deployment recommendation.",
        "All binary and survival approaches are evaluated on the same horizon-mature 2017–2019",
        "test loans. Thresholds are selected on 2015–2016 validation data only.",
        "",
        "## Test-cohort results",
        "",
        "| Horizon | N | Defaults | Best ROC-AUC | Best PR-AUC | Lowest Brier |",
        "|---:|---:|---:|---|---|---|",
    ]
    for horizon, group in test.groupby("horizon_months"):
        baseline = group.iloc[0]
        best_roc = group.loc[group["roc_auc"].idxmax()]
        best_pr = group.loc[group["pr_auc"].idxmax()]
        best_brier = group.loc[group["brier_score"].idxmin()]
        lines.append(
            f"| {int(horizon)} | {int(baseline['n']):,} | {int(baseline['defaults']):,} "
            f"| {best_roc['model']} / {best_roc['strategy']} ({best_roc['roc_auc']:.4f}) "
            f"| {best_pr['model']} / {best_pr['strategy']} ({best_pr['pr_auc']:.4f}) "
            f"| {best_brier['model']} / {best_brier['strategy']} ({best_brier['brier_score']:.6f}) |"
        )
    lines.extend(
        [
            "",
            "## Training cohort sizes",
            "",
            "| Horizon | Strategy | Training loans | Defaults |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in training_summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon_months} | {row.strategy} | {row.training_count:,} | {row.defaults:,} |"
        )
    lines.extend(
        [
            "",
            "## Label strategy comparison",
            "",
            "| Horizon | Naive defaults | Complete-eligible loans | Complete defaults | Retrospective naive false negatives | Same-population disagreements |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in label_comparison_frame.itertuples(index=False):
        lines.append(
            f"| {row.horizon_months} | {row.naive_training_defaults:,} "
            f"| {row.complete_eligible_loans:,} | {row.complete_horizon_defaults:,} "
            f"| {row.retrospective_naive_false_negatives:,} "
            f"| {row.common_population_label_disagreements:,} |"
        )
    matching_path = utils.TABLE_DIR / "matched_sampling_sensitivity.csv"
    if matching_path.exists():
        matching = pd.read_csv(matching_path)
        lines.extend(
            [
                "",
                "## Repeated matched-naive sampling",
                "",
                "Five fixed matched samples per horizon preserve exact counts within vintage, FICO, LTV, and occupancy strata. The table reports the observed naive-label prevalence range; broader feature-balance diagnostics are in `matched_sampling_balance.csv`. This is sample-composition sensitivity, not a causal estimate.",
                "",
                "| Horizon | Seeds | Matched sample size | Naive default-rate range | Complete default rate |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for horizon, group in matching.groupby("horizon_months", sort=True):
            lines.append(
                f"| {int(horizon)} | {group['sampling_seed'].nunique()} "
                f"| {int(group['sample_size'].iloc[0]):,} "
                f"| {group['naive_default_rate'].min():.4%}–{group['naive_default_rate'].max():.4%} "
                f"| {group['complete_default_rate'].iloc[0]:.4%} |"
            )
    survival_summary_path = utils.TABLE_DIR / "survival_training_summary.csv"
    if survival_summary_path.exists():
        survival_summary = pd.read_csv(survival_summary_path).iloc[0]
        lines.extend(
            [
                "",
                "## Delayed-entry survival risk set",
                "",
                f"The 36-month Cox risk set contains {int(survival_summary['survival_eligible_rows']):,} "
                f"of {int(survival_summary['training_label_rows']):,} cutoff-qualified training loans; "
                f"{int(survival_summary['survival_excluded_rows']):,} "
                f"({survival_summary['survival_exclusion_rate']:.2%}) are excluded. "
                f"{int(survival_summary['delayed_entry_rows']):,} "
                f"({survival_summary['delayed_entry_rate_of_risk_set']:.2%}) have entry after month 1. "
                f"It contains {int(survival_summary['default_events']):,} defaults, "
                f"{int(survival_summary['voluntary_payoff_events']):,} payoff events, and "
                f"{int(survival_summary['right_censored_or_other_termination_rows']):,} "
                f"right-censored/other-termination observations "
                f"({survival_summary['censoring_rate_of_risk_set']:.2%} of the Cox risk set).",
            ]
        )
    lines.extend(
        [
            "",
            "## Survival prediction evaluation",
            "",
            "The Cox model is evaluated using its predicted default cumulative incidence at each horizon. "
            "For this binary scoring, the test denominator is restricted to loans with fully ascertainable "
            "horizon outcomes and acquisition entry in month 1; voluntary payoff is a competing non-default "
            "outcome, while early other termination is excluded rather than mislabeled as a control. "
            "Consequently, the horizon-specific ROC-AUC, PR-AUC, and Brier score are complete-outcome "
            "cumulative/dynamic discrimination and prediction-error summaries, not IPCW estimates for "
            "the full censored test population. Results are in `survival_evaluation.csv`.",
            "",
            "Calibration intercept/slope are descriptive estimates on the held-out common test cohorts; "
            "they are not used to recalibrate predictions. Reliability diagrams are generated by "
            "`12_make_paper_figures.py`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Decision costs",
            "",
            "Costs are normalized hypothetical values. They are not observed dollar losses.",
            "The review-to-missed-default cost ratios range from 0.001 to 100.0 in a broad hypothetical sensitivity grid. No lender-specific utility evidence was available to justify point values. Thresholds are selected on validation data at exact observed score boundaries, including no review.",
            f"No-review was the validation-minimum-cost action for {no_review_validation:,} of "
            f"{len(thresholds):,} model/strategy/horizon/scenario combinations; applying the "
            f"validation thresholds produced zero test reviews in {no_review_test:,} of "
            f"{len(costs):,} combinations. This occurs when validation-estimated avoided "
            "missed-default cost does not offset review costs under the scenario.",
            "See `test_decision_costs.csv` for all model and scenario results.",
            "",
            "## Paired bootstrap",
            "",
            f"Paired vintage-stratified loan bootstrap replicates: {utils.pilot_settings()['bootstrap']['replicates']}.",
            "Intervals are paired candidate-minus-naive-full-Logistic-Regression differences, resampling loans within each observed vintage. Negative values favor the candidate for Brier score and cost; positive values favor the candidate for ROC-AUC and PR-AUC. Cost intervals cover review-to-missed-default ratios 0.05 and 1.0; the full ratio grid is reported descriptively. Inference is conditional on the 2017–2019 vintage mix. Results are in `paired_bootstrap.csv`.",
            "",
            "## Interpretation limits",
            "",
            "- Performance histories begin at acquisition, so survival estimates condition on delayed entry.",
            "- Binary maturity requires entry at month 1; the survival model uses delayed-entry times.",
            "- Cox preprocessing pools categories with fewer than 100 records and omits one-hot",
            "  indicators with fewer than 100 records in the minority category.",
            "- These are pilot fits and should not be described as evidence for a material research effect",
            "  unless metric, ranking, segment, and decision-cost results support that claim.",
            "- The decision-cost ratios are hypothetical and should be replaced or justified before paper claims.",
            "",
            "## Cost leaders by horizon and scenario",
            "",
            "| Horizon | Scenario | Lowest test cost | Model / strategy |",
            "|---:|---|---:|---|",
        ]
    )
    for (horizon, scenario), group in costs.groupby(["horizon_months", "cost_scenario"]):
        best = group.loc[group["test_mean_cost"].idxmin()]
        lines.append(
            f"| {int(horizon)} | {scenario} | {best['test_mean_cost']:.6f} "
            f"| {best['model']} / {best['strategy']} |"
        )
    supported_differences = bootstrap.loc[
        (bootstrap["metric"] == "brier_score")
        & ((bootstrap["ci_2_5"] > 0) | (bootstrap["ci_97_5"] < 0))
    ]
    lines.extend(
        [
            "",
            "## Paired differences outside the 95% interval around zero",
            "",
            "Brier-score differences below are relative to naive-full Logistic Regression.",
        ]
    )
    if supported_differences.empty:
        lines.append("No candidate Brier-score interval excluded zero in the pilot bootstrap.")
    else:
        for row in supported_differences.itertuples(index=False):
            lines.append(
                f"- {int(row.horizon_months)}m {row.candidate_model}/{row.candidate_strategy}: "
                f"delta={row.delta_candidate_minus_reference:.6g}, "
                f"95% CI [{row.ci_2_5:.6g}, {row.ci_97_5:.6g}]."
            )
    discrimination_differences = bootstrap.loc[
        bootstrap["metric"].isin(["roc_auc", "pr_auc"])
        & ((bootstrap["ci_2_5"] > 0) | (bootstrap["ci_97_5"] < 0))
    ]
    lines.extend(
        [
            "",
            "Discrimination intervals excluding zero (positive favors candidate):",
        ]
    )
    if discrimination_differences.empty:
        lines.append("No candidate ROC-AUC or PR-AUC interval excluded zero.")
    else:
        for row in discrimination_differences.itertuples(index=False):
            lines.append(
                f"- {int(row.horizon_months)}m {row.metric} "
                f"{row.candidate_model}/{row.candidate_strategy}: "
                f"delta={row.delta_candidate_minus_reference:.6g}, "
                f"95% CI [{row.ci_2_5:.6g}, {row.ci_97_5:.6g}]."
            )
    cost_differences = bootstrap.loc[
        (bootstrap["metric"] == "decision_cost")
        & ((bootstrap["ci_2_5"] > 0) | (bootstrap["ci_97_5"] < 0))
    ]
    lines.extend(
        [
            "",
            "Decision-cost intervals excluding zero (negative favors candidate):",
        ]
    )
    if cost_differences.empty:
        lines.append("No candidate decision-cost interval excluded zero.")
    else:
        for row in cost_differences.itertuples(index=False):
            lines.append(
                f"- {int(row.horizon_months)}m {row.scenario} "
                f"{row.candidate_model}/{row.candidate_strategy}: "
                f"delta={row.delta_candidate_minus_reference:.6g}, "
                f"95% CI [{row.ci_2_5:.6g}, {row.ci_97_5:.6g}]."
            )
    path = utils.config.RESULTS_DIR / "pilot_results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_paths = [
        utils.PREDICTION_DIR / "binary_pilot_predictions.parquet",
        utils.PREDICTION_DIR / "survival_pilot_predictions.parquet",
    ]
    missing = [path for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot predictions: {missing}")
    predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_paths], ignore_index=True
    )
    threshold_path = utils.TABLE_DIR / "validation_selected_thresholds.csv"
    cost_path = utils.TABLE_DIR / "test_decision_costs.csv"
    metrics_path = utils.TABLE_DIR / "pilot_metrics.csv"
    for path in (threshold_path, cost_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(f"Run the preceding pilot analysis stage first: {path}")
    thresholds = pd.read_csv(threshold_path)
    costs = pd.read_csv(cost_path)
    metrics = pd.read_csv(metrics_path)
    bootstrap = paired_bootstrap(predictions, thresholds)
    segments = segment_analysis(predictions)
    bootstrap.to_csv(utils.TABLE_DIR / "paired_bootstrap.csv", index=False)
    segments.to_csv(utils.TABLE_DIR / "pilot_segment_metrics.csv", index=False)
    create_results_report(metrics, costs, bootstrap)
    print(f"Saved {len(bootstrap):,} paired-bootstrap intervals.")
    print(f"Saved {len(segments):,} vintage/borrower-segment rows.")
    return bootstrap, segments


if __name__ == "__main__":
    run_robustness()
