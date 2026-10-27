"""Evaluate frozen pilot models on previously inspected 2020-2022 vintages."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pilot_model_utils as utils


EXPLORATORY_DIR = utils.config.RESULTS_DIR / "exploratory_late_vintages"
MODEL_STRATEGIES = (
    ("logistic_regression", "naive_full"),
    ("logistic_regression", "complete_eligible"),
    ("logistic_regression", "matched_naive"),
    ("xgboost", "naive_full"),
    ("xgboost", "complete_eligible"),
    ("xgboost", "matched_naive"),
)


def _load_script(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluation helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_survival_predictions(
    years: list[int], training_count: int
) -> list[pd.DataFrame]:
    survival = _load_script("08_train_survival_models.py", "late_vintage_survival")
    artifact_path = utils.MODEL_DIR / "cause_specific_cox_36m.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing frozen Cox artifact: {artifact_path}")
    artifact = joblib.load(artifact_path)
    features = utils.load_features(years)
    incidence = survival.predict_default_cumulative_incidence(
        features,
        artifact["preprocessor"],
        artifact["models"],
        max_month=int(artifact["max_month"]),
    )
    feature_index = pd.MultiIndex.from_frame(features[["vintage", "loan_identifier"]])
    rows = []
    for horizon in utils.HORIZONS:
        mature = utils.load_mature_cohort(years, horizon)
        probabilities = pd.Series(
            incidence[:, horizon - 1],
            index=feature_index,
            name="probability",
        ).reindex(pd.MultiIndex.from_frame(mature[["vintage", "loan_identifier"]]))
        if probabilities.isna().any():
            raise ValueError(f"Cox predictions do not cover the mature {horizon}m cohort.")
        rows.append(
            pd.DataFrame(
                {
                    "split": "exploratory_late_test",
                    "model": "cause_specific_cox",
                    "strategy": "delayed_entry_competing_risk",
                    "horizon_months": horizon,
                    "vintage": mature["vintage"].to_numpy(),
                    "loan_identifier": mature["loan_identifier"].to_numpy(),
                    "y_true": mature["y_true"].to_numpy(dtype="int8"),
                    "probability": probabilities.to_numpy(dtype=float),
                    "training_count": training_count,
                }
            )
        )
    return rows


def _make_binary_predictions(
    years: list[int], training_counts: pd.Series
) -> list[pd.DataFrame]:
    rows = []
    for horizon in utils.HORIZONS:
        mature = utils.load_mature_cohort(years, horizon)
        for model, strategy in MODEL_STRATEGIES:
            model_path = utils.MODEL_DIR / f"{model}_{strategy}_{horizon}m.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing frozen binary model: {model_path}")
            estimator = joblib.load(model_path)
            probabilities = estimator.predict_proba(
                mature[utils.FEATURE_COLUMNS]
            )[:, 1]
            rows.append(
                pd.DataFrame(
                    {
                        "split": "exploratory_late_test",
                        "model": model,
                        "strategy": strategy,
                        "horizon_months": horizon,
                        "vintage": mature["vintage"].to_numpy(),
                        "loan_identifier": mature["loan_identifier"].to_numpy(),
                        "y_true": mature["y_true"].to_numpy(dtype="int8"),
                        "probability": probabilities,
                        "training_count": int(training_counts.loc[(horizon, strategy)]),
                    }
                )
            )
    return rows


def _validate_common_cohorts(predictions: pd.DataFrame) -> None:
    evaluator = _load_script("09_evaluate_models.py", "late_vintage_evaluator")
    evaluator._validate_common_cohorts(predictions)
    if not np.isfinite(predictions["probability"]).all():
        raise ValueError("Exploratory predictions contain nonfinite probabilities.")
    if predictions["probability"].lt(0).any() or predictions["probability"].gt(1).any():
        raise ValueError("Exploratory predictions fall outside [0, 1].")


def _calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    evaluator = _load_script("09_evaluate_models.py", "late_vintage_evaluator_metrics")
    rows = []
    for (horizon, model, strategy), group in predictions.groupby(
        ["horizon_months", "model", "strategy"], sort=True
    ):
        rows.append(
            {
                "horizon_months": int(horizon),
                "model": model,
                "strategy": strategy,
                **evaluator.evaluate_group(group),
            }
        )
    return pd.DataFrame(rows)


def _calculate_costs(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    threshold_lookup = thresholds.set_index(
        ["model", "strategy", "horizon_months", "cost_scenario"]
    )["threshold"]
    rows = []
    for (horizon, model, strategy), group in predictions.groupby(
        ["horizon_months", "model", "strategy"], sort=True
    ):
        y = group["y_true"].to_numpy(dtype="int8")
        probability = group["probability"].to_numpy(dtype=float)
        for scenario, parameters in settings["decision_cost_scenarios"].items():
            threshold = float(
                threshold_lookup.loc[(model, strategy, int(horizon), scenario)]
            )
            cost = utils.cost_value(
                y,
                probability,
                threshold,
                missed_default=float(parameters["missed_default_cost"]),
                review=float(parameters["review_cost_per_flag"]),
            )
            rows.append(
                {
                    "horizon_months": int(horizon),
                    "model": model,
                    "strategy": strategy,
                    "cost_scenario": scenario,
                    "validation_selected_threshold": threshold,
                    "test_n": len(group),
                    "test_defaults": int(y.sum()),
                    "test_mean_cost": cost,
                    "review_rate": float((probability >= threshold).mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    metrics: pd.DataFrame,
    costs: pd.DataFrame,
    bootstrap: pd.DataFrame,
    years: list[int],
    outcome_cutoff: str,
) -> None:
    lines = [
        "# Exploratory Later-Vintage Evaluation",
        "",
        "**Exploratory only; not a confirmatory holdout.** Outcome coverage, maturity, and "
        "default-rate summaries for these vintages were previously generated in "
        "`tables/mature_outcome_coverage_asof_20260331.csv`.",
        "",
        f"- Vintages: {', '.join(map(str, years))}",
        f"- Outcome snapshot: {outcome_cutoff}",
        "- Models: frozen pilot model artifacts trained on 2005-2014; no refitting or "
        "model selection on these vintages.",
        "- Thresholds: previously selected using 2015-2016 validation data.",
        "- Cohort: common complete-horizon test population for each horizon; payoff is a "
        "non-default competing outcome, and early other termination is excluded.",
        "- Cost values are hypothetical normalized scenarios, not observed dollar losses.",
        "- Active pilot settings now match the exact validation thresholds and 1,000-replicate "
        "bootstrap outputs; pre-audit metadata is retained in the settings JSON.",
        "",
        "## Common-cohort metrics",
        "",
        "| Horizon | Model | Strategy | N | Defaults | ROC-AUC | PR-AUC | Brier | ECE |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.horizon_months} | {row.model} | {row.strategy} | {row.n:,} | "
            f"{row.defaults:,} | {row.roc_auc:.4f} | {row.pr_auc:.4f} | "
            f"{row.brier_score:.6f} | {row.expected_calibration_error:.6f} |"
        )
    leaders = costs.loc[costs.groupby(
        ["horizon_months", "cost_scenario"]
    )["test_mean_cost"].idxmin()]
    lines.extend(
        [
            "",
            "## Lowest descriptive test cost",
            "",
            "| Horizon | Scenario | Mean cost | Model | Strategy |",
            "|---:|---|---:|---|---|",
        ]
    )
    for row in leaders.sort_values(["horizon_months", "cost_scenario"]).itertuples(index=False):
        lines.append(
            f"| {row.horizon_months} | {row.cost_scenario} | {row.test_mean_cost:.6f} | "
            f"{row.model} | {row.strategy} |"
        )
    nonzero = bootstrap.loc[
        (bootstrap["ci_2_5"] > 0) | (bootstrap["ci_97_5"] < 0)
    ]
    interval_counts = (
        nonzero.groupby("metric").size().to_dict()
        if not nonzero.empty
        else {}
    )
    lines.extend(
        [
            "",
            "## Paired uncertainty and segments",
            "",
            f"Paired, vintage-stratified bootstrap intervals use "
            f"{int(bootstrap['bootstrap_replicates'].max()):,} replicates. "
            "Intervals are exploratory and unadjusted; inference is conditional on the "
            "2020–2022 vintage mix.",
            f"Intervals excluding zero by metric: {json.dumps(interval_counts, sort_keys=True)}.",
            "Vintage/FICO/occupancy segment summaries are in `segment_metrics.csv`; "
            "paired comparison intervals are in `paired_bootstrap.csv`.",
            "",
            "These descriptive later-vintage results do not support confirmatory inference: "
            "the vintages' outcomes were previously inspected, and the tested cost scenarios "
            "are hypothetical.",
        ]
    )
    (EXPLORATORY_DIR / "exploratory_late_vintage_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    years = list(utils.config.LATE_TEST_YEARS)
    settings_path = utils.MODEL_DIR / "pilot_settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"Missing frozen pilot settings: {settings_path}")
    artifact_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings = utils.pilot_settings()
    thresholds_path = utils.TABLE_DIR / "validation_selected_thresholds.csv"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing validation-selected thresholds: {thresholds_path}")
    binary_cohort_path = utils.TABLE_DIR / "binary_training_cohort_summary.csv"
    survival_cohort_path = utils.TABLE_DIR / "survival_training_summary.csv"
    for path in (binary_cohort_path, survival_cohort_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen training cohort summary: {path}")
    binary_training_counts = pd.read_csv(binary_cohort_path).set_index(
        ["horizon_months", "strategy"]
    )["training_count"]
    survival_training_count = int(
        pd.read_csv(survival_cohort_path).loc[0, "survival_eligible_rows"]
    )
    predictions = pd.concat(
        _make_binary_predictions(years, binary_training_counts)
        + _make_survival_predictions(years, survival_training_count),
        ignore_index=True,
    )
    _validate_common_cohorts(predictions)
    metrics = _calculate_metrics(predictions)
    thresholds = pd.read_csv(thresholds_path)
    configured_scenarios = set(settings["decision_cost_scenarios"])
    threshold_scenarios = set(thresholds["cost_scenario"].astype(str))
    if configured_scenarios != threshold_scenarios:
        raise ValueError(
            "Current cost configuration does not match the validation threshold table: "
            f"configured={sorted(configured_scenarios)}, "
            f"thresholds={sorted(threshold_scenarios)}"
        )
    costs = _calculate_costs(predictions, thresholds, settings)
    robustness = _load_script("11_robustness.py", "late_vintage_robustness")
    bootstrap = robustness.paired_bootstrap(
        predictions,
        thresholds,
        split_name="exploratory_late_test",
    )
    segments = robustness.segment_analysis(
        predictions,
        years=years,
        split_name="exploratory_late_test",
    )
    predictions.to_parquet(EXPLORATORY_DIR / "predictions.parquet", index=False)
    metrics.to_csv(EXPLORATORY_DIR / "metrics.csv", index=False)
    costs.to_csv(EXPLORATORY_DIR / "decision_costs.csv", index=False)
    bootstrap.to_csv(EXPLORATORY_DIR / "paired_bootstrap.csv", index=False)
    segments.to_csv(EXPLORATORY_DIR / "segment_metrics.csv", index=False)
    protocol = {
        "evaluation": "exploratory_late_vintage",
        "vintages": years,
        "prior_outcome_inspection": (
            "mature outcome coverage, maturity, and default-rate summaries exist "
            "in results/tables/mature_outcome_coverage_asof_20260331.csv"
        ),
        "model_artifacts": str(utils.MODEL_DIR),
        "training_vintages": settings["training_vintages"],
        "validation_vintages": settings["validation_vintages"],
        "model_settings_snapshot": settings,
        "stored_model_settings_cost_scenarios": sorted(
            artifact_settings.get("decision_cost_scenarios", {})
        ),
        "stored_model_settings_bootstrap": artifact_settings.get("bootstrap"),
        "superseded_metadata_before_methodology_audit": artifact_settings.get(
            "superseded_metadata_before_methodology_audit"
        ),
        "stored_model_settings_warning": (
            "The pre-existing model settings file has older cost-scenario and bootstrap "
            "metadata than the current validation threshold table and robustness outputs."
            if set(artifact_settings.get("decision_cost_scenarios", {}))
            != configured_scenarios
            else None
        ),
        "threshold_source": str(thresholds_path),
        "retrospective_outcome_cutoff": settings["retrospective_outcome_cutoff"],
        "horizons_months": utils.HORIZONS,
        "models": list(settings["binary_models"]) + ["cause_specific_cox"],
        "cost_scenarios": settings["decision_cost_scenarios"],
        "bootstrap": settings["bootstrap"],
        "segment_dimensions": ["vintage", "fico_band", "occupancy_status"],
        "interpretation": "Descriptive/exploratory; not a confirmatory holdout.",
    }
    (EXPLORATORY_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    _write_report(
        metrics,
        costs,
        bootstrap,
        years,
        settings["retrospective_outcome_cutoff"],
    )
    print(f"Saved exploratory predictions and results to {EXPLORATORY_DIR}")
    return metrics, costs


if __name__ == "__main__":
    run_evaluation()
