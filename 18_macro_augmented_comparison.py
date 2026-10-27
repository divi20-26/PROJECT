"""Train and compare isolated baseline/macro XGBoost experiments on matched cohorts."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import pilot_model_utils as utils
from research_extensions import (
    MACRO_FEATURES,
    horizon_importance_table,
    merge_macro_asof,
    require_existing_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MACRO_DIR = utils.DATA_DIR / "macro"
OUTPUT_DIR = utils.config.RESULTS_DIR / "macro_experiment"
MODEL_DIR = utils.config.MODEL_DIR / "macro_experiment"
FIGURE_DIR = OUTPUT_DIR / "figures"
OBSERVATIONS_PATH = MACRO_DIR / "macro_observations_asof_snapshot.csv"
SEED = utils.SEED
BOOTSTRAP_REPLICATES = 1000


def _load_binary_builder():
    script = PROJECT_ROOT / "07_train_binary_models.py"
    spec = importlib.util.spec_from_file_location("frozen_binary_builder", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the frozen model builder at {script}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_estimator


@lru_cache(maxsize=8)
def _origination_date_proxy_cached(years: tuple[int, ...]) -> pd.DataFrame:
    from schema_contract import map_frame

    frames = []
    for year in years:
        path = utils.config.EXTRACTED_DIR / f"vintage_{year}" / "origination.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw origination data for vintage {year}: {path}. "
                "Run the existing extraction pipeline first."
            )
        raw = map_frame(pd.read_parquet(path), "origination")
        first_payment = raw["first_payment_date"].astype("string").str.strip()
        month_start = pd.to_datetime(first_payment + "01", format="%Y%m%d", errors="coerce")
        if month_start.isna().any():
            raise ValueError(f"Invalid first_payment_date values in vintage {year}.")
        frames.append(
            pd.DataFrame(
                {
                    "vintage": year,
                    "loan_identifier": raw["loan_identifier"].astype("string"),
                    "prediction_date_proxy": month_start - pd.DateOffset(months=2),
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["vintage", "loan_identifier"]).any():
        raise ValueError("Origination date source contains duplicate loan keys.")
    return result


def _origination_date_proxy(years: list[int]) -> pd.DataFrame:
    normalized_years = tuple(sorted(set(int(year) for year in years)))
    return _origination_date_proxy_cached(normalized_years).copy()


def attach_macro(frame: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    dates = _origination_date_proxy(frame["vintage"].astype(int).unique().tolist())
    merged = frame.merge(
        dates,
        on=["vintage", "loan_identifier"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Some model-eligible loans lack their origination date proxy.")
    merged = merged.drop(columns="_merge")
    return merge_macro_asof(merged, macro)


def _make_augmented_estimator(seed: int, features: list[str], builder) -> Pipeline:
    numeric = [name for name in utils.NUMERIC_COLUMNS if name in features]
    numeric.extend(name for name in MACRO_FEATURES if name in features)
    categorical = [name for name in utils.CATEGORICAL_COLUMNS if name in features]
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    reference_model = builder("xgboost", seed, utils.FEATURE_COLUMNS).named_steps["model"]
    return Pipeline([("preprocessor", preprocessor), ("model", clone(reference_model))])


def _calibration(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    model = LogisticRegression(C=1e6, max_iter=300, solver="lbfgs")
    model.fit(logits.reshape(-1, 1), y_true)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _metrics(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(dtype="int8")
    probability = frame["probability"].to_numpy(dtype=float)
    if np.unique(y).size != 2:
        raise ValueError("Macro experiment evaluation cohort must contain both outcomes.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Macro experiment predictions contain invalid probabilities.")
    intercept, slope = _calibration(y, probability)
    return {
        "n": len(frame),
        "defaults": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "expected_calibration_error": utils.expected_calibration_error(y, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _alignment_audit(frame: pd.DataFrame, split: str, horizon: int) -> list[dict]:
    rows = []
    for feature in MACRO_FEATURES:
        available = frame[feature].notna()
        availability_date = pd.to_datetime(
            frame[f"{feature}_available_date"], errors="coerce"
        )
        prediction_date = pd.to_datetime(frame["prediction_date_proxy"], errors="raise")
        age = pd.to_numeric(frame[f"{feature}_age_days"], errors="coerce")
        rows.append(
            {
                "split": split,
                "horizon_months": horizon,
                "feature": feature,
                "loans": len(frame),
                "available_observations": int(available.sum()),
                "missing_observations": int((~available).sum()),
                "future_availability_violations": int(
                    (availability_date.notna() & availability_date.gt(prediction_date)).sum()
                ),
                "maximum_observed_age_days": float(age.max()) if age.notna().any() else None,
                "prediction_date_proxy": "first_payment_month minus two calendar months",
            }
        )
    return rows


def _metric_vector(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(dtype="int8")
    p = frame["probability"].to_numpy(dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def _weighted_binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    group_inverse: np.ndarray,
    group_count: int,
    weights: np.ndarray,
) -> dict[str, float] | None:
    positives = np.bincount(
        group_inverse, weights=weights * y, minlength=group_count
    )
    totals = np.bincount(group_inverse, weights=weights, minlength=group_count)
    negatives = totals - positives
    positive_total = positives.sum()
    negative_total = negatives.sum()
    if positive_total == 0 or negative_total == 0:
        return None
    negative_before = np.cumsum(negatives) - negatives
    roc_auc = np.sum(positives * (negative_before + 0.5 * negatives)) / (
        positive_total * negative_total
    )
    positive_descending = positives[::-1]
    total_descending = totals[::-1]
    cumulative_positive = np.cumsum(positive_descending)
    cumulative_total = np.cumsum(total_descending)
    average_precision = np.sum(
        (positive_descending / positive_total)
        * np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
    )
    brier = np.dot(weights, np.square(probability - y)) / weights.sum()
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(average_precision),
        "brier_score": float(brier),
    }


def paired_bootstrap(
    predictions: pd.DataFrame, horizon: int, replicates: int, seed: int
) -> pd.DataFrame:
    reference = predictions.loc[
        predictions["model"].eq("baseline") & predictions["horizon_months"].eq(horizon)
    ].copy()
    augmented = predictions.loc[
        predictions["model"].eq("macro_augmented")
        & predictions["horizon_months"].eq(horizon)
    ].copy()
    keys = ["vintage", "loan_identifier"]
    reference = reference.set_index(keys).sort_index()
    augmented = augmented.set_index(keys).sort_index()
    if not reference.index.equals(augmented.index) or not reference["y_true"].equals(augmented["y_true"]):
        raise ValueError("Baseline and macro models must use exactly the same test loans and labels.")

    observed = _metric_vector(augmented)
    baseline_observed = _metric_vector(reference)
    ref_y = reference["y_true"].to_numpy(dtype=np.int8)
    ref_p = reference["probability"].to_numpy(dtype=float)
    aug_y = augmented["y_true"].to_numpy(dtype=np.int8)
    aug_p = augmented["probability"].to_numpy(dtype=float)
    if not np.array_equal(ref_y, aug_y):
        raise ValueError("Paired bootstrap outcomes are not aligned.")
    ref_groups, ref_inverse = np.unique(ref_p, return_inverse=True)
    aug_groups, aug_inverse = np.unique(aug_p, return_inverse=True)
    vintage_positions = [
        np.flatnonzero(reference.index.get_level_values("vintage").to_numpy() == vintage)
        for vintage in sorted(reference.index.get_level_values("vintage").unique())
    ]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in observed}
    rows = []

    for _ in range(replicates):
        weights = np.zeros(len(reference), dtype=np.int32)
        for vintage_index in vintage_positions:
            sampled = rng.integers(0, len(vintage_index), size=len(vintage_index))
            weights[vintage_index] = np.bincount(sampled, minlength=len(vintage_index))
        baseline_values = _weighted_binary_metrics(
            ref_y, ref_p, ref_inverse, len(ref_groups), weights
        )
        macro_values = _weighted_binary_metrics(
            aug_y, aug_p, aug_inverse, len(aug_groups), weights
        )
        if baseline_values is None or macro_values is None:
            continue
        for metric in draws:
            draws[metric].append(macro_values[metric] - baseline_values[metric])
    for metric, values in draws.items():
        difference = observed[metric] - baseline_observed[metric]
        if not values:
            raise ValueError(f"No valid bootstrap replicates for {metric}.")
        rows.append(
            {
                "horizon_months": horizon,
                "metric": metric,
                "baseline_value": baseline_observed[metric],
                "macro_augmented_value": observed[metric],
                "difference_macro_minus_baseline": difference,
                "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)),
                "valid_bootstrap_replicates": len(values),
                "requested_bootstrap_replicates": replicates,
                "seed": seed,
                "interpretation": "paired exploratory interval; not a confirmatory p-value",
            }
        )
    return pd.DataFrame(rows)


def run(
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    shap_sample: int = 500,
    reuse_existing: bool = False,
) -> None:
    require_existing_inputs([OBSERVATIONS_PATH])
    macro = pd.read_csv(OBSERVATIONS_PATH, parse_dates=["observation_date", "available_date"])
    macro = macro.rename(columns={"max_age_days": "max_age_days"})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training = attach_macro(utils.load_training_data(), macro)
    builder = _load_binary_builder()
    prediction_frames = []
    predictions_path = OUTPUT_DIR / "baseline_macro_predictions.parquet"
    expected_model_paths = [
        MODEL_DIR / f"xgboost_{name}_complete_eligible_{horizon}m.joblib"
        for horizon in utils.HORIZONS
        for name in ("baseline", "macro_augmented")
    ]
    if reuse_existing:
        require_existing_inputs([predictions_path, *expected_model_paths])
    cohort_rows = []
    alignment_rows = []
    paired_rows = []
    shap_importance = []
    model_paths = []
    test_frames: dict[int, pd.DataFrame] = {}
    metadata_rows = []
    all_macro = training[MACRO_FEATURES].notna().all(axis=1)
    print(
        f"Training loans with all three dated macro indicators: "
        f"{int(all_macro.sum()):,}/{len(training):,}",
        flush=True,
    )

    for horizon in utils.HORIZONS:
        train = training.loc[
            training[f"complete_horizon_eligible_{horizon}m"].fillna(False)
        ].copy()
        train["y_true"] = train[f"complete_horizon_label_{horizon}m"].astype("int8")
        train_n_before = len(train)
        alignment_rows.extend(_alignment_audit(train, "training", horizon))
        train = train.loc[train[MACRO_FEATURES].notna().all(axis=1)].copy()
        if train["y_true"].nunique() != 2:
            raise ValueError(f"Macro-matched training data do not contain both classes at {horizon}m.")
        validation = attach_macro(utils.load_mature_cohort(utils.VALIDATION_YEARS, horizon), macro)
        test = attach_macro(utils.load_mature_cohort(utils.TEST_YEARS, horizon), macro)
        validation_before, test_before = len(validation), len(test)
        alignment_rows.extend(_alignment_audit(validation, "validation", horizon))
        alignment_rows.extend(_alignment_audit(test, "test", horizon))
        validation = validation.loc[validation[MACRO_FEATURES].notna().all(axis=1)].copy()
        test = test.loc[test[MACRO_FEATURES].notna().all(axis=1)].copy()
        if test.empty or validation.empty:
            raise ValueError(f"No fully macro-observed validation/test loans at {horizon}m.")
        if validation["y_true"].nunique() != 2 or test["y_true"].nunique() != 2:
            raise ValueError(f"Macro-matched validation/test lacks both classes at {horizon}m.")
        test_frames[horizon] = test
        cohort_rows.append(
            {
                "horizon_months": horizon,
                "split": "training",
                "rows_before_macro_availability_filter": train_n_before,
                "rows_after_filter": len(train),
                "rows_excluded": train_n_before - len(train),
                "prediction_date_proxy": "first_payment_month minus two calendar months",
            }
        )
        cohort_rows.extend(
            [
                {
                    "horizon_months": horizon,
                    "split": "validation",
                    "rows_before_macro_availability_filter": validation_before,
                    "rows_after_filter": len(validation),
                    "rows_excluded": validation_before - len(validation),
                    "prediction_date_proxy": "first_payment_month minus two calendar months",
                },
                {
                    "horizon_months": horizon,
                    "split": "test",
                    "rows_before_macro_availability_filter": test_before,
                    "rows_after_filter": len(test),
                    "rows_excluded": test_before - len(test),
                    "prediction_date_proxy": "first_payment_month minus two calendar months",
                },
            ]
        )
        active = utils.active_feature_columns(train)
        augmented_features = active + MACRO_FEATURES
        model_paths.extend(
            str((MODEL_DIR / f"xgboost_{name}_complete_eligible_{horizon}m.joblib").relative_to(
                utils.PROJECT_ROOT
            ))
            for name in ("baseline", "macro_augmented")
        )
        if not reuse_existing:
            baseline_model = builder("xgboost", SEED + horizon, active)
            augmented_model = _make_augmented_estimator(
                SEED + horizon, augmented_features, builder
            )
            y_train = train["y_true"].to_numpy(dtype="int8")
            baseline_model.fit(train[active], y_train)
            augmented_model.fit(train[augmented_features], y_train)
            for name, estimator, features in (
                ("baseline", baseline_model, active),
                ("macro_augmented", augmented_model, augmented_features),
            ):
                model_path = MODEL_DIR / f"xgboost_{name}_complete_eligible_{horizon}m.joblib"
                joblib.dump(estimator, model_path, compress=3)
                for split, frame in (("validation", validation), ("test", test)):
                    probabilities = estimator.predict_proba(frame[features])[:, 1]
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "split": split,
                                "model": name,
                                "horizon_months": horizon,
                                "vintage": frame["vintage"].to_numpy(),
                                "loan_identifier": frame["loan_identifier"].to_numpy(),
                                "y_true": frame["y_true"].to_numpy(dtype="int8"),
                                "probability": probabilities,
                            }
                        )
                    )
        metadata_rows.append(
            {
                "horizon_months": horizon,
                "training_count": len(train),
                "training_defaults": int(train["y_true"].sum()),
                "feature_count_baseline": len(active),
                "feature_count_augmented": len(augmented_features),
                "random_seed": SEED + horizon,
                "xgboost_settings": {
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "tree_method": "hist",
                },
            }
        )
        print(
            f"Fitted matched baseline and macro models at {horizon}m "
            f"(train={len(train):,}, validation={len(validation):,}, test={len(test):,}).",
            flush=True,
        )

    if reuse_existing:
        predictions = pd.read_parquet(predictions_path)
    else:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_parquet(predictions_path, index=False)
    required_prediction_columns = {
        "split",
        "model",
        "horizon_months",
        "vintage",
        "loan_identifier",
        "y_true",
        "probability",
    }
    if missing := required_prediction_columns.difference(predictions.columns):
        raise ValueError(f"Saved macro predictions are missing columns: {sorted(missing)}")
    for horizon, test in test_frames.items():
        expected_keys = pd.MultiIndex.from_frame(test[["vintage", "loan_identifier"]])
        for model_name in ("baseline", "macro_augmented"):
            observed = predictions.loc[
                predictions["split"].eq("test")
                & predictions["horizon_months"].eq(horizon)
                & predictions["model"].eq(model_name)
            ]
            keys = pd.MultiIndex.from_frame(observed[["vintage", "loan_identifier"]])
            if not keys.sort_values().equals(expected_keys.sort_values()):
                raise ValueError(
                    f"Saved {model_name}/{horizon}m predictions do not match the current "
                    "macro-available test cohort."
                )
    pd.DataFrame(cohort_rows).to_csv(OUTPUT_DIR / "matched_cohort_report.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(
        OUTPUT_DIR / "macro_date_alignment_validation.csv", index=False
    )

    metric_rows = []
    for (split, horizon, model), frame in predictions.groupby(
        ["split", "horizon_months", "model"], sort=True
    ):
        metric_rows.append(
            {"split": split, "horizon_months": horizon, "model": model, **_metrics(frame)}
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT_DIR / "baseline_vs_macro_metrics.csv", index=False)
    for horizon in utils.HORIZONS:
        paired_rows.append(
            paired_bootstrap(
                predictions.loc[predictions["split"].eq("test")],
                horizon,
                bootstrap_replicates,
                SEED + horizon,
            )
        )
    intervals = pd.concat(paired_rows, ignore_index=True)
    intervals.to_csv(OUTPUT_DIR / "paired_performance_differences.csv", index=False)

    # Explain macro-augmented models on a fixed sample from the common matched test cohort.
    explainer_path = PROJECT_ROOT / "16_explainability_and_profiles.py"
    spec = importlib.util.spec_from_file_location("extension_explainability", explainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load explainability utilities from {explainer_path}.")
    explainability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(explainability)
    shap_module = explainability._load_shap()
    for horizon, test in test_frames.items():
        sample = test.sample(n=min(shap_sample, len(test)), random_state=SEED + horizon)
        estimator = joblib.load(
            MODEL_DIR / f"xgboost_macro_augmented_complete_eligible_{horizon}m.joblib"
        )
        (
            transformed,
            family,
            _,
            _,
            _,
            additivity_error,
            _,
        ) = explainability._explain(estimator, sample, shap_module)
        features = list(estimator.named_steps["preprocessor"].feature_names_in_)
        shap_importance.append(
            pd.DataFrame(
                {
                    "horizon_months": horizon,
                    "feature": features,
                    "mean_abs_shap": np.mean(np.abs(family), axis=0),
                    "mean_signed_shap": np.mean(family, axis=0),
                    "sample_size": len(sample),
                    "shap_output_scale": "raw XGBoost margin (log-odds)",
                    "model": "macro_augmented",
                    "max_additivity_error": additivity_error,
                }
            )
        )
    shap_table = pd.concat(shap_importance, ignore_index=True)
    shap_table.to_csv(OUTPUT_DIR / "macro_augmented_shap_importance.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric in zip(axes, ("roc_auc", "pr_auc", "brier_score")):
        subset = intervals.loc[intervals["metric"].eq(metric)].sort_values("horizon_months")
        y = subset["difference_macro_minus_baseline"].to_numpy()
        lower = y - subset["ci_95_lower"].to_numpy()
        upper = subset["ci_95_upper"].to_numpy() - y
        axis.errorbar(
            subset["horizon_months"],
            y,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            capsize=4,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Prediction horizon (months)")
        axis.set_ylabel("Macro minus baseline")
    test_sizes = [
        int(
            predictions.loc[
                predictions["split"].eq("test")
                & predictions["horizon_months"].eq(horizon)
                & predictions["model"].eq("baseline"),
                "loan_identifier",
            ].nunique()
        )
        for horizon in utils.HORIZONS
    ]
    figure.suptitle(
        "Paired performance differences on common exploratory test loans "
        f"(n={test_sizes[0]:,}/{test_sizes[1]:,}/{test_sizes[2]:,} at 12/24/36m)"
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "baseline_vs_macro_performance_differences.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    report_rows = []
    for horizon in utils.HORIZONS:
        for metric in ("roc_auc", "pr_auc", "brier_score"):
            row = intervals.loc[
                intervals["horizon_months"].eq(horizon) & intervals["metric"].eq(metric)
            ].iloc[0]
            report_rows.append(
                f"| {horizon} | {metric} | {row['baseline_value']:.6f} | "
                f"{row['macro_augmented_value']:.6f} | "
                f"{row['difference_macro_minus_baseline']:.6f} "
                f"[{row['ci_95_lower']:.6f}, {row['ci_95_upper']:.6f}] |"
            )
    report = [
        "# Macro-augmented pilot comparison",
        "",
        "This is an isolated, exploratory model branch; frozen baseline artifacts are not overwritten.",
        "Baseline and augmented XGBoost models use the same complete-eligible label strategy,",
        "training/validation/test vintages, hyperparameters, seed, and macro-complete cohort.",
        "The 2017–2019 test vintages were previously examined, so these comparisons are not confirmatory.",
        "",
        "The exact origination date is unavailable in Release 47. The experiment uses first-payment",
        "month minus two calendar months as an approximate decision-date proxy. Macro release dates",
        "are conservatively modeled with fixed lags (UNRATE 7 days after month end, real GDP 45 days",
        "after quarter end, and FHFA HPI 90 days after quarter end). FRED current-vintage values can",
        "include later revisions; this is not a real-time-vintage backtest. All measures are national.",
        "",
        "| Horizon | Metric | Baseline | Macro augmented | Difference (paired 95% interval) |",
        "|---:|---|---:|---:|---:|",
        *report_rows,
        "",
        "Differences are empirical and should be judged on practical size, paired interval,",
        "calibration, and the documented timing limitations. SHAP values are model associations,",
        "not causal effects. See `macro_augmented_shap_importance.csv` for the macro indicator",
        "contributions and `matched_cohort_report.csv` for sample exclusions.",
    ]
    (OUTPUT_DIR / "macro_experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata = {
        "status": "completed",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_preserved": True,
        "design": {
            "training_vintages": utils.TRAIN_YEARS,
            "validation_vintages": utils.VALIDATION_YEARS,
            "test_vintages": utils.TEST_YEARS,
            "horizons_months": utils.HORIZONS,
            "label_strategy": "complete_eligible",
            "model_family": "XGBoost",
            "cohort": "identical rows with all three macro indicators available for each split and horizon",
            "threshold_tuning": "none; no test-tuned thresholds or hyperparameters",
        },
        "macro_series": ["UNRATE", "GDPC1", "USSTHPI"],
        "macro_manifest": "data/macro/macro_data_manifest.json",
        "date_alignment": "first-payment month minus two months; fixed modeled release lags; backward as-of join; per-series maximum age; no future available_date",
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed_by_horizon": {horizon: SEED + horizon for horizon in utils.HORIZONS},
            "method": "paired loan resampling within observed vintage",
        },
        "shap_sample_per_horizon": shap_sample,
        "reused_saved_models_and_predictions": reuse_existing,
        "models": metadata_rows,
        "artifacts": {
            "predictions": str(predictions_path.relative_to(utils.PROJECT_ROOT)),
            "models": model_paths,
            "metrics": "results/macro_experiment/baseline_vs_macro_metrics.csv",
            "date_alignment_audit": "results/macro_experiment/macro_date_alignment_validation.csv",
        },
        "python_version": platform.python_version(),
        "package_versions": {
            "xgboost": importlib.metadata.version("xgboost"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "shap": importlib.metadata.version("shap"),
        },
        "limitations": [
            "No exact origination date field exists; first-payment month is only a proxy.",
            "Fixed release delays are conservative approximations, not archived release timestamps.",
            "FRED latest values are revised historical data, not real-time vintages.",
            "National macro variables do not encode state/property-local conditions.",
            "Existing 2017-2019 results have prior exposure and are exploratory.",
        ],
    }
    (OUTPUT_DIR / "macro_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Macro comparison and report saved under {OUTPUT_DIR}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--shap-sample", type=int, default=500)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse isolated macro model artifacts and predictions if already present.",
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 1 or args.shap_sample < 1:
        raise ValueError("Bootstrap replicates and SHAP sample must be positive.")
    run(args.bootstrap_replicates, args.shap_sample, args.reuse_existing)


if __name__ == "__main__":
    main()
