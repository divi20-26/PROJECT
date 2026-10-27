"""Fit delayed-entry cause-specific Cox models and predict default incidence."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import pilot_model_utils as utils


MAX_MONTH = 36
CAUSES = {"default": 1, "voluntary_payoff": 2}


def prepare_survival_training() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    training = utils.load_training_data()
    eligible_column = f"survival_eligible_{MAX_MONTH}m"
    required = {
        eligible_column,
        f"survival_entry_month_{MAX_MONTH}m",
        f"survival_exit_month_{MAX_MONTH}m",
        f"survival_event_code_{MAX_MONTH}m",
    }
    missing = required.difference(training.columns)
    if missing:
        raise ValueError(f"Training labels lack delayed-entry fields: {sorted(missing)}")
    eligible = training.loc[training[eligible_column].fillna(False)].copy()
    eligible["cox_entry"], eligible["cox_exit"] = utils.cox_time_bounds(
        eligible[f"survival_entry_month_{MAX_MONTH}m"],
        eligible[f"survival_exit_month_{MAX_MONTH}m"],
    )
    eligible["observed_event_code"] = pd.to_numeric(
        eligible[f"survival_event_code_{MAX_MONTH}m"], errors="raise"
    ).astype(int)
    eligible = eligible.loc[eligible["cox_exit"] > eligible["cox_entry"]].copy()
    if eligible.empty or not eligible["observed_event_code"].isin([0, 1, 2]).all():
        raise ValueError("Invalid or empty delayed-entry survival training sample.")
    active_features = utils.active_feature_columns(eligible)
    preprocessor = utils.make_preprocessor(
        scale_numeric=True,
        feature_columns=active_features,
        categorical_min_frequency=100,
    )
    matrix = preprocessor.fit_transform(eligible[utils.FEATURE_COLUMNS])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    names = preprocessor.get_feature_names_out()
    encoded = pd.DataFrame(matrix, columns=names, index=eligible.index)
    # Rare one-hot levels provide too little risk-set information for stable Cox estimates.
    varying = encoded.nunique(dropna=False) > 1
    category_counts = encoded.sum(axis=0)
    minority_counts = np.minimum(category_counts, len(encoded) - category_counts)
    rare_indicator = encoded.columns.str.startswith("categorical__") & (
        minority_counts < 100
    )
    encoded = encoded.loc[:, varying & ~rare_indicator]
    metadata = eligible[["cox_entry", "cox_exit", "observed_event_code"]].reset_index(drop=True)
    encoded = encoded.reset_index(drop=True)
    return encoded, metadata, preprocessor


def fit_cause_specific_models(
    encoded: pd.DataFrame, metadata: pd.DataFrame
) -> dict[str, CoxPHFitter]:
    models: dict[str, CoxPHFitter] = {}
    for cause_name, cause_code in CAUSES.items():
        fit_data = encoded.copy()
        fit_data["cox_entry"] = metadata["cox_entry"]
        fit_data["cox_exit"] = metadata["cox_exit"]
        fit_data["event"] = metadata["observed_event_code"].eq(cause_code).astype("int8")
        if int(fit_data["event"].sum()) == 0:
            raise ValueError(f"No {cause_name} events in survival training data.")
        model = CoxPHFitter(penalizer=0.1)
        model.fit(
            fit_data,
            duration_col="cox_exit",
            event_col="event",
            entry_col="cox_entry",
            batch_mode=True,
            show_progress=False,
        )
        models[cause_name] = model
        print(
            f"Fitted cause-specific Cox for {cause_name}: "
            f"{len(fit_data):,} loans, {int(fit_data['event'].sum()):,} events",
            flush=True,
        )
    return models


def _baseline_cumulative_hazard(model: CoxPHFitter, timeline: np.ndarray) -> np.ndarray:
    baseline = model.baseline_cumulative_hazard_.iloc[:, 0]
    x = baseline.index.to_numpy(dtype=float)
    y = baseline.to_numpy(dtype=float)
    if (
        len(x) == 0
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or (np.diff(x) <= 0).any()
        or (y < 0).any()
        or (np.diff(y) < -1e-12).any()
    ):
        raise ValueError("Cox baseline cumulative hazard must be finite, nonnegative, and monotone.")
    return np.interp(timeline, x, y, left=0.0, right=float(y[-1]))


def predict_default_cumulative_incidence(
    features: pd.DataFrame,
    preprocessor: object,
    models: dict[str, CoxPHFitter],
    max_month: int = MAX_MONTH,
    batch_size: int = 20_000,
) -> np.ndarray:
    """Return loan-by-month cumulative incidence from the two cause-specific hazards."""
    if set(models) != set(CAUSES):
        raise ValueError(f"Expected cause-specific Cox models for {sorted(CAUSES)}.")
    if max_month < 1 or batch_size < 1:
        raise ValueError("Prediction horizon and batch size must be positive.")
    timeline = np.arange(1, max_month + 1, dtype=float)
    baseline = {
        cause: _baseline_cumulative_hazard(model, timeline)
        for cause, model in models.items()
    }
    predictions = np.empty((len(features), max_month), dtype=float)
    feature_matrix = preprocessor.transform(features[utils.FEATURE_COLUMNS])
    feature_names = list(models["default"].params_.index)

    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        block = feature_matrix[start:stop]
        if hasattr(block, "toarray"):
            block = block.toarray()
        encoded = pd.DataFrame(block, columns=preprocessor.get_feature_names_out())
        encoded = encoded.loc[:, feature_names]
        relative_risk = {}
        for cause, model in models.items():
            relative_risk[cause] = model.predict_partial_hazard(encoded).to_numpy(dtype=float)

        cumulative_default = relative_risk["default"][:, None] * baseline["default"][None, :]
        cumulative_payoff = relative_risk["voluntary_payoff"][:, None] * baseline[
            "voluntary_payoff"
        ][None, :]
        delta_default = np.maximum(np.diff(cumulative_default, axis=1, prepend=0.0), 0.0)
        delta_payoff = np.maximum(np.diff(cumulative_payoff, axis=1, prepend=0.0), 0.0)
        delta_total = delta_default + delta_payoff
        interval_event_probability = -np.expm1(-delta_total)
        default_share = np.divide(
            delta_default,
            delta_total,
            out=np.zeros_like(delta_default),
            where=delta_total > 0,
        )
        survival_before = np.exp(
            -np.cumsum(delta_total, axis=1) + delta_total
        )
        incidence = np.cumsum(
            survival_before * interval_event_probability * default_share, axis=1
        )
        if (
            not np.isfinite(incidence).all()
            or (incidence < -1e-10).any()
            or (incidence > 1 + 1e-10).any()
            or (np.diff(incidence, axis=1) < -1e-10).any()
        ):
            raise RuntimeError("Predicted default cumulative incidence violates probability bounds.")
        predictions[start:stop] = np.clip(incidence, 0.0, 1.0)
    return predictions


def train_survival_pilot() -> pd.DataFrame:
    utils.ensure_output_directories()
    settings_path = utils.MODEL_DIR / "pilot_settings.json"
    settings = utils.pilot_settings()
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        settings.update(existing)
    encoded, metadata, preprocessor = prepare_survival_training()
    settings["survival_encoded_columns"] = list(encoded.columns)
    settings["survival_delayed_entry_rows"] = int((metadata["cox_entry"] > 0).sum())
    settings["survival_training_rows"] = len(metadata)
    (utils.MODEL_DIR / "pilot_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    training = utils.load_training_data()
    summary = pd.DataFrame(
        [
            {
                "training_label_rows": len(training),
                "survival_eligible_rows": len(metadata),
                "survival_excluded_rows": len(training) - len(metadata),
                "delayed_entry_rows": int((metadata["cox_entry"] > 0).sum()),
                "right_censored_or_other_termination_rows": int(
                    (metadata["observed_event_code"] == 0).sum()
                ),
                "default_events": int((metadata["observed_event_code"] == 1).sum()),
                "voluntary_payoff_events": int((metadata["observed_event_code"] == 2).sum()),
                "survival_exclusion_rate": (len(training) - len(metadata)) / len(training),
                "delayed_entry_rate_of_risk_set": float((metadata["cox_entry"] > 0).mean()),
                "censoring_rate_of_risk_set": float(
                    (metadata["observed_event_code"] == 0).mean()
                ),
            }
        ]
    )
    summary.to_csv(utils.TABLE_DIR / "survival_training_summary.csv", index=False)
    models = fit_cause_specific_models(encoded, metadata)
    artifact = {
        "preprocessor": preprocessor,
        "models": models,
        "encoded_feature_names": list(models["default"].params_.index),
        "max_month": MAX_MONTH,
    }
    joblib.dump(artifact, utils.MODEL_DIR / "cause_specific_cox_36m.joblib", compress=3)

    predictions: list[pd.DataFrame] = []
    for split_name, years in (
        ("validation", utils.VALIDATION_YEARS),
        ("test", utils.TEST_YEARS),
    ):
        features = utils.load_features(years)
        incidence = predict_default_cumulative_incidence(
            features,
            preprocessor,
            models,
            max_month=MAX_MONTH,
        )
        for horizon in utils.HORIZONS:
            mature = utils.load_mature_cohort(years, horizon)
            index = pd.MultiIndex.from_frame(features[["vintage", "loan_identifier"]])
            risk_series = pd.Series(
                incidence[:, horizon - 1],
                index=index,
                name="probability",
            )
            mature_index = pd.MultiIndex.from_frame(mature[["vintage", "loan_identifier"]])
            probabilities = risk_series.reindex(mature_index)
            if probabilities.isna().any():
                raise ValueError("Survival predictions failed to cover the common mature cohort.")
            predictions.append(
                pd.DataFrame(
                    {
                        "split": split_name,
                        "model": "cause_specific_cox",
                        "strategy": "delayed_entry_competing_risk",
                        "horizon_months": horizon,
                        "vintage": mature["vintage"].to_numpy(),
                        "loan_identifier": mature["loan_identifier"].to_numpy(),
                        "y_true": mature["y_true"].to_numpy(dtype="int8"),
                        "probability": probabilities.to_numpy(),
                        "training_count": len(metadata),
                    }
                )
            )
        print(f"Generated survival predictions for {split_name}.", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    output_path = utils.PREDICTION_DIR / "survival_pilot_predictions.parquet"
    result.to_parquet(output_path, index=False)
    print(f"Saved {len(result):,} survival predictions to {output_path}")
    return result


if __name__ == "__main__":
    train_survival_pilot()
