"""Fit pilot Logistic Regression and XGBoost models for the frozen label strategies."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import pilot_model_utils as utils


def build_estimator(model_name: str, seed: int, feature_columns: list[str]) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    elif model_name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown binary model: {model_name}")
    return Pipeline(
        [
            (
                "preprocessor",
                utils.make_preprocessor(
                    scale_numeric=model_name == "logistic_regression",
                    feature_columns=feature_columns,
                ),
            ),
            ("model", estimator),
        ]
    )


def _predict_rows(
    estimator: Pipeline,
    frame: pd.DataFrame,
    split_name: str,
    model_name: str,
    strategy: str,
    horizon: int,
    training_count: int,
) -> pd.DataFrame:
    probabilities = estimator.predict_proba(frame[utils.FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame(
        {
            "split": split_name,
            "model": model_name,
            "strategy": strategy,
            "horizon_months": horizon,
            "vintage": frame["vintage"].to_numpy(),
            "loan_identifier": frame["loan_identifier"].to_numpy(),
            "y_true": frame["y_true"].to_numpy(dtype="int8"),
            "probability": probabilities,
            "training_count": training_count,
        }
    )


def train_pilot() -> pd.DataFrame:
    utils.ensure_output_directories()
    settings = utils.pilot_settings()
    (utils.MODEL_DIR / "pilot_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    training = utils.load_training_data()
    predictions: list[pd.DataFrame] = []
    balance_frames = []
    cohort_rows = []
    model_feature_transforms = {}

    for horizon in utils.HORIZONS:
        naive_column = f"naive_label_{horizon}m"
        eligible_column = f"complete_horizon_eligible_{horizon}m"
        complete_column = f"complete_horizon_label_{horizon}m"
        full = training.copy()
        full["y_true"] = full[naive_column].astype("int8")
        complete = training.loc[training[eligible_column].fillna(False)].copy()
        if complete.empty:
            raise ValueError(f"No complete-horizon training records for {horizon} months.")
        complete["y_true"] = complete[complete_column].astype("int8")
        matched, balance = utils.matched_naive_sample(
            full,
            complete,
            seed=utils.SEED + horizon,
        )
        matched["y_true"] = matched[naive_column].astype("int8")
        balance["horizon_months"] = horizon
        balance_frames.append(balance)

        validation = utils.load_mature_cohort(utils.VALIDATION_YEARS, horizon)
        test = utils.load_mature_cohort(utils.TEST_YEARS, horizon)
        for model_name in ("logistic_regression", "xgboost"):
            for strategy, fit_frame in (
                ("naive_full", full),
                ("complete_eligible", complete),
                ("matched_naive", matched),
            ):
                active_features = utils.active_feature_columns(fit_frame)
                y_train = fit_frame["y_true"].to_numpy(dtype="int8")
                if np.unique(y_train).size != 2:
                    raise ValueError(
                        f"{model_name}/{strategy}/{horizon}m training labels contain one class."
                    )
                estimator = build_estimator(model_name, utils.SEED + horizon, active_features)
                print(
                    f"Fitting {model_name} {strategy} {horizon}m: "
                    f"{len(fit_frame):,} loans, {int(y_train.sum()):,} defaults",
                    flush=True,
                )
                estimator.fit(fit_frame[utils.FEATURE_COLUMNS], y_train)
                model_feature_transforms[f"{model_name}:{strategy}:{horizon}m"] = {
                    "active_origination_features": active_features,
                    "encoded_columns": list(
                        estimator.named_steps["preprocessor"].get_feature_names_out()
                    ),
                }
                model_path = (
                    utils.MODEL_DIR
                    / f"{model_name}_{strategy}_{horizon}m.joblib"
                )
                joblib.dump(estimator, model_path, compress=3)
                predictions.append(
                    _predict_rows(
                        estimator,
                        validation,
                        "validation",
                        model_name,
                        strategy,
                        horizon,
                        len(fit_frame),
                    )
                )
                predictions.append(
                    _predict_rows(
                        estimator,
                        test,
                        "test",
                        model_name,
                        strategy,
                        horizon,
                        len(fit_frame),
                    )
                )
                del estimator
        for strategy, fit_frame in (
            ("naive_full", full),
            ("complete_eligible", complete),
            ("matched_naive", matched),
        ):
            for vintage, group in fit_frame.groupby("vintage"):
                cohort_rows.append(
                    {
                        "horizon_months": horizon,
                        "strategy": strategy,
                        "vintage": int(vintage),
                        "training_count": len(group),
                        "default_count": int(group["y_true"].sum()),
                        "default_rate": float(group["y_true"].mean()) if len(group) else None,
                    }
                )
        print(f"Completed binary horizon {horizon}m.", flush=True)

    settings["active_feature_whitelist"] = utils.active_feature_columns(training)
    settings["features_missing_for_all_training_loans"] = sorted(
        set(utils.FEATURE_COLUMNS).difference(settings["active_feature_whitelist"])
    )
    settings["model_feature_transforms"] = model_feature_transforms
    (utils.MODEL_DIR / "pilot_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    result = pd.concat(predictions, ignore_index=True)
    output_path = utils.PREDICTION_DIR / "binary_pilot_predictions.parquet"
    result.to_parquet(output_path, index=False)
    pd.concat(balance_frames, ignore_index=True).to_csv(
        utils.TABLE_DIR / "matched_naive_balance.csv", index=False
    )
    pd.DataFrame(cohort_rows).to_csv(
        utils.TABLE_DIR / "binary_training_cohort_summary.csv", index=False
    )
    print(f"Saved {len(result):,} validation/test predictions to {output_path}")
    return result


if __name__ == "__main__":
    train_pilot()
