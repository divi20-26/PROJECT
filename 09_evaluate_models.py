"""Evaluate all pilot predictions on the shared mature validation and test cohorts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

import pilot_model_utils as utils


def calibration_summary(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    calibration = LogisticRegression(C=1e6, max_iter=300, solver="lbfgs")
    calibration.fit(logits.reshape(-1, 1), y_true)
    return float(calibration.intercept_[0]), float(calibration.coef_[0, 0])


def evaluate_group(group: pd.DataFrame) -> dict:
    y_true = group["y_true"].to_numpy(dtype="int8")
    probability = group["probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Predictions contain nonfinite or out-of-range probabilities.")
    if np.unique(y_true).size != 2:
        raise ValueError("Evaluation cohort must include both event and non-event outcomes.")
    intercept, slope = calibration_summary(y_true, probability)
    return {
        "n": len(group),
        "defaults": int(y_true.sum()),
        "default_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "expected_calibration_error": utils.expected_calibration_error(y_true, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _validate_common_cohorts(predictions: pd.DataFrame) -> None:
    keys = ["split", "horizon_months"]
    for (split, horizon), group in predictions.groupby(keys):
        reference = None
        reference_label = None
        for (_, model, strategy), candidate in group.groupby(["split", "model", "strategy"]):
            indexed = candidate.set_index(["vintage", "loan_identifier"])["y_true"].sort_index()
            if indexed.index.has_duplicates:
                raise ValueError(f"Duplicate prediction keys for {split}/{horizon}/{model}/{strategy}.")
            if reference is None:
                reference, reference_label = indexed.index, indexed
            elif not indexed.index.equals(reference) or not indexed.equals(reference_label):
                raise ValueError(
                    f"Pilot approaches do not share identical {split} {horizon}m test labels."
                )


def evaluate_predictions() -> pd.DataFrame:
    paths = [
        utils.PREDICTION_DIR / "binary_pilot_predictions.parquet",
        utils.PREDICTION_DIR / "survival_pilot_predictions.parquet",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot prediction files: {missing}")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    _validate_common_cohorts(predictions)

    rows = []
    group_cols = ["split", "horizon_months", "model", "strategy"]
    for key, group in predictions.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row.update(evaluate_group(group))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(utils.TABLE_DIR / "pilot_metrics.csv", index=False)
    test_metrics = metrics.loc[metrics["split"].eq("test")].copy()
    test_metrics["roc_auc_rank"] = test_metrics.groupby("horizon_months")["roc_auc"].rank(
        ascending=False, method="min"
    )
    test_metrics["brier_rank"] = test_metrics.groupby("horizon_months")["brier_score"].rank(
        ascending=True, method="min"
    )
    test_metrics.to_csv(utils.TABLE_DIR / "pilot_model_rankings.csv", index=False)
    survival_metrics = test_metrics.loc[
        test_metrics["model"].eq("cause_specific_cox")
    ].copy()
    survival_metrics["evaluation_target"] = (
        "default by horizon; voluntary payoff is a competing non-default outcome; "
        "early other termination is excluded; acquisition entry month must equal 1"
    )
    survival_metrics.to_csv(utils.TABLE_DIR / "survival_evaluation.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Saved metrics and rankings in {utils.TABLE_DIR}")
    return metrics


if __name__ == "__main__":
    evaluate_predictions()
