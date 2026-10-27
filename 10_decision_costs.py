"""Select decision thresholds on validation data and report costs on test data."""

from __future__ import annotations

import numpy as np
import pandas as pd

import pilot_model_utils as utils


COST_SCENARIOS = utils.pilot_settings()["decision_cost_scenarios"]


def _best_threshold(
    validation: pd.DataFrame, missed_default: float, review: float
) -> tuple[float, float]:
    y_true = validation["y_true"].to_numpy(dtype="int8")
    probabilities = validation["probability"].to_numpy(dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot select a decision threshold on an empty validation cohort.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Validation probabilities must be finite and in [0, 1].")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Decision-cost labels must be binary.")
    if missed_default < 0 or review < 0:
        raise ValueError("Decision costs must be nonnegative.")

    values, counts = np.unique(probabilities, return_counts=True)
    positive_counts = np.bincount(
        np.searchsorted(values, probabilities), weights=y_true, minlength=len(values)
    ).astype(np.int64)
    flagged_count = np.cumsum(counts[::-1])
    flagged_defaults = np.cumsum(positive_counts[::-1])
    thresholds = np.concatenate(
        (values[::-1], [np.nextafter(probabilities.max(), np.inf)])
    )
    missed_defaults = int(y_true.sum()) - np.concatenate((flagged_defaults, [0]))
    flagged_count = np.concatenate((flagged_count, [0]))
    costs = (
        missed_defaults * missed_default + flagged_count * review
    ) / len(y_true)
    best_cost = float(costs.min())
    best_threshold = float(thresholds[np.flatnonzero(costs == best_cost)[-1]])
    return best_threshold, best_cost


def calculate_decision_costs() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [
        utils.PREDICTION_DIR / "binary_pilot_predictions.parquet",
        utils.PREDICTION_DIR / "survival_pilot_predictions.parquet",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot prediction files: {missing}")
    predictions = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    group_cols = ["model", "strategy", "horizon_months"]
    threshold_rows = []
    cost_rows = []
    for key, validation in predictions.loc[predictions["split"].eq("validation")].groupby(
        group_cols, sort=True
    ):
        model, strategy, horizon = key
        test_mask = (
            predictions["split"].eq("test")
            & predictions["model"].eq(model)
            & predictions["strategy"].eq(strategy)
            & predictions["horizon_months"].eq(horizon)
        )
        test = predictions.loc[test_mask]
        if test.empty:
            raise ValueError(f"No common test predictions for {key}.")
        for scenario, parameters in COST_SCENARIOS.items():
            threshold, validation_cost = _best_threshold(
                validation,
                parameters["missed_default_cost"],
                parameters["review_cost_per_flag"],
            )
            y_test = test["y_true"].to_numpy(dtype="int8")
            probability_test = test["probability"].to_numpy(dtype=float)
            flagged = probability_test >= threshold
            false_negative = (y_test == 1) & ~flagged
            true_positive_flags = (y_test == 1) & flagged
            false_positive_flags = (y_test == 0) & flagged
            test_cost = utils.cost_value(
                y_test,
                probability_test,
                threshold,
                parameters["missed_default_cost"],
                parameters["review_cost_per_flag"],
            )
            threshold_rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "horizon_months": int(horizon),
                    "cost_scenario": scenario,
                    "threshold": threshold,
                    "validation_mean_cost": validation_cost,
                    "validation_n": len(validation),
                    "validation_flagged": int(
                        (validation["probability"].to_numpy(dtype=float) >= threshold).sum()
                    ),
                    "validation_review_rate": float(
                        (validation["probability"].to_numpy(dtype=float) >= threshold).mean()
                    ),
                    "validation_no_review_action": bool(
                        (validation["probability"].to_numpy(dtype=float) >= threshold).sum() == 0
                    ),
                }
            )
            cost_rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "horizon_months": int(horizon),
                    "cost_scenario": scenario,
                    "threshold": threshold,
                    "test_n": len(test),
                    "test_defaults": int(y_test.sum()),
                    "test_flagged": int(flagged.sum()),
                    "test_flagged_defaults": int(true_positive_flags.sum()),
                    "test_false_positive_flags": int(false_positive_flags.sum()),
                    "test_missed_defaults": int(false_negative.sum()),
                    "test_mean_cost": test_cost,
                    "test_total_normalized_cost": test_cost * len(test),
                }
            )
    thresholds = pd.DataFrame(threshold_rows)
    costs = pd.DataFrame(cost_rows)
    thresholds.to_csv(utils.TABLE_DIR / "validation_selected_thresholds.csv", index=False)
    costs.to_csv(utils.TABLE_DIR / "test_decision_costs.csv", index=False)
    print(costs.to_string(index=False))
    return thresholds, costs


if __name__ == "__main__":
    calculate_decision_costs()
