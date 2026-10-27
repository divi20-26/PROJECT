"""Quantify matched-naive sample variation across fixed sampling seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd

import pilot_model_utils as utils


REPEATS = 5


def _balance_rows(
    matched: pd.DataFrame,
    complete: pd.DataFrame,
    horizon: int,
    seed: int,
) -> list[dict]:
    rows = []
    for feature in utils.FEATURE_COLUMNS:
        missingness_difference = abs(
            float(matched[feature].isna().mean() - complete[feature].isna().mean())
        )
        if matched[feature].notna().sum() == 0 and complete[feature].notna().sum() == 0:
            rows.append(
                {
                    "horizon_months": horizon,
                    "sampling_seed": seed,
                    "feature": feature,
                    "diagnostic": "not_observed_in_training",
                    "difference": np.nan,
                    "missingness_rate_difference": missingness_difference,
                }
            )
            continue
        if feature in utils.NUMERIC_COLUMNS:
            matched_values = pd.to_numeric(matched[feature], errors="coerce")
            complete_values = pd.to_numeric(complete[feature], errors="coerce")
            matched_mean = float(matched_values.mean())
            complete_mean = float(complete_values.mean())
            matched_var = float(matched_values.var(ddof=1))
            complete_var = float(complete_values.var(ddof=1))
            pooled_sd = np.sqrt((matched_var + complete_var) / 2)
            difference = (
                (matched_mean - complete_mean) / pooled_sd if pooled_sd > 0 else np.nan
            )
            rows.append(
                {
                    "horizon_months": horizon,
                    "sampling_seed": seed,
                    "feature": feature,
                    "diagnostic": "standardized_mean_difference",
                    "difference": difference,
                    "missingness_rate_difference": missingness_difference,
                }
            )
            continue

        matched_values = matched[feature].astype("string").fillna("__MISSING__")
        complete_values = complete[feature].astype("string").fillna("__MISSING__")
        categories = set(matched_values.unique()) | set(complete_values.unique())
        proportion_difference = max(
            abs(float(matched_values.eq(category).mean() - complete_values.eq(category).mean()))
            for category in categories
        )
        rows.append(
            {
                "horizon_months": horizon,
                "sampling_seed": seed,
                "feature": feature,
                "diagnostic": "maximum_absolute_category_proportion_difference",
                "difference": proportion_difference,
                "missingness_rate_difference": missingness_difference,
            }
        )
    return rows


def run_repeated_matching() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = utils.load_training_data()
    balance_rows: list[dict] = []
    sample_rows: list[dict] = []

    for horizon in utils.HORIZONS:
        naive_column = f"naive_label_{horizon}m"
        eligible_column = f"complete_horizon_eligible_{horizon}m"
        complete_label_column = f"complete_horizon_label_{horizon}m"
        full = training.copy()
        full["y_naive"] = full[naive_column].astype("int8")
        complete = training.loc[training[eligible_column].fillna(False)].copy()
        complete["y_complete"] = complete[complete_label_column].astype("int8")
        for repeat in range(REPEATS):
            seed = utils.SEED + horizon if repeat == 0 else utils.SEED + horizon + 1000 * repeat
            matched, _ = utils.matched_naive_sample(full, complete, seed=seed)
            matched["y_naive"] = matched[naive_column].astype("int8")
            if len(matched) != len(complete):
                raise AssertionError("Repeated matched sample does not match eligible sample size.")
            sample_rows.append(
                {
                    "horizon_months": horizon,
                    "sampling_seed": seed,
                    "sample_size": len(matched),
                    "naive_default_count": int(matched["y_naive"].sum()),
                    "naive_default_rate": float(matched["y_naive"].mean()),
                    "complete_default_count": int(complete["y_complete"].sum()),
                    "complete_default_rate": float(complete["y_complete"].mean()),
                    "naive_minus_complete_default_rate": float(
                        matched["y_naive"].mean() - complete["y_complete"].mean()
                    ),
                    "vintage_strata_exactly_balanced": True,
                }
            )
            balance_rows.extend(_balance_rows(matched, complete, horizon, seed))

    samples = pd.DataFrame(sample_rows)
    balance = pd.DataFrame(balance_rows)
    samples.to_csv(utils.TABLE_DIR / "matched_sampling_sensitivity.csv", index=False)
    balance.to_csv(utils.TABLE_DIR / "matched_sampling_balance.csv", index=False)
    print(samples.to_string(index=False))
    print(f"Saved {len(balance):,} feature-balance diagnostics.")
    return samples, balance


if __name__ == "__main__":
    run_repeated_matching()
