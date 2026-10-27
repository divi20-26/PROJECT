"""Shared, leakage-aware utilities for research extensions to the frozen pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


MACRO_FEATURES = ["unemployment_rate", "real_gdp", "national_house_price_index"]
MACRO_SOURCE = {
    "UNRATE": {
        "feature": "unemployment_rate",
        "source": "U.S. Bureau of Labor Statistics via FRED",
        "url": "https://fred.stlouisfed.org/series/UNRATE",
        "frequency": "monthly",
        "units": "percent, seasonally adjusted",
        "geography": "United States",
        "availability_lag_days": 7,
        "max_age_days": 90,
    },
    "GDPC1": {
        "feature": "real_gdp",
        "source": "U.S. Bureau of Economic Analysis via FRED",
        "url": "https://fred.stlouisfed.org/series/GDPC1",
        "frequency": "quarterly",
        "units": "billions of chained 2017 dollars, seasonally adjusted annual rate",
        "geography": "United States",
        "availability_lag_days": 45,
        "max_age_days": 180,
    },
    "USSTHPI": {
        "feature": "national_house_price_index",
        "source": "U.S. Federal Housing Finance Agency via FRED",
        "url": "https://fred.stlouisfed.org/series/USSTHPI",
        "frequency": "quarterly",
        "units": "index, 1980:Q1=100, not seasonally adjusted",
        "geography": "United States",
        "availability_lag_days": 90,
        "max_age_days": 365,
    },
}


def prepare_macro_series(raw_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create macro observations with conservative modeled availability dates."""
    expected = set(MACRO_SOURCE)
    missing = expected.difference(raw_frames)
    if missing:
        raise FileNotFoundError(f"Missing required FRED series: {sorted(missing)}")
    rows = []
    for series_id, definition in MACRO_SOURCE.items():
        frame = raw_frames[series_id].copy()
        if not {"observation_date", series_id}.issubset(frame.columns):
            raise ValueError(
                f"{series_id} input must have observation_date and {series_id} columns."
            )
        observation_date = pd.to_datetime(frame["observation_date"], errors="coerce")
        value = pd.to_numeric(frame[series_id], errors="coerce")
        valid = observation_date.notna() & value.notna()
        frame = pd.DataFrame(
            {
                "series_id": series_id,
                "feature": definition["feature"],
                "observation_date": observation_date[valid].to_numpy(),
                "value": value[valid].to_numpy(dtype=float),
            }
        )
        if frame.empty:
            raise ValueError(f"{series_id} contains no valid observations.")
        if frame["observation_date"].duplicated().any():
            raise ValueError(f"{series_id} contains duplicate observation dates.")
        if definition["frequency"] == "monthly":
            period_end = frame["observation_date"] + pd.offsets.MonthEnd(0)
        else:
            period_end = frame["observation_date"] + pd.offsets.QuarterEnd(0)
        frame["available_date"] = (
            period_end + pd.to_timedelta(definition["availability_lag_days"], unit="D")
        )
        frame["max_age_days"] = int(definition["max_age_days"])
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["feature", "available_date"], ignore_index=True
    )
    if result.duplicated(["feature", "available_date"]).any():
        raise ValueError("Prepared macro observations have duplicate feature/date keys.")
    return result


def merge_macro_asof(
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    date_column: str = "prediction_date_proxy",
) -> pd.DataFrame:
    """As-of merge national series, enforcing modeled availability and age limits."""
    required_loan_columns = {"vintage", "loan_identifier", date_column}
    required_macro_columns = {
        "feature",
        "observation_date",
        "available_date",
        "value",
        "max_age_days",
    }
    if missing := required_loan_columns.difference(loans.columns):
        raise ValueError(f"Loan frame is missing required columns: {sorted(missing)}")
    if missing := required_macro_columns.difference(macro.columns):
        raise ValueError(f"Macro frame is missing required columns: {sorted(missing)}")
    keys = ["vintage", "loan_identifier"]
    if loans.duplicated(keys).any():
        raise ValueError("Loan input has duplicate vintage-loan identifiers.")
    if macro.duplicated(["feature", "available_date"]).any():
        raise ValueError("Macro input has duplicate feature-availability dates.")

    result = loans.copy()
    result["__merge_order"] = np.arange(len(result))
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    if result[date_column].isna().any():
        raise ValueError("Loan prediction-date proxy contains missing or invalid dates.")
    result = result.sort_values(date_column, kind="stable").copy()
    for feature in MACRO_FEATURES:
        series = macro.loc[macro["feature"].eq(feature)].copy()
        if series.empty:
            raise FileNotFoundError(f"Macro series for feature {feature!r} is missing.")
        series["available_date"] = pd.to_datetime(series["available_date"], errors="raise")
        series["observation_date"] = pd.to_datetime(series["observation_date"], errors="raise")
        series = series.sort_values("available_date", kind="stable")
        right = series[["available_date", "observation_date", "value", "max_age_days"]]
        joined = pd.merge_asof(
            result[[date_column]].sort_values(date_column, kind="stable"),
            right,
            left_on=date_column,
            right_on="available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        age = (joined[date_column] - joined["available_date"]).dt.days
        valid = joined["available_date"].notna() & age.le(joined["max_age_days"])
        result[feature] = joined["value"].where(valid).to_numpy()
        result[f"{feature}_observation_date"] = joined["observation_date"].where(valid).to_numpy()
        result[f"{feature}_available_date"] = joined["available_date"].where(valid).to_numpy()
        result[f"{feature}_age_days"] = age.where(valid).to_numpy()
    result = result.sort_values("__merge_order", kind="stable").drop(columns="__merge_order")
    if result.duplicated(keys).any() or len(result) != len(loans):
        raise ValueError("Macro merge changed loan grain or row count.")
    for feature in MACRO_FEATURES:
        available = pd.to_datetime(result[f"{feature}_available_date"], errors="coerce")
        if (available.notna() & available.gt(result[date_column])).any():
            raise ValueError(f"Future {feature} information was merged to a loan.")
    return result


def transformed_feature_families(preprocessor, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return model-input names and their exact origination-feature family."""
    names = list(preprocessor.get_feature_names_out())
    numeric = list(preprocessor.transformers_[0][2])
    categorical = list(preprocessor.transformers_[1][2])
    numeric_slice = preprocessor.output_indices_["numeric"]
    categorical_slice = preprocessor.output_indices_["categorical"]
    families: list[str | None] = [None] * len(names)
    for offset, name in enumerate(numeric):
        families[numeric_slice.start + offset] = name
    if categorical:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        category_names = list(encoder.get_feature_names_out(categorical))
        start = categorical_slice.start
        cursor = start
        for name, categories in zip(categorical, encoder.categories_):
            for _ in categories:
                families[cursor] = name
                cursor += 1
        if cursor - start != len(category_names):
            raise ValueError("One-hot encoded columns could not be mapped to input families.")
    if len(names) != len(families) or any(name is None for name in families):
        raise ValueError("Transformed model features do not map to the fitted input columns.")
    if not set(families).issubset(set(feature_columns)):
        raise ValueError("Transformed feature families include columns outside the whitelist.")
    return names, [str(name) for name in families]


def aggregate_shap_by_family(
    shap_values: np.ndarray, families: list[str], expected_features: list[str]
) -> np.ndarray:
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(families):
        raise ValueError("SHAP matrix shape does not match the transformed feature mapping.")
    if set(families).difference(expected_features):
        raise ValueError("SHAP feature families are not in the declared feature whitelist.")
    aggregated = np.zeros((values.shape[0], len(expected_features)), dtype=float)
    positions = {name: index for index, name in enumerate(expected_features)}
    for column, family in enumerate(families):
        aggregated[:, positions[family]] += values[:, column]
    if not np.isfinite(aggregated).all():
        raise ValueError("Aggregated SHAP values contain nonfinite values.")
    return aggregated


def horizon_importance_table(
    importance_frames: Iterable[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate aligned absolute-importance ranks and Spearman consistency."""
    combined = pd.concat(list(importance_frames), ignore_index=True)
    required = {"horizon_months", "feature", "mean_abs_shap", "mean_signed_shap"}
    if not required.issubset(combined.columns):
        raise ValueError(f"Importance input is missing columns: {sorted(required.difference(combined.columns))}")
    if combined.duplicated(["horizon_months", "feature"]).any():
        raise ValueError("Importance input has duplicate horizon-feature rows.")
    horizons = sorted(combined["horizon_months"].unique())
    if len(horizons) < 2:
        raise ValueError("At least two prediction horizons are required.")
    combined["rank"] = combined.groupby("horizon_months")["mean_abs_shap"].rank(
        ascending=False, method="min"
    )
    wide = combined.pivot(index="feature", columns="horizon_months", values="mean_abs_shap")
    rank_wide = combined.pivot(index="feature", columns="horizon_months", values="rank")
    if wide.isna().any().any():
        raise ValueError("Every horizon must contain the same feature set.")
    result = pd.DataFrame({"feature": wide.index})
    for horizon in horizons:
        result[f"mean_abs_shap_{horizon}m"] = wide[horizon].to_numpy()
        total_importance = float(wide[horizon].sum())
        result[f"relative_importance_{horizon}m"] = (
            wide[horizon].to_numpy() / total_importance
            if total_importance > 0
            else np.zeros(len(wide))
        )
        result[f"rank_{horizon}m"] = rank_wide[horizon].to_numpy()
    for horizon in horizons[1:]:
        result[f"rank_change_{horizon}m_vs_{horizons[0]}m"] = (
            rank_wide[horizon] - rank_wide[horizons[0]]
        ).to_numpy()
    direction = combined.pivot(index="feature", columns="horizon_months", values="mean_signed_shap")
    for horizon in horizons:
        result[f"mean_signed_shap_{horizon}m"] = direction[horizon].to_numpy()
    consistency = []
    for left_index, left in enumerate(horizons):
        for right in horizons[left_index + 1 :]:
            correlation = rank_wide[left].corr(rank_wide[right], method="spearman")
            consistency.append(
                {
                    "horizon_a_months": left,
                    "horizon_b_months": right,
                    "spearman_rank_correlation": float(correlation),
                    "feature_count": len(rank_wide),
                }
            )
    return result.sort_values(f"rank_{horizons[0]}m"), pd.DataFrame(consistency)


def choose_explanation_clusters(
    validation_values: np.ndarray,
    seed: int,
    candidates: Iterable[int] = range(2, 7),
) -> tuple[pd.DataFrame, KMeans, float]:
    """Choose K-means k by validation silhouette and estimate seed stability."""
    values = np.asarray(validation_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("Validation explanations must be a finite two-dimensional matrix.")
    diagnostics = []
    for k in candidates:
        if k < 2 or k >= len(values):
            raise ValueError(f"Invalid candidate cluster count {k} for {len(values)} rows.")
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(values)
        score = float(silhouette_score(values, labels))
        counts = np.bincount(labels, minlength=k)
        diagnostics.append(
            {
                "n_clusters": k,
                "silhouette_score": score,
                "minimum_cluster_size": int(counts.min()),
                "maximum_cluster_size": int(counts.max()),
                "minimum_cluster_fraction": float(counts.min() / len(values)),
            }
        )
    if not diagnostics:
        raise ValueError("No valid candidate cluster counts were evaluated.")
    best_k = max(diagnostics, key=lambda row: row["silhouette_score"])["n_clusters"]
    selected = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(values)
    repeated = []
    for alternate_seed in range(seed, seed + 5):
        labels = KMeans(n_clusters=best_k, random_state=alternate_seed, n_init=10).fit_predict(values)
        repeated.append(labels)
    stability = float(
        np.mean(
            [
                adjusted_rand_score(repeated[left], repeated[right])
                for left in range(len(repeated))
                for right in range(left + 1, len(repeated))
            ]
        )
    )
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics["selected"] = diagnostics["n_clusters"].eq(best_k)
    diagnostics["validation_sample_size"] = len(values)
    diagnostics["stability_mean_pairwise_ari"] = np.where(
        diagnostics["selected"], stability, np.nan
    )
    return diagnostics, selected, stability


def require_existing_inputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required extension inputs are missing: {missing}")
