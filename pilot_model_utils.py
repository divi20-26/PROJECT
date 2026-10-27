"""Shared cohort, feature, metric, and reproducibility helpers for the pilot."""

from __future__ import annotations

import json
import importlib.util
import platform
from pathlib import Path
from typing import Any
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "00_config.py"
_config_spec = importlib.util.spec_from_file_location("pilot_project_config", CONFIG_PATH)
if _config_spec is None or _config_spec.loader is None:
    raise RuntimeError(f"Could not load project configuration from {CONFIG_PATH}")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)

DATA_DIR = config.DATA_DIR
FEATURE_DIR = DATA_DIR / "features"
OUTCOME_DIR = DATA_DIR / "outcomes"
LABEL_DIR = DATA_DIR / "labels"
MODEL_DIR = config.MODEL_DIR / "pilot"
PREDICTION_DIR = config.RESULTS_DIR / "predictions"
TABLE_DIR = config.TABLES_DIR
HORIZONS = list(config.PRIMARY_HORIZONS)
TRAIN_YEARS = list(config.PRIMARY_TRAIN_YEARS)
VALIDATION_YEARS = list(config.PRIMARY_VALIDATION_YEARS)
TEST_YEARS = list(config.PRIMARY_TEST_YEARS)
SEED = int(config.RANDOM_SEED)

FEATURE_DICTIONARY_PATH = FEATURE_DIR / "feature_dictionary_release_47.json"
if not FEATURE_DICTIONARY_PATH.exists():
    raise FileNotFoundError(f"Missing frozen feature dictionary: {FEATURE_DICTIONARY_PATH}")
with FEATURE_DICTIONARY_PATH.open(encoding="utf-8") as _file:
    FEATURE_COLUMNS = [item["feature"] for item in json.load(_file)]

NUMERIC_COLUMNS = [
    "classic_fico",
    "vantagescore_4",
    "mortgage_insurance_percentage",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
]
CATEGORICAL_COLUMNS = [name for name in FEATURE_COLUMNS if name not in NUMERIC_COLUMNS]
IDENTIFIER_COLUMNS = {"loan_identifier", "vintage", "model_eligible", "model_eligibility_reason"}
if set(FEATURE_COLUMNS) & IDENTIFIER_COLUMNS:
    raise ValueError("The frozen feature whitelist contains an identifier or nonfeature column.")
if set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS) != set(FEATURE_COLUMNS):
    raise ValueError("Numeric and categorical pilot columns do not partition the frozen whitelist.")


def ensure_output_directories() -> None:
    for directory in (MODEL_DIR, PREDICTION_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_features(years: list[int], eligible_only: bool = True) -> pd.DataFrame:
    frames = []
    for year in years:
        path = FEATURE_DIR / f"features_{year}_release_47.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing Release 47 feature file for vintage {year}: {path}")
        frame = pd.read_parquet(path)
        required = {"loan_identifier", "model_eligible", *FEATURE_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame = frame.copy()
        frame["vintage"] = int(year)
        if eligible_only:
            frame = frame.loc[frame["model_eligible"].fillna(False)].copy()
        for name in NUMERIC_COLUMNS:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
        for name in CATEGORICAL_COLUMNS:
            values = frame[name].astype(object)
            frame[name] = values.where(pd.notna(values), np.nan)
        frames.append(frame[["vintage", "loan_identifier", *FEATURE_COLUMNS]])
    if not frames:
        raise ValueError("No feature vintages requested.")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["vintage", "loan_identifier"]).any():
        raise ValueError("Feature inputs contain duplicate vintage-loan keys.")
    return result


def load_training_data() -> pd.DataFrame:
    features = load_features(TRAIN_YEARS)
    labels_path = LABEL_DIR / "label_strategies_asof_20141231.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing frozen training labels: {labels_path}")
    labels = pd.read_parquet(labels_path)
    required = {
        "vintage",
        "loan_identifier",
        *(f"naive_label_{horizon}m" for horizon in HORIZONS),
        *(f"complete_horizon_eligible_{horizon}m" for horizon in HORIZONS),
        *(f"complete_horizon_label_{horizon}m" for horizon in HORIZONS),
        *(f"survival_eligible_{horizon}m" for horizon in HORIZONS),
        *(f"survival_entry_month_{horizon}m" for horizon in HORIZONS),
        *(f"survival_exit_month_{horizon}m" for horizon in HORIZONS),
        *(f"survival_event_code_{horizon}m" for horizon in HORIZONS),
    }
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Frozen label table is missing required columns: {sorted(missing)}")
    merged = features.merge(
        labels,
        on=["vintage", "loan_identifier"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise ValueError(
            f"Some cutoff-qualified training labels lack an eligible feature row: "
            f"{len(labels)} labels, {len(merged)} joined."
        )
    return merged


def _complete_horizon_label(row: dict[str, Any], horizon: int) -> tuple[bool, int | None]:
    entry = row.get("entry_month")
    if entry is None or pd.isna(entry) or int(entry) != 1:
        return False, None
    outcome_type = row.get("outcome_type")
    outcome_month = row.get("outcome_month")
    last_month = row.get("last_observed_month")
    has_outcome = outcome_month is not None and not pd.isna(outcome_month)
    has_last = last_month is not None and not pd.isna(last_month)
    if outcome_type == "default" and has_outcome and int(outcome_month) < int(entry):
        return False, None
    if outcome_type == "default" and has_outcome and int(outcome_month) <= horizon:
        return True, 1
    if outcome_type == "voluntary_payoff" and has_outcome and int(outcome_month) <= horizon:
        return True, 0
    if outcome_type == "other_termination" and has_outcome and int(outcome_month) <= horizon:
        return False, None
    if has_outcome and int(outcome_month) > horizon:
        return True, 0
    if has_last and int(last_month) >= horizon:
        return True, 0
    return False, None


def load_mature_cohort(years: list[int], horizon: int) -> pd.DataFrame:
    """Join frozen features to outcomes and retain horizon-mature test labels."""
    features = load_features(years)
    outcomes = []
    for year in years:
        path = OUTCOME_DIR / f"outcomes_{year}_asof_20260331.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing retrospective outcomes for vintage {year}: {path}")
        outcome = pd.read_parquet(
            path,
            columns=[
                "loan_identifier",
                "entry_month",
                "outcome_type",
                "outcome_month",
                "last_observed_month",
            ],
        )
        outcome["vintage"] = int(year)
        outcomes.append(outcome)
    outcome_frame = pd.concat(outcomes, ignore_index=True)
    merged = features.merge(
        outcome_frame,
        on=["vintage", "loan_identifier"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Some eligible feature rows have no corresponding outcome record.")
    label_data = merged[["entry_month", "outcome_type", "outcome_month", "last_observed_month"]]
    labels = [
        _complete_horizon_label(row, horizon)
        for row in label_data.to_dict(orient="records")
    ]
    merged["y_true"] = [result[1] if result[0] else pd.NA for result in labels]
    merged = merged.loc[merged["y_true"].notna()].copy()
    merged["y_true"] = merged["y_true"].astype("int8")
    return merged.drop(columns=["_merge"])


def active_feature_columns(training_frame: pd.DataFrame) -> list[str]:
    active = [name for name in FEATURE_COLUMNS if training_frame[name].notna().any()]
    if not active:
        raise ValueError("No frozen origination features are observed in the training cohort.")
    return active


def make_preprocessor(
    scale_numeric: bool = True,
    feature_columns: list[str] | None = None,
    categorical_min_frequency: int | None = None,
) -> ColumnTransformer:
    selected = FEATURE_COLUMNS if feature_columns is None else feature_columns
    numeric = [name for name in NUMERIC_COLUMNS if name in selected]
    categorical = [name for name in CATEGORICAL_COLUMNS if name in selected]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                min_frequency=categorical_min_frequency,
            ),
        ),
    ]
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def fico_band(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 619, 679, 739, np.inf]
    labels = ["lt620", "620_679", "680_739", "740plus"]
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels).astype("string").fillna("missing")


def ltv_band(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 60, 80, 90, np.inf]
    labels = ["le60", "60_80", "80_90", "gt90"]
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels).astype("string").fillna("missing")


MATCH_STRATA = ["vintage", "fico_band", "ltv_band", "occupancy_status"]


def add_matching_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fico_band"] = fico_band(result["classic_fico"])
    result["ltv_band"] = ltv_band(result["original_ltv"])
    result["occupancy_status"] = result["occupancy_status"].astype("string").fillna("missing")
    return result


def matched_naive_sample(
    full_training: pd.DataFrame,
    complete_eligible: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample naive records to exactly match complete-case counts in frozen strata."""
    full = add_matching_strata(full_training)
    eligible = add_matching_strata(complete_eligible)
    targets = eligible.groupby(MATCH_STRATA, dropna=False, observed=True).size()
    sampled_parts = []
    target_counts = targets.to_dict()
    seen_targets = set()
    for key, available in full.groupby(
        MATCH_STRATA, dropna=False, observed=True, sort=False
    ):
        target_n = target_counts.get(key, 0)
        if not target_n:
            continue
        seen_targets.add(key)
        if len(available) < int(target_n):
            raise ValueError(f"Matched stratum {key} has fewer naive rows than target.")
        sampled_parts.append(available.sample(n=int(target_n), random_state=seed))
    missing_targets = set(target_counts).difference(seen_targets)
    if missing_targets:
        raise ValueError(
            f"Matched strata have no naive rows: {sorted(missing_targets, key=str)[:5]}"
        )
    matched = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else full.iloc[0:0].copy()
    balance = (
        matched.groupby(MATCH_STRATA, dropna=False, observed=True)
        .size()
        .rename("matched_count")
        .to_frame()
        .join(targets.rename("complete_count"), how="outer")
        .fillna(0)
        .reset_index()
    )
    if not balance["matched_count"].eq(balance["complete_count"]).all():
        raise AssertionError("Matched naive sample does not exactly balance complete-case strata.")
    if len(matched) != len(complete_eligible):
        raise AssertionError("Matched naive sample size differs from complete-horizon sample.")
    return matched, balance


def cox_time_bounds(entry_month: pd.Series, exit_month: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert discrete monthly entry/exit to lifelines' start-stop interval convention."""
    entry = pd.to_numeric(entry_month, errors="raise").astype(float) - 1.0
    exit_time = pd.to_numeric(exit_month, errors="raise").astype(float)
    if entry.isna().any() or exit_time.isna().any() or (exit_time <= entry).any():
        raise ValueError("Cox exit time must be observed and strictly after delayed entry.")
    if (entry < 0).any():
        raise ValueError("Cox delayed entry cannot precede month 1.")
    return entry, exit_time


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            error += mask.mean() * abs(float(y_true[mask].mean() - probability[mask].mean()))
    return float(error)


def cost_value(y_true: np.ndarray, probability: np.ndarray, threshold: float, missed_default: float, review: float) -> float:
    flagged = probability >= threshold
    false_negative = (y_true == 1) & ~flagged
    return float((false_negative.sum() * missed_default + flagged.sum() * review) / len(y_true))


def pilot_settings() -> dict[str, Any]:
    return {
        "schema_version": "freddie_mac_sflld_release_47",
        "protocol_status": "frozen_baseline_experiment",
        "primary_comparison": (
            "Each binary label strategy and delayed-entry competing-risk Cox versus "
            "naive-full Logistic Regression on identical mature test loans at each horizon."
        ),
        "primary_metrics": ["roc_auc", "pr_auc", "brier_score"],
        "calibration_metrics": [
            "expected_calibration_error",
            "calibration_intercept",
            "calibration_slope",
        ],
        "training_cutoff": "2014-12-31",
        "retrospective_outcome_cutoff": "2026-03-31",
        "training_vintages": TRAIN_YEARS,
        "validation_vintages": VALIDATION_YEARS,
        "test_vintages": TEST_YEARS,
        "horizons_months": HORIZONS,
        "feature_whitelist": FEATURE_COLUMNS,
        "categorical_missing_transform": "documented missing/sentinel values become a dedicated __MISSING__ category",
        "excluded_columns": sorted(IDENTIFIER_COLUMNS),
        "matched_strata": MATCH_STRATA,
        "matched_sampling_sensitivity": {
            "repeats_per_horizon": 5,
            "seeds": "primary seed SEED+horizon, then SEED+horizon+1000*k for k=1..4",
            "balance_diagnostics": [
                "numeric standardized mean difference",
                "maximum absolute category proportion difference",
                "feature missingness-rate difference",
            ],
        },
        "random_seed": SEED,
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "lifelines",
                "pyarrow",
                "matplotlib",
            )
        },
        "binary_models": {
            "logistic_regression": {"C": 1.0, "max_iter": 1000, "class_weight": None},
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "n_jobs": -1,
            },
        },
        "cox": {
            "delayed_entry_time": "max(survival_entry_month - 1, 0)",
            "exit_time": "survival_exit_month",
            "causes": {"default": 1, "voluntary_payoff": 2},
            "other_causes_handling": "censored for each cause-specific fit",
            "penalizer": 0.1,
            "categorical_min_frequency": 100,
            "drop_rare_onehot_indicators_below_count": 100,
            "max_month": 36,
            "cumulative_incidence": "piecewise-constant competing cause-specific hazards",
        },
        "decision_cost_scenarios": {
            f"review_to_missed_cost_{ratio:g}": {
                "missed_default_cost": 1.0,
                "review_cost_per_flag": ratio,
            }
            for ratio in (
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                25.0,
                100.0,
            )
        },
        "decision_cost_interpretation": (
            "Predeclared normalized hypothetical cost ratios; not estimated dollar losses "
            "or empirically justified lender utilities."
        ),
        "bootstrap_decision_cost_scenarios": [
            "review_to_missed_cost_0.05",
            "review_to_missed_cost_1",
        ],
        "decision_threshold_selection": (
            "Minimum mean validation cost over exact observed validation-score boundaries, "
            "including the no-review action; ties prefer the highest threshold."
        ),
        "bootstrap": {
            "replicates": 1000,
            "seed": SEED,
            "sampling": "paired loan bootstrap stratified within vintage, with replacement",
            "interval": "percentile 95% confidence interval",
        },
        "important_limitations": [
            "Performance histories begin at acquisition; delayed entry conditions survival results on observed active loans.",
            "Binary mature labels require entry at month 1 and an ascertainable outcome at the horizon.",
            "Manual-review and missed-default costs are normalized hypothetical scenario values, not observed dollar losses.",
        ],
    }
