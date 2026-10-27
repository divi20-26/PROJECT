"""Exploratory censor-aware survival metrics for frozen Cox predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import pilot_model_utils as utils
from importlib.util import module_from_spec, spec_from_file_location


METRIC_DIR = utils.config.RESULTS_DIR / "survival_censor_aware"
EVALUATION_PERIODS = {
    "2017-2019_exploratory": [2017, 2018, 2019],
    "2020-2022_exploratory": [2020, 2021, 2022],
}
CENSORING_TYPES = {"other_termination", "right_censored"}
EVENT_TYPES = {"default", "voluntary_payoff", *CENSORING_TYPES}


def _survival_module():
    path = Path(__file__).resolve().parent / "08_train_survival_models.py"
    spec = spec_from_file_location("censor_aware_survival_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load survival prediction code: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_rows(years: list[int]) -> pd.DataFrame:
    frames = []
    features = utils.load_features(years)
    for year in years:
        path = utils.OUTCOME_DIR / f"outcomes_{year}_asof_20260331.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing outcome file: {path}")
        outcome = pd.read_parquet(
            path,
            columns=[
                "loan_identifier",
                "entry_month",
                "last_observed_month",
                "outcome_month",
                "outcome_type",
            ],
        )
        outcome["vintage"] = year
        frames.append(outcome)
    outcomes = pd.concat(frames, ignore_index=True)
    joined = features.merge(
        outcomes,
        on=["vintage", "loan_identifier"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Eligible feature rows are missing outcomes for survival metrics.")
    joined = joined.drop(columns="_merge")
    observed_types = set(joined["outcome_type"].dropna().unique())
    if observed_types.difference(EVENT_TYPES):
        raise ValueError(f"Unexpected survival outcome types: {sorted(observed_types - EVENT_TYPES)}")
    joined["entry_month"] = pd.to_numeric(joined["entry_month"], errors="raise")
    joined["last_observed_month"] = pd.to_numeric(
        joined["last_observed_month"], errors="raise"
    )
    joined["outcome_month"] = pd.to_numeric(joined["outcome_month"], errors="coerce")
    joined["exit_month"] = joined["outcome_month"].where(
        joined["outcome_type"].ne("right_censored"),
        joined["last_observed_month"],
    )
    valid = (
        joined["entry_month"].eq(1)
        & joined["exit_month"].notna()
        & joined["exit_month"].ge(joined["entry_month"])
        & joined["outcome_type"].isin(EVENT_TYPES)
    )
    result = joined.loc[valid].copy()
    result["exit_month"] = result["exit_month"].astype(int)
    return result


def _censoring_survival_before(
    exit_month: np.ndarray,
    censor_event: np.ndarray,
    query_month: np.ndarray,
) -> np.ndarray:
    """Estimate discrete-time marginal G(t-) from observed competing-risk histories."""
    exit_month = np.asarray(exit_month, dtype=int)
    censor_event = np.asarray(censor_event, dtype=bool)
    query_month = np.asarray(query_month, dtype=int)
    if len(exit_month) == 0 or len(exit_month) != len(censor_event):
        raise ValueError("Censoring estimator requires aligned, nonempty observations.")
    if (exit_month < 1).any() or (query_month < 1).any():
        raise ValueError("Survival times must be positive integer months.")
    survival = 1.0
    values: dict[int, float] = {}
    for month in range(1, int(max(exit_month.max(), query_month.max())) + 1):
        values[month] = survival
        risk_count = int(np.count_nonzero(exit_month >= month))
        censor_count = int(np.count_nonzero((exit_month == month) & censor_event))
        if risk_count == 0 or censor_count > risk_count:
            raise RuntimeError("Invalid censoring risk set.")
        survival *= 1 - censor_count / risk_count
    return np.asarray([values[int(month)] for month in query_month], dtype=float)


def _ipcw_horizon_metrics(
    outcome_type: np.ndarray,
    exit_month: np.ndarray,
    horizon: int,
    probability: np.ndarray,
) -> dict[str, float | int]:
    outcome_type = np.asarray(outcome_type, dtype=str)
    exit_month = np.asarray(exit_month, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not (len(outcome_type) == len(exit_month) == len(probability)) or len(exit_month) == 0:
        raise ValueError("IPCW inputs must be aligned and nonempty.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Survival probabilities must be finite and in [0, 1].")
    if horizon < 1:
        raise ValueError("Evaluation horizon must be positive.")

    censoring = np.isin(outcome_type, list(CENSORING_TYPES))
    cumulative_g = _censoring_survival_before(exit_month, censoring, exit_month)
    horizon_g = float(
        _censoring_survival_before(
            exit_month,
            censoring,
            np.asarray([horizon], dtype=int),
        )[0]
    )
    observed_by_horizon = (
        np.isin(outcome_type, ["default", "voluntary_payoff"]) & (exit_month <= horizon)
    )
    observed_controls = (exit_month >= horizon) & (
        (outcome_type != "other_termination") | (exit_month > horizon)
    )
    included = observed_by_horizon | observed_controls
    included_exit = exit_month[included]
    labels = (
        (outcome_type[included] == "default") & (included_exit <= horizon)
    ).astype("int8")
    weights = np.where(
        included_exit <= horizon,
        1.0 / cumulative_g[included],
        1.0 / horizon_g,
    )
    scored_probability = probability[included]
    if np.unique(labels).size != 2 or not np.isfinite(weights).all():
        raise ValueError(f"IPCW horizon {horizon}m does not have both classes or valid weights.")
    return {
        "n_observed_for_horizon": int(included.sum()),
        "defaults": int(labels.sum()),
        "observed_censored_before_horizon": int((~included).sum()),
        "censoring_survival_at_horizon_left_limit": horizon_g,
        "max_ipcw_weight": float(weights.max()),
        "roc_auc_ipcw": float(
            roc_auc_score(labels, scored_probability, sample_weight=weights)
        ),
        "pr_auc_ipcw": float(
            average_precision_score(labels, scored_probability, sample_weight=weights)
        ),
        "brier_score_ipcw": float(
            brier_score_loss(labels, scored_probability, sample_weight=weights)
        ),
    }


def evaluate_period(period: str, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_path = utils.MODEL_DIR / "cause_specific_cox_36m.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing fitted Cox models: {artifact_path}")
    artifact = joblib.load(artifact_path)
    model_module = _survival_module()
    rows = _load_eval_rows(years)
    incidence = model_module.predict_default_cumulative_incidence(
        rows,
        artifact["preprocessor"],
        artifact["models"],
        max_month=36,
    )
    metric_rows = []
    concordance_rows = []
    outcome_types = rows["outcome_type"].to_numpy(dtype=str)
    exit_month = rows["exit_month"].to_numpy(dtype=int)
    for horizon in utils.HORIZONS:
        metrics = _ipcw_horizon_metrics(
            outcome_types,
            exit_month,
            horizon,
            incidence[:, horizon - 1],
        )
        metric_rows.append(
            {
                "evaluation_period": period,
                "vintages": ",".join(map(str, years)),
                "horizon_months": horizon,
                **metrics,
            }
        )
    thirty_six_month_prediction = incidence[:, 35]
    concordance_rows.append(
        {
            "evaluation_period": period,
            "vintages": ",".join(map(str, years)),
            "entry_rule": "entry_month == 1; entry after month 1 excluded",
            "risk_score": "36-month predicted default cumulative incidence",
            "competing_events": "voluntary payoff treated as censored for cause-specific concordance",
            "concordance_index": float(
                concordance_index(
                    exit_month,
                    -thirty_six_month_prediction,
                    event_observed=(outcome_types == "default"),
                )
            ),
            "n": len(rows),
            "defaults": int((outcome_types == "default").sum()),
        }
    )
    return pd.DataFrame(metric_rows), pd.DataFrame(concordance_rows)


def run_survival_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    concordance = []
    for period, years in EVALUATION_PERIODS.items():
        period_metrics, period_concordance = evaluate_period(period, years)
        metrics.append(period_metrics)
        concordance.append(period_concordance)
    result = pd.concat(metrics, ignore_index=True)
    concordance_result = pd.concat(concordance, ignore_index=True)
    result.to_csv(METRIC_DIR / "ipcw_time_dependent_metrics.csv", index=False)
    concordance_result.to_csv(METRIC_DIR / "cause_specific_concordance.csv", index=False)
    protocol = {
        "metric_status": "Exploratory; both included vintage groups have been inspected.",
        "horizons_months": utils.HORIZONS,
        "evaluation_entry_rule": "entry_month == 1",
        "default_event": "default before or at horizon",
        "competing_nondefault": "voluntary payoff before or at horizon",
        "censoring": sorted(CENSORING_TYPES),
        "ipcw_assumption": (
            "Marginal independent censoring, estimated as a discrete-time reverse "
            "Kaplan-Meier curve within each evaluation period; no truncation or "
            "conditional censoring model."
        ),
        "concordance": (
            "Cause-specific Harrell concordance using 36-month predicted cumulative "
            "incidence as the risk score; payoff and other termination are censored."
        ),
        "limitations": [
            "Censoring may depend on borrower and loan characteristics; marginal IPCW can be biased if independent censoring fails.",
            "C-index treats competing payoff as censored and is not a competing-risk concordance estimand.",
            "Acquisition histories cannot recover unobserved pre-acquisition performance.",
            "Metrics are exploratory, not confirmatory, because the evaluation vintages were previously inspected.",
        ],
    }
    (METRIC_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(concordance_result.to_string(index=False))
    return result, concordance_result


if __name__ == "__main__":
    run_survival_metrics()
