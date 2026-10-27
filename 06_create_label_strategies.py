"""Create cutoff-aware binary labels and delayed-entry competing-risk inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "00_config.py"
config_spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config)

OUTCOME_DIR = config.DATA_DIR / "outcomes"
LABEL_DIR = config.DATA_DIR / "labels"
HORIZONS = config.PRIMARY_HORIZONS


def cutoff_tag(value: str) -> str:
    return value.replace("-", "")


def qualifies_origination_by_cutoff(frame: pd.DataFrame, cutoff: str) -> pd.Series:
    cutoff_month = cutoff_tag(cutoff)[:6]
    return frame["first_payment_date"].astype("string").str.slice(0, 6).le(cutoff_month)


def complete_horizon_label(row: dict, horizon: int) -> tuple[bool, int | None]:
    entry = row.get("entry_month")
    if entry is None or pd.isna(entry):
        return False, None
    if int(entry) != 1:
        return False, None

    outcome_type = row.get("outcome_type")
    outcome_month = row.get("outcome_month")
    last_month = row.get("last_observed_month")
    has_outcome_month = outcome_month is not None and not pd.isna(outcome_month)
    has_last_month = last_month is not None and not pd.isna(last_month)
    if (
        outcome_type == "default"
        and has_outcome_month
        and int(outcome_month) < int(entry)
    ):
        return False, None
    if outcome_type == "default" and has_outcome_month and int(outcome_month) <= horizon:
        return True, 1
    if outcome_type == "voluntary_payoff" and has_outcome_month and int(outcome_month) <= horizon:
        return True, 0
    if outcome_type == "other_termination" and has_outcome_month and int(outcome_month) <= horizon:
        return False, None
    if has_outcome_month and int(outcome_month) > horizon:
        return True, 0
    if has_last_month and int(last_month) >= horizon:
        return True, 0
    return False, None


def mature_exclusion_reason(row: dict, horizon: int) -> str | None:
    eligible, _ = complete_horizon_label(row, horizon)
    if eligible:
        return None
    entry = row.get("entry_month")
    if entry is None or pd.isna(entry):
        return "missing_entry"
    entry_month = int(entry)
    if entry_month != 1:
        return "delayed_entry"
    outcome_type = row.get("outcome_type")
    outcome_month = row.get("outcome_month")
    last_month = row.get("last_observed_month")
    has_outcome_month = outcome_month is not None and not pd.isna(outcome_month)
    has_last_month = last_month is not None and not pd.isna(last_month)
    if (
        outcome_type == "default"
        and has_outcome_month
        and int(outcome_month) < entry_month
    ):
        return "default_before_entry"
    if (
        outcome_type == "other_termination"
        and has_outcome_month
        and int(outcome_month) <= horizon
    ):
        return "other_termination_within_horizon"
    if outcome_type == "right_censored" and (
        not has_last_month or int(last_month) < horizon
    ):
        return "right_censored_before_horizon"
    if not has_last_month:
        return "no_observed_horizon_followup"
    return "other_unresolved"


def censor_aware_input(row: dict, horizon: int) -> dict:
    entry = row.get("entry_month")
    followup = row.get("followup_months")
    if entry is None or pd.isna(entry) or followup is None or pd.isna(followup):
        return {
            "survival_eligible": False,
            "survival_entry_month": None,
            "survival_exit_month": None,
            "survival_event_code": None,
            "survival_outcome_by_horizon": None,
            "survival_exclusion_reason": "no_observed_followup",
        }
    entry_month = int(entry)
    exit_month = int(followup)
    event_code = int(row["survival_event_code"])
    if exit_month < entry_month:
        return {
            "survival_eligible": False,
            "survival_entry_month": None,
            "survival_exit_month": None,
            "survival_event_code": None,
            "survival_outcome_by_horizon": None,
            "survival_exclusion_reason": (
                "default_before_risk_entry"
                if row.get("outcome_type") == "default"
                else "terminal_event_before_risk_entry"
            ),
        }
    horizon_event_code = event_code if exit_month <= horizon else 0
    horizon_exit = min(exit_month, horizon)
    if entry_month > horizon:
        eligible = False
        horizon_event_code = None
        horizon_exit = None
        exclusion_reason = "delayed_entry_after_horizon"
    else:
        eligible = True
        exclusion_reason = None
    return {
        "survival_eligible": eligible,
        "survival_entry_month": entry_month if eligible else None,
        "survival_exit_month": horizon_exit,
        "survival_event_code": event_code,
        "survival_outcome_by_horizon": horizon_event_code,
        "survival_exclusion_reason": exclusion_reason,
    }


def generate_strategy_rows(training: pd.DataFrame, evaluation: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    if training["loan_identifier"].duplicated().any() or evaluation["loan_identifier"].duplicated().any():
        raise ValueError("Outcome inputs must contain at most one row per loan identifier.")
    evaluation_columns = [
        "loan_identifier",
        "outcome_type",
        "outcome_month",
        "entry_month",
        "last_observed_month",
        "followup_months",
        "survival_event_code",
    ]
    evaluation = evaluation[evaluation_columns].rename(
        columns={
            "outcome_type": "evaluation_outcome_type",
            "outcome_month": "evaluation_outcome_month",
            "entry_month": "evaluation_entry_month",
            "last_observed_month": "evaluation_last_observed_month",
            "followup_months": "evaluation_followup_months",
            "survival_event_code": "evaluation_survival_event_code",
        }
    )
    joined = training.merge(evaluation, on="loan_identifier", how="left", validate="one_to_one")
    joined = joined.loc[qualifies_origination_by_cutoff(joined, cutoff)].copy()

    result_rows = []
    for row in joined.to_dict(orient="records"):
        result = {
            "vintage": row["vintage"],
            "loan_identifier": row["loan_identifier"],
            "first_payment_date": row["first_payment_date"],
            "training_cutoff": cutoff,
            "training_outcome_type": row["outcome_type"],
            "training_outcome_month": row["outcome_month"],
            "training_entry_month": row["entry_month"],
            "training_last_observed_month": row["last_observed_month"],
            "training_followup_months": row["followup_months"],
            "evaluation_outcome_type": row.get("evaluation_outcome_type"),
            "evaluation_followup_months": row.get("evaluation_followup_months"),
            "evaluation_entry_month": row.get("evaluation_entry_month"),
        }
        for horizon in HORIZONS:
            event_by_cutoff = (
                row.get("outcome_type") == "default"
                and row.get("outcome_month") is not None
                and not pd.isna(row.get("outcome_month"))
                and int(row["outcome_month"]) <= horizon
            )
            naive = int(bool(event_by_cutoff))
            eligible, complete_label = complete_horizon_label(row, horizon)

            evaluation_row = {
                "entry_month": row.get("evaluation_entry_month"),
                "outcome_type": row.get("evaluation_outcome_type"),
                "outcome_month": row.get("evaluation_outcome_month"),
                "last_observed_month": row.get("evaluation_last_observed_month"),
            }
            evaluation_eligible, evaluation_label = complete_horizon_label(evaluation_row, horizon)
            survival = censor_aware_input(row, horizon)

            result[f"naive_label_{horizon}m"] = naive
            result[f"complete_horizon_eligible_{horizon}m"] = eligible
            result[f"complete_horizon_label_{horizon}m"] = complete_label
            result[f"evaluation_mature_eligible_{horizon}m"] = evaluation_eligible
            result[f"evaluation_mature_label_{horizon}m"] = evaluation_label
            result[f"naive_false_negative_{horizon}m"] = bool(
                evaluation_eligible and evaluation_label == 1 and naive == 0
            )
            result[f"survival_eligible_{horizon}m"] = survival["survival_eligible"]
            result[f"survival_entry_month_{horizon}m"] = survival["survival_entry_month"]
            result[f"survival_exit_month_{horizon}m"] = survival["survival_exit_month"]
            result[f"survival_event_code_{horizon}m"] = survival["survival_outcome_by_horizon"]
            result[f"survival_exclusion_reason_{horizon}m"] = survival["survival_exclusion_reason"]
        result_rows.append(result)
    return pd.DataFrame(result_rows)


def summarize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (vintage, cutoff), group in labels.groupby(["vintage", "training_cutoff"], dropna=False):
        for horizon in HORIZONS:
            complete_mask = group[f"complete_horizon_eligible_{horizon}m"]
            mature_mask = group[f"evaluation_mature_eligible_{horizon}m"]
            unresolved_count = int((~complete_mask).sum())
            mature_count = int(mature_mask.sum())
            training_exclusion_reasons = [
                mature_exclusion_reason(
                    {
                        "entry_month": row["training_entry_month"],
                        "outcome_type": row["training_outcome_type"],
                        "outcome_month": row["training_outcome_month"],
                        "last_observed_month": row["training_last_observed_month"],
                    },
                    horizon,
                )
                for row in group.to_dict(orient="records")
            ]
            training_exclusion_counts = {
                reason: training_exclusion_reasons.count(reason)
                for reason in (
                    "missing_entry",
                    "delayed_entry",
                    "default_before_entry",
                    "other_termination_within_horizon",
                    "right_censored_before_horizon",
                    "no_observed_horizon_followup",
                    "other_unresolved",
                )
            }
            if sum(training_exclusion_counts.values()) != unresolved_count:
                raise ValueError(
                    f"Training maturity exclusions do not reconcile for vintage {vintage}, "
                    f"horizon {horizon}."
                )
            false_negatives = int(group[f"naive_false_negative_{horizon}m"].sum())
            naive_complete_disagreements = int(
                (
                    complete_mask
                    & (
                        group[f"naive_label_{horizon}m"]
                        != group[f"complete_horizon_label_{horizon}m"]
                    )
                ).sum()
            )
            training_event_count = int(
                (
                    (group["training_outcome_type"] == "default")
                    & (group["training_followup_months"] <= horizon)
                ).sum()
            )
            training_payoff_count = int(
                (
                    (group["training_outcome_type"] == "voluntary_payoff")
                    & (group["training_followup_months"] <= horizon)
                ).sum()
            )
            evaluation_event_count = int(
                (
                    mature_mask
                    & (group[f"evaluation_mature_label_{horizon}m"] == 1)
                ).sum()
            )
            summaries.append(
                {
                    "vintage": int(vintage),
                    "training_cutoff": cutoff,
                    "horizon_months": horizon,
                    "origination_loans_at_cutoff": len(group),
                    "loans_with_observed_history": int(group["training_entry_month"].notna().sum()),
                    "default_events_observed_by_cutoff_within_horizon": training_event_count,
                    "voluntary_payoffs_by_cutoff_within_horizon": training_payoff_count,
                    "naive_positive_count": int(group[f"naive_label_{horizon}m"].sum()),
                    "complete_horizon_eligible_count": int(complete_mask.sum()),
                    "complete_horizon_excluded_count": unresolved_count,
                    "complete_horizon_excluded_delayed_entry_count": training_exclusion_counts["delayed_entry"],
                    "complete_horizon_excluded_default_before_entry_count": training_exclusion_counts[
                        "default_before_entry"
                    ],
                    "complete_horizon_excluded_other_termination_within_horizon_count": training_exclusion_counts[
                        "other_termination_within_horizon"
                    ],
                    "complete_horizon_excluded_right_censored_before_horizon_count": training_exclusion_counts[
                        "right_censored_before_horizon"
                    ],
                    "complete_horizon_excluded_missing_entry_count": training_exclusion_counts["missing_entry"],
                    "complete_horizon_excluded_no_observed_horizon_followup_count": training_exclusion_counts[
                        "no_observed_horizon_followup"
                    ],
                    "complete_horizon_excluded_other_unresolved_count": training_exclusion_counts[
                        "other_unresolved"
                    ],
                    "complete_horizon_exclusion_reconciliation_count": sum(
                        training_exclusion_counts.values()
                    ),
                    "training_censored_or_unresolved_rate": unresolved_count / len(group) if len(group) else None,
                    "naive_vs_complete_disagreement_count": naive_complete_disagreements,
                    "naive_vs_complete_disagreement_rate_of_eligible": (
                        naive_complete_disagreements / int(complete_mask.sum())
                        if complete_mask.sum()
                        else None
                    ),
                    "evaluation_mature_count": mature_count,
                    "evaluation_mature_event_count": evaluation_event_count,
                    "evaluation_right_censored_count": int(
                        (group["evaluation_outcome_type"] == "right_censored").sum()
                    ),
                    "evaluation_right_censored_before_horizon_count": int(
                        (
                            (group["evaluation_outcome_type"] == "right_censored")
                            & (
                                group["evaluation_followup_months"]
                                >= group["evaluation_entry_month"]
                            )
                            & (group["evaluation_followup_months"] < horizon)
                        ).sum()
                    ),
                    "evaluation_right_censoring_rate_before_horizon": (
                        int(
                            (
                                (group["evaluation_outcome_type"] == "right_censored")
                                & (
                                    group["evaluation_followup_months"]
                                    >= group["evaluation_entry_month"]
                                )
                                & (group["evaluation_followup_months"] < horizon)
                            ).sum()
                        )
                        / len(group)
                        if len(group)
                        else None
                    ),
                    "evaluation_other_termination_before_horizon_count": int(
                        (
                            (group["evaluation_outcome_type"] == "other_termination")
                            & (
                                group["evaluation_followup_months"]
                                >= group["evaluation_entry_month"]
                            )
                            & (group["evaluation_followup_months"] <= horizon)
                        ).sum()
                    ),
                    "evaluation_censoring_count_before_horizon": int(
                        (
                            (group["evaluation_outcome_type"] == "right_censored")
                            & (
                                group["evaluation_followup_months"]
                                >= group["evaluation_entry_month"]
                            )
                            & (group["evaluation_followup_months"] < horizon)
                        ).sum()
                        + (
                            (group["evaluation_outcome_type"] == "other_termination")
                            & (
                                group["evaluation_followup_months"]
                                >= group["evaluation_entry_month"]
                            )
                            & (group["evaluation_followup_months"] <= horizon)
                        ).sum()
                    ),
                    "evaluation_censoring_rate_before_horizon": (
                        (
                            (
                                (group["evaluation_outcome_type"] == "right_censored")
                                & (
                                    group["evaluation_followup_months"]
                                    >= group["evaluation_entry_month"]
                                )
                                & (group["evaluation_followup_months"] < horizon)
                            ).sum()
                            + (
                                (group["evaluation_outcome_type"] == "other_termination")
                                & (
                                    group["evaluation_followup_months"]
                                    >= group["evaluation_entry_month"]
                                )
                                & (group["evaluation_followup_months"] <= horizon)
                            ).sum()
                        )
                        / len(group)
                        if len(group)
                        else None
                    ),
                    "default_before_risk_entry_count": int(
                        (
                            (group["training_outcome_type"] == "default")
                            & (group["training_followup_months"] < group["training_entry_month"])
                        ).sum()
                    ),
                    "retrospective_naive_false_negative_count": false_negatives,
                    "retrospective_naive_false_negative_rate_of_mature": (
                        false_negatives / mature_count if mature_count else None
                    ),
                    "delayed_entry_count": int((group["training_entry_month"] > 1).sum()),
                    "survival_eligible_count": int(group[f"survival_eligible_{horizon}m"].sum()),
                }
            )
    return pd.DataFrame(summaries)


def summarize_label_sensitivity(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (vintage, cutoff), group in labels.groupby(["vintage", "training_cutoff"], dropna=False):
        for horizon in HORIZONS:
            complete_mask = group[f"complete_horizon_eligible_{horizon}m"].astype(bool)
            immature_mask = ~complete_mask
            eval_mature_mask = group[f"evaluation_mature_eligible_{horizon}m"].astype(bool)
            eval_default_mask = (
                group[f"evaluation_mature_label_{horizon}m"].eq(1).fillna(False)
            )
            later_mature_immature = immature_mask & eval_mature_mask
            later_default_immature = later_mature_immature & eval_default_mask
            common_disagreement = (
                complete_mask
                & group[f"naive_label_{horizon}m"].ne(
                    group[f"complete_horizon_label_{horizon}m"]
                )
            )
            immature_count = int(immature_mask.sum())
            later_mature_count = int(later_mature_immature.sum())
            later_default_count = int(later_default_immature.sum())
            rows.append(
                {
                    "vintage": int(vintage),
                    "training_cutoff": cutoff,
                    "horizon_months": horizon,
                    "training_cohort_count": len(group),
                    "training_complete_horizon_eligible_count": int(complete_mask.sum()),
                    "training_immature_or_ineligible_count": immature_count,
                    "naive_positive_count_within_immature_group": int(
                        (
                            immature_mask
                            & group[f"naive_label_{horizon}m"].eq(1)
                        ).sum()
                    ),
                    "immature_group_mature_at_evaluation_count": later_mature_count,
                    "immature_group_not_mature_at_evaluation_count": int(
                        (immature_mask & ~eval_mature_mask).sum()
                    ),
                    "retrospective_default_count_within_immature_group": later_default_count,
                    "retrospective_default_rate_of_all_cutoff_immature": (
                        later_default_count / immature_count if immature_count else None
                    ),
                    "retrospective_default_rate_among_immature_matured_at_evaluation": (
                        later_default_count / later_mature_count if later_mature_count else None
                    ),
                    "naive_false_negative_count": int(
                        group[f"naive_false_negative_{horizon}m"].sum()
                    ),
                    "common_population_training_count": int(complete_mask.sum()),
                    "naive_vs_complete_disagreement_on_common_population": int(
                        common_disagreement.sum()
                    ),
                    "same_population_labels_identical": not bool(common_disagreement.any()),
                }
            )

    report = pd.DataFrame(rows)
    totals = []
    for horizon, group in report.groupby("horizon_months"):
        cohort = int(group["training_cohort_count"].sum())
        immature = int(group["training_immature_or_ineligible_count"].sum())
        later_mature = int(group["immature_group_mature_at_evaluation_count"].sum())
        later_defaults = int(group["retrospective_default_count_within_immature_group"].sum())
        common_disagreements = int(
            group["naive_vs_complete_disagreement_on_common_population"].sum()
        )
        totals.append(
            {
                "vintage": "ALL",
                "training_cutoff": group["training_cutoff"].iloc[0],
                "horizon_months": int(horizon),
                "training_cohort_count": cohort,
                "training_complete_horizon_eligible_count": int(
                    group["training_complete_horizon_eligible_count"].sum()
                ),
                "training_immature_or_ineligible_count": immature,
                "naive_positive_count_within_immature_group": int(
                    group["naive_positive_count_within_immature_group"].sum()
                ),
                "immature_group_mature_at_evaluation_count": later_mature,
                "immature_group_not_mature_at_evaluation_count": int(
                    group["immature_group_not_mature_at_evaluation_count"].sum()
                ),
                "retrospective_default_count_within_immature_group": later_defaults,
                "retrospective_default_rate_of_all_cutoff_immature": (
                    later_defaults / immature if immature else None
                ),
                "retrospective_default_rate_among_immature_matured_at_evaluation": (
                    later_defaults / later_mature if later_mature else None
                ),
                "naive_false_negative_count": int(group["naive_false_negative_count"].sum()),
                "common_population_training_count": int(
                    group["common_population_training_count"].sum()
                ),
                "naive_vs_complete_disagreement_on_common_population": common_disagreements,
                "same_population_labels_identical": common_disagreements == 0,
            }
        )
    return pd.concat([report, pd.DataFrame(totals)], ignore_index=True)


def summarize_mature_coverage(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vintage, group in evaluation.groupby("vintage"):
        for horizon in HORIZONS:
            eligibility = [
                complete_horizon_label(row, horizon)
                for row in group.to_dict(orient="records")
            ]
            exclusion_reasons = [
                mature_exclusion_reason(row, horizon)
                for row in group.to_dict(orient="records")
            ]
            mature = [eligible for eligible, _ in eligibility]
            labels = [label for _, label in eligibility]
            mature_count = sum(mature)
            exclusion_counts = {
                reason: exclusion_reasons.count(reason)
                for reason in (
                    "missing_entry",
                    "delayed_entry",
                    "default_before_entry",
                    "other_termination_within_horizon",
                    "right_censored_before_horizon",
                    "no_observed_horizon_followup",
                    "other_unresolved",
                )
            }
            excluded_count = len(group) - mature_count
            if sum(exclusion_counts.values()) != excluded_count:
                raise ValueError(
                    f"Mature eligibility exclusions do not reconcile for vintage {vintage}, "
                    f"horizon {horizon}: {sum(exclusion_counts.values())} categorized, "
                    f"{excluded_count} excluded."
                )
            rows.append(
                {
                    "vintage": int(vintage),
                    "evaluation_cutoff": str(group["information_cutoff"].iloc[0]),
                    "horizon_months": horizon,
                    "origination_loans": len(group),
                    "loans_with_observed_history": int(group["entry_month"].notna().sum()),
                    "mature_eligible_count": int(mature_count),
                    "mature_ineligible_count": int(excluded_count),
                    "immature_or_left_truncated_count": int(excluded_count),
                    "mature_excluded_missing_entry_count": exclusion_counts["missing_entry"],
                    "mature_excluded_delayed_entry_count": exclusion_counts["delayed_entry"],
                    "mature_excluded_default_before_entry_count": exclusion_counts["default_before_entry"],
                    "mature_excluded_other_termination_within_horizon_count": exclusion_counts[
                        "other_termination_within_horizon"
                    ],
                    "mature_excluded_right_censored_before_horizon_count": exclusion_counts[
                        "right_censored_before_horizon"
                    ],
                    "mature_excluded_no_observed_horizon_followup_count": exclusion_counts[
                        "no_observed_horizon_followup"
                    ],
                    "mature_excluded_other_unresolved_count": exclusion_counts["other_unresolved"],
                    "mature_exclusion_reconciliation_count": sum(exclusion_counts.values()),
                    "mature_eligible_rate": mature_count / len(group) if len(group) else None,
                    "mature_default_event_count": int(
                        sum(eligible and label == 1 for eligible, label in eligibility)
                    ),
                    "default_before_risk_entry_count": int(
                        (
                            (group["outcome_type"] == "default")
                            & (group["followup_months"] < group["entry_month"])
                        ).sum()
                    ),
                    "mature_event_rate": (
                        sum(eligible and label == 1 for eligible, label in eligibility) / mature_count
                        if mature_count
                        else None
                    ),
                    "voluntary_payoff_within_horizon_count": int(
                        (
                            (group["outcome_type"] == "voluntary_payoff")
                            & (group["outcome_month"] <= horizon)
                        ).sum()
                    ),
                    "other_termination_within_horizon_count": int(
                        (
                            (group["outcome_type"] == "other_termination")
                            & (group["outcome_month"] <= horizon)
                        ).sum()
                    ),
                    "right_censored_count": int((group["outcome_type"] == "right_censored").sum()),
                    "right_censored_before_horizon_count": int(
                        (
                            (group["outcome_type"] == "right_censored")
                            & (group["followup_months"] >= group["entry_month"])
                            & (group["followup_months"] < horizon)
                        ).sum()
                    ),
                    "right_censoring_rate_before_horizon": (
                        int(
                            (
                                (group["outcome_type"] == "right_censored")
                                & (group["followup_months"] >= group["entry_month"])
                                & (group["followup_months"] < horizon)
                            ).sum()
                        )
                        / len(group)
                        if len(group)
                        else None
                    ),
                    "other_termination_censored_within_horizon_count": int(
                        (
                            (group["outcome_type"] == "other_termination")
                            & (group["followup_months"] >= group["entry_month"])
                            & (group["followup_months"] <= horizon)
                        ).sum()
                    ),
                    "censoring_count_within_horizon": int(
                        (
                            (group["outcome_type"] == "right_censored")
                            & (group["followup_months"] >= group["entry_month"])
                            & (group["followup_months"] < horizon)
                        ).sum()
                        + (
                            (group["outcome_type"] == "other_termination")
                            & (group["followup_months"] >= group["entry_month"])
                            & (group["followup_months"] <= horizon)
                        ).sum()
                    ),
                    "censoring_rate_within_horizon": (
                        (
                            (
                                (group["outcome_type"] == "right_censored")
                                & (group["followup_months"] >= group["entry_month"])
                                & (group["followup_months"] < horizon)
                            ).sum()
                            + (
                                (group["outcome_type"] == "other_termination")
                                & (group["followup_months"] >= group["entry_month"])
                                & (group["followup_months"] <= horizon)
                            ).sum()
                        )
                        / len(group)
                        if len(group)
                        else None
                    ),
                    "left_truncated_count": int((group["entry_month"] > 1).sum()),
                }
            )
    return pd.DataFrame(rows)


def _load_outcomes(years: list[int], cutoff: str) -> pd.DataFrame:
    frames = []
    tag = cutoff_tag(cutoff)
    for year in years:
        path = OUTCOME_DIR / f"outcomes_{year}_asof_{tag}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required outcome table missing for {year} at {cutoff}: {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2005, 2025)))
    parser.add_argument(
        "--evaluation-years",
        nargs="+",
        type=int,
        help="Vintages used for retrospective maturity coverage (defaults to --years).",
    )
    parser.add_argument("--training-cutoff", default="2014-12-31")
    parser.add_argument("--evaluation-cutoff", default="2026-03-31")
    args = parser.parse_args()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    training = _load_outcomes(args.years, args.training_cutoff)
    evaluation_years = args.evaluation_years or args.years
    evaluation = _load_outcomes(evaluation_years, args.evaluation_cutoff)
    labels = generate_strategy_rows(training, evaluation, args.training_cutoff)
    tag = cutoff_tag(args.training_cutoff)
    output_path = LABEL_DIR / f"label_strategies_asof_{tag}.parquet"
    labels.to_parquet(output_path, index=False)
    summary = summarize_labels(labels)
    report_path = config.TABLES_DIR / f"label_distortion_report_asof_{tag}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_path, index=False)
    sensitivity = summarize_label_sensitivity(labels)
    sensitivity_path = config.TABLES_DIR / f"label_sensitivity_report_asof_{tag}.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    coverage = summarize_mature_coverage(evaluation)
    coverage_path = config.TABLES_DIR / (
        f"mature_outcome_coverage_asof_{cutoff_tag(args.evaluation_cutoff)}.csv"
    )
    coverage.to_csv(coverage_path, index=False)

    settings = {
        "training_cutoff": args.training_cutoff,
        "retrospective_evaluation_cutoff": args.evaluation_cutoff,
        "training_vintages": args.years,
        "evaluation_coverage_vintages": evaluation_years,
        "horizons_months": HORIZONS,
        "strategy_a": "A loan is positive only for a qualifying default observed by the training cutoff within the horizon; all other cutoff-known loans receive zero.",
        "strategy_b": "Use only loans with entry at month 1 and an ascertainable horizon outcome; early voluntary payoff is a competing event and establishes no later default; unresolved/other terminations and defaults before risk entry are excluded.",
        "strategy_c": "Retain observed time at risk with delayed entry, default event code 1, voluntary payoff competing event code 2, and administrative/other termination censoring code 0. Exclude terminal/default events before risk entry and record the exclusion reason.",
        "retrospective_false_negative": "Naive zero label at training cutoff, later mature 1 label at evaluation cutoff.",
        "label_sensitivity": (
            "Report immature-at-training subgroup size, the count with mature evaluation outcomes, "
            "retrospective defaults and rates over both denominators. Separately compare naive and "
            "complete-horizon targets on the shared complete-horizon-eligible training population; "
            "agreement is expected by construction."
        ),
        "censoring_report": "Censoring before a horizon includes observed history ending strictly before the horizon and other termination code 15/16/96 on or before the horizon, provided the event is not before risk entry; rates use all origination rows in the vintage as denominator.",
        "outputs": [str(output_path), str(report_path)],
        "label_sensitivity_report": str(sensitivity_path),
        "coverage_report": str(coverage_path),
    }
    settings_path = LABEL_DIR / f"label_strategy_settings_asof_{tag}.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved labels: {output_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved label sensitivity report: {sensitivity_path}")
    print(f"Saved mature outcome coverage: {coverage_path}")
    print(f"Saved settings: {settings_path}")


if __name__ == "__main__":
    main()
