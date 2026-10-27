"""Build loan-level default, competing-payoff, delayed-entry, and censoring outcomes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd

from schema_contract import SCHEMA, map_frame


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "00_config.py"
config_spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config)

EXTRACTED_DIR = config.EXTRACTED_DIR
INVENTORY_DIR = config.INVENTORY_DIR
OUTCOME_DIR = config.DATA_DIR / "outcomes"
PRIMARY_DEFAULT_ZBC = ("02", "03", "09")
PAYOFF_ZBC = ("01",)
OTHER_TERMINATION_ZBC = ("15", "16", "96")
KNOWN_ZBC = PRIMARY_DEFAULT_ZBC + PAYOFF_ZBC + OTHER_TERMINATION_ZBC
DEFAULT_AS_OF = "2026-03-31"


def parse_month(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if len(text) != 6 or not text.isdigit():
        raise ValueError(f"Expected YYYYMM month, got {value!r}.")
    year, month = int(text[:4]), int(text[4:])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid YYYYMM month: {value!r}.")
    return year * 12 + month


def relative_month(value: object, first_payment: object) -> int | None:
    value_month = parse_month(value)
    first_month = parse_month(first_payment)
    if value_month is None or first_month is None:
        return None
    return value_month - first_month + 1


def choose_observed_outcome(row: dict) -> tuple[str, str | None, int | None]:
    candidates = []
    if row.get("default_event_month") is not None:
        candidates.append((row["default_event_month"], 1, "default"))
    if row.get("payoff_month") is not None:
        candidates.append((row["payoff_month"], 2, "voluntary_payoff"))
    if row.get("other_termination_month") is not None:
        candidates.append((row["other_termination_month"], 3, "other_termination"))
    if not candidates:
        return "right_censored", None, None
    # Monthly data cannot order events within the same month; classify a qualifying
    # credit event in a tied month as the event for the primary research endpoint.
    time_month, _, kind = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    date_field = {
        "default": "default_event_date",
        "voluntary_payoff": "payoff_date",
        "other_termination": "other_termination_date",
    }[kind]
    return kind, row.get(date_field), time_month


def finalize_outcome_row(row: dict) -> dict:
    first_payment = row.get("first_payment_date")
    first_observed = row.get("first_observed_period")
    last_observed = row.get("last_observed_period")

    row["entry_month"] = max(1, relative_month(first_observed, first_payment) or 1) if first_observed else None
    row["last_observed_month"] = relative_month(last_observed, first_payment) if last_observed else None
    row["left_truncated"] = bool(row["entry_month"] and row["entry_month"] > 1)

    for date_field, month_field in (
        ("default_event_date", "default_event_month"),
        ("payoff_date", "payoff_month"),
        ("other_termination_date", "other_termination_month"),
    ):
        row[month_field] = relative_month(row.get(date_field), first_payment)

    outcome_type, outcome_date, outcome_month = choose_observed_outcome(row)
    row["outcome_type"] = outcome_type
    row["outcome_date"] = outcome_date
    row["outcome_month"] = outcome_month
    if outcome_type == "right_censored":
        row["followup_months"] = row["last_observed_month"]
    else:
        row["followup_months"] = outcome_month
    row["event_observed"] = outcome_type == "default"
    row["competing_payoff"] = outcome_type == "voluntary_payoff"
    row["censored"] = outcome_type in {"right_censored", "other_termination"}
    row["survival_event_code"] = {
        "default": 1,
        "voluntary_payoff": 2,
        "right_censored": 0,
        "other_termination": 0,
    }[outcome_type]
    return row


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def aggregate_performance(perf_path: Path, cutoff_yyyymm: str) -> pd.DataFrame:
    cutoff = cutoff_yyyymm.replace("-", "")[:6]
    if len(cutoff) != 6 or not cutoff.isdigit():
        raise ValueError(f"Cutoff must be YYYY-MM-DD or YYYYMM, got {cutoff_yyyymm!r}.")
    path_sql = _sql_string(str(perf_path))
    default_codes = ",".join(_sql_string(code) for code in PRIMARY_DEFAULT_ZBC)
    payoff_codes = ",".join(_sql_string(code) for code in PAYOFF_ZBC)
    other_codes = ",".join(_sql_string(code) for code in OTHER_TERMINATION_ZBC)
    known_codes = ",".join(_sql_string(code) for code in KNOWN_ZBC)

    connection = duckdb.connect()
    try:
        unexpected_codes = connection.execute(
            f"""
            SELECT DISTINCT lpad(trim(coalesce("8", '')), 2, '0') AS code
            FROM read_parquet({path_sql})
            WHERE trim(coalesce("1", '')) <= ?
              AND trim(coalesce("8", '')) <> ''
              AND lpad(trim(coalesce("8", '')), 2, '0') NOT IN ({known_codes})
            """,
            [cutoff],
        ).fetchall()
        if unexpected_codes:
            raise ValueError(
                f"Unmapped nonblank zero-balance codes in {perf_path}: "
                f"{[row[0] for row in unexpected_codes]}"
            )

        missing_effective_date = connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet({path_sql})
            WHERE trim(coalesce("1", '')) <= ?
              AND lpad(trim(coalesce("8", '')), 2, '0') IN ({known_codes})
              AND nullif(trim(coalesce("9", '')), '') IS NULL
            """,
            [cutoff],
        ).fetchone()[0]
        if missing_effective_date:
            raise ValueError(
                f"{missing_effective_date} terminal records in {perf_path} have a documented "
                "zero-balance code but no Zero Balance Effective Date."
            )

        query = f"""
        WITH p AS (
            SELECT
                trim("0") AS loan_identifier,
                trim("1") AS period,
                trim(coalesce("3", '')) AS delinquency_status,
                lpad(trim(coalesce("8", '')), 2, '0') AS zero_balance_code,
                nullif(trim(coalesce("9", '')), '') AS zero_balance_effective_date
            FROM read_parquet({path_sql})
            WHERE trim(coalesce("1", '')) <= {_sql_string(cutoff)}
        )
        SELECT
            loan_identifier,
            min(period) AS first_observed_period,
            max(period) AS last_observed_period,
            min(CASE
                WHEN delinquency_status = 'RA'
                  OR coalesce(try_cast(delinquency_status AS INTEGER), 0) >= 3
                THEN period END) AS first_default_date_status,
            arg_min(period, CASE
                WHEN delinquency_status = 'RA'
                  OR coalesce(try_cast(delinquency_status AS INTEGER), 0) >= 3
                THEN period END) AS first_default_report_period_status,
            min(CASE WHEN zero_balance_code IN ({default_codes})
                THEN zero_balance_effective_date END) AS first_default_date_terminal,
            arg_min(period, CASE WHEN zero_balance_code IN ({default_codes})
                THEN zero_balance_effective_date END) AS first_default_report_period_terminal,
            min(CASE WHEN zero_balance_code IN ({payoff_codes})
                THEN zero_balance_effective_date END) AS payoff_date,
            arg_min(period, CASE WHEN zero_balance_code IN ({payoff_codes})
                THEN zero_balance_effective_date END) AS payoff_report_period,
            min(CASE WHEN zero_balance_code IN ({other_codes})
                THEN zero_balance_effective_date END) AS other_termination_date,
            arg_min(period, CASE WHEN zero_balance_code IN ({other_codes})
                THEN zero_balance_effective_date END) AS other_termination_report_period
        FROM p
        GROUP BY loan_identifier
        """
        return connection.execute(query).fetchdf()
    finally:
        connection.close()


def build_vintage_outcomes(year: int, cutoff: str) -> pd.DataFrame:
    vintage_dir = EXTRACTED_DIR / f"vintage_{year}"
    orig_path = vintage_dir / "origination.parquet"
    perf_path = vintage_dir / "performance.parquet"
    if not orig_path.exists() or not perf_path.exists():
        raise FileNotFoundError(
            f"Missing extracted parquet inputs for vintage {year}; expected {orig_path} and {perf_path}."
        )
    orig = map_frame(pd.read_parquet(orig_path), "origination")
    if orig["loan_identifier"].duplicated().any():
        raise ValueError(f"Duplicate origination loan identifiers in vintage {year}.")
    performance = aggregate_performance(perf_path, cutoff)
    joined = orig[["loan_identifier", "first_payment_date"]].merge(
        performance, on="loan_identifier", how="left", validate="one_to_one"
    )

    rows = []
    for record in joined.to_dict(orient="records"):
        default_options = []
        for date_field, report_field in (
            ("first_default_date_status", "first_default_report_period_status"),
            ("first_default_date_terminal", "first_default_report_period_terminal"),
        ):
            if record.get(date_field) and record.get(report_field):
                default_options.append((str(record[date_field]), str(record[report_field])))
        if default_options:
            default_date, default_report_period = min(default_options, key=lambda item: item[0])
        else:
            default_date = default_report_period = None

        row = {
            "vintage": year,
            "loan_identifier": record["loan_identifier"],
            "first_payment_date": record["first_payment_date"],
            "first_observed_period": record.get("first_observed_period"),
            "last_observed_period": record.get("last_observed_period"),
            "default_event_date": default_date,
            "default_report_period": default_report_period,
            "payoff_date": record.get("payoff_date"),
            "payoff_report_period": record.get("payoff_report_period"),
            "other_termination_date": record.get("other_termination_date"),
            "other_termination_report_period": record.get("other_termination_report_period"),
            "information_cutoff": cutoff,
        }
        if row["default_event_date"] and row["first_observed_period"]:
            if str(row["default_event_date"]) < str(row["first_observed_period"]):
                raise ValueError(
                    f"Default event predates first observed performance for loan "
                    f"{row['loan_identifier']}: {row['default_event_date']} < {row['first_observed_period']}"
                )
        rows.append(finalize_outcome_row(row))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2005, 2025)))
    parser.add_argument("--cutoff", default=DEFAULT_AS_OF)
    args = parser.parse_args()
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for year in args.years:
        outcomes = build_vintage_outcomes(year, args.cutoff)
        cutoff_tag = args.cutoff.replace("-", "")
        path = OUTCOME_DIR / f"outcomes_{year}_asof_{cutoff_tag}.parquet"
        outcomes.to_parquet(path, index=False)
        outputs.append({"year": year, "loans": len(outcomes), "output": str(path)})
        print(outputs[-1])

    settings = {
        "schema_version": SCHEMA["schema_version"],
        "release": SCHEMA["release"],
        "information_cutoff": args.cutoff,
        "primary_default_definition": (
            "First monthly status with numeric delinquency status >= 3 (90+ days), REO Acquisition "
            "status RA, or Zero Balance Code 02/03/09; event month uses Zero Balance Effective Date "
            "for terminal events and Period for status events."
        ),
        "competing_payoff_codes": {"01": "voluntary prepayment or maturity"},
        "other_termination_codes_censored": {
            "15": "whole loan sale",
            "16": "reperforming loan securitization",
            "96": "confirmed defect prior to another credit event",
        },
        "entry_and_followup": (
            "Time origin is First Payment Date. Performance begins at acquisition, so entry is delayed "
            "to the first observed Period (month index, minimum 1); the analysis is conditional on "
            "loans observed active at acquisition. The unobserved origination-to-acquisition interval "
            "is not imputed. If a qualifying default is already recorded before entry, retain its "
            "observed event in the outcome table but exclude it from complete-horizon and survival "
            "risk sets."
        ),
        "monthly_ties": "A primary credit event takes precedence over another event recorded in the same month.",
        "outputs": outputs,
    }
    settings_path = OUTCOME_DIR / f"outcome_definitions_asof_{args.cutoff.replace('-', '')}.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Saved outcome definitions to {settings_path}")


if __name__ == "__main__":
    main()
