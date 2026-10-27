"""Create an origination-only feature table and auditable feature dictionary."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from schema_contract import SCHEMA, map_frame


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "00_config.py"
config_spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config)

EXTRACTED_DIR = config.EXTRACTED_DIR
FEATURE_DIR = config.DATA_DIR / "features"

FEATURE_SPEC = [
    ("classic_fico", "numeric", ["", "9999"], "Origination credit score."),
    ("vantagescore_4", "numeric", ["", "9999"], "Additional origination-time credit score; retain only when disclosed."),
    ("first_time_homebuyer_indicator", "categorical", ["", "9"], "Documented Y/N indicator; 9 is not available/not applicable."),
    ("mortgage_insurance_percentage", "numeric", ["", "999"], "Origination MI coverage; zero is a valid no-MI value."),
    ("number_of_units", "categorical", ["", "99"], "Property unit count; official valid values are 1-4."),
    ("occupancy_status", "categorical", ["", "9"], "Origination occupancy status; documented categories P/I/S."),
    ("original_cltv", "numeric", ["", "999"], "Origination combined loan-to-value percentage."),
    ("original_dti", "numeric", ["", "999"], "Origination debt-to-income percentage."),
    ("original_upb", "numeric", [""], "Original unpaid principal balance; no additional sentinel specified in the guide."),
    ("original_ltv", "numeric", ["", "999"], "Origination loan-to-value percentage."),
    ("original_interest_rate", "numeric", [""], "Interest rate at origination."),
    ("channel", "categorical", ["", "9"], "Origination channel; 9 is not available."),
    ("amortization_type", "categorical", [""], "Origination amortization type."),
    ("property_state", "categorical", [""], "State where the property is located."),
    ("property_type", "categorical", [""], "Origination property type."),
    ("loan_purpose", "categorical", [""], "Origination loan purpose."),
    ("original_loan_term", "numeric", [""], "Original scheduled loan term in months."),
    ("number_of_borrowers", "categorical", ["", "99"], "Borrower count; 99 is not available."),
    ("super_conforming_flag", "categorical", [""], "Origination-time Y/N program flag; N is a valid non-super-conforming value."),
    ("special_eligibility_program", "categorical", [""], "Origination-time program indicator; blank is documented as not available/not applicable."),
    ("harp_indicator", "categorical", [""], "Origination-time Y/N Relief Refinance indicator."),
    ("property_valuation_method", "categorical", ["", "7"], "Origination-time valuation method; 7 is not available."),
    ("interest_only_indicator", "categorical", [""], "Origination-time Y/N interest-only indicator.")
]

FEATURE_NAMES = [spec[0] for spec in FEATURE_SPEC]
NUMERIC_BOUNDS = {
    "classic_fico": (300, 850),
    "vantagescore_4": (300, 850),
}
LEAKAGE_FIELDS = {
    "loan_identifier",
    "first_payment_date",
    "maturity_date",
    "pre_harp_loan_sequence_number",
    "seller_name",
    "postal_code",
    "msa",
}


def feature_dictionary() -> list[dict]:
    return [
        {
            "feature": name,
            "source_field": name,
            "source_position": next(
                field["position"] for field in SCHEMA["origination"] if field["name"] == name
            ),
            "data_type": data_type,
            "missing_value_codes": missing_codes,
            "transform": (
                "parse numeric; replace only documented sentinels/blanks and documented out-of-range credit scores with missing"
                if data_type == "numeric"
                else "retain as categorical string; replace only documented sentinel and blank codes with missing"
            ),
            "documented_valid_range": NUMERIC_BOUNDS.get(name),
            "reason": reason,
            "leakage_review": "origination table field; excludes identifiers and subsequent performance data",
        }
        for name, data_type, missing_codes, reason in FEATURE_SPEC
    ]


def build_feature_frame(origination: pd.DataFrame) -> pd.DataFrame:
    orig = map_frame(origination, "origination")
    if not set(FEATURE_NAMES).issubset(orig.columns):
        raise ValueError("Feature whitelist contains a field absent from the Release 47 origination schema.")
    if set(FEATURE_NAMES) & LEAKAGE_FIELDS:
        raise ValueError("Feature leakage audit failed: identifier or excluded field is selected.")

    output = pd.DataFrame(index=orig.index)
    output["loan_identifier"] = orig["loan_identifier"]
    for name, data_type, missing_codes, _ in FEATURE_SPEC:
        values = orig[name].astype("string").str.strip()
        values = values.mask(values.isin(missing_codes), pd.NA)
        if data_type == "numeric":
            parsed = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & parsed.isna()
            if invalid.any():
                raise ValueError(
                    f"Unexpected nonnumeric values in feature {name}: "
                    f"{values[invalid].drop_duplicates().head(5).tolist()}"
                )
            if name in NUMERIC_BOUNDS:
                minimum, maximum = NUMERIC_BOUNDS[name]
                parsed = parsed.mask((parsed < minimum) | (parsed > maximum), pd.NA)
            output[name] = parsed
        else:
            output[name] = values
    output["model_eligible"] = (
        orig["amortization_type"].astype("string").str.strip().eq("FRM")
        & orig["interest_only_indicator"].astype("string").str.strip().eq("N")
    )
    output["model_eligibility_reason"] = output["model_eligible"].map(
        {True: "fully_amortizing_fixed_rate", False: "not_fully_amortizing_fixed_rate_or_missing_flag"}
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2005, 2025)))
    args = parser.parse_args()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    dict_path = FEATURE_DIR / "feature_dictionary_release_47.json"
    dict_path.write_text(json.dumps(feature_dictionary(), indent=2), encoding="utf-8")
    outputs = []
    eligibility_summary = []
    for year in args.years:
        path = EXTRACTED_DIR / f"vintage_{year}" / "origination.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing origination Parquet for vintage {year}: {path}")
        features = build_feature_frame(pd.read_parquet(path))
        output_path = FEATURE_DIR / f"features_{year}_release_47.parquet"
        features.to_parquet(output_path, index=False)
        eligible_count = int(features["model_eligible"].sum())
        outputs.append(
            {
                "year": year,
                "loans": len(features),
                "model_eligible": eligible_count,
                "output": str(output_path),
            }
        )
        eligibility_summary.append(
            {
                "vintage": year,
                "origination_loans": len(features),
                "model_eligible_count": eligible_count,
                "model_ineligible_count": len(features) - eligible_count,
                "model_eligible_rate": eligible_count / len(features) if len(features) else None,
            }
        )
        print(outputs[-1])
    summary_path = config.TABLES_DIR / "feature_eligibility_release_47.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eligibility_summary).to_csv(summary_path, index=False)
    print(f"Saved feature dictionary to {dict_path}")
    print(f"Saved feature eligibility summary to {summary_path}")


if __name__ == "__main__":
    main()
