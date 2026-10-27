from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

CONFIG_PATH = Path(__file__).with_name("00_config.py")
spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

DATA_DIR = config.DATA_DIR
EXTRACTED_DIR = config.EXTRACTED_DIR
INVENTORY_DIR = config.INVENTORY_DIR
RAW_ZIP_DIR = config.RAW_ZIP_DIR
ensure_directories = config.ensure_directories
stage_raw_zips = config.stage_raw_zips
find_raw_zip_files = config.find_raw_zip_files


def load_year_parquet(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = EXTRACTED_DIR / f"vintage_{year}"
    orig = pd.read_parquet(base / "origination.parquet")
    perf = pd.read_parquet(base / "performance.parquet")
    return orig, perf


def check_unique_loan_id(orig: pd.DataFrame, loan_id_col: int = 19) -> None:
    seq_values = orig.iloc[:, loan_id_col]
    duplicates = seq_values.duplicated().sum()
    if duplicates:
        raise ValueError(f"Origination table has {duplicates} duplicate loan sequence numbers in column {loan_id_col}.")


def check_unique_perf_key(perf: pd.DataFrame, loan_id_col: int = 0, period_col: int = 1) -> None:
    seq_values = perf.iloc[:, loan_id_col]
    period_values = perf.iloc[:, period_col]
    duplicates = perf.duplicated(subset=[perf.columns[loan_id_col], perf.columns[period_col]], keep=False).sum()
    if duplicates:
        raise ValueError(f"Performance table has {duplicates} duplicate (loan_id, reporting_period) rows.")


def validate_reporting_periods(perf: pd.DataFrame, period_col: int = 1) -> None:
    period_values = perf.iloc[:, period_col]
    bad = perf[~period_values.astype(str).str.match(r"^\d{6}$", na=False)]
    if not bad.empty:
        raise ValueError(f"Found invalid reporting periods in performance table: {bad.head(5).to_dict(orient='records')}")


def main() -> None:
    ensure_directories()
    stage_raw_zips()
    summary: list[dict] = []
    for year in range(2005, 2025):
        year_zip = RAW_ZIP_DIR / f"sample_{year}.zip"
        root_zip = Path(__file__).resolve().parent / f"sample_{year}.zip"
        if not year_zip.exists() and not root_zip.exists():
            continue
        base = EXTRACTED_DIR / f"vintage_{year}"
        if not base.exists():
            continue
        orig, perf = load_year_parquet(year)
        check_unique_loan_id(orig, loan_id_col=19)
        check_unique_perf_key(perf, loan_id_col=0, period_col=1)
        validate_reporting_periods(perf, period_col=1)

        match_count = perf.iloc[:, 0].isin(orig.iloc[:, 19]).sum()
        summary.append(
            {
                "year": year,
                "orig_rows": int(orig.shape[0]),
                "perf_rows": int(perf.shape[0]),
                "matched_perf_rows": int(match_count),
                "unique_orig_ids": int(orig.iloc[:, 19].nunique()),
                "unique_perf_ids": int(perf.iloc[:, 0].nunique()),
                "unique_perf_pairs": int(perf.iloc[:, [0, 1]].drop_duplicates().shape[0]),
            }
        )

    output_path = INVENTORY_DIR / "data_quality_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary[:3], indent=2))
    print(f"Saved quality summary to {output_path}.")


if __name__ == "__main__":
    main()
