from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ZIP_DIR = DATA_DIR / "raw_zips"
EXTRACTED_DIR = DATA_DIR / "extracted"
INVENTORY_DIR = DATA_DIR / "inventory"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOG_DIR = PROJECT_ROOT / "logs"

PRIMARY_HORIZONS = [12, 24, 36]
SELECTED_VINTAGES = list(range(2005, 2025))
PRIMARY_TRAIN_YEARS = list(range(2005, 2015))
PRIMARY_VALIDATION_YEARS = [2015, 2016]
PRIMARY_TEST_YEARS = [2017, 2018, 2019]
LATE_TEST_YEARS = [2020, 2021, 2022]
SHORT_HORIZON_YEARS = [2023, 2024]

FILE_LAYOUT_NOTE = (
    "The Freddie Mac sample ZIP archives are pipe-delimited flat files with no header row. "
    "Field positions and names are verified against the July 2026 Release 47 workbook and "
    "official file-header archive. The extraction pipeline reads raw data as strings and "
    "rejects rows that do not match the versioned Release 47 schema contract."
)

EXPECTED_FILE_LAYOUT = {
    "origination": {
        "kind": "origination",
        "required_tokens": 31,
        "filename_markers": ("orig",),
    },
    "performance": {
        "kind": "performance",
        "required_tokens": 35,
        "filename_markers": ("perf",),
    },
}

REPORTING_PERIOD_FORMAT = "%Y%m"
RANDOM_SEED = 42


def ensure_directories() -> None:
    for directory in [
        DATA_DIR,
        RAW_ZIP_DIR,
        EXTRACTED_DIR,
        INVENTORY_DIR,
        MODEL_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def find_raw_zip_files() -> list[Path]:
    search_roots = [
        PROJECT_ROOT,
        DATA_DIR,
        RAW_ZIP_DIR,
    ]
    seen = set()
    results: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("sample_*.zip")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                results.append(path)
    return results


def stage_raw_zips() -> list[Path]:
    ensure_directories()
    results = find_raw_zip_files()
    if not results:
        return []
    for source in results:
        target = RAW_ZIP_DIR / source.name
        if not target.exists() or source.stat().st_size != target.stat().st_size:
            target.write_bytes(source.read_bytes())
    return sorted(RAW_ZIP_DIR.glob("sample_*.zip"))


if __name__ == "__main__":
    ensure_directories()
    print("Project root:", PROJECT_ROOT)
    print("Available sample vintages:", SELECTED_VINTAGES)
    print(FILE_LAYOUT_NOTE)
    print("Raw zips detected:", [p.name for p in find_raw_zip_files()[:5]])
