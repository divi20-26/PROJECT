from __future__ import annotations

import importlib.util
import json
import zipfile
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
EXPECTED_FILE_LAYOUT = config.EXPECTED_FILE_LAYOUT
ensure_directories = config.ensure_directories
stage_raw_zips = config.stage_raw_zips
find_raw_zip_files = config.find_raw_zip_files


def detect_kind(filename: str) -> str:
    lower = filename.lower()
    if "orig" in lower:
        return "origination"
    if "perf" in lower:
        return "performance"
    raise ValueError(f"Unrecognized archive member: {filename!r}.")


def extract_zip(zip_path: Path, output_dir: Path) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.namelist() if member.endswith(".txt")]
        if len(members) != 2:
            raise ValueError(f"Expected 2 .txt members in {zip_path.name}, found {len(members)}: {members}")

        for member in members:
            kind = detect_kind(member)
            expected_width = EXPECTED_FILE_LAYOUT[kind]["required_tokens"]
            raw_text = archive.read(member).decode("latin1")
            lines = [line for line in raw_text.splitlines() if line.strip()]
            if not lines:
                raise ValueError(f"Archive member {member} in {zip_path.name} is empty.")
            widths = [len(line.split("|")) for line in lines]
            if len(set(widths)) != 1 or widths[0] != expected_width:
                raise ValueError(
                    f"Schema mismatch for {zip_path.name}:{member}. "
                    f"Expected {expected_width} pipe-delimited fields, observed counts={sorted(set(widths))}."
                )
            destination = output_dir / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(member))
            extracted[kind] = destination
    return extracted


def read_raw_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "N/A"],
        engine="python",
    )
    if df.empty:
        raise ValueError(f"File {path} contained no rows.")
    return df


def convert_to_parquet(year: int) -> dict[str, Path]:
    candidates = [
        DATA_DIR / f"sample_{year}.zip",
        RAW_ZIP_DIR / f"sample_{year}.zip",
        Path(__file__).resolve().parent / f"sample_{year}.zip",
    ]
    zip_path = next((p for p in candidates if p.exists()), None)
    if zip_path is None:
        zip_path_candidates = find_raw_zip_files()
        zip_path = next((p for p in zip_path_candidates if p.name == f"sample_{year}.zip"), None)
    if zip_path is None:
        raise FileNotFoundError(f"Missing sample ZIP for year {year} in any expected directory.")

    extracted_dir = EXTRACTED_DIR / f"vintage_{year}"
    files = extract_zip(zip_path, extracted_dir)
    outputs = {}
    for kind, path in files.items():
        df = read_raw_file(path)
        expected_width = EXPECTED_FILE_LAYOUT[kind]["required_tokens"]
        if df.shape[1] != expected_width:
            raise ValueError(
                f"Expected {expected_width} columns for {kind} data in {path.name}; "
                f"actual columns={df.shape[1]}."
            )
        parquet_path = extracted_dir / f"{kind}.parquet"
        df.to_parquet(parquet_path, index=False)
        outputs[kind] = parquet_path
    return outputs


def main() -> None:
    ensure_directories()
    stage_raw_zips()
    for year in range(2005, 2025):
        year_zip = RAW_ZIP_DIR / f"sample_{year}.zip"
        root_zip = Path(__file__).resolve().parent / f"sample_{year}.zip"
        if not year_zip.exists() and not root_zip.exists():
            continue
        outputs = convert_to_parquet(year)
        print(f"Year {year}: {outputs}")

    manifest_path = INVENTORY_DIR / "extraction_manifest.json"
    manifest_path.write_text(json.dumps({"status": "ok", "years_processed": list(range(2005, 2025))}, indent=2), encoding="utf-8")
    print(f"Saved extraction manifest to {manifest_path}.")


if __name__ == "__main__":
    main()
