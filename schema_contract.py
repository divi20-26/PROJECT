from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "sflld_release_47.json"
HEADER_ZIP_PATH = PROJECT_ROOT / "data" / "documentation" / "file_headers_july_2026.zip"


def load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


SCHEMA = load_schema()


def field_names(file_type: str) -> list[str]:
    if file_type not in ("origination", "performance"):
        raise ValueError(f"Unknown SFLLD file type: {file_type}")
    return [field["name"] for field in SCHEMA[file_type]]


def normalize_header(value: str) -> str:
    value = value.replace("®", "").replace("™", "")
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def validate_header_archive(header_zip: Path = HEADER_ZIP_PATH) -> dict[str, list[str]]:
    if not header_zip.exists():
        raise FileNotFoundError(f"Official July 2026 header archive not found: {header_zip}")
    expected_members = {
        "origination_data_file_header.txt": "origination",
        "performance_data_file_header.txt": "performance",
    }
    validated: dict[str, list[str]] = {}
    with zipfile.ZipFile(header_zip) as archive:
        for member, file_type in expected_members.items():
            if member not in archive.namelist():
                raise ValueError(f"Required official header member missing: {member}")
            header = archive.read(member).decode("utf-8-sig").strip().split("|")
            expected = SCHEMA["header_names"][file_type]
            if len(header) != len(expected):
                raise ValueError(
                    f"Official header field count mismatch for {file_type}: "
                    f"observed {len(header)}, expected {len(expected)}."
                )
            normalized_header = [normalize_header(part) for part in header]
            normalized_schema = [normalize_header(part) for part in expected]
            if normalized_header != normalized_schema:
                differences = [
                    (index + 1, left, right)
                    for index, (left, right) in enumerate(zip(normalized_header, normalized_schema))
                    if left != right
                ]
                raise ValueError(
                    f"Official header does not match Release 47 {file_type} mapping. "
                    f"Observed fields={len(header)}, expected={len(expected)}, differences={differences[:5]}"
                )
            validated[file_type] = header
    return validated


def validate_row(file_type: str, line: str, source: str = "<record>") -> list[str]:
    fields = line.rstrip("\r\n").split("|")
    expected = field_names(file_type)
    if len(fields) != len(expected):
        raise ValueError(
            f"Schema mismatch in {source}: {len(fields)} fields observed, "
            f"{len(expected)} required for Release 47 {file_type}."
        )
    return fields


def map_frame(frame: pd.DataFrame, file_type: str) -> pd.DataFrame:
    expected = field_names(file_type)
    if frame.shape[1] != len(expected):
        raise ValueError(
            f"Schema mismatch for {file_type}: {frame.shape[1]} columns observed, "
            f"{len(expected)} required for Release 47."
        )
    mapped = frame.copy()
    mapped.columns = expected
    return mapped


def validate_key_fields(file_type: str, fields: list[str], source: str) -> None:
    if file_type == "origination":
        loan_id = fields[19]
        first_payment = fields[1]
        if not re.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}", loan_id):
            raise ValueError(f"Invalid Release 47 origination loan identifier in {source}: {loan_id!r}")
        if not re.fullmatch(r"\d{6}", first_payment):
            raise ValueError(f"Invalid first payment period in {source}: {first_payment!r}")
    elif file_type == "performance":
        loan_id = fields[0]
        period = fields[1]
        if not re.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}", loan_id):
            raise ValueError(f"Invalid Release 47 performance loan identifier in {source}: {loan_id!r}")
        if not re.fullmatch(r"\d{6}", period):
            raise ValueError(f"Invalid performance period in {source}: {period!r}")


def validate_archive(zip_path: Path, full_scan: bool = True) -> dict:
    counts = {}
    with zipfile.ZipFile(zip_path) as archive:
        members_by_type = {}
        for member in archive.namelist():
            lower = Path(member).name.lower()
            if lower.endswith(".txt") and "orig" in lower:
                members_by_type["origination"] = member
            elif lower.endswith(".txt") and "perf" in lower:
                members_by_type["performance"] = member

        if set(members_by_type) != {"origination", "performance"}:
            raise ValueError(f"{zip_path.name} must contain one origination and one performance text file.")

        for file_type, member in members_by_type.items():
            rows = 0
            with archive.open(member) as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    if not raw_line.strip():
                        continue
                    line = raw_line.decode("latin1")
                    fields = validate_row(file_type, line, f"{zip_path.name}:{member}:{line_number}")
                    if rows == 0:
                        validate_key_fields(file_type, fields, f"{zip_path.name}:{member}:{line_number}")
                    rows += 1
                    if not full_scan and rows >= 5:
                        break
            counts[file_type] = {"member": member, "records_checked": rows}
    return {"zip_file": zip_path.name, "schema_version": SCHEMA["schema_version"], "files": counts}
