from __future__ import annotations

import csv
import importlib.util
import zipfile
from collections import Counter
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("00_config.py")
spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

DATA_DIR = config.DATA_DIR
INVENTORY_DIR = config.INVENTORY_DIR
EXPECTED_FILE_LAYOUT = config.EXPECTED_FILE_LAYOUT
ensure_directories = config.ensure_directories
find_raw_zip_files = config.find_raw_zip_files
stage_raw_zips = config.stage_raw_zips


def find_zip_archives(root: Path | str = None) -> list[Path]:
    if root is not None:
        roots = [Path(root)]
    else:
        roots = []
    roots.extend([
        Path(__file__).resolve().parent,
        DATA_DIR,
        DATA_DIR / "raw_zips",
    ])
    seen = set()
    files: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.glob("sample_*.zip")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    if not files:
        files = find_raw_zip_files()
    return files


def detect_layout_kind(filename: str) -> str:
    lower = filename.lower()
    for kind, metadata in EXPECTED_FILE_LAYOUT.items():
        if any(marker in lower for marker in metadata["filename_markers"]):
            return kind
    raise ValueError(f"Could not classify archive member {filename!r}. Expected orig/perf markers.")


def row_widths_for_lines(lines: list[str]) -> Counter:
    widths = [len(line.split("|")) for line in lines if line.strip()]
    return Counter(widths)


def inspect_archive(zip_path: Path) -> list[dict]:
    inventory: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member_name in sorted(archive.namelist()):
            if not member_name.endswith(".txt"):
                continue
            raw = archive.read(member_name)
            text = raw.decode("latin1")
            lines = text.splitlines()
            nonempty_lines = [line for line in lines if line.strip()]
            widths = row_widths_for_lines(nonempty_lines)
            dominant_width = widths.most_common(1)[0][0] if widths else 0
            kind = detect_layout_kind(member_name)
            expected_width = EXPECTED_FILE_LAYOUT[kind]["required_tokens"]
            if dominant_width and dominant_width != expected_width:
                raise ValueError(
                    f"Unexpected width in {zip_path.name}:{member_name}: "
                    f"observed {dominant_width}, expected {expected_width}."
                )

            sample_rows = nonempty_lines[:5]
            file_size_bytes = archive.getinfo(member_name).file_size
            inventory.append(
                {
                    "zip_file": zip_path.name,
                    "archive_member": member_name,
                    "kind": kind,
                    "file_size_bytes": file_size_bytes,
                    "record_count": len(nonempty_lines),
                    "first_5_record_widths": [len(row.split("|")) for row in sample_rows],
                    "expected_width": expected_width,
                    "observed_width": dominant_width,
                    "has_malformed_rows": bool(widths and len(widths) > 1),
                    "sample_records": sample_rows,
                }
            )
    return inventory


def write_inventory(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "zip_file",
        "archive_member",
        "kind",
        "file_size_bytes",
        "record_count",
        "first_5_record_widths",
        "expected_width",
        "observed_width",
        "has_malformed_rows",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, "") for key in fieldnames
            })


def main() -> None:
    ensure_directories()
    stage_raw_zips()
    zip_files = find_zip_archives(DATA_DIR)
    if not zip_files:
        raise FileNotFoundError(f"No sample ZIP files found in the project or data directories.")

    all_rows: list[dict] = []
    for zip_path in zip_files:
        all_rows.extend(inspect_archive(zip_path))

    output_path = INVENTORY_DIR / "zip_inventory.csv"
    write_inventory(all_rows, output_path)

    print(f"Inspected {len(zip_files)} ZIP files and {len(all_rows)} archive members.")
    print(f"Saved inventory to {output_path}.")
    for row in all_rows[:10]:
        print(f"{row['zip_file']} :: {row['archive_member']} :: {row['kind']} :: rows={row['record_count']} :: width={row['observed_width']}")


if __name__ == "__main__":
    main()
