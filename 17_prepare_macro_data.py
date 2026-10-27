"""Download and prepare official FRED macroeconomic series for the isolated experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import pilot_model_utils as utils
from research_extensions import MACRO_SOURCE, prepare_macro_series


MACRO_DIR = utils.DATA_DIR / "macro"
SOURCE_DIR = MACRO_DIR / "source"
OBSERVATIONS_PATH = MACRO_DIR / "macro_observations_asof_snapshot.csv"
MANIFEST_PATH = MACRO_DIR / "macro_data_manifest.json"
FRED_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    "&cosd=2004-01-01"
)


def _download_series(series_id: str, destination: Path) -> None:
    url = FRED_CSV_URL.format(series_id=series_id)
    curl = shutil.which("curl")
    if curl:
        result = subprocess.run(
            [curl, "--fail", "--location", "--max-time", "120", "--silent", "--show-error", url],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Could not download {series_id} from FRED with curl: {detail}. "
                f"Download the CSV from {MACRO_SOURCE[series_id]['url']} and save it as {destination}."
            )
        content = result.stdout
    else:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "FreddieMacResearchExtension/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Could not download {series_id} from FRED. Download the CSV from "
                f"{MACRO_SOURCE[series_id]['url']} and save it as {destination}."
            ) from error
    if not content.startswith(b"observation_date,"):
        raise ValueError(f"Unexpected FRED CSV response for {series_id}.")
    destination.write_bytes(content)


def prepare(download: bool = False) -> pd.DataFrame:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    raw_frames = {}
    source_manifest = {}
    for series_id, definition in MACRO_SOURCE.items():
        raw_path = SOURCE_DIR / f"{series_id}.csv"
        if download or not raw_path.exists():
            _download_series(series_id, raw_path)
        frame = pd.read_csv(raw_path)
        if not {"observation_date", series_id}.issubset(frame.columns):
            raise ValueError(
                f"{raw_path} must contain observation_date and {series_id}; "
                "provide the untouched CSV export from the listed official FRED series URL."
            )
        raw_frames[series_id] = frame[["observation_date", series_id]]
        source_manifest[series_id] = {
            **definition,
            "fred_series_id": series_id,
            "series_page": definition["url"],
            "download_url": FRED_CSV_URL.format(series_id=series_id),
            "raw_file": str(raw_path.relative_to(utils.PROJECT_ROOT)),
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "observation_count": int(frame[series_id].notna().sum()),
            "first_observation": str(frame["observation_date"].min()),
            "last_observation": str(frame["observation_date"].max()),
        }

    observations = prepare_macro_series(raw_frames)
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    observations.to_csv(OBSERVATIONS_PATH, index=False)
    manifest = {
        "status": "prepared_latest_vintage_snapshot",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "FRED CSV download of the latest available series values",
        "requires_api_key": False,
        "national_geography_limitation": (
            "All selected indicators are national series. They are not state- or "
            "property-market-specific measures."
        ),
        "availability_policy": (
            "UNRATE: month-end plus 7 days; GDPC1: quarter-end plus 45 days; "
            "USSTHPI: quarter-end plus 90 days. These conservative fixed lags are "
            "modeled availability dates, not a complete archive of historical release "
            "timestamps."
        ),
        "revision_limitation": (
            "FRED CSV provides current revised observations, not historical real-time "
            "data vintages. Historical GDP and other revised series may therefore "
            "contain information revised after the modeled prediction date."
        ),
        "loan_date_limitation": (
            "Release 47 has first-payment month but no exact origination date. The macro "
            "experiment uses first_payment_month minus two calendar months as an "
            "origination-decision-date proxy; this is approximate and not an exact "
            "real-time reconstruction."
        ),
        "sources": source_manifest,
        "prepared_observations_file": str(
            OBSERVATIONS_PATH.relative_to(utils.PROJECT_ROOT)
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(observations):,} dated macro observations: {OBSERVATIONS_PATH}")
    print(f"Data manifest: {MANIFEST_PATH}")
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh local CSV series from FRED; otherwise reuse existing raw files.",
    )
    args = parser.parse_args()
    prepare(download=args.download)


if __name__ == "__main__":
    main()
