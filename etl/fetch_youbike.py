"""Fetch YouBike 2.0 real-time station status data."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
from typing import Any, Dict, Optional

import requests

API_URL = "https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json"
OUTPUT_DIR = pathlib.Path("data/raw")


class FetchError(RuntimeError):
    """Raised when the YouBike endpoint cannot be reached."""


def fetch_snapshot(session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch a snapshot of the YouBike 2.0 JSON feed."""
    client = session or requests.Session()
    response = client.get(API_URL, timeout=10)
    if response.status_code != 200:
        raise FetchError(f"Unexpected status code {response.status_code} from YouBike API")
    return response.json()


def save_snapshot(payload: Dict[str, Any], output_dir: pathlib.Path, timestamp: Optional[dt.datetime] = None) -> pathlib.Path:
    """Persist the payload to disk for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or dt.datetime.utcnow()
    filename = ts.strftime("youbike_%Y%m%dT%H%M%SZ.json")
    path = output_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a YouBike 2.0 snapshot.")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_DIR, help="Directory to store the snapshot")
    parser.add_argument("--dry-run", action="store_true", help="Print the payload size instead of writing to disk")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logging.info("Fetching YouBike snapshot from %s", API_URL)
    try:
        payload = fetch_snapshot()
    except (requests.RequestException, FetchError) as err:
        logging.error("Failed to fetch snapshot: %s", err)
        raise SystemExit(1) from err

    if args.dry_run:
        stations = payload.get("retVal", {})
        logging.info("Fetched %d stations (dry-run)", len(stations))
        return

    path = save_snapshot(payload, args.output)
    logging.info("Saved snapshot to %s", path)


if __name__ == "__main__":
    main()
