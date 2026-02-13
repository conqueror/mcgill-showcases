#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import requests

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net"


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYC TLC sample parquet + zone lookup")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.month <= 12:
        raise SystemExit("--month must be in [1, 12]")

    ym = f"{args.year:04d}-{args.month:02d}"
    trip_url = f"{TLC_BASE_URL}/trip-data/yellow_tripdata_{ym}.parquet"
    zone_url = f"{TLC_BASE_URL}/misc/taxi_zone_lookup.csv"

    raw_dir = Path("data/raw")
    trip_path = raw_dir / f"yellow_tripdata_{ym}.parquet"
    zone_path = raw_dir / "taxi_zone_lookup.csv"

    print(f"Downloading {trip_url} -> {trip_path}")
    _download_file(trip_url, trip_path)

    print(f"Downloading {zone_url} -> {zone_path}")
    _download_file(zone_url, zone_path)

    print("Done.")


if __name__ == "__main__":
    main()
