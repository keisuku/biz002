from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/daily"
CHECKSUM_PATTERN = re.compile(r"^([a-fA-F0-9]{64})\s+\*?(.+)$")


@dataclass(frozen=True)
class DownloadRecord:
    symbol: str
    day: str
    url: str
    path: str
    bytes: int
    sha256: str
    checksum_verified: bool
    status: str


def iter_days(start: date, end: date):
    if end < start:
        raise ValueError("end must be on or after start")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def archive_url(
    symbol: str,
    day: date,
    dataset: str = "aggTrades",
    interval: str | None = None,
) -> str:
    symbol = symbol.upper()
    if dataset == "klines":
        if not interval:
            raise ValueError("interval is required for klines")
        filename = f"{symbol}-{interval}-{day.isoformat()}.zip"
        return f"{BASE_URL}/{dataset}/{symbol}/{interval}/{filename}"
    filename = f"{symbol}-{dataset}-{day.isoformat()}.zip"
    return f"{BASE_URL}/{dataset}/{symbol}/{filename}"


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "momentum-ignition-research/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_checksum(payload: bytes, expected_filename: str) -> str:
    text = payload.decode("utf-8").strip()
    match = CHECKSUM_PATTERN.match(text)
    if not match:
        raise ValueError(f"Unsupported checksum format: {text!r}")
    digest, filename = match.groups()
    if Path(filename).name != expected_filename:
        raise ValueError(f"Checksum names {filename!r}, expected {expected_filename!r}")
    return digest.lower()


def download_day(
    symbol: str,
    day: date,
    output_root: str | Path,
    *,
    dataset: str = "aggTrades",
    interval: str | None = None,
    overwrite: bool = False,
) -> DownloadRecord:
    symbol = symbol.upper()
    url = archive_url(symbol, day, dataset, interval)
    filename = Path(url).name
    destination = Path(output_root) / dataset / symbol
    if interval:
        destination /= interval
    destination = destination / f"{day:%Y}" / f"{day:%m}" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    expected_sha = parse_checksum(_fetch_bytes(f"{url}.CHECKSUM"), filename)
    if destination.exists() and not overwrite:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise ValueError(f"Existing archive checksum mismatch: {destination}")
        return DownloadRecord(
            symbol,
            day.isoformat(),
            url,
            str(destination),
            destination.stat().st_size,
            digest,
            True,
            "cached",
        )

    payload = _fetch_bytes(url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha:
        raise ValueError(f"Downloaded archive checksum mismatch: {url}")
    destination.write_bytes(payload)
    return DownloadRecord(
        symbol,
        day.isoformat(),
        url,
        str(destination),
        len(payload),
        digest,
        True,
        "downloaded",
    )


def run_download(
    symbol: str,
    start: date,
    end: date,
    output_root: str | Path = "data/raw",
    *,
    dataset: str = "aggTrades",
    interval: str | None = None,
) -> list[DownloadRecord]:
    records: list[DownloadRecord] = []
    for day in iter_days(start, end):
        try:
            records.append(
                download_day(
                    symbol,
                    day,
                    output_root,
                    dataset=dataset,
                    interval=interval,
                )
            )
        except urllib.error.HTTPError as exc:
            records.append(
                DownloadRecord(
                    symbol.upper(),
                    day.isoformat(),
                    archive_url(symbol, day, dataset, interval),
                    "",
                    0,
                    "",
                    False,
                    f"http_{exc.code}",
                )
            )
    report_path = Path("reports") / "phase0" / f"{symbol.upper()}-download-manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )
    return records


def _date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download verified Binance futures archives.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument("--dataset", default="aggTrades")
    parser.add_argument("--interval")
    parser.add_argument("--output-root", default="data/raw")
    args = parser.parse_args()
    records = run_download(
        args.symbol,
        args.start,
        args.end,
        args.output_root,
        dataset=args.dataset,
        interval=args.interval,
    )
    print(json.dumps([asdict(record) for record in records], indent=2))


if __name__ == "__main__":
    main()
