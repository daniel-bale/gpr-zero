#!/usr/bin/env python3
"""Download only the bounded WARC validation selection and record provenance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(record: dict, log_dir: Path) -> dict:
    destination = Path(record["warc_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    expected = record["head"]["content_length"]
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    reused = destination.exists()
    initial_bytes = destination.stat().st_size if reused else (partial.stat().st_size if partial.exists() else 0)
    curl_result = None
    log_path = log_dir / (destination.name + ".curl.log")
    headers_path = log_dir / (destination.name + ".headers.txt")
    if not reused:
        command = [
            "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "3",
            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
            "--continue-at", "-", "--silent", "--show-error",
            "--dump-header", str(headers_path), "--output", str(partial),
            "--write-out", "%{http_code} %{size_download} %{time_total} %{url_effective}\n",
            record["url"],
        ]
        with log_path.open("a") as log:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True)
            log.write(result.stdout)
        curl_result = result.stdout.strip()
        if result.returncode:
            raise RuntimeError(f"curl failed ({result.returncode}): {log_path}")
        actual = partial.stat().st_size
        if actual != expected:
            raise RuntimeError(f"Size mismatch: {partial}: {actual} != {expected}")
        partial.replace(destination)
    if destination.stat().st_size != expected:
        raise RuntimeError(f"Existing file has unexpected size: {destination}")
    digest = sha256(destination)
    integrity = subprocess.run(["gzip", "-t", str(destination)], capture_output=True, text=True)
    if integrity.returncode:
        raise RuntimeError(f"gzip integrity failed: {destination}: {integrity.stderr}")
    return {
        "relative_path": record["relative_path"], "url": record["url"],
        "path": str(destination.resolve()), "bytes": expected, "sha256": digest,
        "gzip_integrity": "passed", "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "reused_complete_file": reused, "initial_local_bytes": initial_bytes,
        "curl_result": curl_result, "curl_log": str(log_path.resolve()),
        "response_headers": str(headers_path.resolve()), "head": record["head"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("replication/warc_validation/selection.json"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    records = selection["records"]
    if not 1 <= len(records) <= 3:
        raise ValueError("This validation downloader is limited to three complete WARC files")
    log_dir = args.selection.parent / "download_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    output = args.selection.parent / "download_provenance.json"
    prior = json.loads(output.read_text()) if output.exists() else None
    if prior and prior.get("complete"):
        if prior.get("selection_sha256") != sha256(args.selection):
            raise ValueError("Existing completed download proof belongs to another selection")
        if {r["relative_path"] for r in prior["records"]} != {r["relative_path"] for r in records}:
            raise ValueError("Existing completed download proof has a different file set")
    print(f"Downloading {len(records)} WARC files, {sum(r['head']['content_length'] for r in records):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, record, log_dir): record for record in records}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if prior and prior.get("complete"):
                previous = next(r for r in prior["records"] if r["relative_path"] == result["relative_path"])
                for key in ("url", "path", "bytes", "sha256", "gzip_integrity"):
                    if previous[key] != result[key]:
                        raise ValueError(f"Existing completed proof differs in {key}: {result['relative_path']}")
                print(f"Reverified {result['relative_path']}; existing completed provenance retained", flush=True)
                continue
            payload = {
                "selection": str(args.selection.resolve()), "selection_sha256": sha256(args.selection),
                "curl_version": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0],
                "records": sorted(results, key=lambda r: r["relative_path"]),
                "complete": len(results) == len(records),
                "total_bytes": sum(r["bytes"] for r in results),
                "note": "Multipart ETags are recorded as server metadata, not treated as simple MD5 checksums.",
            }
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(output)
            print(f"Verified {result['relative_path']}: {result['bytes']:,} bytes; SHA256 {result['sha256']}; gzip passed", flush=True)


if __name__ == "__main__":
    main()
