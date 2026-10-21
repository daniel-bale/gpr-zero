#!/usr/bin/env python3
"""Verify recovered reference inputs against the original transfer manifest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "replication/recovered_inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failures = []
    total_bytes = 0
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "error": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
                total_bytes += len(block)
        if digest.hexdigest() != item["sha256"] or path.stat().st_size != item["bytes"]:
            failures.append({"path": item["path"], "actual_sha256": digest.hexdigest(), "expected_sha256": item["sha256"]})
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "files": len(manifest["files"]), "bytes_read": total_bytes,
              "all_unchanged": not failures, "failures": failures}
    output = root / "replication/raw_rebuild/input_integrity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
