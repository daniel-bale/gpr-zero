#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyarrow==20.0.0"]
# ///
"""Read Parquet footers only; verify inventory and schema before full parsing."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "replication/raw_rebuild"
    manifest = json.loads((output / "news/full/manifest.json").read_text())
    source = Path(manifest["input_dir"])
    required = {"warc_id", "warc_date", "url", "domain", "outlet", "is_english", "html_content_bytes"}
    years = defaultdict(Counter)
    schemas = Counter()
    errors = []
    for i, entry in enumerate(manifest["source_files"], 1):
        path = source / entry["path"]
        stat = path.stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            errors.append({"path": entry["path"], "error": "Source metadata changed"})
        try:
            with pq.ParquetFile(path) as parquet:
                metadata = parquet.metadata
                schema = parquet.schema_arrow
                missing = required - set(schema.names)
                if missing:
                    errors.append({"path": entry["path"], "missing_columns": sorted(missing)})
                schemas[str(schema)] += 1
                year = Path(entry["path"]).parts[0]
                years[year].update(files=1, rows=metadata.num_rows, row_groups=metadata.num_row_groups,
                                   bytes=stat.st_size, empty_files=int(metadata.num_rows == 0))
        except Exception as error:
            errors.append({"path": entry["path"], "error": str(error)})
        if i % 5000 == 0:
            print(f"Read {i}/{len(manifest['source_files'])} footers", flush=True)
    totals = sum(years.values(), Counter())
    report = {"source_root": str(source), "source_inventory_sha256": manifest["source_inventory_sha256"],
              "footer_only": True, "totals": dict(totals), "by_year": dict(years),
              "schemas": [{"schema": schema, "files": count} for schema, count in schemas.items()],
              "errors": errors, "all_files_valid": not errors and totals["files"] == len(manifest["source_files"])}
    (output / "source_parquet_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"totals": totals, "errors": errors, "all_files_valid": report["all_files_valid"]}, indent=2))
    if not report["all_files_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
