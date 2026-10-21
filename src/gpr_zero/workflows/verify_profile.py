"""Verify the supplied thesis profile without modifying production artifacts."""

import argparse
import json
from pathlib import Path

from gpr_zero.verification import verify_profile


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gpr verify thesis-profile", description=__doc__
    )
    parser.add_argument("profile", type=Path, help="Completed thesis_profile directory")
    args = parser.parse_args(argv)
    report = verify_profile(args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1
