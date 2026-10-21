"""Main entry point for GPR-Zero."""

import argparse
import importlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# One command catalog for installed workflow modules.
WORKFLOWS = {
    "verify": {
        "thesis-profile": (
            "verify_profile",
            "main",
            "Compare outputs with the pinned thesis profile",
        ),
    },
    "ingest": {
        "warc": ("ingest_warc", "main", "Extract article HTML from WARC archives"),
        "html": (
            "ingest_html",
            "main",
            "Parse extracted HTML into the article database",
        ),
    },
    "dictionaries": {
        "step1": ("batch_generate_step1", "main", "Generate role-based dictionaries"),
        "step2": (
            "batch_generate_step2",
            "main",
            "Generate fixed-category dictionaries",
        ),
    },
    "cache": {
        "rebuild": (
            "compute_all_dictionary_timeseries",
            "rebuild_cache",
            "Explicitly rebuild the binary matrix",
        ),
    },
    "index": {
        "all": (
            "compute_all_dictionary_timeseries",
            "main",
            "Calculate all-outlet time series from the cache",
        ),
        "three-outlets": (
            "compute_all_dictionary_timeseries_3outlets",
            "main",
            "Calculate the thesis three-outlet time series",
        ),
        "categories": (
            "calculate_category_gpr_complete",
            "main",
            "Calculate Step 2 category indices",
        ),
        "newspapers": (
            "calculate_newspaper_specific_step2_gpr",
            "main",
            "Calculate newspaper indices and comparisons",
        ),
        "median": (
            "compute_step2_median_3outlets",
            "main",
            "Combine three-outlet Step 2 indices",
        ),
        "verify-categories": (
            "verify_step2_category_contributions_3outlets",
            "main",
            "Verify category contributions for the three-outlet cohort",
        ),
    },
    "spikes": {
        "detect": (
            "detect_gpr_spikes_unified",
            "main",
            "Detect and compare weekly spike events",
        ),
        "summaries": (
            "create_step2_unique_spikes_summary",
            "main",
            "Select unique-event summaries",
        ),
        "articles": (
            "analyze_step2_spike_articles_normalized",
            "main",
            "Analyze articles within spike windows",
        ),
        "words": (
            "calculate_spike_word_lift_scores_v3",
            "main",
            "Calculate spike keyword lift",
        ),
        "category-peaks": (
            "analyze_category_peak_events",
            "main",
            "Analyze category peaks with the thesis Step 2 vocabulary",
        ),
        "category-peaks-updated": (
            "analyze_category_peak_events_updated",
            "main",
            "Analyze category peaks with the complete vocabulary",
        ),
    },
}

_DELEGATED_PARSERS = {"ingest", "verify"}


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = logging.getLevelNamesMapping().get(log_level.upper())
    if numeric_level is None:
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gpr", description="GPR thesis workflows using one project configuration."
    )
    parser.add_argument(
        "--project", type=Path, help="Project TOML file (default: discover gpr.toml)"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    groups = parser.add_subparsers(dest="command", required=True)
    groups.add_parser("config", help="Show resolved project paths without opening data")
    for group, tasks in WORKFLOWS.items():
        group_parser = groups.add_parser(group, help=f"{group.capitalize()} workflows")
        actions = group_parser.add_subparsers(dest="action", required=True)
        for action, (module, function, description) in tasks.items():
            task_parser = actions.add_parser(
                action,
                help=description,
                description=description,
                add_help=group not in _DELEGATED_PARSERS,
            )
            task_parser.set_defaults(workflow_module=module, workflow_function=function)
    return parser


def main(argv=None):
    """Run a named workflow with one resolved project configuration."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    from gpr_zero.project import Project, use_project

    parser = build_parser()
    args, workflow_arguments = parser.parse_known_args(arguments)
    if workflow_arguments and args.command not in _DELEGATED_PARSERS:
        parser.error(f"unrecognized arguments: {' '.join(workflow_arguments)}")
    setup_logging(args.log_level)
    try:
        project = Project.load(args.project)
        if args.command == "config":
            import json

            print(
                json.dumps(
                    {
                        "root": str(project.root),
                        "data": str(project.data_dir),
                        "database": str(project.database),
                        "dictionaries": str(project.dictionaries_dir),
                        "step2_dictionaries": str(project.step2_dictionaries_dir),
                        "llm_config": str(project.llm_config),
                    },
                    indent=2,
                )
            )
            return 0
        with use_project(project):
            module = importlib.import_module(
                f"gpr_zero.workflows.{args.workflow_module}"
            )
            workflow = getattr(module, args.workflow_function)
            if args.command in _DELEGATED_PARSERS:
                status = workflow(argv=workflow_arguments)
            else:
                status = workflow()
        return 0 if status is None else status
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        return 1
