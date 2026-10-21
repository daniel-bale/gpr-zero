"""Reconstructed replay of the documented May/June 2025 extraction history.

This is not a recovered original module. Base extraction uses the recovered May
selector set plus the documented pre-May30 China Times and Taipei Times fixes.
It then replays the two recovered June1 UPDATE stages over their exact source
directory month/outlet eligibility. Previously rejected rows remain rejected.
Original repair functions are compiled unchanged from verified source ASTs;
font setup, database writes, and log-directory side effects are not executed.
"""

import ast
import csv
import hashlib
import importlib.util
import io
import logging
import re
from pathlib import Path
from types import SimpleNamespace

from selectolax.lexbor import LexborHTMLParser


ROOT = Path(__file__).parent


def verified_text(relative, digest):
    path = ROOT / relative
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Historical dependency changed: {path}")
    return content.decode()


def load_original(name, relative, digest):
    verified_text(relative, digest)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAY = load_original("replay_may", "stage2_parse_html_load_db.2025-05.py",
                    "a9d95a9ef4360b2f025a27e422c5f5c9e6a96666fe716e9d017c5fd80ead3761")
BASE = load_original("replay_base", "stage2_parse_html_load_db.2025-06.py",
                     "23db8348b3b4555ebca8827e10485894cc27ef5ef109e9088568c8f6add5c043")
# These two fixes are documented before the May30 full extraction in the
# recovered data_integrity01.md/data_integrity02.md. Other June selectors are
# excluded from base extraction and applied only by their recorded UPDATE jobs.
BASE.DOMAIN_SELECTORS = {
    **MAY.DOMAIN_SELECTORS,
    "chinatimes.com": BASE.DOMAIN_SELECTORS["chinatimes.com"],
    "taipeitimes.com": BASE.DOMAIN_SELECTORS["taipeitimes.com"],
}


def load_repair(relative, digest):
    tree = ast.parse(verified_text(relative, digest), filename=relative)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"clean_text", "extract_content_with_enhanced_selectors"}:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"ENHANCED_SELECTORS", "REMAINING_PROBLEM_FILES"}
            for target in node.targets
        ):
            selected.append(node)
    namespace = {"re": re, "logging": logging, "LexborHTMLParser": LexborHTMLParser}
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"), namespace)
    return SimpleNamespace(**namespace)


PHASE4 = load_repair("cache_provenance/targeted_reprocess_problem_samples.py",
                     "c47a75d93dc407f62d428f191afd3c53491003a33aaa3a90636dc9d3752b8861")
PHASE5 = load_repair("cache_provenance/targeted_reprocess_remaining_problems.py",
                     "6383b5082c7e36855412c654e9a77af2ef7cf22a20db3ac47c2a4679b527d484")
eligibility_text = verified_text("cache_provenance/problem_outlets_sub95.csv",
                                "89b767cb68f81cf122728834446893de79d881ea0f1a29a36816185c16e4f29e")
PHASE4_PAIRS = {(row["year_month"], row["outlet"]) for row in csv.DictReader(io.StringIO(eligibility_text))}
if len(PHASE4_PAIRS) != 49:
    raise ValueError("Expected the49historical outlet-month eligibility pairs")
DOMAIN_HINTS = {"People News (民報)": "peoplenews", "Central News Agency (中央社)": "cna",
                "Liberty Times (自由時報)": "ltn", "Now News (今日新聞)": "nownews",
                "United Daily (聯合報)": "udn", "Taiwan News": "taiwannews"}


def slug(outlet):
    return outlet.replace(" ", "_").replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace("/", "_")


PHASE5_PAIRS = {(month, outlet) for month, outlet in PHASE4_PAIRS
                if any(filename.startswith(f"{month}_{slug(outlet)}_") for filename in PHASE5.REMAINING_PROBLEM_FILES)}
if len(PHASE5_PAIRS) != 7:
    raise ValueError("Expected the7historical phase5outlet-month pairs")


def prepare_source(parquet_file, source_path):
    parts = Path(source_path).parts
    month = f"{parts[0]}-{parts[1]}" if len(parts) >= 3 else None
    # The extractor uses exact outlet labels first, then domain hints only when
    # a source shard has zero rows for that requested outlet.
    outlets = set()
    if any(period == month for period, _ in PHASE4_PAIRS):
        for batch in parquet_file.iter_batches(columns=["outlet"], batch_size=8192):
            outlets.update(batch.column(0).to_pylist())
    return {"_source_path": source_path, "_source_month": month, "_source_outlets": frozenset(outlets)}


def eligible(row, pairs):
    month = row.get("_source_month")
    present = row.get("_source_outlets", frozenset())
    for period, outlet in sorted(pairs):
        if period != month:
            continue
        if outlet in present:
            if row["outlet"] == outlet:
                return True
        elif DOMAIN_HINTS[outlet] in (row.get("domain") or ""):
            return True
    return False


def extract_article_content_selectolax(html_content_bytes, url, domain):
    """Base extraction only; full historical replay requires row/source context."""
    return BASE.extract_article_content_selectolax(html_content_bytes, url, domain)


def extract_article_record(row):
    original = extract_article_content_selectolax(row["html_content_bytes"], row["url"], row["domain"])
    if not (original.get("title") or original.get("content")):
        return original  # Historical repair SQL UPDATE never inserts new IDs.
    result = original
    for pairs, repair in ((PHASE4_PAIRS, PHASE4), (PHASE5_PAIRS, PHASE5)):
        if eligible(row, pairs):
            updated = repair.extract_content_with_enhanced_selectors(row["html_content_bytes"], row["url"], row["domain"])
            if updated["title"] or (updated["content"] and len(updated["content"]) > 100):
                result = updated
    return result
