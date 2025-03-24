#!/usr/bin/env python
"""
Script to transform dictionary keywords into search strings.

This script loads a dictionary file, transforms it to search strings using the rules:
1. Prioritize processing of bracketed keywords
2. All keywords are enclosed in double quotation marks for consistency
3. Keywords are grouped by category with "or" and different categories are joined with "and"
4. The script verifies that keywords don't already contain double quotes

Usage:
    uv run scripts/generate_search_strings.py --input data/dictionaries/yeu_tong_lau_2024.json --output data/search_strings/yeu_tong_lau_2024.json [--query war_threat] [--plain-text]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add the src directory to sys.path to import the package
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from gpr_zero.utils import transform_dict_to_search_strings


def main():
    """Run the script to transform dictionary to search strings."""
    parser = argparse.ArgumentParser(
        description="Transform dictionary to search strings"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dictionary JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output search strings JSON file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help="Optional: Specific query type to include in output (e.g., war_threat, peace_threat)",
    )
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Also create a plain text file with unescaped quotes for easy copying",
    )
    args = parser.parse_args()

    # Ensure the input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        # Load the dictionary
        with open(args.input, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        # Transform dictionary to search strings (all query types)
        all_search_strings = transform_dict_to_search_strings(dictionary)

        # Filter to specific query if requested
        if args.query:
            if args.query in all_search_strings:
                search_strings = {args.query: all_search_strings[args.query]}
                description = f"Search string for '{args.query}' generated from {os.path.basename(args.input)}"
            else:
                print(
                    f"Warning: Query type '{args.query}' not found in generated search strings"
                )
                search_strings = {}
                description = f"Empty result as query '{args.query}' not found in {os.path.basename(args.input)}"
        else:
            search_strings = all_search_strings
            description = f"Search strings for all query types generated from {os.path.basename(args.input)}"

        # Save the JSON search strings output
        with open(args.output, "w", encoding="utf-8") as f:
            # Create a result object with metadata and search strings
            result = {
                "metadata": {
                    "source": dictionary.get("metadata", {}).get("source", "Unknown"),
                    "description": description,
                    "created_at": dictionary.get("metadata", {}).get(
                        "created", "Unknown"
                    ),
                },
                "search_strings": search_strings,
            }
            json.dump(result, f, ensure_ascii=False, indent=2)

        # If plain text output requested, create a .txt file with copy-friendly format
        if args.plain_text:
            # Create txt filename from json filename
            txt_filename = os.path.splitext(args.output)[0] + ".txt"
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(f"# Search strings from {os.path.basename(args.input)}\n")
                f.write(
                    f"# Generated: {dictionary.get('metadata', {}).get('created', 'Unknown')}\n\n"
                )

                for query_name, search_string in search_strings.items():
                    f.write(f"## {query_name}\n")
                    # Write the search string directly (no JSON escaping)
                    f.write(f"{search_string}\n\n")

            print(f"Plain text search strings saved to {txt_filename}")

        # Determine appropriate message based on query parameter
        if args.query and args.query in all_search_strings:
            success_msg = f"Successfully generated search string for '{args.query}'"
        elif args.query:
            success_msg = (
                f"Query '{args.query}' not found. No search strings generated."
            )
        else:
            success_msg = f"Successfully generated search strings for {len(search_strings)} query types"

        print(f"{success_msg} and saved to {args.output}")

        # Display a preview of generated search strings
        if search_strings:
            print("\nSearch string preview:")
            for query_name, search_string in list(search_strings.items())[
                :3
            ]:  # Show first 3 entries
                print(
                    f"{query_name}: {search_string[:100]}..."
                    if len(search_string) > 100
                    else f"{query_name}: {search_string}"
                )

            if len(search_strings) > 3:
                print(f"... and {len(search_strings) - 3} more queries")

    except json.JSONDecodeError:
        print(f"Error: {args.input} is not a valid JSON file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
