#!/usr/bin/env python3
"""
Parse news articles from RTF files to NDJSON format.
"""

import os
import json
import re
import glob
import logging
import subprocess
from datetime import datetime
from pathlib import Path
import tempfile
from striprtf.striprtf import rtf_to_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rtf_parser.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_article_id(text: str) -> str | None:
    """Extract article ID from text, trying multiple patterns."""
    # From our tests, we found that this pattern works correctly
    id_match = re.search(r"文章編號:\s*\[(\d+)\]", text)
    if id_match:
        return id_match.group(1)

    # Try some variations as fallbacks
    patterns = [
        r"文章編號\s*:\s*\[(\d+)\]",  # Space before colon
        r"文章編號：\s*\[(\d+)\]",  # Full-width colon
        r"\[(\d{15})\]",  # Just the 15-digit ID in brackets
        r"編號.*?\[(\d+)\]",  # Simplified pattern with any chars between
        r"文章編號\s*[:：]?\s*(\d+)",  # No brackets
        r"\d+年\d+月\d+日\s+(\d{15})",  # Date followed by ID with no brackets
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def clean_content(content: str) -> str:
    """Clean and normalize article content."""
    # Remove excessive whitespace and normalize line breaks
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"^\s+", "", content, flags=re.MULTILINE)
    return content.strip()


def read_rtf_file(file_path: str) -> str:
    """Read an RTF file and convert it to plain text."""
    try:
        logger.info(f"Reading RTF file: {file_path}")

        # Method 1: Use striprtf library (faster but simpler)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            rtf_text = f.read()
            text = rtf_to_text(rtf_text)

        # If text appears corrupted or empty, try fallback method
        if not text or len(text) < 100:
            logger.warning(
                f"striprtf produced possibly corrupted output for {file_path}, trying fallback method"
            )

            # Method 2: Use textutil (macOS) or unrtf (Linux) as fallback
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Try unrtf (Linux)
                subprocess.run(
                    ["unrtf", "--text", file_path, "-o", temp_path], check=True
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                try:
                    # Try textutil (macOS)
                    subprocess.run(
                        [
                            "textutil",
                            "-convert",
                            "txt",
                            "-output",
                            temp_path,
                            file_path,
                        ],
                        check=True,
                    )
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    logger.error(f"Failed to convert RTF using external tools: {e}")
                    return text  # Return the original striprtf output as last resort

            # Read the converted text file
            with open(temp_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

        return text

    except Exception as e:
        logger.error(f"Error reading RTF file {file_path}: {e}")
        raise


def parse_rtf_file(file_path: str) -> list:
    """Parse an RTF file and return a list of article dictionaries."""
    logger.info(f"Parsing RTF file: {file_path}")
    articles = []

    try:
        # Read RTF file and convert to text
        content = read_rtf_file(file_path)

        # Debug: Print first few hundred characters to see the format
        logger.debug(f"Sample content: {content[:500]}...")

        # Extract article count from first line if possible
        first_line = content.split("\n")[0].strip()
        expected_count = None
        count_match = re.search(r".*?:\s*(\d+)", first_line)
        if count_match:
            try:
                expected_count = int(count_match.group(1))
                logger.info(f"File claims to contain {expected_count} articles")
            except ValueError:
                pass

        # Try determining the newspaper from the file name or content
        newspaper = "自由時報"  # Default for Liberty Times files
        if "LT-" in file_path or "Liberty" in file_path:
            newspaper = "自由時報"

        # First attempt: Try standard patterns for article starts
        found_articles = False

        # Try multiple patterns to match article starts in various formats
        article_patterns = [
            # Standard format
            r"^(\d+)\.\s+(.+?)\s+\|\s+(\d{4}-\d{2}-\d{2})",
            # Allow different pipe characters
            r"^(\d+)\.\s+(.+?)\s+[|｜]\s+(\d{4}-\d{2}-\d{2})",
            # Allow different periods
            r"^(\d+)[\.。]\s+(.+?)\s+[|｜]\s+(\d{4}-\d{2}-\d{2})",
            # Most flexible pattern
            r"^(\d+)[\.。]?\s*(.+?)\s*[|｜]?\s*(\d{4}-\d{2}-\d{2})",
            # Liberty Times specific patterns
            r"^\d+\.\s+自由時報\s+\d{4}年\d{1,2}月\d{1,2}日",
            r"^\d+\.\s+自由時報電子報\s+\d{4}年\d{1,2}月\d{1,2}日",
        ]

        for pattern in article_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            if matches:
                article_starts = []
                for match in matches:
                    if len(match.groups()) >= 3:
                        article_num = match.group(1)
                        paper = match.group(2)
                        date = match.group(3)
                    else:
                        article_num = "Unknown"
                        paper = newspaper
                        date = "1900-01-01"  # Default date

                    start_pos = match.start()
                    article_starts.append((start_pos, article_num, paper, date))

                logger.info(f"Found {len(matches)} articles using pattern: {pattern}")
                found_articles = True
                break  # Use the first pattern that finds matches

        # If no articles found with the standard patterns, try Liberty Times specific approach
        if not found_articles:
            logger.info(
                "No articles found with standard patterns. Trying Liberty Times specific approach."
            )

            # For Liberty Times RTF, try looking for date patterns as article delimiters
            date_patterns = [
                r"\d{4}年\d{1,2}月\d{1,2}日",  # Chinese format date
                r"\d{4}-\d{2}-\d{2}",  # ISO format date
            ]

            # Find all dates as potential article boundaries
            date_positions = []
            for pattern in date_patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    date_positions.append((match.start(), match.group(0)))

            if date_positions:
                date_positions.sort()  # Sort by position
                logger.info(
                    f"Found {len(date_positions)} dates as potential article boundaries"
                )

                # Create article chunks based on dates
                article_chunks = []
                for i in range(len(date_positions) - 1):
                    start_pos = date_positions[i][0]
                    end_pos = date_positions[i + 1][0]
                    date_str = date_positions[i][1]
                    chunk = content[start_pos:end_pos]
                    article_chunks.append(
                        (start_pos, i + 1, newspaper, date_str, chunk)
                    )

                # Last chunk to the end of the file
                if date_positions:
                    last_pos = date_positions[-1][0]
                    last_date = date_positions[-1][1]
                    last_chunk = content[last_pos:]
                    article_chunks.append(
                        (
                            last_pos,
                            len(date_positions),
                            newspaper,
                            last_date,
                            last_chunk,
                        )
                    )

                logger.info(f"Created {len(article_chunks)} article chunks from dates")

                # Process each chunk
                for chunk_pos, article_num, paper, date_str, chunk in article_chunks:
                    # Only process if chunk is substantial
                    if len(chunk) < 50:
                        continue

                    # Convert Chinese date to ISO format if needed
                    if "年" in date_str:
                        try:
                            # Convert Chinese date format to ISO
                            date_parts = re.search(
                                r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str
                            )
                            if date_parts:
                                year = int(date_parts.group(1))
                                month = int(date_parts.group(2))
                                day = int(date_parts.group(3))
                                date_str = f"{year:04d}-{month:02d}-{day:02d}"
                        except Exception as e:
                            logger.warning(f"Error converting date {date_str}: {e}")

                    # Extract ID
                    article_id = extract_article_id(chunk)
                    if not article_id:
                        # Use position and date as fallback ID
                        article_id = f"LT-{date_str}-{article_num}"

                    # Extract headline and content - look for the first substantial paragraph
                    lines = chunk.split("\n")
                    headline = ""
                    content_lines = []
                    found_headline = False

                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue

                        # Skip the first line with the date
                        if i == 0 and (
                            date_str in line or "年" in line or "月" in line
                        ):
                            continue

                        # First substantial line is the headline
                        if not found_headline and len(line) > 5:
                            headline = line
                            found_headline = True
                            continue

                        # Everything else is content
                        if found_headline:
                            content_lines.append(line)

                    content_text = clean_content("\n".join(content_lines))

                    # Create article dictionary
                    if headline and content_text:
                        article = {
                            "id": article_id,
                            "newspaper": newspaper,
                            "date": date_str,
                            "page": "",  # Not available
                            "category": "",  # Not available
                            "author": "",  # Not available
                            "headline": headline,
                            "content": content_text,
                            "source_file": os.path.basename(file_path),
                        }

                        articles.append(article)
                        logger.debug(
                            f"Extracted article {article_num} with ID {article_id}: {headline[:30]}..."
                        )

                # We've processed with the Liberty Times specific approach
                found_articles = True

        if not found_articles or not articles:
            logger.warning(f"No articles successfully extracted from {file_path}")
            return articles

        logger.info(f"Successfully parsed {len(articles)} articles from {file_path}")
        if expected_count and len(articles) != expected_count:
            logger.warning(
                f"Expected {expected_count} articles but found {len(articles)}"
            )

        return articles

    except Exception as e:
        logger.error(f"Error parsing RTF file {file_path}: {str(e)}")
        return []


def write_to_ndjson(articles: list, output_file: str):
    """Write articles to NDJSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Remove existing file if it exists
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            logger.info(f"Removed existing file: {output_file}")
        except Exception as e:
            logger.error(f"Failed to remove file {output_file}: {str(e)}")

    # Write mode
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(articles)} articles to {output_file}")


def process_rtf_files(input_dir: str, output_dir: str):
    """Process all RTF files in the input directory."""
    # Get all .rtf files in the input directory
    rtf_files = glob.glob(os.path.join(input_dir, "*.rtf"))

    if not rtf_files:
        logger.warning(f"No RTF files found in {input_dir}")
        return

    logger.info(f"Found {len(rtf_files)} RTF files to process")

    # Process each file and write to separate NDJSON file
    total_articles = 0
    for file_path in rtf_files:
        file_name = os.path.basename(file_path)
        output_file = os.path.join(
            output_dir, f"{os.path.splitext(file_name)[0]}.ndjson"
        )

        articles = parse_rtf_file(file_path)
        if articles:
            write_to_ndjson(articles, output_file)
            total_articles += len(articles)

    logger.info(f"Total articles processed from RTF files: {total_articles}")


if __name__ == "__main__":
    # Get directories from environment variables or use defaults
    INPUT_DIR = os.environ.get("NEWS_RAW_DIR", "data/news/raw")  # Same as TXT files
    OUTPUT_DIR = os.environ.get("NEWS_OUTPUT_DIR", "data/news/processed")

    process_rtf_files(INPUT_DIR, OUTPUT_DIR)
    logger.info("RTF parsing completed")
