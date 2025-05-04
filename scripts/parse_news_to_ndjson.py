#!/usr/bin/env python3
"""
Parse news articles from TXT files to NDJSON format.
"""

import os
import json
import re
import glob
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/news/output/simple_gpr.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def extract_article_id(text: str) -> str | None:
    """Extract article ID from text, trying multiple patterns."""
    # Regular expression pattern for article ID
    id_match = re.search(r"文章編號\s*[:]?\s*\[(\d+)\]", text)
    if id_match:
        return id_match.group(1)

    # Try some variations as fallback
    patterns = [
        r"文章編號：\s*\[(\d+)\]",  # Full-width colon
        r"\[(\d{15})\]",  # Just the 15-digit ID in brackets
        r"編號.*?\[(\d+)\]",  # Simplified pattern with any chars between
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


def parse_txt_file(file_path: str) -> list:
    """Parse a news TXT file and return a list of article dictionaries."""
    logger.info(f"Parsing file: {file_path}")
    articles = []
    newspaper = ""

    try:
        with open(file_path, "r", encoding="big5", errors="replace") as f:
            content = f.read()

        # Try to determine the newspaper from the header
        # This can vary by file format, so we try multiple patterns
        newspaper_match = re.search(
            r"^.*?(\d+).*?(自由時報|聯合報|中國時報|工商時報).*?(\d+).*?",
            content,
            re.MULTILINE,
        )
        if newspaper_match:
            newspaper = newspaper_match.group(2)
            logger.info(f"Detected newspaper: {newspaper}")

        # Split the content by article, assuming each article starts with a number followed by a period
        # We use different patterns to capture variations in file formats
        article_pattern = r"^\d+\.\s+"
        articles_raw = re.split(article_pattern, content, flags=re.MULTILINE)[
            1:
        ]  # Skip the header part

        logger.info(f"Found {len(articles_raw)} potential articles in {file_path}")

        for i, article_text in enumerate(articles_raw):
            try:
                # Extract article ID
                article_id = extract_article_id(article_text)

                if not article_id:
                    logger.warning(f"No article ID found for article {i + 1}, skipping")
                    continue

                # Split the article into lines for further processing
                lines = article_text.split("\n")

                # First line often contains newspaper and date information
                first_line = lines[0].strip() if lines else ""

                # Try to extract newspaper and date from the first line
                # Format: <newspaper> | <date>
                header_parts = first_line.split("|")

                extracted_newspaper = newspaper  # Default to the file's newspaper
                date_str = ""

                if len(header_parts) >= 2:
                    # Extract newspaper if not already determined
                    if not newspaper and header_parts[0].strip():
                        extracted_newspaper = header_parts[0].strip()

                    # Extract date
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", header_parts[1])
                    if date_match:
                        date_str = date_match.group(1)

                # If we still don't have a date, try a broader search
                if not date_str:
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", article_text)
                    if date_match:
                        date_str = date_match.group(1)

                if not date_str:
                    logger.warning(
                        f"No date found for article {i + 1}, using placeholder"
                    )
                    date_str = "1900-01-01"  # Placeholder date

                # Extract metadata from the second line (if it exists)
                page = ""
                category = ""
                author = ""

                if len(lines) >= 2:
                    meta_line = lines[1].strip()
                    if "|" in meta_line:
                        meta_parts = [p.strip() for p in meta_line.split("|")]

                        # Page is typically the second part
                        if len(meta_parts) > 1:
                            page = meta_parts[1]

                        # Category is typically the third part
                        if len(meta_parts) > 2:
                            category = meta_parts[2]

                        # Author info might be in the fourth part
                        if len(meta_parts) > 3:
                            if "By" in meta_parts[3]:
                                author = meta_parts[3].replace("By", "").strip()
                            else:
                                author = meta_parts[3].strip()

                # Extract headline and content
                headline = ""
                content_text = ""  # Initialize as empty string
                headline_line_idx = -1
                content_start_idx = -1
                content_end_idx = len(lines)

                # Find headline (first non-empty line after metadata - typically line index 2 or later)
                for idx in range(2, min(10, len(lines))): # Search in first few lines
                    line_strip = lines[idx].strip()
                    if line_strip and not line_strip.startswith("--"):
                        headline = line_strip
                        headline_line_idx = idx
                        break
                
                # Find content start (first meaningful line after headline and any separators)
                if headline_line_idx != -1:
                    for idx in range(headline_line_idx + 1, len(lines)):
                        line_strip = lines[idx].strip()
                        # Skip empty lines, separators, or potential leftover metadata
                        if line_strip and not line_strip.startswith("--") and "|" not in line_strip:
                            content_start_idx = idx
                            break
                
                # Find end of content (look for start of next article or fallback markers)
                if content_start_idx != -1:
                    for idx in range(content_start_idx, len(lines)):
                        line_strip = lines[idx].strip()
                        # Primary end marker: start of next article (e.g., "2. ")
                        if re.match(r"^\d+\.\s+", line_strip):
                            content_end_idx = idx
                            break
                        # Fallback markers (allow them within content, check only at end?)
                        # Let's rely on the next article marker for now.
                        # if "文章編號" in lines[idx] or "[20" in lines[idx] or lines[idx].startswith("----------")
                        #     content_end_idx = idx
                        #     break
                
                # Extract content between start and end indices
                if content_start_idx != -1 and content_end_idx > content_start_idx:
                    content_lines = lines[content_start_idx:content_end_idx]
                    content_text = clean_content("\n".join(content_lines))
                
                # Sometimes the fields are mixed up - newspapers appear to use different formats
                # Fix common issues based on content of fields
                
                # If headline contains 'A' followed by digits, it's likely a page number
                if headline and re.match(r'^A\d+', headline):
                    if not page:
                        page = headline
                    
                    # Shift everything: headline becomes category, category becomes author
                    if category:
                        headline = category
                        category = ""
                    
                    # If we have an author but not headline, use author as headline
                    if author and not headline:
                        headline = author
                        author = ""
                
                # If headline starts with a name (looks like an author), adjust
                if headline and len(headline.split()) <= 3 and not author:
                    author = headline
                    headline = category if category else ""
                    category = ""
                
                # Create the article dictionary
                article = {
                    "id": article_id,
                    "newspaper": extracted_newspaper,
                    "date": date_str,
                    "page": page,
                    "category": category,
                    "author": author,
                    "headline": headline,
                    "content": content_text,
                    "source_file": os.path.basename(file_path),
                }

                articles.append(article)
                logger.debug(
                    f"Extracted article {i + 1} with ID {article_id}: {headline[:30]}..."
                )

            except Exception as e:
                logger.error(f"Error parsing article {i + 1}: {str(e)}")
                continue

        logger.info(f"Successfully parsed {len(articles)} articles from {file_path}")
        return articles

    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
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

    # Write articles to NDJSON file
    with open(output_file, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(articles)} articles to {output_file}")


def process_txt_files(input_dir: str, output_dir: str):
    """Process all TXT files in the input directory."""
    # Get all .txt files in the input directory
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        logger.warning(f"No TXT files found in {input_dir}")
        return

    logger.info(f"Found {len(txt_files)} TXT files to process")

    # Process each file and write to separate NDJSON file
    total_articles = 0
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        output_file = os.path.join(
            output_dir, f"{os.path.splitext(file_name)[0]}.ndjson"
        )

        articles = parse_txt_file(file_path)
        if articles:
            write_to_ndjson(articles, output_file)
            total_articles += len(articles)

    logger.info(f"Total articles processed from TXT files: {total_articles}")


if __name__ == "__main__":
    # Get directories from environment variables or use defaults
    input_dir = os.environ.get("NEWS_RAW_DIR", "data/news/raw")
    output_dir = os.environ.get("NEWS_OUTPUT_DIR", "data/news/processed")

    process_txt_files(input_dir, output_dir)
    logger.info("TXT parsing completed")
