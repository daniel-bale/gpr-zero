#!/usr/bin/env python3
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cc_news_download.log"),
        logging.StreamHandler()
    ]
)

def generate_date_range(start_date: str, end_date: str) -> list[str]:
    """Generate a list of months between start and end date in format CC-NEWS-YYYY-MM"""
    start_parts = start_date.split('-')
    end_parts = end_date.split('-')
    
    start_year, start_month = int(start_parts[0]), int(start_parts[1])
    end_year, end_month = int(end_parts[0]), int(end_parts[1])
    
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    
    months = []
    while current <= end:
        month_id = f"CC-NEWS-{current.year}-{current.month:02d}"
        months.append(month_id)
        
        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return months

def download_paths(month_id: str, output_dir: Path, retries: int = 3) -> bool:
    """Download the paths file for a month"""
    # The paths file will be downloaded directly to the output directory
    paths_file = output_dir / "warc.paths.gz"
    
    # Check if paths file already exists
    if paths_file.exists():
        logging.info(f"Paths file already exists at {paths_file}, will be overridden")
    
    # Try to download paths file
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Downloading paths for {month_id} (Attempt {attempt}/{retries})")
            
            # Command format: cc-downloader download-paths CC-NEWS-YYYY-MM warc .
            cmd = [
                "cc-downloader", 
                "download-paths", 
                month_id, 
                "warc", 
                str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully downloaded paths for {month_id}")
                # Verify that paths file exists
                if paths_file.exists():
                    return True
                else:
                    logging.error(f"Command succeeded but paths file not found at {paths_file}")
            else:
                logging.error(f"Failed to download paths for {month_id}: {result.stderr}")
        except Exception as e:
            logging.error(f"Error downloading paths for {month_id}: {e}")
        
        # Only sleep if we're going to retry
        if attempt < retries:
            logging.info(f"Retrying in 10 seconds...")
            subprocess.run(["sleep", "10"])
    
    return False

def download_warcs(month_id: str, output_dir: Path, retries: int = 3) -> bool:
    """Download WARC files for a month"""
    # The paths file is expected to be directly in the output directory
    paths_file = output_dir / "warc.paths.gz"
    
    if not paths_file.exists():
        logging.error(f"Paths file not found at {paths_file}")
        return False
    
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Downloading WARC files for {month_id} (Attempt {attempt}/{retries})")
            
            # Command format: cc-downloader download warc.paths.gz .
            cmd = [
                "cc-downloader", 
                "download", 
                str(paths_file), 
                str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully downloaded WARC files for {month_id}")
                return True
            else:
                logging.error(f"Failed to download WARC files for {month_id}: {result.stderr}")
        except Exception as e:
            logging.error(f"Error downloading WARC files for {month_id}: {e}")
        
        # Only sleep if we're going to retry
        if attempt < retries:
            logging.info(f"Retrying in 10 seconds...")
            subprocess.run(["sleep", "10"])
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Download Common Crawl News dataset")
    parser.add_argument("--start-date", default="2022-02", help="Start date in format YYYY-MM")
    parser.add_argument("--end-date", default="2025-03", help="End date in format YYYY-MM")
    parser.add_argument("--output-dir", default="/home5/chuenho/data", help="Output directory")
    parser.add_argument("--resume-from", help="Resume downloading from a specific month (format: CC-NEWS-YYYY-MM)")
    parser.add_argument("--paths-only", action="store_true", help="Only download paths files, not WARC files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate list of months to download
    months = generate_date_range(args.start_date, args.end_date)
    logging.info(f"Planning to download {len(months)} months: {', '.join(months)}")
    
    # Apply resume filter if specified
    if args.resume_from:
        if args.resume_from in months:
            resume_index = months.index(args.resume_from)
            months = months[resume_index:]
            logging.info(f"Resuming from {args.resume_from}, {len(months)} months remaining")
        else:
            logging.warning(f"Resume month {args.resume_from} not found in date range, starting from beginning")
    
    # Download each month
    for i, month in enumerate(months):
        logging.info(f"Processing month {i+1}/{len(months)}: {month}")
        
        # Download paths file
        paths_success = download_paths(month, output_dir)
        
        if not paths_success:
            logging.error(f"Failed to download paths for {month}, skipping WARC download")
            continue
        
        # Download WARC files if not paths-only
        if not args.paths_only:
            warcs_success = download_warcs(month, output_dir)
            if not warcs_success:
                logging.error(f"Failed to download WARC files for {month}")
    
    logging.info("Download process completed")

if __name__ == "__main__":
    main() 