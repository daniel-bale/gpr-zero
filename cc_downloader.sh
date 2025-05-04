#!/bin/bash
# Wrapper script for Common Crawl downloader
# This script makes it easier to call the Common Crawl downloader

# Path to the CC-NEWS data directory
DATA_DIR="/home3/chuenho/data"

# Create the data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Make the Python script executable
chmod +x download_cc_news.py

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Download Common Crawl News data."
    echo ""
    echo "Options:"
    echo "  --all                  Download all data from February 2022 to March 2025"
    echo "  --month YYYY-MM        Download data for a specific month"
    echo "  --range YYYY-MM YYYY-MM  Download data for a specific date range"
    echo "  --resume YYYY-MM       Resume downloading from a specific month"
    echo "  --paths-only           Only download paths files, not WARC files"
    echo "  --help                 Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all                # Download all data from February 2022 to March 2025"
    echo "  $0 --month 2023-01      # Download only January 2023 data"
    echo "  $0 --range 2022-02 2022-12  # Download data from February to December 2022"
    echo "  $0 --resume 2023-06     # Resume downloading from June 2023"
    echo ""
}

# Parse command-line arguments
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "$1" in
    --all)
        echo "Downloading all CC-NEWS data from February 2022 to March 2025"
        ./download_cc_news.py --output-dir "$DATA_DIR"
        ;;
    --month)
        if [ -z "$2" ]; then
            echo "Error: Month not specified"
            usage
            exit 1
        fi
        echo "Downloading CC-NEWS data for month: $2"
        ./download_cc_news.py --start-date "$2" --end-date "$2" --output-dir "$DATA_DIR"
        ;;
    --range)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Error: Start or end date not specified"
            usage
            exit 1
        fi
        echo "Downloading CC-NEWS data from $2 to $3"
        ./download_cc_news.py --start-date "$2" --end-date "$3" --output-dir "$DATA_DIR"
        ;;
    --resume)
        if [ -z "$2" ]; then
            echo "Error: Resume month not specified"
            usage
            exit 1
        fi
        echo "Resuming CC-NEWS download from $2"
        RESUME="$2"
        RESUME_FORMAT="CC-NEWS-$RESUME"
        ./download_cc_news.py --resume-from "$RESUME_FORMAT" --output-dir "$DATA_DIR"
        ;;
    --paths-only)
        shift
        case "$1" in
            --all)
                echo "Downloading all CC-NEWS paths from February 2022 to March 2025"
                ./download_cc_news.py --paths-only --output-dir "$DATA_DIR"
                ;;
            --month)
                if [ -z "$2" ]; then
                    echo "Error: Month not specified"
                    usage
                    exit 1
                fi
                echo "Downloading CC-NEWS paths for month: $2"
                ./download_cc_news.py --start-date "$2" --end-date "$2" --paths-only --output-dir "$DATA_DIR"
                ;;
            --range)
                if [ -z "$2" ] || [ -z "$3" ]; then
                    echo "Error: Start or end date not specified"
                    usage
                    exit 1
                fi
                echo "Downloading CC-NEWS paths from $2 to $3"
                ./download_cc_news.py --start-date "$2" --end-date "$3" --paths-only --output-dir "$DATA_DIR"
                ;;
            --resume)
                if [ -z "$2" ]; then
                    echo "Error: Resume month not specified"
                    usage
                    exit 1
                fi
                echo "Resuming CC-NEWS paths download from $2"
                RESUME="$2"
                RESUME_FORMAT="CC-NEWS-$RESUME"
                ./download_cc_news.py --resume-from "$RESUME_FORMAT" --paths-only --output-dir "$DATA_DIR"
                ;;
            *)
                echo "Error: Invalid option after --paths-only"
                usage
                exit 1
                ;;
        esac
        ;;
    --help)
        usage
        exit 0
        ;;
    *)
        echo "Error: Invalid option: $1"
        usage
        exit 1
        ;;
esac

echo "Done!" 