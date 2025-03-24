"""Data loader for news articles and preprocessing utilities."""

from typing import Dict, List, Optional, Any, Tuple, Union
import os
import json
import pandas as pd
import re
from datetime import datetime


class NewsDataLoader:
    """Loads and preprocesses news article data."""

    def __init__(self, data_dir: str = "data/news", cache_dir: str = "data/cache"):
        """Initialize news data loader.

        Args:
            data_dir: Directory containing news data files
            cache_dir: Directory for caching processed data
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def load_articles(
        self,
        source: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load articles from a specific news source.

        Args:
            source: News source name
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            limit: Optional limit on number of articles
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame containing articles
        """
        # Check if cached data exists
        cache_file = os.path.join(self.cache_dir, f"{source}_articles.csv")
        if use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])

            # Apply date filters if provided
            if start_date:
                df = df[df["date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["date"] <= pd.to_datetime(end_date)]

            # Apply limit if provided
            if limit:
                df = df.head(limit)

            return df

        # Load data files
        source_dir = os.path.join(self.data_dir, source)
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory not found: {source_dir}")

        # Find all data files for the source
        data_files = [
            os.path.join(source_dir, f)
            for f in os.listdir(source_dir)
            if f.endswith(".json") or f.endswith(".csv")
        ]

        # Load articles from each file
        articles = []
        for file_path in data_files:
            file_articles = self._load_file(file_path)
            articles.extend(file_articles)

        # Convert to DataFrame
        df = pd.DataFrame(articles)

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Apply date filters if provided
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        # Apply limit if provided
        if limit:
            df = df.head(limit)

        # Save to cache
        df.to_csv(cache_file, index=False)

        return df

    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load articles from a data file.

        Args:
            file_path: Path to the data file

        Returns:
            List of article dictionaries
        """
        # Load based on file extension
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "articles" in data:
                return data["articles"]
            else:
                return [data]

        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            return df.to_dict("records")

        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def group_by_date(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        text_column: str = "text",
        id_column: str = "id",
    ) -> Dict[str, Dict[str, str]]:
        """Group articles by date.

        Args:
            df: DataFrame containing articles
            date_column: Name of the date column
            text_column: Name of the text column
            id_column: Name of the ID column

        Returns:
            Dictionary mapping dates to article dictionaries
        """
        # Convert date column to string format
        df = df.copy()
        df["date_str"] = df[date_column].dt.strftime("%Y-%m-%d")

        # Group by date
        grouped = {}
        for date, group in df.groupby("date_str"):
            articles = {}
            for _, row in group.iterrows():
                article_id = str(row[id_column])
                articles[article_id] = row[text_column]
            grouped[date] = articles

        return grouped

    def get_total_articles_by_date(
        self,
        source: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get total number of articles by date for normalization.

        Args:
            source: News source name
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            Dictionary mapping dates to total article counts
        """
        # Load all articles for the source
        df = self.load_articles(source, start_date, end_date)

        # Convert date to string format
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

        # Count articles by date
        counts = df.groupby("date_str").size().to_dict()

        return counts

    def preprocess_text(self, text: str) -> str:
        """Preprocess article text.

        Args:
            text: Article text

        Returns:
            Preprocessed text
        """
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
