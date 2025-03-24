"""Calculator for the Geopolitical Risk (GPR) index."""

from typing import Dict, List, Optional, Any, Tuple, Union
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime


class GPRCalculator:
    """Calculates Geopolitical Risk (GPR) index from news articles."""

    def __init__(
        self, dictionary_path: Optional[str] = None, output_dir: str = "data/indices"
    ):
        """Initialize GPR calculator.

        Args:
            dictionary_path: Path to the keyword dictionary JSON file
            output_dir: Directory to save calculated indices
        """
        self.output_dir = output_dir

        # Load dictionary if provided
        self.dictionary = None
        if dictionary_path:
            self.load_dictionary(dictionary_path)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_dictionary(self, dictionary_path: str) -> None:
        """Load a keyword dictionary from a JSON file.

        Args:
            dictionary_path: Path to the dictionary JSON file
        """
        with open(dictionary_path, "r", encoding="utf-8") as f:
            self.dictionary = json.load(f)

    def calculate_article_score(self, article_text: str) -> Dict[str, float]:
        """Calculate GPR score for a single article.

        Args:
            article_text: Text of the article

        Returns:
            Dictionary with category scores and overall score
        """
        if not self.dictionary:
            raise ValueError("Dictionary not loaded. Call load_dictionary first.")

        # Convert to lowercase for case-insensitive matching
        article_lower = article_text.lower()

        # Calculate scores for each category
        category_scores = {}
        for category, keywords in self.dictionary.items():
            # Count occurrences of each keyword
            count = 0
            for keyword in keywords:
                count += article_lower.count(keyword.lower())

            # Normalize by article length
            article_length = len(article_text.split())
            if article_length > 0:
                score = count / article_length
            else:
                score = 0

            category_scores[category] = score

        # Calculate overall score as sum of category scores
        overall_score = sum(category_scores.values())

        return {"overall": overall_score, "categories": category_scores}

    def calculate_daily_index(
        self,
        articles: Dict[str, str],
        date: Union[str, datetime],
        normalize: bool = True,
        total_articles: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate GPR index for a specific date.

        Args:
            articles: Dictionary mapping article IDs to article texts
            date: Date for the index
            normalize: Whether to normalize by total number of articles
            total_articles: Optional total number of articles (for normalization)

        Returns:
            Dictionary with index values
        """
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")

        # Calculate scores for each article
        article_scores = {
            article_id: self.calculate_article_score(article_text)
            for article_id, article_text in articles.items()
        }

        # Initialize category scores
        all_categories = self.dictionary.keys() if self.dictionary else set()
        category_totals = {category: 0.0 for category in all_categories}

        # Sum scores across all articles
        for scores in article_scores.values():
            for category, score in scores["categories"].items():
                category_totals[category] += score

        # Calculate overall index as sum of category totals
        overall_index = sum(category_totals.values())

        # Normalize if requested
        if normalize:
            num_articles = total_articles or len(articles)
            if num_articles > 0:
                overall_index /= num_articles
                category_totals = {
                    category: score / num_articles
                    for category, score in category_totals.items()
                }

        # Create result dictionary
        result = {
            "date": date_str,
            "overall_index": overall_index,
            "category_indices": category_totals,
            "num_articles": len(articles),
            "total_articles": total_articles or len(articles),
        }

        return result

    def calculate_index_series(
        self,
        articles_by_date: Dict[str, Dict[str, str]],
        normalize: bool = True,
        total_articles_by_date: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Calculate GPR index series across multiple dates.

        Args:
            articles_by_date: Dictionary mapping dates to articles dictionaries
            normalize: Whether to normalize by total number of articles
            total_articles_by_date: Optional dictionary of total articles by date

        Returns:
            DataFrame with index values by date
        """
        # Calculate index for each date
        results = []
        for date, articles in articles_by_date.items():
            total_articles = None
            if total_articles_by_date:
                total_articles = total_articles_by_date.get(date)

            result = self.calculate_daily_index(
                articles, date, normalize=normalize, total_articles=total_articles
            )
            results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return df

    def save_index(self, index_df: pd.DataFrame, name: str) -> None:
        """Save index DataFrame to CSV file.

        Args:
            index_df: DataFrame with index values
            name: Base name for the file
        """
        filename = os.path.join(self.output_dir, f"{name}.csv")
        index_df.to_csv(filename)

    def load_index(self, name: str) -> pd.DataFrame:
        """Load a previously saved index.

        Args:
            name: Name of the index file (without .csv extension)

        Returns:
            Loaded index DataFrame
        """
        filename = os.path.join(self.output_dir, f"{name}.csv")
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
