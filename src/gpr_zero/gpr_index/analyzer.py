"""Analyzer for GPR indices - comparison and visualization."""

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GPRAnalyzer:
    """Analyzes and visualizes GPR indices."""

    def __init__(self, output_dir: str = "data/analysis"):
        """Initialize GPR analyzer.

        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_indices(self, index_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load multiple GPR indices for comparison.

        Args:
            index_paths: Dictionary mapping index names to file paths

        Returns:
            Dictionary mapping index names to DataFrames
        """
        indices = {}
        for name, path in index_paths.items():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            indices[name] = df

        return indices

    def compare_indices(
        self, indices: Dict[str, pd.DataFrame], column: str = "overall_index"
    ) -> pd.DataFrame:
        """Compare multiple indices.

        Args:
            indices: Dictionary mapping index names to DataFrames
            column: Column to compare

        Returns:
            DataFrame with indices side by side
        """
        # Extract specified column from each index
        compared = {}
        for name, df in indices.items():
            compared[name] = df[column]

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(compared)

        return comparison_df

    def find_divergence_points(
        self, comparison_df: pd.DataFrame, threshold: float = 0.5, window: int = 1
    ) -> pd.DataFrame:
        """Find points where indices diverge significantly.

        Args:
            comparison_df: DataFrame with indices side by side
            threshold: Threshold for significant divergence
            window: Rolling window for smoothing

        Returns:
            DataFrame with divergence points
        """
        # Calculate rolling means if window > 1
        if window > 1:
            smoothed = comparison_df.rolling(window=window).mean()
        else:
            smoothed = comparison_df

        # Calculate pairwise differences
        columns = smoothed.columns
        divergence = pd.DataFrame(index=smoothed.index)

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                diff_col = f"{col1}_vs_{col2}"
                divergence[diff_col] = (smoothed[col1] - smoothed[col2]).abs()

        # Identify points above threshold
        above_threshold = divergence > threshold

        # Filter to only include rows with at least one divergence
        divergent_points = divergence[above_threshold.any(axis=1)]

        return divergent_points

    def plot_indices(
        self,
        comparison_df: pd.DataFrame,
        title: str = "GPR Index Comparison",
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """Plot multiple indices for comparison.

        Args:
            comparison_df: DataFrame with indices side by side
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib figure
        """
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each column
        for column in comparison_df.columns:
            ax.plot(comparison_df.index, comparison_df[column], label=column)

        # Add labels and title
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Index Value")
        ax.legend(title="Index")

        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save if requested
        if save:
            save_path = os.path.join(
                self.output_dir, filename or "index_comparison.png"
            )
            plt.savefig(save_path, dpi=300)

        return fig

    def correlation_analysis(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between indices.

        Args:
            comparison_df: DataFrame with indices side by side

        Returns:
            Correlation matrix
        """
        return comparison_df.corr()

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Index Correlation Matrix",
        figsize: Tuple[int, int] = (8, 6),
        save: bool = True,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """Plot correlation matrix as a heatmap.

        Args:
            corr_matrix: Correlation matrix
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib figure
        """
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create a heatmap
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)

        # Set tick labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns)
        ax.set_yticklabels(corr_matrix.index)

        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        # Add title
        ax.set_title(title)
        plt.tight_layout()

        # Save if requested
        if save:
            save_path = os.path.join(
                self.output_dir, filename or "correlation_matrix.png"
            )
            plt.savefig(save_path, dpi=300)

        return fig
