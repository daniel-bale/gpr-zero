"""Shared matplotlib utilities for figure generation.

Provides consistent styling, CJK font setup, and figure saving.
"""

# TODO: add switch for matplotlib backend (e.g. "Agg" for headless environments)
# TODO: add different plot backends (e.g. plotly) and corresponding utilities
import logging
from pathlib import Path

import matplotlib.axes as mpl_axes
import matplotlib.dates as mdates
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

from gpr_zero.utils.font_setup import ensure_cjk_font_setup

logger = logging.getLogger(__name__)


def setup_plot_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets up CJK fonts and applies the seaborn whitegrid theme.
    Safe to call multiple times.
    """
    ensure_cjk_font_setup()
    plt.style.use("seaborn-v0_8-whitegrid")


def save_figure(
    fig: mpl_figure.Figure,
    filename: str,
    output_dir: str | Path = "output/figures",
    *,
    dpi: int = 300,
) -> Path:
    """Save a figure to disk.

    Args:
        fig: The matplotlib figure to save.
        filename: Output filename (e.g. ``"gpr_median.png"``).
        output_dir: Directory to save the figure in.
        dpi: Resolution in dots per inch (default 300).

    Returns:
        Path to the saved figure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    output_path = out / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure: %s", output_path)

    return output_path


def format_date_axis(ax: mpl_axes.Axes) -> None:
    """Apply standard date formatting to the x-axis.

    Uses ``YYYY-MM`` major ticks with 3-month intervals and auto-rotated labels.

    Args:
        ax: The matplotlib axes to format.
    """
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def clean_axes(ax: mpl_axes.Axes) -> None:
    """Remove titles and axis labels for clean publication figures.

    Args:
        ax: The matplotlib axes to clean.
    """
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
