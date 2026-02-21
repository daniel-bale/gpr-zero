"""
Font setup utilities for CJK text display in matplotlib plots.
"""
# TODO: this should be write into skills and help user configure language and font, and test by LLM's vision

import logging
import os
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Default font search paths, overridable via CJK_FONT_PATHS env var (colon-separated)
_DEFAULT_FONT_PATHS = (
    "data/fonts/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
)

# System font candidates (in order of preference)
_SYSTEM_FONT_CANDIDATES = (
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
)


def _get_font_search_paths() -> list[Path]:
    """Get font search paths from env var or defaults."""
    env_paths = os.environ.get("CJK_FONT_PATHS")
    if env_paths:
        return [Path(p) for p in env_paths.split(":") if p.strip()]

    project_root = Path(__file__).parent.parent.parent.parent
    paths = []
    for p in _DEFAULT_FONT_PATHS:
        path = Path(p)
        paths.append(path)
        # Also try relative to project root
        if not path.is_absolute():
            paths.append(project_root / path)
    return paths


def _apply_font_config(font_name: str) -> None:
    """Apply matplotlib rcParams for the given font."""
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = [
        font_name,
        "Noto Sans CJK TC",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
        "Arial",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def setup_cjk_font(
    font_path: str | Path | None = None,
) -> tuple[bool, str | None]:
    """
    Set up CJK font for matplotlib to properly display Chinese characters.

    Args:
        font_path: Optional explicit path to a CJK font file.
                   If None, searches CJK_FONT_PATHS env var then defaults.

    Returns:
        Tuple of (success, font_name).
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    # Build search list
    if font_path is not None:
        search_paths = [Path(font_path)]
    else:
        search_paths = _get_font_search_paths()

    # Find first existing font file
    working_path = None
    for path in search_paths:
        if path.exists() and path.is_file():
            working_path = path
            break

    if not working_path:
        return setup_system_cjk_font()

    try:
        fm.fontManager.addfont(str(working_path))

        try:
            prop = fm.FontProperties(fname=str(working_path))
            font_name = prop.get_name()
        except Exception as e:
            logger.warning(f"Could not get font properties for {working_path}: {e}")
            font_name = "Noto Sans CJK TC"

        _apply_font_config(font_name)
        logger.info(f"Configured CJK font: {font_name} from {working_path}")
        return True, font_name

    except Exception as e:
        logger.error(f"Failed to setup CJK font from {working_path}: {e}")
        return setup_system_cjk_font()


def setup_system_cjk_font() -> tuple[bool, str | None]:
    """
    Fallback: Set up CJK font support using system fonts.

    Returns:
        Tuple of (success, font_name).
    """
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in _SYSTEM_FONT_CANDIDATES:
        if font in available_fonts:
            try:
                _apply_font_config(font)
                logger.info(f"Using system CJK font: {font}")
                return True, font
            except Exception as e:
                logger.error(f"Failed to configure system font {font}: {e}")

    logger.warning("No suitable CJK fonts found")
    return False, None


def ensure_cjk_font_setup(
    font_path: str | Path | None = None,
) -> None:
    """
    Convenience function to setup CJK font with simple usage.
    Logs warnings if setup fails but doesn't raise exceptions.

    Args:
        font_path: Optional path to a CJK font file.
    """
    success, _ = setup_cjk_font(font_path)
    if not success:
        logger.warning(
            "CJK font setup failed. Chinese text may not display properly in plots."
        )
