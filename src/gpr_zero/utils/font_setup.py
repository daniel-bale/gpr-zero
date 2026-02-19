"""
Font setup utilities for CJK text display in matplotlib plots.
"""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from typing import Tuple


logger = logging.getLogger(__name__)


def setup_cjk_font(font_path: str | Path = "data/fonts/NotoSansCJKtc-Regular.otf") -> Tuple[bool, str | None]:
    """
    Set up CJK font for matplotlib to properly display Chinese, Japanese, Korean characters.
    
    Args:
        font_path: Path to the CJK font file. Defaults to "data/fonts/NotoSansCJKtc-Regular.otf"
    
    Returns:
        Tuple of (setup_success: bool, font_name: str | None)
        - setup_success: True if font was successfully configured, False otherwise
        - font_name: Name of the configured font if successful, None if failed
    """
    # Suppress matplotlib font warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    font_path = Path(font_path)
    
    # Try multiple possible font paths
    project_root = Path(__file__).parent.parent.parent.parent
    possible_paths = [
        font_path,
        project_root / font_path,  # From project root
        project_root / "data/fonts/NotoSansCJKtc-Regular.otf",  # Explicit path
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf"),
    ]
    
    working_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            working_path = path
            break
    
    if not working_path:
        # Fallback to system CJK fonts
        return setup_system_cjk_font()
    
    try:
        # Register the font with matplotlib
        fm.fontManager.addfont(str(working_path))
        
        try:
            # Get the registered font name
            prop = fm.FontProperties(fname=str(working_path))
            font_name = prop.get_name()
            logger.info(f"Font file {working_path} registered with name: '{font_name}'")
        except Exception as e:
            logger.error(f"Could not get font properties for {working_path}: {e}")
            # Try common Noto Sans CJK names
            font_name = "Noto Sans CJK TC"
            logger.warning(f"Falling back to using font name: {font_name}")
        
        # Configure matplotlib to use the CJK font with multiple fallbacks
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [
            font_name,
            'Noto Sans CJK TC',
            'Noto Sans CJK SC',
            'SimHei',
            'Microsoft YaHei',
            'DejaVu Sans',
            'Arial'
        ]
        plt.rcParams['axes.unicode_minus'] = False
        
        # Test if the font actually works
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "測試", fontsize=12)
            fig.canvas.draw()
            plt.close(fig)
            logger.info(f"Font rendering test passed for: {font_name}")
        except Exception as e:
            logger.warning(f"Font rendering test failed for {font_name}: {e}")
            # Continue anyway, might still work in actual plots
        
        logger.info(f"Successfully configured CJK font: {font_name} from {working_path}")
        return True, font_name
        
    except Exception as e:
        logger.error(f"Failed to setup CJK font from {working_path}: {e}")
        return setup_system_cjk_font()


def setup_system_cjk_font() -> Tuple[bool, str | None]:
    """
    Fallback: Set up CJK font support using system fonts.
    
    Returns:
        Tuple of (success: bool, font_info: str | None)
    """
    # List of CJK fonts to try (in order of preference)
    font_candidates = [
        'Noto Sans CJK TC',
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB',
        'WenQuanYi Micro Hei',
        'Arial Unicode MS',
        'DejaVu Sans'
    ]
    
    # Get available system fonts
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    working_font = None
    for font in font_candidates:
        if font in available_fonts:
            working_font = font
            break
    
    if working_font:
        try:
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [working_font] + [
                'Noto Sans CJK TC',
                'Noto Sans CJK SC',
                'SimHei',
                'Microsoft YaHei',
                'DejaVu Sans',
                'Arial'
            ]
            plt.rcParams['axes.unicode_minus'] = False
            
            # Test the font
            try:
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, "測試", fontsize=12)
                fig.canvas.draw()
                plt.close(fig)
                logger.info(f"System font rendering test passed for: {working_font}")
            except Exception as e:
                logger.warning(f"System font rendering test failed for {working_font}: {e}")
            
            logger.info(f"Using system CJK font: {working_font}")
            return True, working_font
        except Exception as e:
            logger.error(f"Failed to configure system font {working_font}: {e}")
    
    logger.warning("No suitable CJK fonts found")
    return False, None


def test_cjk_rendering() -> bool:
    """
    Test if CJK characters render properly with current font setup.
    
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        # Create a simple test plot
        fig, ax = plt.subplots(figsize=(6, 4))
        
        test_text = "測試中文字體 (Test Chinese Font)"
        ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("CJK Font Test")
        
        # Try to render (this will fail if fonts don't work)
        fig.canvas.draw()
        plt.close(fig)
        
        return True
    except Exception:
        return False


def ensure_cjk_font_setup(font_path: str | Path = "data/fonts/NotoSansCJKtc-Regular.otf") -> None:
    """
    Convenience function to setup CJK font with simple usage.
    Logs warnings if setup fails but doesn't raise exceptions.
    
    Args:
        font_path: Path to the CJK font file. Defaults to "data/fonts/NotoSansCJKtc-Regular.otf"
    """
    success, font_name = setup_cjk_font(font_path)
    if not success:
        logger.warning("CJK font setup failed. Chinese text may not display properly in plots.")

def ensure_cjk_font_setup_strict(font_path: str | Path = "data/fonts/NotoSansCJKtc-Regular.otf") -> bool:
    """
    Strict CJK font setup with comprehensive validation and immediate failure on any error.
    
    Args:
        font_path: Path to the CJK font file. Defaults to "data/fonts/NotoSansCJKtc-Regular.otf"
        
    Returns:
        bool: True if setup successful, False if failed
        
    Raises:
        SystemExit: On critical font setup failures
    """
    import os
    import sys
    
    # Suppress matplotlib font warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
    
    font_path = Path(font_path)
    
    # Comprehensive font file validation
    try:
        # Check file existence
        if not font_path.exists():
            logger.error(f"FATAL: Font file not found: {font_path}")
            sys.exit(2)
        
        # Check file permissions
        if not os.access(font_path, os.R_OK):
            logger.error(f"FATAL: Insufficient permissions to read font file: {font_path}")
            sys.exit(3)
        
        # Check file size (corrupted files are often 0 bytes or extremely small)
        file_size = font_path.stat().st_size
        if file_size < 1024:  # Less than 1KB is suspicious for a font file
            logger.error(f"FATAL: Font file appears corrupted (size: {file_size} bytes): {font_path}")
            sys.exit(4)
        
        # Check file format by extension
        valid_extensions = {'.ttf', '.otf', '.ttc', '.woff', '.woff2'}
        if font_path.suffix.lower() not in valid_extensions:
            logger.error(f"FATAL: Unsupported font format '{font_path.suffix}': {font_path}")
            logger.error(f"Supported formats: {', '.join(valid_extensions)}")
            sys.exit(5)
        
        # Try to read first few bytes to check for corruption
        try:
            with open(font_path, 'rb') as f:
                header = f.read(16)
                if len(header) < 16:
                    logger.error(f"FATAL: Font file appears truncated: {font_path}")
                    sys.exit(6)
        except IOError as e:
            logger.error(f"FATAL: Cannot read font file: {font_path} - {e}")
            sys.exit(7)
        except MemoryError:
            logger.error(f"FATAL: Insufficient memory to read font file: {font_path}")
            sys.exit(8)
        
    except Exception as e:
        logger.error(f"FATAL: Unexpected error during font validation: {e}")
        sys.exit(9)
    
    # Attempt font setup with comprehensive error handling
    try:
        success, font_name = setup_cjk_font(font_path)
        
        if not success:
            logger.error("FATAL: CJK font setup failed - unable to configure Chinese text display")
            sys.exit(10)
        
        logger.info(f"Successfully configured CJK font: {font_name}")
        
        # Test font rendering
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "測試", fontsize=12)
            plt.close(fig)
            logger.info("Font rendering test passed")
            
        except Exception as e:
            logger.error(f"FATAL: Font rendering test failed: {e}")
            sys.exit(15)
        
        return True
        
    except MemoryError:
        logger.error("FATAL: Insufficient memory for font loading")
        sys.exit(11)
    except OSError as e:
        logger.error(f"FATAL: System error during font loading: {e}")
        sys.exit(12)
    except Exception as e:
        logger.error(f"FATAL: Unexpected error during font setup: {e}")
        sys.exit(13)