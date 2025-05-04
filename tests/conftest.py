import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
# This allows tests to import modules from the 'scripts' directory
# using 'from scripts.module import ...'
# Adjust if your scripts are structured differently or if moving to src/

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Optional: Print path for debugging if needed
# print("\nModified sys.path in conftest.py:")
# print("\n".join(sys.path))
# print("---\n") 