from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUT_DIR = PROJECT_DIR / "output"

__all__ = [
    "DATA_DIR",
    "DOCS_DIR",
    "OUTPUT_DIR",
    "PACKAGE_DIR",
    "PROJECT_DIR",
    "SRC_DIR",
]
