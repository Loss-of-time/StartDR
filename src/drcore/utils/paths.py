from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "resource"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUT_DIR = PROJECT_DIR / "output"

__all__ = [
    "DOCS_DIR",
    "OUTPUT_DIR",
    "PACKAGE_DIR",
    "PROJECT_DIR",
    "RESOURCE_DIR",
    "SRC_DIR",
]
