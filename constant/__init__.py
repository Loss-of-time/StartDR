from pathlib import Path

from .CYPHER import *  # noqa: F403

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
