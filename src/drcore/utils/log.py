import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .paths import OUTPUT_DIR, PACKAGE_DIR

LOG_DIR = OUTPUT_DIR / "log"
ROOT_LOGGER_NAME = PACKAGE_DIR.name
CONSOLE = Console()
_CURRENT_LOG_PATH: Path | None = None


def build_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{timestamp}.log"


def get_console() -> Console:
    return CONSOLE


def setup_logging(level: int = logging.INFO) -> Path:
    global _CURRENT_LOG_PATH

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _CURRENT_LOG_PATH is not None and logger.handlers:
        return _CURRENT_LOG_PATH

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = build_log_path()

    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    rich_handler = RichHandler(
        console=CONSOLE,
        show_path=False,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    )

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    _CURRENT_LOG_PATH = log_path
    return log_path
