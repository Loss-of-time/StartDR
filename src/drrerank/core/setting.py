from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CORE_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "resource"
OUTPUT_DIR = PROJECT_DIR / "output"

DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT = RESOURCE_DIR / "patient_candidate"
DEFAULT_TRAIN_INPUT_PATH = (
    DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT
    / "pyserini_bm25_top50"
    / "train.jsonl"
)
DEFAULT_DEV_INPUT_PATH = (
    DEFAULT_TRACEDR_JSONL_OUTPUT_ROOT
    / "pyserini_bm25_top50"
    / "dev.jsonl"
)
DEFAULT_MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
