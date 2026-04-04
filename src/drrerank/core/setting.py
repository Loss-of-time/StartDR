from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = CORE_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "resource"
OUTPUT_DIR = PROJECT_DIR / "output"

DEFAULT_GNN_DATA_INPUT_DIR = RESOURCE_DIR / "patient_candidate" / "pyserini_bm25_top50"
DEFAULT_GNN_DATA_OUTPUT_ROOT = RESOURCE_DIR / "gnn_data"
DEFAULT_TRAIN_INPUT_DIR = RESOURCE_DIR / "gnn_data" / "pyserini_bm25_top50" / "train"
DEFAULT_DEV_INPUT_DIR = RESOURCE_DIR / "gnn_data" / "pyserini_bm25_top50" / "dev"
DEFAULT_MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
DEFAULT_DATA_FILE = "samples.pkl"
