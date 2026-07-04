"""Project path configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"


PATHS = {
    "project_root": PROJECT_ROOT,
    "raw_data": RAW_DATA_DIR,
    "processed_data": PROCESSED_DATA_DIR,
    "sample_data": SAMPLE_DATA_DIR,
    "models": MODELS_DIR,
}
