from src.utils.config import (
    MODELS_DIR,
    PATHS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
)


def test_config_paths_resolve_to_project_directories():
    assert PROJECT_ROOT.name == "world-cup-ai-predictor"
    assert RAW_DATA_DIR == PROJECT_ROOT / "data" / "raw"
    assert PROCESSED_DATA_DIR == PROJECT_ROOT / "data" / "processed"
    assert SAMPLE_DATA_DIR == PROJECT_ROOT / "data" / "sample"
    assert MODELS_DIR == PROJECT_ROOT / "models"


def test_configured_directories_exist():
    for key in ("raw_data", "processed_data", "sample_data", "models"):
        assert PATHS[key].is_dir()
