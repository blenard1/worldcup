import json
import subprocess
import sys
import warnings

import numpy as np
import pytest

from src.data.build_dataset import (
    MODEL_FEATURES,
    TRAINING_DATASET_PATH,
    build_training_dataset,
)
from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.models.train_result_model import (
    RESULT_MODEL_METADATA_PATH,
    RESULT_MODEL_PATH,
    train_result_model,
)
from src.prediction.predict_result import predict_result_proba
from src.utils.config import PROJECT_ROOT


class StrictNumericModel:
    classes_ = [0, 1, 2]

    def predict_proba(self, X):
        assert not X.empty
        assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
        return np.array([[0.4, 0.2, 0.4]])


def _build_sample_training_dataset():
    return build_training_dataset(
        matches_df=load_matches(),
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )


def test_train_result_model_saves_model_and_metadata(tmp_path):
    _build_sample_training_dataset()
    model_path = tmp_path / "result_model.pkl"
    metadata_path = tmp_path / "result_model_metadata.json"

    metadata = train_result_model(
        data_path=TRAINING_DATASET_PATH,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert model_path.is_file()
    assert metadata_path.is_file()
    assert metadata["model_type"] in {"xgboost", "random_forest"}
    assert metadata["features"] == MODEL_FEATURES
    assert 0 <= metadata["accuracy"] <= 1
    assert "created_at" in metadata

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["model_type"] == metadata["model_type"]


def test_predict_result_proba_returns_valid_probabilities(tmp_path):
    from joblib import load

    dataset = _build_sample_training_dataset()
    model_path = tmp_path / "result_model.pkl"
    metadata_path = tmp_path / "result_model_metadata.json"
    train_result_model(
        data_path=TRAINING_DATASET_PATH,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        model = load(model_path)

    probabilities = predict_result_proba(model, dataset.iloc[0])

    assert set(probabilities) == {"team_a_win", "draw", "team_b_win"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in probabilities.values())


def test_predict_result_proba_coerces_mixed_series_features_to_numeric():
    dataset = _build_sample_training_dataset()
    feature_row = dataset.iloc[0].copy()
    feature_row["team_a"] = "France"
    feature_row["team_b"] = "Paraguay"

    probabilities = predict_result_proba(StrictNumericModel(), feature_row)

    assert probabilities == {
        "team_a_win": pytest.approx(0.4),
        "draw": pytest.approx(0.2),
        "team_b_win": pytest.approx(0.4),
    }


def test_predict_result_proba_raises_for_missing_features(tmp_path):
    from joblib import load

    _build_sample_training_dataset()
    model_path = tmp_path / "result_model.pkl"
    metadata_path = tmp_path / "result_model_metadata.json"
    train_result_model(
        data_path=TRAINING_DATASET_PATH,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        model = load(model_path)

    with pytest.raises(ValueError, match=MODEL_FEATURES[0]):
        predict_result_proba(model, {"not_a_model_feature": 1})


def test_train_result_model_handles_one_row_dataset(tmp_path):
    dataset = _build_sample_training_dataset().head(1)
    data_path = tmp_path / "tiny_training_dataset.csv"
    model_path = tmp_path / "tiny_result_model.pkl"
    metadata_path = tmp_path / "tiny_result_model_metadata.json"
    dataset.to_csv(data_path, index=False)

    metadata = train_result_model(
        data_path=data_path,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert model_path.is_file()
    assert metadata["model_type"] in {"xgboost", "random_forest"}
    assert 0 <= metadata["accuracy"] <= 1


def test_train_result_model_cli_saves_default_artifacts():
    _build_sample_training_dataset()
    if RESULT_MODEL_PATH.exists():
        RESULT_MODEL_PATH.unlink()
    if RESULT_MODEL_METADATA_PATH.exists():
        RESULT_MODEL_METADATA_PATH.unlink()

    result = subprocess.run(
        [sys.executable, "-m", "src.models.train_result_model"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "accuracy:" in result.stdout
    assert "class_distribution:" in result.stdout
    assert RESULT_MODEL_PATH.is_file()
    assert RESULT_MODEL_METADATA_PATH.is_file()
