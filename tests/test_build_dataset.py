import subprocess
import sys

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from src.data.build_dataset import (
    MODEL_FEATURES,
    TRAINING_DATASET_PATH,
    build_training_dataset,
    create_advance_label,
    create_result_label,
)
from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.utils.config import PROJECT_ROOT


def test_create_result_label():
    assert create_result_label({"team_a_score": 2, "team_b_score": 1}) == 0
    assert create_result_label({"team_a_score": 1, "team_b_score": 1}) == 1
    assert create_result_label({"team_a_score": 0, "team_b_score": 1}) == 2


def test_create_advance_label():
    assert create_advance_label({"is_knockout": True, "team_a_advanced": True}) == 1
    assert create_advance_label({"is_knockout": True, "team_a_advanced": False}) == 0
    assert create_advance_label({"is_knockout": False, "team_a_advanced": True}) is None
    assert create_advance_label({"is_knockout": True, "team_a_advanced": ""}) is None


def test_build_training_dataset_creates_model_ready_sample_dataset():
    dataset = build_training_dataset(
        matches_df=load_matches(),
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )

    assert TRAINING_DATASET_PATH.is_file()
    assert not dataset.empty
    assert "result_label" in dataset.columns
    assert "advance_label" in dataset.columns
    assert set(dataset["result_label"]).issubset({0, 1, 2})
    assert dataset["date"].is_monotonic_increasing

    for feature in MODEL_FEATURES:
        assert feature in dataset.columns
        assert is_numeric_dtype(dataset[feature])
        assert not dataset[feature].isna().any()


def test_training_dataset_contains_all_feature_families():
    dataset = build_training_dataset(
        matches_df=load_matches(),
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )

    expected_columns = {
        "team_a_elo",
        "elo_diff",
        "team_a_wc_goals_scored",
        "xg_delta",
        "team_a_pressing_level",
        "tactical_edge",
        "h2h_matches_count",
        "h2h_weighted_score",
        "result_label",
    }
    assert expected_columns.issubset(dataset.columns)


def test_training_dataset_uses_pre_match_elo_features():
    dataset = build_training_dataset(
        matches_df=load_matches(),
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )

    first_match = dataset.iloc[0]
    assert first_match["team_a_elo"] == pytest.approx(1500)
    assert first_match["team_b_elo"] == pytest.approx(1500)
    assert first_match["elo_diff"] == pytest.approx(0)


def test_cli_builds_processed_training_dataset():
    if TRAINING_DATASET_PATH.exists():
        TRAINING_DATASET_PATH.unlink()

    result = subprocess.run(
        [sys.executable, "-m", "src.data.build_dataset"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Created" in result.stdout
    assert TRAINING_DATASET_PATH.is_file()

    saved_dataset = pd.read_csv(TRAINING_DATASET_PATH)
    assert "result_label" in saved_dataset.columns
    assert len(saved_dataset) == len(load_matches())
