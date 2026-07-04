import subprocess
import sys
from numbers import Real

from src.data.build_dataset import MODEL_FEATURES, TRAINING_DATASET_PATH, build_training_dataset
from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.models.poisson_model import estimate_team_attack_defense
from src.models.train_result_model import train_result_model
from src.prediction.predict_match import (
    build_match_feature_row,
    format_prediction_output,
)
from src.utils.config import PROJECT_ROOT


def _ensure_prediction_artifacts():
    matches = load_matches()
    build_training_dataset(
        matches_df=matches,
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )
    estimate_team_attack_defense(matches)
    train_result_model(data_path=TRAINING_DATASET_PATH)


def test_build_match_feature_row_handles_unknown_team_gracefully():
    row, team_strength, notes = build_match_feature_row(
        team_a="France",
        team_b="Paraguay",
        stage="R16",
        knockout=True,
        match_date="2026-07-04",
    )

    assert row["team_a"] == "France"
    assert row["team_b"] == "Paraguay"
    assert not team_strength.empty
    assert any("Paraguay" in note for note in notes)

    for feature in MODEL_FEATURES:
        assert feature in row
        assert isinstance(row[feature], Real)


def test_format_prediction_output_is_readable():
    output = format_prediction_output(
        "France",
        "Paraguay",
        {
            "team_a_win": 0.55,
            "draw": 0.25,
            "team_b_win": 0.20,
            "team_a_advance": 0.68,
            "team_b_advance": 0.32,
            "expected_goals": {"team_a": 1.72, "team_b": 0.88},
            "most_likely_scores": [
                {"score": "2-1", "probability": 0.12},
                {"score": "1-0", "probability": 0.10},
                {"score": "1-1", "probability": 0.09},
            ],
            "explanation_factors": [
                "France has a stronger Elo rating.",
                "France has better current tournament xG.",
            ],
        },
        notes=["Limited sample data for Paraguay; neutral defaults used."],
    )

    assert "France vs Paraguay" in output
    assert "90-minute prediction:" in output
    assert "France win: 55.0%" in output
    assert "Advance prediction:" in output
    assert "Expected goals:" in output
    assert "1. France 2-1 Paraguay" in output
    assert "Why:" in output
    assert "- France has a stronger Elo rating." in output
    assert "Data notes:" in output


def test_predict_match_cli_outputs_prediction_for_unknown_team():
    _ensure_prediction_artifacts()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prediction.predict_match",
            "--team-a",
            "France",
            "--team-b",
            "Paraguay",
            "--stage",
            "R16",
            "--knockout",
            "true",
            "--date",
            "2026-07-04",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "France vs Paraguay" in result.stdout
    assert "90-minute prediction:" in result.stdout
    assert "Advance prediction:" in result.stdout
    assert "Most likely scores:" in result.stdout
    assert "Why:" in result.stdout
    assert "Limited sample data for Paraguay" in result.stdout
    assert "RuntimeWarning" not in result.stderr


def test_predict_match_cli_missing_model_exits_with_clear_error(tmp_path):
    missing_model = tmp_path / "missing_result_model.pkl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prediction.predict_match",
            "--team-a",
            "France",
            "--team-b",
            "Paraguay",
            "--stage",
            "R16",
            "--knockout",
            "true",
            "--model-path",
            str(missing_model),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Run `python -m src.models.train_result_model` first." in result.stderr
    assert not missing_model.exists()
