import pytest
from fastapi import HTTPException

import api.main as api_main
from api.main import PredictMatchRequest, app
from src.data.build_dataset import TRAINING_DATASET_PATH, build_training_dataset
from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.models.poisson_model import estimate_team_attack_defense
from src.models.train_result_model import train_result_model


def _ensure_api_prediction_artifacts():
    matches = load_matches()
    build_training_dataset(
        matches_df=matches,
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )
    estimate_team_attack_defense(matches)
    train_result_model(data_path=TRAINING_DATASET_PATH)


def test_root_returns_project_status():
    body = api_main.root()

    assert body["project"] == "World Cup AI Predictor"
    assert body["status"] == "running"


def test_health_returns_ok():
    assert api_main.health() == {"status": "ok"}


def test_teams_returns_available_sample_teams():
    response = api_main.teams()

    teams = response.teams
    assert teams == [
        "Argentina",
        "Belgium",
        "Brazil",
        "Canada",
        "Colombia",
        "Egypt",
        "England",
        "France",
        "Mexico",
        "Morocco",
        "Norway",
        "Paraguay",
        "Portugal",
        "Spain",
        "Switzerland",
        "USA",
    ]
    assert teams == sorted(teams)


def test_predict_match_returns_prediction_payload():
    _ensure_api_prediction_artifacts()

    response = api_main.predict_match_endpoint(
        PredictMatchRequest(
            team_a="France",
            team_b="Paraguay",
            stage="R16",
            is_knockout=True,
        )
    )
    prediction = response.prediction

    assert response.match == "France vs Paraguay"
    assert (
        prediction.team_a_win + prediction.draw + prediction.team_b_win
    ) == pytest.approx(1.0)
    assert (
        prediction.team_a_advance + prediction.team_b_advance
    ) == pytest.approx(1.0)
    assert prediction.expected_goals.team_a > 0
    assert prediction.expected_goals.team_b > 0
    assert prediction.most_likely_scores
    assert prediction.confidence in {"low", "medium", "high"}
    assert prediction.explanation_factors
    assert any("Paraguay" in note for note in prediction.data_notes)


def test_predict_match_missing_model_returns_503(monkeypatch):
    def raise_missing_model():
        raise FileNotFoundError(
            "Trained result model not found. "
            "Run `python -m src.models.train_result_model` first."
        )

    monkeypatch.setattr(api_main, "load_result_model", raise_missing_model)

    with pytest.raises(HTTPException) as exc_info:
        api_main.predict_match_endpoint(
            PredictMatchRequest(
                team_a="France",
                team_b="Paraguay",
                stage="R16",
                is_knockout=True,
            )
        )

    assert exc_info.value.status_code == 503
    assert "Run `python -m src.models.train_result_model` first." in str(
        exc_info.value.detail
    )


def test_openapi_schema_is_available_for_swagger_docs():
    schema = app.openapi()

    assert "/predict-match" in schema["paths"]
    assert "/health" in schema["paths"]
