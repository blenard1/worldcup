import numpy as np
import pytest

from src.data.build_dataset import build_training_dataset
from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.models.poisson_model import estimate_team_attack_defense
from src.prediction.ensemble import (
    elo_probability,
    ensemble_predict,
    h2h_probability_adjustment,
    normalize_probs,
)


class DummyResultModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        return np.tile(np.array([[0.50, 0.20, 0.30]]), (len(X), 1))


def _sample_training_row_and_strength():
    matches = load_matches()
    dataset = build_training_dataset(
        matches_df=matches,
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )
    team_strength = estimate_team_attack_defense(matches)
    return dataset.iloc[8], team_strength


def test_normalize_probs_handles_invalid_totals():
    probabilities = normalize_probs(
        {"team_a_win": 0, "draw": -1, "team_b_win": 0}
    )

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities == {
        "team_a_win": pytest.approx(1 / 3),
        "draw": pytest.approx(1 / 3),
        "team_b_win": pytest.approx(1 / 3),
    }


def test_elo_probability_returns_valid_probabilities():
    probabilities = elo_probability(100)

    assert set(probabilities) == {"team_a_win", "draw", "team_b_win"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["team_a_win"] > probabilities["team_b_win"]
    assert 0.15 <= probabilities["draw"] <= 0.30


def test_h2h_probability_adjustment_is_small():
    base_probs = {"team_a_win": 0.40, "draw": 0.25, "team_b_win": 0.35}
    h2h_features = {
        "h2h_weighted_score": 1.0,
        "h2h_goal_diff": 8,
        "h2h_team_a_wins": 6,
        "h2h_team_b_wins": 0,
    }

    adjusted = h2h_probability_adjustment(base_probs, h2h_features)

    assert sum(adjusted.values()) == pytest.approx(1.0)
    assert adjusted["team_a_win"] > base_probs["team_a_win"]
    assert adjusted["team_a_win"] - base_probs["team_a_win"] <= 0.08


def test_ensemble_predict_returns_complete_prediction_object():
    row, team_strength = _sample_training_row_and_strength()

    prediction = ensemble_predict(
        row,
        result_model=DummyResultModel(),
        team_strength_df=team_strength,
    )

    assert set(prediction) == {
        "team_a_win",
        "draw",
        "team_b_win",
        "team_a_advance",
        "team_b_advance",
        "expected_goals",
        "most_likely_scores",
        "confidence",
        "explanation_factors",
    }
    assert (
        prediction["team_a_win"]
        + prediction["draw"]
        + prediction["team_b_win"]
    ) == pytest.approx(1.0)
    assert prediction["team_a_advance"] + prediction["team_b_advance"] == pytest.approx(
        1.0
    )
    assert prediction["expected_goals"]["team_a"] > 0
    assert prediction["expected_goals"]["team_b"] > 0
    assert len(prediction["most_likely_scores"]) == 5
    assert {"score", "probability"}.issubset(prediction["most_likely_scores"][0])
    assert prediction["confidence"] in {"low", "medium", "high"}
    assert prediction["explanation_factors"]
    assert "90-minute" in prediction["explanation_factors"][0]


def test_ensemble_predict_works_without_result_model():
    row, team_strength = _sample_training_row_and_strength()

    prediction = ensemble_predict(row, team_strength_df=team_strength)

    assert (
        prediction["team_a_win"]
        + prediction["draw"]
        + prediction["team_b_win"]
    ) == pytest.approx(1.0)
    assert all(
        0 <= prediction[key] <= 1
        for key in (
            "team_a_win",
            "draw",
            "team_b_win",
            "team_a_advance",
            "team_b_advance",
        )
    )
