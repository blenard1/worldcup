import pandas as pd
import pytest

from src.data import load_matches
from src.models.poisson_model import (
    TEAM_STRENGTH_PATH,
    estimate_team_attack_defense,
    expected_goals,
    most_likely_scores,
    result_probabilities_from_poisson,
    scoreline_probabilities,
)


def test_estimate_team_attack_defense_saves_team_strength_data():
    matches = load_matches()

    team_strength = estimate_team_attack_defense(matches)

    assert TEAM_STRENGTH_PATH.is_file()
    assert {
        "team",
        "avg_goals_scored",
        "avg_goals_conceded",
        "attack_strength",
        "defense_strength",
    }.issubset(team_strength.columns)
    assert "Argentina" in set(team_strength["team"])
    assert (team_strength["attack_strength"] >= 0).all()
    assert (team_strength["defense_strength"] >= 0).all()


def test_estimate_team_attack_defense_smooths_tiny_zero_scoring_samples():
    matches = pd.DataFrame(
        [
            {
                "team_a": "Low Sample",
                "team_b": "Opponent",
                "team_a_score": 0,
                "team_b_score": 2,
            }
        ]
    )

    team_strength = estimate_team_attack_defense(matches)
    low_sample = team_strength[team_strength["team"] == "Low Sample"].iloc[0]

    assert low_sample["avg_goals_scored"] > 0
    assert low_sample["attack_strength"] > 0


def test_scoreline_probabilities_are_valid():
    probabilities = scoreline_probabilities(1.4, 1.0, max_goals=6)

    assert len(probabilities) == 49
    assert {"team_a_goals", "team_b_goals", "probability"}.issubset(
        probabilities.columns
    )
    assert probabilities["probability"].between(0, 1).all()
    assert probabilities["probability"].sum() <= 1.0
    assert probabilities["probability"].sum() > 0.95


def test_most_likely_scores_return_requested_shape():
    top_scores = most_likely_scores(1.4, 1.0, top_n=5)

    assert len(top_scores) == 5
    assert list(top_scores.columns) == [
        "team_a_goals",
        "team_b_goals",
        "probability",
    ]
    assert top_scores["probability"].is_monotonic_decreasing


def test_result_probabilities_from_poisson_sum_to_one():
    probabilities = result_probabilities_from_poisson(1.5, 1.2)

    assert set(probabilities) == {"team_a_win", "draw", "team_b_win"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in probabilities.values())


def test_stronger_attack_increases_expected_goals():
    team_strength = pd.DataFrame(
        [
            {
                "team": "Strong Attack",
                "avg_goals_scored": 2.0,
                "avg_goals_conceded": 1.0,
                "attack_strength": 1.5,
                "defense_strength": 1.0,
            },
            {
                "team": "Weak Attack",
                "avg_goals_scored": 0.8,
                "avg_goals_conceded": 1.0,
                "attack_strength": 0.7,
                "defense_strength": 1.0,
            },
            {
                "team": "Opponent",
                "avg_goals_scored": 1.0,
                "avg_goals_conceded": 1.2,
                "attack_strength": 1.0,
                "defense_strength": 1.1,
            },
        ]
    )

    strong_lambda, _ = expected_goals("Strong Attack", "Opponent", team_strength)
    weak_lambda, _ = expected_goals("Weak Attack", "Opponent", team_strength)

    assert strong_lambda > weak_lambda


def test_expected_goals_adjust_for_elo_and_tactics():
    team_strength = estimate_team_attack_defense(load_matches())

    neutral_a, neutral_b = expected_goals(
        "Argentina",
        "Croatia",
        team_strength,
        elo_diff=0,
        tactical_edge=0,
    )
    adjusted_a, adjusted_b = expected_goals(
        "Argentina",
        "Croatia",
        team_strength,
        elo_diff=100,
        tactical_edge=2,
    )

    assert adjusted_a > neutral_a
    assert adjusted_b < neutral_b


def test_poisson_model_raises_for_missing_match_columns():
    matches = pd.DataFrame([{"team_a": "Team A", "team_a_score": 1}])

    with pytest.raises(ValueError, match="team_b"):
        estimate_team_attack_defense(matches)
