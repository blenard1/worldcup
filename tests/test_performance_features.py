import logging

import pandas as pd
import pytest

from src.data import load_current_performance, load_matches
from src.features.performance import (
    DEFAULT_PERFORMANCE_VALUES,
    add_performance_features,
)


def test_add_performance_features_adds_team_values_from_sample_data():
    matches = load_matches().head(1)
    performance = load_current_performance()

    enriched = add_performance_features(matches, performance)
    row = enriched.iloc[0]

    assert row["team_a"] == "Argentina"
    assert row["team_b"] == "Saudi Arabia"
    assert row["team_a_wc_goals_scored"] == pytest.approx(9)
    assert row["team_b_wc_goals_scored"] == pytest.approx(3)
    assert row["team_a_xg_per_match"] == pytest.approx(1.9)
    assert row["team_b_xga_per_match"] == pytest.approx(1.7)
    assert row["team_a_shots_per_match"] == pytest.approx(14.2)
    assert row["team_b_possession_avg"] == pytest.approx(44.8)
    assert row["team_a_rest_days"] == pytest.approx(4)


def test_add_performance_features_calculates_difference_features():
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team_a": "Team A",
                "team_b": "Team B",
            }
        ]
    )
    performance = pd.DataFrame(
        [
            {
                "team": "Team A",
                "matches_played": 3,
                "goals_scored": 7,
                "goals_conceded": 2,
                "shots_per_match": 12,
                "shots_on_target_per_match": 5,
                "xg_per_match": 1.8,
                "xga_per_match": 0.7,
                "possession_avg": 58,
                "clean_sheets": 2,
                "cards": 3,
                "rest_days": 5,
            },
            {
                "team": "Team B",
                "matches_played": 3,
                "goals_scored": 4,
                "goals_conceded": 5,
                "shots_per_match": 9,
                "shots_on_target_per_match": 3,
                "xg_per_match": 1.1,
                "xga_per_match": 1.4,
                "possession_avg": 47,
                "clean_sheets": 1,
                "cards": 6,
                "rest_days": 3,
            },
        ]
    )

    enriched = add_performance_features(matches, performance)
    row = enriched.iloc[0]

    assert row["wc_goal_diff_delta"] == pytest.approx(6)
    assert row["xg_delta"] == pytest.approx(0.7)
    assert row["xga_delta"] == pytest.approx(-0.7)
    assert row["shots_delta"] == pytest.approx(3)
    assert row["possession_delta"] == pytest.approx(11)
    assert row["rest_days_delta"] == pytest.approx(2)


def test_missing_teams_use_defaults_and_emit_warning(caplog):
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "team_a": "Known Team",
                "team_b": "Missing Team",
            }
        ]
    )
    performance = pd.DataFrame(
        [
            {
                "team": "Known Team",
                "matches_played": 2,
                "goals_scored": 3,
                "goals_conceded": 1,
                "shots_per_match": 10,
                "shots_on_target_per_match": 4,
                "xg_per_match": 1.4,
                "xga_per_match": 0.6,
                "possession_avg": 55,
                "clean_sheets": 1,
                "cards": 2,
                "rest_days": 4,
            }
        ]
    )

    with caplog.at_level(logging.WARNING):
        enriched = add_performance_features(matches, performance)

    row = enriched.iloc[0]
    assert "Missing Team" in caplog.text
    assert row["team_b_wc_goals_scored"] == DEFAULT_PERFORMANCE_VALUES["goals_scored"]
    assert row["team_b_wc_goals_conceded"] == DEFAULT_PERFORMANCE_VALUES[
        "goals_conceded"
    ]
    assert row["team_b_xg_per_match"] == DEFAULT_PERFORMANCE_VALUES["xg_per_match"]
    assert row["team_b_possession_avg"] == DEFAULT_PERFORMANCE_VALUES["possession_avg"]
    assert row["team_b_rest_days"] == DEFAULT_PERFORMANCE_VALUES["rest_days"]


def test_missing_performance_columns_raise_clear_error():
    matches = pd.DataFrame([{"team_a": "Team A", "team_b": "Team B"}])
    performance = pd.DataFrame(
        [
            {
                "team": "Team A",
                "matches_played": 1,
                "goals_scored": 2,
            }
        ]
    )

    with pytest.raises(ValueError, match="goals_conceded"):
        add_performance_features(matches, performance)
