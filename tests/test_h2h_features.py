import pandas as pd
import pytest

from src.data import load_h2h, load_matches
from src.features.h2h import (
    add_h2h_features,
    calculate_h2h_features,
    get_h2h_matches,
    normalize_pair,
)


def _sample_h2h():
    return pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "team_a": "Team A",
                "team_b": "Team B",
                "team_a_score": 2,
                "team_b_score": 1,
                "tournament": "Sample World Cup",
                "neutral": True,
            },
            {
                "date": "2021-01-01",
                "team_a": "Team B",
                "team_b": "Team A",
                "team_a_score": 0,
                "team_b_score": 0,
                "tournament": "Sample Friendly",
                "neutral": True,
            },
            {
                "date": "2023-01-01",
                "team_a": "Team B",
                "team_b": "Team A",
                "team_a_score": 3,
                "team_b_score": 1,
                "tournament": "Sample World Cup",
                "neutral": True,
            },
        ]
    )


def test_normalize_pair_is_order_independent():
    assert normalize_pair("Brazil", "Argentina") == normalize_pair(
        "Argentina",
        "Brazil",
    )


def test_get_h2h_matches_finds_matches_in_both_directions():
    h2h = _sample_h2h()

    forward = get_h2h_matches(h2h, "Team A", "Team B")
    reverse = get_h2h_matches(h2h, "Team B", "Team A")

    assert len(forward) == 3
    assert len(reverse) == 3
    assert forward["date"].tolist() == reverse["date"].tolist()


def test_get_h2h_matches_does_not_include_future_matches():
    h2h = _sample_h2h()

    filtered = get_h2h_matches(h2h, "Team A", "Team B", before_date="2022-01-01")

    assert len(filtered) == 2
    assert filtered["date"].max() < pd.Timestamp("2022-01-01")


def test_calculate_h2h_features_are_oriented_to_requested_teams():
    h2h = _sample_h2h()

    features = calculate_h2h_features(h2h, "Team A", "Team B")

    assert features["h2h_matches_count"] == 3
    assert features["h2h_team_a_wins"] == 1
    assert features["h2h_team_b_wins"] == 1
    assert features["h2h_draws"] == 1
    assert features["h2h_team_a_goals"] == 3
    assert features["h2h_team_b_goals"] == 4
    assert features["h2h_goal_diff"] == -1
    assert -1 <= features["h2h_weighted_score"] <= 1
    assert features["h2h_weighted_score"] == pytest.approx(-0.384615, abs=1e-6)


def test_calculate_h2h_features_returns_neutral_values_when_no_h2h_exists():
    features = calculate_h2h_features(
        _sample_h2h(),
        "Team A",
        "Unseen Team",
        before_date="2024-01-01",
    )

    assert features == {
        "h2h_matches_count": 0,
        "h2h_team_a_wins": 0,
        "h2h_team_b_wins": 0,
        "h2h_draws": 0,
        "h2h_team_a_goals": 0,
        "h2h_team_b_goals": 0,
        "h2h_goal_diff": 0,
        "h2h_weighted_score": 0.0,
    }


def test_add_h2h_features_uses_only_h2h_before_match_date():
    matches = pd.DataFrame(
        [
            {
                "match_id": "target",
                "date": "2022-01-01",
                "team_a": "Team A",
                "team_b": "Team B",
            }
        ]
    )

    enriched = add_h2h_features(matches, _sample_h2h())
    row = enriched.iloc[0]

    assert row["h2h_matches_count"] == 2
    assert row["h2h_team_a_wins"] == 1
    assert row["h2h_team_b_wins"] == 0
    assert row["h2h_draws"] == 1
    assert row["h2h_team_a_goals"] == 2
    assert row["h2h_team_b_goals"] == 1


def test_add_h2h_features_works_with_sample_data():
    matches = load_matches()
    h2h = load_h2h()

    enriched = add_h2h_features(matches, h2h)

    argentina_croatia = enriched[enriched["match_id"] == "M009"].iloc[0]
    assert argentina_croatia["h2h_matches_count"] == 1
    assert argentina_croatia["h2h_team_b_wins"] == 1
    assert argentina_croatia["h2h_goal_diff"] == -3


def test_missing_h2h_columns_raise_clear_error():
    h2h = _sample_h2h().drop(columns=["team_b_score"])

    with pytest.raises(ValueError, match="team_b_score"):
        get_h2h_matches(h2h, "Team A", "Team B")
