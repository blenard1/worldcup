import pandas as pd
import pytest

from src.features.elo import (
    DEFAULT_ELO_RATING,
    add_elo_features,
    build_elo_history,
    expected_score,
    get_team_elo_before_date,
    update_elo,
)


def test_expected_score_for_equal_ratings_is_even():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_favors_higher_rated_team():
    assert expected_score(1600, 1500) > 0.5
    assert expected_score(1500, 1600) < 0.5


def test_elo_increases_after_win():
    new_a, new_b = update_elo(1500, 1500, 2, 0)

    assert new_a > 1500
    assert new_b < 1500
    assert new_a == pytest.approx(1515)
    assert new_b == pytest.approx(1485)


def test_elo_remains_close_after_draw_between_equal_teams():
    new_a, new_b = update_elo(1500, 1500, 1, 1)

    assert new_a == pytest.approx(1500)
    assert new_b == pytest.approx(1500)


def test_build_elo_history_sorts_matches_by_date():
    matches = pd.DataFrame(
        [
            {
                "match_id": "future",
                "date": "2024-01-10",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 2,
            },
            {
                "match_id": "first",
                "date": "2024-01-01",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 2,
                "team_b_score": 0,
            },
        ]
    )

    history = build_elo_history(matches)
    first_rows = history[history["match_id"] == "first"]

    assert first_rows["pre_elo"].tolist() == [DEFAULT_ELO_RATING, DEFAULT_ELO_RATING]


def test_get_team_elo_before_date_returns_latest_prior_rating():
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 2,
                "team_b_score": 0,
            },
            {
                "match_id": "m2",
                "date": "2024-01-10",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 2,
            },
        ]
    )
    history = build_elo_history(matches)

    assert get_team_elo_before_date(history, "Argentina", "2024-01-01") == 1500
    assert get_team_elo_before_date(history, "Argentina", "2024-01-05") == pytest.approx(
        1515
    )


def test_add_elo_features_uses_pre_match_not_future_or_post_match_data():
    matches = pd.DataFrame(
        [
            {
                "match_id": "m3",
                "date": "2024-03-01",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 0,
                "team_b_score": 3,
            },
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 2,
                "team_b_score": 0,
            },
            {
                "match_id": "m2",
                "date": "2024-02-01",
                "team_a": "Argentina",
                "team_b": "Croatia",
                "team_a_score": 1,
                "team_b_score": 1,
            },
        ]
    )

    history = build_elo_history(matches)
    enriched = add_elo_features(matches, history).set_index("match_id")

    assert enriched.loc["m1", "team_a_elo"] == pytest.approx(1500)
    assert enriched.loc["m1", "team_b_elo"] == pytest.approx(1500)
    assert enriched.loc["m1", "elo_diff"] == pytest.approx(0)

    assert enriched.loc["m2", "team_a_elo"] == pytest.approx(1515)
    assert enriched.loc["m2", "team_b_elo"] == pytest.approx(1500)


def test_build_elo_history_raises_for_missing_columns():
    matches = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "team_a": "Argentina",
                "team_b": "Brazil",
                "team_a_score": 2,
            }
        ]
    )

    with pytest.raises(ValueError, match="team_b_score"):
        build_elo_history(matches)
