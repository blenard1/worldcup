import logging

import pandas as pd
import pytest

from src.data import load_matches, load_tactics
from src.features.tactics import (
    NEUTRAL_TACTICAL_VALUE,
    add_tactical_features,
)


def _tactics_row(
    team,
    pressing,
    buildup,
    counter,
    set_piece_strength,
    set_piece_defense,
    wide_attack,
    transition_defense,
    aerial,
):
    return {
        "team": team,
        "formation": "4-3-3",
        "pressing_level": pressing,
        "defensive_block": "medium",
        "build_up_quality": buildup,
        "counterattack_speed": counter,
        "set_piece_strength": set_piece_strength,
        "set_piece_defense": set_piece_defense,
        "wide_attack": wide_attack,
        "central_attack": 5,
        "transition_defense": transition_defense,
        "aerial_strength": aerial,
    }


def test_add_tactical_features_adds_team_stats_from_sample_data():
    matches = load_matches().head(1)
    tactics = load_tactics()

    enriched = add_tactical_features(matches, tactics)
    row = enriched.iloc[0]

    assert row["team_a"] == "Argentina"
    assert row["team_b"] == "Saudi Arabia"
    assert row["team_a_pressing_level"] == pytest.approx(7)
    assert row["team_b_build_up_quality"] == pytest.approx(6)
    assert row["team_a_counterattack_speed"] == pytest.approx(7)
    assert row["team_b_transition_defense"] == pytest.approx(6)
    assert row["team_a_set_piece_defense"] == pytest.approx(8)
    assert row["team_b_aerial_strength"] == pytest.approx(6)


def test_tactical_matchup_edges_are_calculated():
    matches = pd.DataFrame([{"team_a": "Team A", "team_b": "Team B"}])
    tactics = pd.DataFrame(
        [
            _tactics_row("Team A", 8, 7, 9, 6, 5, 8, 6, 7),
            _tactics_row("Team B", 5, 6, 4, 7, 8, 5, 7, 6),
        ]
    )

    enriched = add_tactical_features(matches, tactics)
    row = enriched.iloc[0]

    assert row["pressing_vs_buildup_a"] == pytest.approx(2)
    assert row["pressing_vs_buildup_b"] == pytest.approx(-2)
    assert row["counter_vs_transition_a"] == pytest.approx(2)
    assert row["counter_vs_transition_b"] == pytest.approx(-2)
    assert row["set_piece_edge_a"] == pytest.approx(-2)
    assert row["set_piece_edge_b"] == pytest.approx(2)
    assert row["wide_attack_edge_a"] == pytest.approx(1)
    assert row["wide_attack_edge_b"] == pytest.approx(-1)
    assert row["aerial_edge_a"] == pytest.approx(1)
    assert row["tactical_edge"] == pytest.approx(1.4)


def test_missing_tactical_profile_uses_neutral_values_and_logs_warning(caplog):
    matches = pd.DataFrame([{"team_a": "Known Team", "team_b": "Missing Team"}])
    tactics = pd.DataFrame(
        [_tactics_row("Known Team", 8, 7, 9, 6, 5, 8, 6, 7)]
    )

    with caplog.at_level(logging.WARNING):
        enriched = add_tactical_features(matches, tactics)

    row = enriched.iloc[0]
    assert "Missing Team" in caplog.text
    assert row["team_b_pressing_level"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_build_up_quality"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_counterattack_speed"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_set_piece_strength"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_set_piece_defense"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_wide_attack"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_transition_defense"] == NEUTRAL_TACTICAL_VALUE
    assert row["team_b_aerial_strength"] == NEUTRAL_TACTICAL_VALUE


def test_missing_tactics_columns_raise_clear_error():
    matches = pd.DataFrame([{"team_a": "Team A", "team_b": "Team B"}])
    tactics = pd.DataFrame(
        [
            {
                "team": "Team A",
                "formation": "4-3-3",
                "pressing_level": 7,
            }
        ]
    )

    with pytest.raises(ValueError, match="defensive_block"):
        add_tactical_features(matches, tactics)
