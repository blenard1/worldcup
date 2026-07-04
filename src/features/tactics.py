"""Tactical matchup feature engineering."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd


logger = logging.getLogger(__name__)

MATCH_TEAM_COLUMNS = ["team_a", "team_b"]
TACTICS_REQUIRED_COLUMNS = [
    "team",
    "formation",
    "pressing_level",
    "defensive_block",
    "build_up_quality",
    "counterattack_speed",
    "set_piece_strength",
    "set_piece_defense",
    "wide_attack",
    "central_attack",
    "transition_defense",
    "aerial_strength",
]

TACTICAL_NUMERIC_COLUMNS = [
    "pressing_level",
    "build_up_quality",
    "counterattack_speed",
    "set_piece_strength",
    "set_piece_defense",
    "wide_attack",
    "transition_defense",
    "aerial_strength",
]

NEUTRAL_TACTICAL_VALUE = 5.0


def add_tactical_features(
    matches_df: pd.DataFrame,
    tactics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add team tactical profile and matchup edge features."""

    _validate_columns(matches_df, MATCH_TEAM_COLUMNS, "matches")
    _validate_columns(tactics_df, TACTICS_REQUIRED_COLUMNS, "tactics")

    enriched = matches_df.copy()
    tactics = _prepare_tactics(tactics_df)
    _log_missing_teams(enriched, tactics)

    for side in ("team_a", "team_b"):
        side_features = _features_for_side(enriched[side], side, tactics)
        enriched = pd.concat([enriched, side_features], axis=1)

    enriched["pressing_vs_buildup_a"] = (
        enriched["team_a_pressing_level"] - enriched["team_b_build_up_quality"]
    )
    enriched["pressing_vs_buildup_b"] = (
        enriched["team_b_pressing_level"] - enriched["team_a_build_up_quality"]
    )
    enriched["counter_vs_transition_a"] = (
        enriched["team_a_counterattack_speed"]
        - enriched["team_b_transition_defense"]
    )
    enriched["counter_vs_transition_b"] = (
        enriched["team_b_counterattack_speed"]
        - enriched["team_a_transition_defense"]
    )
    enriched["set_piece_edge_a"] = (
        enriched["team_a_set_piece_strength"] - enriched["team_b_set_piece_defense"]
    )
    enriched["set_piece_edge_b"] = (
        enriched["team_b_set_piece_strength"] - enriched["team_a_set_piece_defense"]
    )
    enriched["wide_attack_edge_a"] = (
        enriched["team_a_wide_attack"] - enriched["team_b_transition_defense"]
    )
    enriched["wide_attack_edge_b"] = (
        enriched["team_b_wide_attack"] - enriched["team_a_transition_defense"]
    )
    enriched["aerial_edge_a"] = (
        enriched["team_a_aerial_strength"] - enriched["team_b_aerial_strength"]
    )

    enriched["tactical_edge"] = (
        (
            enriched["pressing_vs_buildup_a"]
            - enriched["pressing_vs_buildup_b"]
        )
        + (
            enriched["counter_vs_transition_a"]
            - enriched["counter_vs_transition_b"]
        )
        + (enriched["set_piece_edge_a"] - enriched["set_piece_edge_b"])
        + (enriched["wide_attack_edge_a"] - enriched["wide_attack_edge_b"])
        + enriched["aerial_edge_a"]
    ) / 5

    return enriched


def _prepare_tactics(tactics_df: pd.DataFrame) -> pd.DataFrame:
    tactics = tactics_df.copy()
    tactics = tactics.drop_duplicates(subset=["team"], keep="last")

    for column in TACTICAL_NUMERIC_COLUMNS:
        tactics[column] = pd.to_numeric(tactics[column], errors="coerce")
        tactics[column] = tactics[column].fillna(NEUTRAL_TACTICAL_VALUE)

    return tactics


def _features_for_side(
    teams: pd.Series,
    side: str,
    tactics_df: pd.DataFrame,
) -> pd.DataFrame:
    lookup = tactics_df[["team", *TACTICAL_NUMERIC_COLUMNS]]
    merged = pd.DataFrame({"team": teams}).merge(lookup, on="team", how="left")

    for column in TACTICAL_NUMERIC_COLUMNS:
        merged[column] = merged[column].fillna(NEUTRAL_TACTICAL_VALUE)

    output = pd.DataFrame(index=teams.index)
    for column in TACTICAL_NUMERIC_COLUMNS:
        output[f"{side}_{column}"] = merged[column].to_numpy()

    return output


def _log_missing_teams(matches_df: pd.DataFrame, tactics_df: pd.DataFrame) -> None:
    teams_in_matches = set(matches_df["team_a"]).union(set(matches_df["team_b"]))
    teams_with_tactics = set(tactics_df["team"])
    missing_teams = sorted(teams_in_matches - teams_with_tactics)

    if missing_teams:
        logger.warning(
            "Missing tactical profiles for teams: %s. Using neutral tactical values.",
            ", ".join(missing_teams),
        )


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{dataset_name} data is missing required columns: {missing}")
