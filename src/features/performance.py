"""Current tournament performance feature engineering."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd


logger = logging.getLogger(__name__)

MATCH_TEAM_COLUMNS = ["team_a", "team_b"]
PERFORMANCE_REQUIRED_COLUMNS = [
    "team",
    "matches_played",
    "goals_scored",
    "goals_conceded",
    "shots_per_match",
    "shots_on_target_per_match",
    "xg_per_match",
    "xga_per_match",
    "possession_avg",
    "clean_sheets",
    "cards",
    "rest_days",
]

PERFORMANCE_FEATURE_MAP = {
    "goals_scored": "wc_goals_scored",
    "goals_conceded": "wc_goals_conceded",
    "xg_per_match": "xg_per_match",
    "xga_per_match": "xga_per_match",
    "shots_per_match": "shots_per_match",
    "possession_avg": "possession_avg",
    "rest_days": "rest_days",
}

DEFAULT_PERFORMANCE_VALUES = {
    "matches_played": 0.0,
    "goals_scored": 0.0,
    "goals_conceded": 0.0,
    "shots_per_match": 0.0,
    "shots_on_target_per_match": 0.0,
    "xg_per_match": 0.0,
    "xga_per_match": 0.0,
    "possession_avg": 50.0,
    "clean_sheets": 0.0,
    "cards": 0.0,
    "rest_days": 0.0,
}


def add_performance_features(
    matches_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add current tournament performance features for both teams.

    In production usage, ``performance_df`` must only contain statistics from
    matches played before the fixture being predicted.
    """

    _validate_columns(matches_df, MATCH_TEAM_COLUMNS, "matches")
    _validate_columns(
        performance_df,
        PERFORMANCE_REQUIRED_COLUMNS,
        "current tournament performance",
    )

    enriched = matches_df.copy()
    performance = _prepare_performance(performance_df)
    _log_missing_teams(enriched, performance)

    for side in ("team_a", "team_b"):
        side_features = _features_for_side(enriched[side], side, performance)
        enriched = pd.concat([enriched, side_features], axis=1)

    enriched["wc_goal_diff_delta"] = (
        enriched["team_a_wc_goals_scored"]
        - enriched["team_a_wc_goals_conceded"]
        - enriched["team_b_wc_goals_scored"]
        + enriched["team_b_wc_goals_conceded"]
    )
    enriched["xg_delta"] = (
        enriched["team_a_xg_per_match"] - enriched["team_b_xg_per_match"]
    )
    enriched["xga_delta"] = (
        enriched["team_a_xga_per_match"] - enriched["team_b_xga_per_match"]
    )
    enriched["shots_delta"] = (
        enriched["team_a_shots_per_match"] - enriched["team_b_shots_per_match"]
    )
    enriched["possession_delta"] = (
        enriched["team_a_possession_avg"] - enriched["team_b_possession_avg"]
    )
    enriched["rest_days_delta"] = (
        enriched["team_a_rest_days"] - enriched["team_b_rest_days"]
    )

    return enriched


def _prepare_performance(performance_df: pd.DataFrame) -> pd.DataFrame:
    performance = performance_df.copy()
    performance = performance.drop_duplicates(subset=["team"], keep="last")

    for column, default_value in DEFAULT_PERFORMANCE_VALUES.items():
        performance[column] = pd.to_numeric(performance[column], errors="coerce")
        performance[column] = performance[column].fillna(default_value)

    return performance


def _features_for_side(
    teams: pd.Series,
    side: str,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    lookup = performance_df[["team", *PERFORMANCE_FEATURE_MAP.keys()]]
    merged = pd.DataFrame({"team": teams}).merge(lookup, on="team", how="left")

    for column, default_value in DEFAULT_PERFORMANCE_VALUES.items():
        if column in merged.columns:
            merged[column] = merged[column].fillna(default_value)

    output = pd.DataFrame(index=teams.index)
    for source_column, feature_suffix in PERFORMANCE_FEATURE_MAP.items():
        output[f"{side}_{feature_suffix}"] = merged[source_column].to_numpy()

    return output


def _log_missing_teams(matches_df: pd.DataFrame, performance_df: pd.DataFrame) -> None:
    teams_in_matches = set(matches_df["team_a"]).union(set(matches_df["team_b"]))
    teams_with_performance = set(performance_df["team"])
    missing_teams = sorted(teams_in_matches - teams_with_performance)

    if missing_teams:
        logger.warning(
            "Missing current tournament performance data for teams: %s. "
            "Using default performance values.",
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
