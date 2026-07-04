"""Elo rating utilities for historical team strength features."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


DEFAULT_ELO_RATING = 1500.0
ELO_REQUIRED_MATCH_COLUMNS = [
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
]
ELO_HISTORY_COLUMNS = ["date", "match_id", "team", "pre_elo", "post_elo"]


def get_initial_rating(team: str) -> float:
    """Return the starting Elo rating for a team."""

    return DEFAULT_ELO_RATING


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate Team A's expected result against Team B."""

    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: int | float,
    score_b: int | float,
    k: float = 30,
) -> tuple[float, float]:
    """Update two Elo ratings from a match scoreline."""

    if score_a > score_b:
        actual_a = 1.0
    elif score_a == score_b:
        actual_a = 0.5
    else:
        actual_a = 0.0

    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1 - expected_a
    actual_b = 1 - actual_a

    new_rating_a = rating_a + k * (actual_a - expected_a)
    new_rating_b = rating_b + k * (actual_b - expected_b)
    return float(new_rating_a), float(new_rating_b)


def build_elo_history(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Build chronological pre- and post-match Elo records for each team.

    The returned history has two rows per match, one for each team. Ratings are
    updated only after both pre-match ratings are recorded, which keeps feature
    generation leakage-safe.
    """

    _validate_columns(matches_df, ELO_REQUIRED_MATCH_COLUMNS, "matches")

    matches = _prepare_matches(matches_df)
    ratings: dict[str, float] = {}
    history_rows: list[dict[str, Any]] = []

    for match in matches.itertuples(index=False):
        team_a = str(match.team_a)
        team_b = str(match.team_b)
        pre_elo_a = ratings.get(team_a, get_initial_rating(team_a))
        pre_elo_b = ratings.get(team_b, get_initial_rating(team_b))
        post_elo_a, post_elo_b = update_elo(
            pre_elo_a,
            pre_elo_b,
            match.team_a_score,
            match.team_b_score,
        )

        history_rows.extend(
            [
                {
                    "date": match.date,
                    "match_id": match.match_id,
                    "team": team_a,
                    "pre_elo": pre_elo_a,
                    "post_elo": post_elo_a,
                },
                {
                    "date": match.date,
                    "match_id": match.match_id,
                    "team": team_b,
                    "pre_elo": pre_elo_b,
                    "post_elo": post_elo_b,
                },
            ]
        )

        ratings[team_a] = post_elo_a
        ratings[team_b] = post_elo_b

    return pd.DataFrame(history_rows, columns=ELO_HISTORY_COLUMNS)


def get_team_elo_before_date(
    elo_history_df: pd.DataFrame,
    team: str,
    date: str | pd.Timestamp,
) -> float:
    """Return a team's latest post-match Elo strictly before a date."""

    if elo_history_df.empty:
        return DEFAULT_ELO_RATING

    _validate_columns(elo_history_df, ELO_HISTORY_COLUMNS, "Elo history")

    target_date = pd.to_datetime(date)
    history = elo_history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    team_history = history[
        (history["team"] == team) & (history["date"] < target_date)
    ].sort_values(["date", "match_id"])

    if team_history.empty:
        return DEFAULT_ELO_RATING

    return float(team_history.iloc[-1]["post_elo"])


def add_elo_features(
    matches_df: pd.DataFrame,
    elo_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add pre-match Elo features to a match dataframe."""

    _validate_columns(matches_df, ELO_REQUIRED_MATCH_COLUMNS, "matches")
    _validate_columns(elo_history_df, ELO_HISTORY_COLUMNS, "Elo history")

    has_input_match_id = "match_id" in matches_df.columns
    matches = _prepare_matches(matches_df)
    feature_rows = []

    for match in matches.itertuples(index=False):
        team_a_elo = _get_pre_match_elo(
            elo_history_df,
            match.match_id,
            str(match.team_a),
            match.date,
        )
        team_b_elo = _get_pre_match_elo(
            elo_history_df,
            match.match_id,
            str(match.team_b),
            match.date,
        )
        feature_rows.append(
            {
                "match_id": match.match_id,
                "team_a_elo": team_a_elo,
                "team_b_elo": team_b_elo,
                "elo_diff": team_a_elo - team_b_elo,
            }
        )

    feature_frame = pd.DataFrame(feature_rows)
    enriched = matches_df.copy()
    enriched["_elo_match_id"] = _match_ids_for(matches_df)
    enriched = enriched.merge(
        feature_frame,
        left_on="_elo_match_id",
        right_on="match_id",
        how="left",
        suffixes=("", "_elo_history"),
    )
    enriched = enriched.drop(columns=["_elo_match_id"])

    if "match_id_elo_history" in enriched.columns:
        enriched = enriched.drop(columns=["match_id_elo_history"])
    elif not has_input_match_id and "match_id" in enriched.columns:
        enriched = enriched.drop(columns=["match_id"])

    return enriched


def _prepare_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["match_id"] = _match_ids_for(matches)
    matches["_original_order"] = range(len(matches))
    return matches.sort_values(["date", "_original_order"]).drop(
        columns=["_original_order"]
    )


def _match_ids_for(matches_df: pd.DataFrame) -> pd.Series:
    if "match_id" in matches_df.columns:
        return matches_df["match_id"]

    return pd.Series(
        [f"row_{index}" for index in matches_df.index],
        index=matches_df.index,
        dtype="object",
    )


def _get_pre_match_elo(
    elo_history_df: pd.DataFrame,
    match_id: str,
    team: str,
    date: pd.Timestamp,
) -> float:
    rows = elo_history_df[
        (elo_history_df["match_id"] == match_id) & (elo_history_df["team"] == team)
    ]
    if not rows.empty:
        return float(rows.iloc[0]["pre_elo"])

    return get_team_elo_before_date(elo_history_df, team, date)


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
