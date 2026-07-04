"""Head-to-head feature engineering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


H2H_REQUIRED_COLUMNS = [
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "tournament",
    "neutral",
]
MATCH_REQUIRED_COLUMNS = ["date", "team_a", "team_b"]

NEUTRAL_H2H_FEATURES = {
    "h2h_matches_count": 0,
    "h2h_team_a_wins": 0,
    "h2h_team_b_wins": 0,
    "h2h_draws": 0,
    "h2h_team_a_goals": 0,
    "h2h_team_b_goals": 0,
    "h2h_goal_diff": 0,
    "h2h_weighted_score": 0.0,
}


def normalize_pair(team_a: str, team_b: str) -> tuple[str, str]:
    """Return a stable team-pair key independent of home/away ordering."""

    return tuple(sorted((team_a, team_b)))


def get_h2h_matches(
    h2h_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return historical matches between two teams in either direction."""

    _validate_columns(h2h_df, H2H_REQUIRED_COLUMNS, "head-to-head")

    history = h2h_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    pair = normalize_pair(team_a, team_b)
    pair_mask = history.apply(
        lambda row: normalize_pair(row["team_a"], row["team_b"]) == pair,
        axis=1,
    )
    filtered = history[pair_mask]

    if before_date is not None:
        filtered = filtered[filtered["date"] < pd.to_datetime(before_date)]

    return filtered.sort_values("date").reset_index(drop=True)


def calculate_h2h_features(
    h2h_df: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date: str | pd.Timestamp | None = None,
    max_matches: int = 10,
) -> dict[str, int | float]:
    """Calculate recent, weighted H2H features oriented to the input teams."""

    matches = get_h2h_matches(h2h_df, team_a, team_b, before_date)
    if matches.empty or max_matches <= 0:
        return dict(NEUTRAL_H2H_FEATURES)

    recent_matches = matches.sort_values("date").tail(max_matches).reset_index(drop=True)
    features = dict(NEUTRAL_H2H_FEATURES)
    features["h2h_matches_count"] = len(recent_matches)

    weighted_score_total = 0.0
    total_weight = 0.0

    for recency_index, match in enumerate(recent_matches.itertuples(index=False), start=1):
        oriented = _orient_match(match, team_a, team_b)
        team_a_goals = oriented["team_a_goals"]
        team_b_goals = oriented["team_b_goals"]

        features["h2h_team_a_goals"] += team_a_goals
        features["h2h_team_b_goals"] += team_b_goals

        if team_a_goals > team_b_goals:
            features["h2h_team_a_wins"] += 1
            result_score = 1.0
        elif team_a_goals < team_b_goals:
            features["h2h_team_b_wins"] += 1
            result_score = -1.0
        else:
            features["h2h_draws"] += 1
            result_score = 0.0

        recency_weight = recency_index / len(recent_matches)
        tournament_weight = _tournament_weight(str(match.tournament))
        weight = recency_weight * tournament_weight
        weighted_score_total += result_score * weight
        total_weight += weight

    features["h2h_goal_diff"] = (
        features["h2h_team_a_goals"] - features["h2h_team_b_goals"]
    )
    features["h2h_weighted_score"] = (
        weighted_score_total / total_weight if total_weight else 0.0
    )

    return features


def add_h2h_features(matches_df: pd.DataFrame, h2h_df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe H2H features for every match row."""

    _validate_columns(matches_df, MATCH_REQUIRED_COLUMNS, "matches")
    _validate_columns(h2h_df, H2H_REQUIRED_COLUMNS, "head-to-head")

    enriched = matches_df.copy()
    feature_rows = []

    for match in enriched.itertuples(index=False):
        feature_rows.append(
            calculate_h2h_features(
                h2h_df,
                str(match.team_a),
                str(match.team_b),
                before_date=match.date,
            )
        )

    features = pd.DataFrame(feature_rows, index=enriched.index)
    return pd.concat([enriched, features], axis=1)


def _orient_match(match: Any, team_a: str, team_b: str) -> dict[str, int]:
    if match.team_a == team_a and match.team_b == team_b:
        return {
            "team_a_goals": int(match.team_a_score),
            "team_b_goals": int(match.team_b_score),
        }

    if match.team_a == team_b and match.team_b == team_a:
        return {
            "team_a_goals": int(match.team_b_score),
            "team_b_goals": int(match.team_a_score),
        }

    raise ValueError("Cannot orient a match that does not contain the requested teams.")


def _tournament_weight(tournament: str) -> float:
    tournament_lower = tournament.lower()
    if "world cup" in tournament_lower:
        return 1.25
    if "continental" in tournament_lower or "nations league" in tournament_lower:
        return 1.10
    if "friendly" in tournament_lower:
        return 0.75
    return 1.0


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
