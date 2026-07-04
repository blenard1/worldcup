"""Poisson-style score prediction model."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from scipy.stats import poisson

from src.data.load_data import load_matches
from src.utils.config import MODELS_DIR


TEAM_STRENGTH_PATH = MODELS_DIR / "team_strength.csv"
MATCH_REQUIRED_COLUMNS = [
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
]
TEAM_STRENGTH_COLUMNS = [
    "team",
    "avg_goals_scored",
    "avg_goals_conceded",
    "attack_strength",
    "defense_strength",
]
PRIOR_MATCHES = 4


def estimate_team_attack_defense(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate team attack and defense strength from historical scorelines.

    ``attack_strength`` is a team's goals scored per match relative to the
    overall average goals scored. ``defense_strength`` is goals conceded per
    match relative to the overall average goals conceded. Values above 1 mean
    stronger attack or weaker defense, respectively.

    The sample dataset is intentionally small, so raw one-match averages can be
    misleading. A simple prior shrinks every team toward the tournament-wide
    average before strengths are calculated.
    """

    _validate_columns(matches_df, MATCH_REQUIRED_COLUMNS, "matches")
    team_rows = []

    for match in matches_df.itertuples(index=False):
        team_rows.extend(
            [
                {
                    "team": match.team_a,
                    "goals_scored": float(match.team_a_score),
                    "goals_conceded": float(match.team_b_score),
                },
                {
                    "team": match.team_b,
                    "goals_scored": float(match.team_b_score),
                    "goals_conceded": float(match.team_a_score),
                },
            ]
        )

    long_matches = pd.DataFrame(team_rows)
    team_totals = (
        long_matches.groupby("team", as_index=False)
        .agg(
            goals_scored=("goals_scored", "sum"),
            goals_conceded=("goals_conceded", "sum"),
            matches_played=("goals_scored", "size"),
        )
        .sort_values("team")
        .reset_index(drop=True)
    )

    overall_goals_scored = max(long_matches["goals_scored"].mean(), 0.1)
    overall_goals_conceded = max(long_matches["goals_conceded"].mean(), 0.1)
    team_strength = team_totals[["team"]].copy()
    team_strength["avg_goals_scored"] = (
        team_totals["goals_scored"] + PRIOR_MATCHES * overall_goals_scored
    ) / (team_totals["matches_played"] + PRIOR_MATCHES)
    team_strength["avg_goals_conceded"] = (
        team_totals["goals_conceded"] + PRIOR_MATCHES * overall_goals_conceded
    ) / (team_totals["matches_played"] + PRIOR_MATCHES)
    team_strength["attack_strength"] = (
        team_strength["avg_goals_scored"] / overall_goals_scored
    )
    team_strength["defense_strength"] = (
        team_strength["avg_goals_conceded"] / overall_goals_conceded
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    team_strength.to_csv(TEAM_STRENGTH_PATH, index=False)
    return team_strength[TEAM_STRENGTH_COLUMNS]


def expected_goals(
    team_a: str,
    team_b: str,
    team_strength_df: pd.DataFrame,
    elo_diff: float = 0,
    tactical_edge: float = 0,
) -> tuple[float, float]:
    """Estimate expected goals for both teams.

    Formula:
    - Start from the global average team goals in the strength table.
    - Multiply by attacking strength of the scoring team.
    - Multiply by defensive weakness of the opponent.
    - Apply small Elo and tactical adjustments so these features influence
      expected goals without overwhelming historical scoring strength.
    """

    _validate_columns(team_strength_df, TEAM_STRENGTH_COLUMNS, "team strength")

    baseline_goals = max(team_strength_df["avg_goals_scored"].mean(), 0.1)
    team_a_strength = _lookup_team_strength(team_strength_df, team_a)
    team_b_strength = _lookup_team_strength(team_strength_df, team_b)

    elo_adjustment = _bounded_multiplier(1 + (elo_diff / 400) * 0.08, 0.8, 1.2)
    tactical_adjustment = _bounded_multiplier(1 + tactical_edge * 0.03, 0.8, 1.2)
    team_a_adjustment = elo_adjustment * tactical_adjustment
    team_b_adjustment = (1 / elo_adjustment) * (1 / tactical_adjustment)

    lambda_a = (
        baseline_goals
        * team_a_strength["attack_strength"]
        * team_b_strength["defense_strength"]
        * team_a_adjustment
    )
    lambda_b = (
        baseline_goals
        * team_b_strength["attack_strength"]
        * team_a_strength["defense_strength"]
        * team_b_adjustment
    )

    return max(float(lambda_a), 0.05), max(float(lambda_b), 0.05)


def scoreline_probabilities(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 6,
) -> pd.DataFrame:
    """Return scoreline probabilities for 0..max_goals for both teams."""

    if max_goals < 0:
        raise ValueError("max_goals must be non-negative.")

    rows = []
    for team_a_goals in range(max_goals + 1):
        probability_a = poisson.pmf(team_a_goals, lambda_a)
        for team_b_goals in range(max_goals + 1):
            probability_b = poisson.pmf(team_b_goals, lambda_b)
            rows.append(
                {
                    "team_a_goals": team_a_goals,
                    "team_b_goals": team_b_goals,
                    "probability": float(probability_a * probability_b),
                }
            )

    return pd.DataFrame(rows)


def most_likely_scores(
    lambda_a: float,
    lambda_b: float,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the most likely scorelines sorted by probability."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    probabilities = scoreline_probabilities(lambda_a, lambda_b)
    return probabilities.sort_values("probability", ascending=False).head(top_n)


def result_probabilities_from_poisson(
    lambda_a: float,
    lambda_b: float,
) -> dict[str, float]:
    """Return W/D/L probabilities implied by the scoreline distribution."""

    probabilities = scoreline_probabilities(lambda_a, lambda_b)
    team_a_win = probabilities[
        probabilities["team_a_goals"] > probabilities["team_b_goals"]
    ]["probability"].sum()
    draw = probabilities[
        probabilities["team_a_goals"] == probabilities["team_b_goals"]
    ]["probability"].sum()
    team_b_win = probabilities[
        probabilities["team_a_goals"] < probabilities["team_b_goals"]
    ]["probability"].sum()

    total = team_a_win + draw + team_b_win
    if total <= 0:
        return {"team_a_win": 1 / 3, "draw": 1 / 3, "team_b_win": 1 / 3}

    return {
        "team_a_win": float(team_a_win / total),
        "draw": float(draw / total),
        "team_b_win": float(team_b_win / total),
    }


def main() -> None:
    """Estimate and save team strength data from sample matches."""

    team_strength = estimate_team_attack_defense(load_matches())
    print(f"Created {TEAM_STRENGTH_PATH} with {len(team_strength)} teams.")


def _lookup_team_strength(team_strength_df: pd.DataFrame, team: str) -> pd.Series:
    rows = team_strength_df[team_strength_df["team"] == team]
    if not rows.empty:
        return rows.iloc[0]

    return pd.Series(
        {
            "team": team,
            "avg_goals_scored": team_strength_df["avg_goals_scored"].mean(),
            "avg_goals_conceded": team_strength_df["avg_goals_conceded"].mean(),
            "attack_strength": 1.0,
            "defense_strength": 1.0,
        }
    )


def _bounded_multiplier(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


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


if __name__ == "__main__":
    main()
