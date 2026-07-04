"""Final ensemble prediction engine."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.models.poisson_model import (
    TEAM_STRENGTH_COLUMNS,
    TEAM_STRENGTH_PATH,
    expected_goals,
    most_likely_scores,
    result_probabilities_from_poisson,
)
from src.prediction.predict_result import predict_result_proba


PROBABILITY_KEYS = ["team_a_win", "draw", "team_b_win"]
RESULT_MODEL_WEIGHT = 0.45
POISSON_WEIGHT = 0.25
ELO_WEIGHT = 0.20
ADJUSTMENT_WEIGHT = 0.10


def normalize_probs(prob_dict: dict[str, float]) -> dict[str, float]:
    """Return probabilities normalized to sum to 1."""

    cleaned = {
        key: max(float(prob_dict.get(key, 0.0)), 0.0)
        for key in PROBABILITY_KEYS
    }
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 1 / len(PROBABILITY_KEYS) for key in PROBABILITY_KEYS}

    return {key: cleaned[key] / total for key in PROBABILITY_KEYS}


def elo_probability(elo_diff: float) -> dict[str, float]:
    """Convert Elo difference into W/D/L probabilities with a draw baseline."""

    elo_expected_a = 1 / (1 + 10 ** (-(float(elo_diff)) / 400))
    draw_probability = 0.26 - min(abs(float(elo_diff)), 300) / 300 * 0.08
    draw_probability = min(max(draw_probability, 0.16), 0.30)
    decisive_probability = 1 - draw_probability

    return normalize_probs(
        {
            "team_a_win": decisive_probability * elo_expected_a,
            "draw": draw_probability,
            "team_b_win": decisive_probability * (1 - elo_expected_a),
        }
    )


def h2h_probability_adjustment(
    base_probs: dict[str, float],
    h2h_features: dict[str, float],
) -> dict[str, float]:
    """Apply a small H2H nudge to existing probabilities."""

    adjusted = normalize_probs(base_probs)
    weighted_score = _get_float(h2h_features, "h2h_weighted_score")
    goal_diff = _get_float(h2h_features, "h2h_goal_diff")
    wins_delta = _get_float(h2h_features, "h2h_team_a_wins") - _get_float(
        h2h_features,
        "h2h_team_b_wins",
    )

    # Keep H2H intentionally modest: even a strong H2H signal can move the
    # win probability by only a few points before final normalization.
    adjustment = (
        weighted_score * 0.04
        + max(min(goal_diff, 5), -5) * 0.005
        + max(min(wins_delta, 3), -3) * 0.006
    )
    adjustment = max(min(adjustment, 0.06), -0.06)

    adjusted["team_a_win"] += adjustment
    adjusted["team_b_win"] -= adjustment
    adjusted["draw"] -= abs(adjustment) * 0.15
    return normalize_probs(adjusted)


def ensemble_predict(
    match_features: pd.Series | dict[str, Any],
    result_model: Any | None = None,
    team_strength_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the final ensemble prediction object for one match."""

    features = _as_dict(match_features)
    team_a = str(features.get("team_a", "Team A"))
    team_b = str(features.get("team_b", "Team B"))
    elo_diff = _get_float(features, "elo_diff")
    tactical_edge = _get_float(features, "tactical_edge")

    result_probs = (
        predict_result_proba(result_model, match_features)
        if result_model is not None
        else {key: 1 / 3 for key in PROBABILITY_KEYS}
    )

    team_strength = _load_or_create_team_strength(team_a, team_b, team_strength_df)
    lambda_a, lambda_b = expected_goals(
        team_a,
        team_b,
        team_strength,
        elo_diff=elo_diff,
        tactical_edge=tactical_edge,
    )
    elo_probs = elo_probability(elo_diff)

    final_lambda_a, final_lambda_b = _adjust_expected_goals_with_model_signals(
        lambda_a,
        lambda_b,
        features,
        result_probs,
        elo_probs,
    )
    final_probs = result_probabilities_from_poisson(final_lambda_a, final_lambda_b)

    strength_share_a = _strength_share_from_elo(elo_diff, tactical_edge)
    team_a_advance = final_probs["team_a_win"] + final_probs["draw"] * strength_share_a
    team_b_advance = final_probs["team_b_win"] + final_probs["draw"] * (
        1 - strength_share_a
    )
    advance_probs = normalize_probs(
        {
            "team_a_win": team_a_advance,
            "draw": 0,
            "team_b_win": team_b_advance,
        }
    )

    return {
        "team_a_win": final_probs["team_a_win"],
        "draw": final_probs["draw"],
        "team_b_win": final_probs["team_b_win"],
        "team_a_advance": advance_probs["team_a_win"],
        "team_b_advance": advance_probs["team_b_win"],
        "expected_goals": {
            "team_a": float(final_lambda_a),
            "team_b": float(final_lambda_b),
        },
        "most_likely_scores": _format_scorelines(final_lambda_a, final_lambda_b),
        "confidence": _confidence_level(final_probs, final_lambda_a, final_lambda_b),
        "explanation_factors": _explanation_factors(
            features,
            team_a,
            team_b,
            final_probs,
            final_lambda_a,
            final_lambda_b,
        ),
    }


def _tactical_probability_adjustment(
    base_probs: dict[str, float],
    tactical_edge: float,
) -> dict[str, float]:
    adjusted = normalize_probs(base_probs)
    adjustment = max(min(float(tactical_edge) * 0.015, 0.05), -0.05)
    adjusted["team_a_win"] += adjustment
    adjusted["team_b_win"] -= adjustment
    adjusted["draw"] -= abs(adjustment) * 0.10
    return normalize_probs(adjusted)


def _adjust_expected_goals_with_model_signals(
    lambda_a: float,
    lambda_b: float,
    features: dict[str, Any],
    result_probs: dict[str, float],
    elo_probs: dict[str, float],
) -> tuple[float, float]:
    """Apply non-score model signals before deriving final score probabilities.

    This keeps the public odds, expected goals, and exact scorelines coherent:
    the 90-minute probabilities are calculated from the same adjusted goal
    distribution that produces the displayed scorelines.
    """

    result_edge = result_probs["team_a_win"] - result_probs["team_b_win"]
    elo_edge = elo_probs["team_a_win"] - elo_probs["team_b_win"]
    xg_delta = _get_float(features, "xg_delta")
    shots_delta = _get_float(features, "shots_delta")
    set_piece_delta = _get_float(features, "set_piece_edge_a") - _get_float(
        features,
        "set_piece_edge_b",
    )
    tactical_edge = _get_float(features, "tactical_edge")
    h2h_score = _get_float(features, "h2h_weighted_score")

    signal_edge = (
        result_edge * 0.14
        + elo_edge * 0.08
        + _bounded(xg_delta, -1.5, 1.5) * 0.06
        + _bounded(shots_delta / 6, -1.0, 1.0) * 0.03
        + _bounded(set_piece_delta / 6, -1.0, 1.0) * 0.04
        + _bounded(tactical_edge / 6, -1.0, 1.0) * 0.05
        + _bounded(h2h_score, -1.0, 1.0) * 0.03
    )
    signal_edge = _bounded(signal_edge, -0.28, 0.28)

    team_a_multiplier = math.exp(signal_edge)
    team_b_multiplier = math.exp(-signal_edge)

    return (
        max(float(lambda_a) * team_a_multiplier, 0.05),
        max(float(lambda_b) * team_b_multiplier, 0.05),
    )


def _load_or_create_team_strength(
    team_a: str,
    team_b: str,
    team_strength_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if team_strength_df is not None:
        return team_strength_df

    if TEAM_STRENGTH_PATH.exists():
        return pd.read_csv(TEAM_STRENGTH_PATH)

    return pd.DataFrame(
        [
            _neutral_strength_row(team_a),
            _neutral_strength_row(team_b),
        ],
        columns=TEAM_STRENGTH_COLUMNS,
    )


def _neutral_strength_row(team: str) -> dict[str, float | str]:
    return {
        "team": team,
        "avg_goals_scored": 1.2,
        "avg_goals_conceded": 1.2,
        "attack_strength": 1.0,
        "defense_strength": 1.0,
    }


def _format_scorelines(lambda_a: float, lambda_b: float) -> list[dict[str, float | str]]:
    scores = most_likely_scores(lambda_a, lambda_b, top_n=5)
    return [
        {
            "score": f"{int(row.team_a_goals)}-{int(row.team_b_goals)}",
            "probability": float(row.probability),
        }
        for row in scores.itertuples(index=False)
    ]


def _strength_share_from_elo(elo_diff: float, tactical_edge: float) -> float:
    adjusted_diff = float(elo_diff) + float(tactical_edge) * 15
    share = 1 / (1 + 10 ** (-(adjusted_diff) / 400))
    return max(min(share, 0.75), 0.25)


def _confidence_level(
    probabilities: dict[str, float],
    lambda_a: float,
    lambda_b: float,
) -> str:
    max_probability = max(probabilities.values())
    expected_goal_gap = abs(lambda_a - lambda_b)

    if max_probability >= 0.58 and expected_goal_gap >= 0.45:
        return "high"
    if max_probability >= 0.43 or expected_goal_gap >= 0.25:
        return "medium"
    return "low"


def _explanation_factors(
    features: dict[str, Any],
    team_a: str,
    team_b: str,
    probabilities: dict[str, float],
    lambda_a: float,
    lambda_b: float,
) -> list[str]:
    factors = []
    elo_diff = _get_float(features, "elo_diff")
    xg_delta = _get_float(features, "xg_delta")
    set_piece_a = _get_float(features, "set_piece_edge_a")
    set_piece_b = _get_float(features, "set_piece_edge_b")
    h2h_score = _get_float(features, "h2h_weighted_score")
    tactical_edge = _get_float(features, "tactical_edge")

    leader_key = max(PROBABILITY_KEYS, key=lambda key: probabilities[key])
    if leader_key == "team_a_win":
        factors.append(
            f"{team_a} has the highest 90-minute win probability at "
            f"{_format_percent(probabilities[leader_key])}."
        )
        factors.append(
            f"That {team_a} win probability adds up all {team_a}-winning exact "
            "scores, not only the single most likely scoreline."
        )
    elif leader_key == "team_b_win":
        factors.append(
            f"{team_b} has the highest 90-minute win probability at "
            f"{_format_percent(probabilities[leader_key])}."
        )
        factors.append(
            f"That {team_b} win probability adds up all {team_b}-winning exact "
            "scores, not only the single most likely scoreline."
        )
    else:
        factors.append(
            "The draw is the single most likely 90-minute outcome at "
            f"{_format_percent(probabilities[leader_key])}."
        )

    expected_goal_gap = float(lambda_a) - float(lambda_b)
    if expected_goal_gap > 0.15:
        factors.append(
            f"The score model projects {team_a} higher on expected goals "
            f"({lambda_a:.2f} vs {lambda_b:.2f})."
        )
    elif expected_goal_gap < -0.15:
        factors.append(
            f"The score model projects {team_b} higher on expected goals "
            f"({lambda_b:.2f} vs {lambda_a:.2f})."
        )

    team_a_edges = []
    team_b_edges = []

    if elo_diff > 25:
        team_a_edges.append("Elo")
    elif elo_diff < -25:
        team_b_edges.append("Elo")

    if xg_delta > 0.15:
        team_a_edges.append("current-tournament xG input")
    elif xg_delta < -0.15:
        team_b_edges.append("current-tournament xG input")

    if set_piece_a - set_piece_b > 0.75:
        team_a_edges.append("set pieces")
    elif set_piece_b - set_piece_a > 0.75:
        team_b_edges.append("set pieces")

    if h2h_score > 0.15:
        team_a_edges.append("H2H")
    elif h2h_score < -0.15:
        team_b_edges.append("H2H")

    if tactical_edge > 0.75:
        team_a_edges.append("tactical matchup")
    elif tactical_edge < -0.75:
        team_b_edges.append("tactical matchup")

    if team_a_edges and team_b_edges:
        factors.append(
            f"Mixed matchup signals: {team_a} leads on "
            f"{_join_labels(team_a_edges)}, while {team_b} leads on "
            f"{_join_labels(team_b_edges)}."
        )
    elif team_a_edges:
        factors.append(
            f"Other matchup signals favor {team_a}: {_join_labels(team_a_edges)}."
        )
    elif team_b_edges:
        factors.append(
            f"Other matchup signals favor {team_b}: {_join_labels(team_b_edges)}."
        )

    if len(factors) < 2:
        factors.append(
            "The final probability blends result model, score model, Elo, tactical, "
            "and H2H signals."
        )

    return factors[:4]


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _bounded(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _as_dict(match_features: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(match_features, pd.Series):
        return match_features.to_dict()
    return dict(match_features)


def _get_float(values: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = values.get(key, default)
    if value is None or pd.isna(value):
        return default
    return float(value)
