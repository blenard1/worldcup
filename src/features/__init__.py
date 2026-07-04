"""Feature engineering package."""

from src.features.elo import (
    add_elo_features,
    build_elo_history,
    expected_score,
    get_initial_rating,
    get_team_elo_before_date,
    update_elo,
)
from src.features.h2h import (
    add_h2h_features,
    calculate_h2h_features,
    get_h2h_matches,
    normalize_pair,
)
from src.features.performance import add_performance_features
from src.features.tactics import add_tactical_features

__all__ = [
    "add_elo_features",
    "add_h2h_features",
    "add_performance_features",
    "add_tactical_features",
    "build_elo_history",
    "calculate_h2h_features",
    "expected_score",
    "get_initial_rating",
    "get_h2h_matches",
    "get_team_elo_before_date",
    "normalize_pair",
    "update_elo",
]
