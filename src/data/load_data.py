"""Data loading utilities for World Cup AI Predictor sample datasets."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import SAMPLE_DATA_DIR


MATCHES_REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "tournament",
    "stage",
    "neutral",
    "city",
    "country",
    "is_knockout",
    "team_a_advanced",
]

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

CURRENT_PERFORMANCE_REQUIRED_COLUMNS = [
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

H2H_REQUIRED_COLUMNS = [
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "tournament",
    "neutral",
]


def _validate_required_columns(
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


def _load_csv(
    path: str | Path | None,
    default_filename: str,
    required_columns: Iterable[str],
    dataset_name: str,
) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else SAMPLE_DATA_DIR / default_filename
    dataframe = pd.read_csv(csv_path)
    _validate_required_columns(dataframe, required_columns, dataset_name)
    return dataframe


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    """Load match-level sample data."""

    return _load_csv(path, "matches.csv", MATCHES_REQUIRED_COLUMNS, "matches")


def load_tactics(path: str | Path | None = None) -> pd.DataFrame:
    """Load team tactical profile sample data."""

    return _load_csv(path, "team_tactics.csv", TACTICS_REQUIRED_COLUMNS, "team tactics")


def load_current_performance(path: str | Path | None = None) -> pd.DataFrame:
    """Load current tournament performance sample data."""

    return _load_csv(
        path,
        "current_worldcup_performance.csv",
        CURRENT_PERFORMANCE_REQUIRED_COLUMNS,
        "current World Cup performance",
    )


def load_h2h(path: str | Path | None = None) -> pd.DataFrame:
    """Load head-to-head sample data."""

    return _load_csv(path, "h2h.csv", H2H_REQUIRED_COLUMNS, "head-to-head")
