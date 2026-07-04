"""Build the model-ready training dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.features.elo import add_elo_features, build_elo_history
from src.features.h2h import add_h2h_features
from src.features.performance import add_performance_features
from src.features.tactics import add_tactical_features
from src.utils.config import PROCESSED_DATA_DIR


TRAINING_DATASET_PATH = PROCESSED_DATA_DIR / "training_dataset.csv"

MODEL_FEATURES = [
    "team_a_elo",
    "team_b_elo",
    "elo_diff",
    "team_a_wc_goals_scored",
    "team_b_wc_goals_scored",
    "team_a_wc_goals_conceded",
    "team_b_wc_goals_conceded",
    "team_a_xg_per_match",
    "team_b_xg_per_match",
    "team_a_xga_per_match",
    "team_b_xga_per_match",
    "team_a_shots_per_match",
    "team_b_shots_per_match",
    "team_a_possession_avg",
    "team_b_possession_avg",
    "team_a_rest_days",
    "team_b_rest_days",
    "wc_goal_diff_delta",
    "xg_delta",
    "xga_delta",
    "shots_delta",
    "possession_delta",
    "rest_days_delta",
    "team_a_pressing_level",
    "team_b_pressing_level",
    "team_a_build_up_quality",
    "team_b_build_up_quality",
    "team_a_counterattack_speed",
    "team_b_counterattack_speed",
    "team_a_set_piece_strength",
    "team_b_set_piece_strength",
    "team_a_set_piece_defense",
    "team_b_set_piece_defense",
    "team_a_wide_attack",
    "team_b_wide_attack",
    "team_a_transition_defense",
    "team_b_transition_defense",
    "team_a_aerial_strength",
    "team_b_aerial_strength",
    "pressing_vs_buildup_a",
    "pressing_vs_buildup_b",
    "counter_vs_transition_a",
    "counter_vs_transition_b",
    "set_piece_edge_a",
    "set_piece_edge_b",
    "wide_attack_edge_a",
    "wide_attack_edge_b",
    "aerial_edge_a",
    "tactical_edge",
    "h2h_matches_count",
    "h2h_team_a_wins",
    "h2h_team_b_wins",
    "h2h_draws",
    "h2h_team_a_goals",
    "h2h_team_b_goals",
    "h2h_goal_diff",
    "h2h_weighted_score",
]

CONTEXT_COLUMNS = [
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


def create_result_label(row: pd.Series | dict[str, Any]) -> int:
    """Create the 3-class match result label.

    Returns 0 for Team A win, 1 for draw, and 2 for Team B win.
    """

    team_a_score = float(row["team_a_score"])
    team_b_score = float(row["team_b_score"])

    if team_a_score > team_b_score:
        return 0
    if team_a_score == team_b_score:
        return 1
    return 2


def create_advance_label(row: pd.Series | dict[str, Any]) -> int | None:
    """Create the knockout advancement label for Team A."""

    is_knockout = _parse_bool(row.get("is_knockout"))
    if not is_knockout:
        return None

    team_a_advanced = _parse_bool(row.get("team_a_advanced"))
    if team_a_advanced is None:
        return None

    return 1 if team_a_advanced else 0


def build_training_dataset(
    matches_df: pd.DataFrame,
    tactics_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build and save the model-ready training dataset."""

    matches = matches_df.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    elo_history = build_elo_history(matches)
    dataset = add_elo_features(matches, elo_history)
    dataset = add_performance_features(dataset, performance_df)
    dataset = add_tactical_features(dataset, tactics_df)
    dataset = add_h2h_features(dataset, h2h_df)

    dataset["result_label"] = dataset.apply(create_result_label, axis=1)
    dataset["advance_label"] = dataset.apply(create_advance_label, axis=1)

    _validate_model_features(dataset)
    for column in MODEL_FEATURES:
        dataset[column] = pd.to_numeric(dataset[column], errors="raise")

    output_columns = [
        *[column for column in CONTEXT_COLUMNS if column in dataset.columns],
        *MODEL_FEATURES,
        "result_label",
        "advance_label",
    ]
    training_dataset = dataset[output_columns].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    training_dataset.to_csv(TRAINING_DATASET_PATH, index=False)
    return training_dataset


def main() -> None:
    """Build the sample training dataset from the command line."""

    training_dataset = build_training_dataset(
        matches_df=load_matches(),
        tactics_df=load_tactics(),
        performance_df=load_current_performance(),
        h2h_df=load_h2h(),
    )
    print(
        f"Created {TRAINING_DATASET_PATH} "
        f"with {len(training_dataset)} rows and {len(training_dataset.columns)} columns."
    )


def _parse_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        if normalized == "":
            return None
    return None


def _validate_model_features(dataset: pd.DataFrame) -> None:
    missing_features = [
        column for column in MODEL_FEATURES if column not in dataset.columns
    ]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Training dataset is missing model features: {missing}")


if __name__ == "__main__":
    main()
