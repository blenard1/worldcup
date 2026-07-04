"""Command-line match prediction tool."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.build_dataset import MODEL_FEATURES
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
from src.models.poisson_model import (
    TEAM_STRENGTH_PATH,
    estimate_team_attack_defense,
)
from src.models.train_result_model import RESULT_MODEL_PATH
from src.prediction.ensemble import ensemble_predict


TEAM_ALIASES = {
    "USA": "United States",
    "US": "United States",
    "U.S.A.": "United States",
}


def build_match_feature_row(
    team_a: str,
    team_b: str,
    stage: str,
    knockout: bool,
    match_date: str | None = None,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Build model features for one requested match."""

    matches = load_matches()
    tactics = load_tactics()
    performance = load_current_performance()
    h2h = load_h2h()
    display_team_a = team_a
    display_team_b = team_b
    canonical_team_a = _canonical_team_name(team_a)
    canonical_team_b = _canonical_team_name(team_b)

    prediction_date = match_date or pd.Timestamp.today().date().isoformat()
    requested_match = pd.DataFrame(
        [
            {
                "match_id": "PREDICTED_MATCH",
                "date": prediction_date,
                "team_a": canonical_team_a,
                "team_b": canonical_team_b,
                "team_a_score": 0,
                "team_b_score": 0,
                "tournament": "Prediction",
                "stage": stage,
                "neutral": True,
                "city": "",
                "country": "",
                "is_knockout": knockout,
                "team_a_advanced": "",
            }
        ]
    )

    elo_history = build_elo_history(matches)
    features = add_elo_features(requested_match, elo_history)
    with _suppress_feature_warnings():
        features = add_performance_features(features, performance)
        features = add_tactical_features(features, tactics)
    features = add_h2h_features(features, h2h)

    for column in MODEL_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")

    features["team_a"] = display_team_a
    features["team_b"] = display_team_b

    team_strength = _add_team_strength_aliases(
        _load_or_create_team_strength(matches),
        [
            (canonical_team_a, display_team_a),
            (canonical_team_b, display_team_b),
        ],
    )
    notes = _data_coverage_notes(
        display_team_a,
        display_team_b,
        canonical_team_a,
        canonical_team_b,
        matches,
        tactics,
        performance,
    )
    return features.iloc[0], team_strength, notes


def load_result_model(model_path: str | Path = RESULT_MODEL_PATH) -> Any:
    """Load the trained result model or raise a clear setup error."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained result model not found at {path}. "
            "Run `python -m src.models.train_result_model` first."
        )

    from joblib import load

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return load(path)


def predict_match(
    team_a: str,
    team_b: str,
    stage: str,
    knockout: bool,
    match_date: str | None = None,
    model_path: str | Path = RESULT_MODEL_PATH,
) -> str:
    """Create a formatted prediction for one match."""

    model = load_result_model(model_path)
    match_features, team_strength, notes = build_match_feature_row(
        team_a=team_a,
        team_b=team_b,
        stage=stage,
        knockout=knockout,
        match_date=match_date,
    )
    prediction = ensemble_predict(
        match_features,
        result_model=model,
        team_strength_df=team_strength,
    )
    return format_prediction_output(team_a, team_b, prediction, notes)


def format_prediction_output(
    team_a: str,
    team_b: str,
    prediction: dict[str, Any],
    notes: list[str] | None = None,
) -> str:
    """Format a prediction object for terminal output."""

    score_lines = []
    for index, scoreline in enumerate(prediction["most_likely_scores"][:3], start=1):
        team_a_goals, team_b_goals = scoreline["score"].split("-")
        score_lines.append(
            f"{index}. {team_a} {team_a_goals}-{team_b_goals} {team_b}"
        )

    reasons = prediction["explanation_factors"][:3]
    note_lines = [f"- {note}" for note in (notes or [])]
    reason_lines = [f"- {reason}" for reason in reasons]

    sections = [
        f"{team_a} vs {team_b}",
        "",
        "90-minute prediction:",
        f"{team_a} win: {_format_percent(prediction['team_a_win'])}",
        f"Draw: {_format_percent(prediction['draw'])}",
        f"{team_b} win: {_format_percent(prediction['team_b_win'])}",
        "",
        "Advance prediction:",
        f"{team_a}: {_format_percent(prediction['team_a_advance'])}",
        f"{team_b}: {_format_percent(prediction['team_b_advance'])}",
        "",
        "Expected goals:",
        f"{team_a}: {prediction['expected_goals']['team_a']:.2f}",
        f"{team_b}: {prediction['expected_goals']['team_b']:.2f}",
        "",
        "Most likely scores:",
        *score_lines,
        "",
        "Why:",
        *reason_lines,
    ]

    if note_lines:
        sections.extend(["", "Data notes:", *note_lines])

    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output = predict_match(
            team_a=args.team_a,
            team_b=args.team_b,
            stage=args.stage,
            knockout=_parse_bool_arg(args.knockout),
            match_date=args.date,
            model_path=args.model_path,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 1

    print(output)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a World Cup match.")
    parser.add_argument("--team-a", required=True, help="Team A name")
    parser.add_argument("--team-b", required=True, help="Team B name")
    parser.add_argument("--stage", default="Group", help="Match stage")
    parser.add_argument(
        "--knockout",
        default="false",
        help="Whether this is a knockout match: true or false",
    )
    parser.add_argument("--date", default=None, help="Prediction match date")
    parser.add_argument(
        "--model-path",
        default=str(RESULT_MODEL_PATH),
        help="Path to trained result_model.pkl",
    )
    return parser


def _load_or_create_team_strength(matches: pd.DataFrame) -> pd.DataFrame:
    if TEAM_STRENGTH_PATH.exists():
        return pd.read_csv(TEAM_STRENGTH_PATH)
    return estimate_team_attack_defense(matches)


def _add_team_strength_aliases(
    team_strength: pd.DataFrame,
    alias_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    output = team_strength.copy()
    rows_to_add = []

    for canonical_team, display_team in alias_pairs:
        if canonical_team == display_team:
            continue
        if display_team in set(output["team"]):
            continue

        canonical_rows = output[output["team"] == canonical_team]
        if canonical_rows.empty:
            continue

        alias_row = canonical_rows.iloc[0].copy()
        alias_row["team"] = display_team
        rows_to_add.append(alias_row)

    if rows_to_add:
        output = pd.concat([output, pd.DataFrame(rows_to_add)], ignore_index=True)

    return output


def _data_coverage_notes(
    display_team_a: str,
    display_team_b: str,
    canonical_team_a: str,
    canonical_team_b: str,
    matches: pd.DataFrame,
    tactics: pd.DataFrame,
    performance: pd.DataFrame,
) -> list[str]:
    known_match_teams = set(matches["team_a"]).union(set(matches["team_b"]))
    known_tactics = set(tactics["team"])
    known_performance = set(performance["team"])
    notes = []

    for display_team, canonical_team in (
        (display_team_a, canonical_team_a),
        (display_team_b, canonical_team_b),
    ):
        missing_sources = []
        if canonical_team not in known_match_teams:
            missing_sources.append("historical match")
        if canonical_team not in known_tactics:
            missing_sources.append("tactical")
        if canonical_team not in known_performance:
            missing_sources.append("current tournament")

        if missing_sources:
            notes.append(
                f"Limited sample data for {display_team}; neutral defaults used for "
                f"{', '.join(missing_sources)} inputs."
            )

    return notes


def _canonical_team_name(team: str) -> str:
    return TEAM_ALIASES.get(team.strip(), team.strip())


def _parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError("--knockout must be true or false")


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


@contextmanager
def _suppress_feature_warnings():
    logger_names = ["src.features.performance", "src.features.tactics"]
    loggers = [logging.getLogger(name) for name in logger_names]
    previous_levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level in zip(loggers, previous_levels):
            logger.setLevel(level)


if __name__ == "__main__":
    raise SystemExit(main())
