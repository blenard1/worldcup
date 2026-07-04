"""Knockout tournament simulation using ensemble predictions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prediction.ensemble import ensemble_predict, normalize_probs
from src.prediction.predict_match import build_match_feature_row, load_result_model
from src.models.train_result_model import RESULT_MODEL_PATH
from src.utils.config import PROCESSED_DATA_DIR, SAMPLE_DATA_DIR


DEFAULT_FIXTURES_PATH = SAMPLE_DATA_DIR / "knockout_fixtures.csv"
SIMULATION_RESULTS_PATH = PROCESSED_DATA_DIR / "simulation_results.csv"
FIXTURE_REQUIRED_COLUMNS = ["match_id", "round", "team_a", "team_b", "next_match_id"]
ROUND_REACH_COLUMN = {
    "quarterfinal": "reach_quarterfinal",
    "semifinal": "reach_semifinal",
    "final": "reach_final",
}
RESULT_COLUMNS = [
    "team",
    "reach_quarterfinal",
    "reach_semifinal",
    "reach_final",
    "champion",
]


PredictionProvider = Callable[[str, str, str], dict[str, Any]]


def simulate_match(
    team_a: str,
    team_b: str,
    prediction: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> str:
    """Choose a knockout winner from prediction probabilities."""

    generator = rng or np.random.default_rng()
    if "team_a_advance" in prediction and "team_b_advance" in prediction:
        probabilities = {
            "team_a_win": prediction["team_a_advance"],
            "draw": 0,
            "team_b_win": prediction["team_b_advance"],
        }
    else:
        match_probs = normalize_probs(
            {
                "team_a_win": prediction.get("team_a_win", 0),
                "draw": prediction.get("draw", 0),
                "team_b_win": prediction.get("team_b_win", 0),
            }
        )
        probabilities = {
            "team_a_win": match_probs["team_a_win"] + match_probs["draw"] * 0.5,
            "draw": 0,
            "team_b_win": match_probs["team_b_win"] + match_probs["draw"] * 0.5,
        }

    advance_probs = normalize_probs(probabilities)
    return (
        team_a
        if generator.random() < advance_probs["team_a_win"]
        else team_b
    )


def simulate_bracket(
    fixtures_df: pd.DataFrame,
    n_simulations: int = 10000,
    prediction_provider: PredictionProvider | None = None,
    random_state: int | None = 42,
    output_path: str | Path = SIMULATION_RESULTS_PATH,
) -> pd.DataFrame:
    """Simulate a knockout bracket and return round/champion probabilities."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    _validate_columns(fixtures_df, FIXTURE_REQUIRED_COLUMNS, "fixtures")
    fixtures = fixtures_df.copy().reset_index(drop=True)
    fixtures["match_id"] = fixtures["match_id"].astype(str)
    fixtures["next_match_id"] = fixtures["next_match_id"].fillna("").astype(str)

    rng = np.random.default_rng(random_state)
    provider = prediction_provider or _default_prediction_provider()
    prediction_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}

    for _ in range(n_simulations):
        _simulate_once(fixtures, provider, prediction_cache, counts, rng)

    results = _counts_to_probabilities(counts, n_simulations)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    fixtures_path = Path(args.fixtures)
    if not fixtures_path.exists():
        print(f"Fixtures CSV not found at {fixtures_path}", file=sys.stderr)
        return 1

    fixtures = pd.read_csv(fixtures_path)
    results = simulate_bracket(
        fixtures,
        n_simulations=args.simulations,
        output_path=args.output,
    )
    champion_preview = results.sort_values("champion", ascending=False).head(5)
    print(f"Created {args.output} with {len(results)} teams.")
    print("Top champion probabilities:")
    for row in champion_preview.itertuples(index=False):
        print(f"- {row.team}: {row.champion:.1%}")
    return 0


def _simulate_once(
    fixtures: pd.DataFrame,
    provider: PredictionProvider,
    prediction_cache: dict[tuple[str, str, str], dict[str, Any]],
    counts: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> str:
    slots = _initial_slots(fixtures)
    processed: set[str] = set()
    champion = ""

    while len(processed) < len(fixtures):
        progressed = False
        for fixture in fixtures.itertuples(index=False):
            match_id = str(fixture.match_id)
            if match_id in processed:
                continue

            team_a = slots[match_id]["team_a"]
            team_b = slots[match_id]["team_b"]
            if not team_a or not team_b:
                continue

            _record_round_reach(counts, str(fixture.round), team_a)
            _record_round_reach(counts, str(fixture.round), team_b)

            cache_key = (team_a, team_b, str(fixture.round))
            prediction = prediction_cache.get(cache_key)
            if prediction is None:
                prediction = provider(team_a, team_b, str(fixture.round))
                prediction_cache[cache_key] = prediction

            winner = simulate_match(team_a, team_b, prediction, rng)
            processed.add(match_id)
            progressed = True

            next_match_id = str(fixture.next_match_id).strip()
            if next_match_id:
                _advance_to_next_match(slots, next_match_id, winner)
            else:
                champion = winner
                _ensure_team(counts, winner)
                counts[winner]["champion"] += 1

        if not progressed:
            unresolved = sorted(set(fixtures["match_id"]) - processed)
            raise ValueError(
                "Could not resolve bracket. Check fixture team slots and "
                f"next_match_id links. Unresolved matches: {', '.join(unresolved)}"
            )

    return champion


def _default_prediction_provider() -> PredictionProvider:
    result_model = _load_model_if_available()

    def provider(team_a: str, team_b: str, round_name: str) -> dict[str, Any]:
        match_features, team_strength, _notes = build_match_feature_row(
            team_a=team_a,
            team_b=team_b,
            stage=round_name,
            knockout=True,
        )
        return ensemble_predict(
            match_features,
            result_model=result_model,
            team_strength_df=team_strength,
        )

    return provider


def _load_model_if_available() -> Any | None:
    if not RESULT_MODEL_PATH.exists():
        return None
    return load_result_model(RESULT_MODEL_PATH)


def _initial_slots(fixtures: pd.DataFrame) -> dict[str, dict[str, str | None]]:
    slots = {}
    for fixture in fixtures.itertuples(index=False):
        slots[str(fixture.match_id)] = {
            "team_a": _team_or_none(fixture.team_a),
            "team_b": _team_or_none(fixture.team_b),
        }
    return slots


def _team_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    team = str(value).strip()
    if not team or team.lower().startswith("winner"):
        return None
    return team


def _advance_to_next_match(
    slots: dict[str, dict[str, str | None]],
    next_match_id: str,
    winner: str,
) -> None:
    if next_match_id not in slots:
        raise ValueError(f"next_match_id {next_match_id} does not exist in fixtures.")

    if slots[next_match_id]["team_a"] is None:
        slots[next_match_id]["team_a"] = winner
        return
    if slots[next_match_id]["team_b"] is None:
        slots[next_match_id]["team_b"] = winner
        return

    raise ValueError(f"Next match {next_match_id} already has two teams.")


def _record_round_reach(
    counts: dict[str, dict[str, int]],
    round_name: str,
    team: str,
) -> None:
    round_key = _normalize_round(round_name)
    column = ROUND_REACH_COLUMN.get(round_key)
    if column is None:
        _ensure_team(counts, team)
        return

    _ensure_team(counts, team)
    counts[team][column] += 1


def _normalize_round(round_name: str) -> str:
    normalized = round_name.strip().lower().replace("-", " ").replace("_", " ")
    if normalized in {"qf", "quarter final", "quarterfinal", "quarter finals"}:
        return "quarterfinal"
    if normalized in {"sf", "semi final", "semifinal", "semi finals"}:
        return "semifinal"
    if normalized in {"f", "final", "finals"}:
        return "final"
    return normalized


def _ensure_team(counts: dict[str, dict[str, int]], team: str) -> None:
    if team not in counts:
        counts[team] = {
            "reach_quarterfinal": 0,
            "reach_semifinal": 0,
            "reach_final": 0,
            "champion": 0,
        }


def _counts_to_probabilities(
    counts: dict[str, dict[str, int]],
    n_simulations: int,
) -> pd.DataFrame:
    rows = []
    for team, team_counts in counts.items():
        rows.append(
            {
                "team": team,
                "reach_quarterfinal": team_counts["reach_quarterfinal"]
                / n_simulations,
                "reach_semifinal": team_counts["reach_semifinal"]
                / n_simulations,
                "reach_final": team_counts["reach_final"] / n_simulations,
                "champion": team_counts["champion"] / n_simulations,
            }
        )

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    return (
        pd.DataFrame(rows, columns=RESULT_COLUMNS)
        .sort_values(["champion", "reach_final", "team"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate a World Cup knockout bracket.")
    parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_FIXTURES_PATH),
        help="Fixture CSV path with match_id,round,team_a,team_b,next_match_id",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of bracket simulations to run.",
    )
    parser.add_argument(
        "--output",
        default=str(SIMULATION_RESULTS_PATH),
        help="Output CSV path for simulation probabilities.",
    )
    return parser


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
    raise SystemExit(main())
