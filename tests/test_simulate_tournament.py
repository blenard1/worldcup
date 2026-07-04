import subprocess
import sys

import pandas as pd
import pytest

from src.prediction.simulate_tournament import (
    simulate_bracket,
    simulate_match,
)
from src.utils.config import PROJECT_ROOT


def _fake_bracket():
    return pd.DataFrame(
        [
            {
                "match_id": "SF_1",
                "round": "Semifinal",
                "team_a": "Alpha",
                "team_b": "Delta",
                "next_match_id": "FINAL",
            },
            {
                "match_id": "SF_2",
                "round": "Semifinal",
                "team_a": "Bravo",
                "team_b": "Charlie",
                "next_match_id": "FINAL",
            },
            {
                "match_id": "FINAL",
                "round": "Final",
                "team_a": "",
                "team_b": "",
                "next_match_id": "",
            },
        ]
    )


def _ranked_prediction_provider(team_a, team_b, round_name):
    ranking = {"Alpha": 4, "Bravo": 3, "Charlie": 2, "Delta": 1}
    team_a_share = (
        0.9 if ranking.get(team_a, 0) >= ranking.get(team_b, 0) else 0.1
    )
    return {
        "team_a_win": team_a_share,
        "draw": 0.0,
        "team_b_win": 1 - team_a_share,
        "team_a_advance": team_a_share,
        "team_b_advance": 1 - team_a_share,
    }


def test_simulate_match_uses_advance_probabilities():
    winner = simulate_match(
        "Alpha",
        "Delta",
        {"team_a_advance": 1.0, "team_b_advance": 0.0},
    )

    assert winner == "Alpha"


def test_simulate_bracket_returns_probabilities_and_saves_csv(tmp_path):
    output_path = tmp_path / "simulation_results.csv"

    results = simulate_bracket(
        _fake_bracket(),
        n_simulations=500,
        prediction_provider=_ranked_prediction_provider,
        random_state=7,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert {
        "team",
        "reach_quarterfinal",
        "reach_semifinal",
        "reach_final",
        "champion",
    }.issubset(results.columns)
    assert results["champion"].sum() == pytest.approx(1.0)
    assert set(results["team"]) == {"Alpha", "Bravo", "Charlie", "Delta"}
    assert results.loc[results["team"] == "Alpha", "champion"].iloc[0] > 0.5


def test_simulate_bracket_counts_reaching_later_rounds():
    results = simulate_bracket(
        _fake_bracket(),
        n_simulations=200,
        prediction_provider=_ranked_prediction_provider,
        random_state=3,
    )

    alpha = results[results["team"] == "Alpha"].iloc[0]
    delta = results[results["team"] == "Delta"].iloc[0]

    assert alpha["reach_semifinal"] == pytest.approx(1.0)
    assert delta["reach_semifinal"] == pytest.approx(1.0)
    assert alpha["reach_final"] > delta["reach_final"]


def test_simulate_bracket_raises_for_missing_fixture_columns():
    fixtures = pd.DataFrame([{"match_id": "M1", "team_a": "A"}])

    with pytest.raises(ValueError, match="round"):
        simulate_bracket(
            fixtures,
            n_simulations=10,
            prediction_provider=_ranked_prediction_provider,
        )


def test_simulate_tournament_cli_creates_output_csv(tmp_path):
    fixtures_path = tmp_path / "fixtures.csv"
    output_path = tmp_path / "simulation_results.csv"
    _fake_bracket().to_csv(fixtures_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prediction.simulate_tournament",
            "--fixtures",
            str(fixtures_path),
            "--simulations",
            "10",
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Created" in result.stdout
    assert output_path.is_file()

    saved = pd.read_csv(output_path)
    assert "champion" in saved.columns
    assert saved["champion"].sum() == pytest.approx(1.0)
