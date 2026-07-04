"""Streamlit entrypoint for deploying the predictor as a single app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.build_dataset import TRAINING_DATASET_PATH, build_training_dataset
from src.data.load_data import (
    load_current_performance,
    load_h2h,
    load_matches,
    load_tactics,
)
from src.models.poisson_model import estimate_team_attack_defense
from src.models.train_result_model import RESULT_MODEL_PATH, train_result_model
from src.prediction.ensemble import ensemble_predict
from src.prediction.predict_match import build_match_feature_row, load_result_model
from src.prediction.simulate_tournament import DEFAULT_FIXTURES_PATH


STAGES = ["Group", "R16", "Quarterfinal", "Semifinal", "Final"]


st.set_page_config(
    page_title="Atrox World Cup AI Predictor",
    page_icon="WC",
    layout="wide",
)


@st.cache_data
def available_teams() -> list[str]:
    fixtures = pd.read_csv(DEFAULT_FIXTURES_PATH)
    teams = set(fixtures["team_a"].dropna()).union(set(fixtures["team_b"].dropna()))
    return sorted(str(team) for team in teams if str(team).strip())


@st.cache_resource(show_spinner="Preparing sample ML model...")
def load_or_prepare_model():
    matches = load_matches()
    if not TRAINING_DATASET_PATH.exists():
        build_training_dataset(
            matches_df=matches,
            tactics_df=load_tactics(),
            performance_df=load_current_performance(),
            h2h_df=load_h2h(),
        )
    estimate_team_attack_defense(matches)
    if not RESULT_MODEL_PATH.exists():
        train_result_model(data_path=TRAINING_DATASET_PATH)
    return load_result_model()


def percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def render_prediction(
    team_a: str,
    team_b: str,
    prediction: dict,
    data_notes: list[str],
) -> None:
    st.subheader(f"{team_a} vs {team_b}")

    cols = st.columns(5)
    cols[0].metric(f"{team_a} 90-min win", percent(prediction["team_a_win"]))
    cols[1].metric("90-min draw", percent(prediction["draw"]))
    cols[2].metric(f"{team_b} 90-min win", percent(prediction["team_b_win"]))
    cols[3].metric(f"{team_a} advance", percent(prediction["team_a_advance"]))
    cols[4].metric(f"{team_b} advance", percent(prediction["team_b_advance"]))

    left, right = st.columns(2)
    with left:
        st.markdown("### Expected goals")
        st.write(f"{team_a}: {prediction['expected_goals']['team_a']:.2f}")
        st.write(f"{team_b}: {prediction['expected_goals']['team_b']:.2f}")

    with right:
        st.markdown("### Most likely exact scores")
        for index, scoreline in enumerate(prediction["most_likely_scores"], start=1):
            st.write(
                f"{index}. {team_a} {scoreline['score']} {team_b} "
                f"({percent(scoreline['probability'])})"
            )

    st.markdown("### Why this result")
    for factor in prediction["explanation_factors"]:
        st.write(f"- {factor}")

    if data_notes:
        st.markdown("### Data notes")
        for note in data_notes:
            st.write(f"- {note}")


def main() -> None:
    st.title("Atrox World Cup AI Predictor")
    st.caption(
        "Sample-data deployment for the World Cup AI Predictor. "
        "Predictions are probabilities, not guarantees."
    )

    teams = available_teams()
    with st.sidebar:
        st.header("Match")
        team_a = st.selectbox("Team A", teams, index=teams.index("Morocco"))
        default_team_b_index = teams.index("Canada") if "Canada" in teams else 1
        team_b = st.selectbox("Team B", teams, index=default_team_b_index)
        stage = st.selectbox("Stage", STAGES, index=STAGES.index("R16"))
        is_knockout = st.checkbox("Knockout match", value=True)
        run_prediction = st.button("Predict match", type="primary")

    if team_a == team_b:
        st.warning("Choose two different teams.")
        return

    if not run_prediction:
        st.info("Choose a matchup in the sidebar and click Predict match.")
        return

    model = load_or_prepare_model()
    match_features, team_strength, data_notes = build_match_feature_row(
        team_a=team_a,
        team_b=team_b,
        stage=stage,
        knockout=is_knockout,
    )
    prediction = ensemble_predict(
        match_features,
        result_model=model,
        team_strength_df=team_strength,
    )
    render_prediction(team_a, team_b, prediction, data_notes)


if __name__ == "__main__":
    main()
