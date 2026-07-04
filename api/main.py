"""FastAPI backend for World Cup AI Predictor."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.prediction.ensemble import ensemble_predict
from src.prediction.predict_match import build_match_feature_row, load_result_model
from src.prediction.simulate_tournament import DEFAULT_FIXTURES_PATH


app = FastAPI(
    title="World Cup AI Predictor",
    description="AI/ML football match prediction API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictMatchRequest(BaseModel):
    team_a: str = Field(..., min_length=1)
    team_b: str = Field(..., min_length=1)
    stage: str = Field(default="Group", min_length=1)
    is_knockout: bool = False


class ExpectedGoals(BaseModel):
    team_a: float
    team_b: float


class ScorelineProbability(BaseModel):
    score: str
    probability: float


class PredictionPayload(BaseModel):
    team_a_win: float
    draw: float
    team_b_win: float
    team_a_advance: float
    team_b_advance: float
    expected_goals: ExpectedGoals
    most_likely_scores: list[ScorelineProbability]
    confidence: Literal["low", "medium", "high"]
    explanation_factors: list[str]
    data_notes: list[str] = Field(default_factory=list)


class PredictMatchResponse(BaseModel):
    match: str
    prediction: PredictionPayload


class TeamsResponse(BaseModel):
    teams: list[str]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": "World Cup AI Predictor",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams", response_model=TeamsResponse)
def teams() -> TeamsResponse:
    return TeamsResponse(teams=available_teams())


@app.post("/predict-match", response_model=PredictMatchResponse)
def predict_match_endpoint(request: PredictMatchRequest) -> PredictMatchResponse:
    try:
        result_model = load_result_model()
        match_features, team_strength, notes = build_match_feature_row(
            team_a=request.team_a,
            team_b=request.team_b,
            stage=request.stage,
            knockout=request.is_knockout,
        )
        prediction = ensemble_predict(
            match_features,
            result_model=result_model,
            team_strength_df=team_strength,
        )
        prediction["data_notes"] = notes
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictMatchResponse(
        match=f"{request.team_a} vs {request.team_b}",
        prediction=PredictionPayload(**prediction),
    )


def available_teams() -> list[str]:
    fixtures = pd.read_csv(DEFAULT_FIXTURES_PATH)
    teams = set(fixtures["team_a"].dropna()).union(set(fixtures["team_b"].dropna()))
    return sorted(str(team) for team in teams if str(team).strip())
