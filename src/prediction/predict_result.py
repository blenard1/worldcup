"""Prediction helpers for the match result model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.build_dataset import MODEL_FEATURES


RESULT_PROBABILITY_KEYS = {
    0: "team_a_win",
    1: "draw",
    2: "team_b_win",
}


def predict_result_proba(model: Any, feature_row: pd.Series | dict | pd.DataFrame) -> dict[str, float]:
    """Return Team A win, draw, and Team B win probabilities."""

    features = _feature_frame(feature_row)
    raw_probabilities = model.predict_proba(features)
    model_classes = list(getattr(model, "classes_", RESULT_PROBABILITY_KEYS.keys()))

    probabilities = {label: 0.0 for label in RESULT_PROBABILITY_KEYS}
    for index, label in enumerate(model_classes):
        if label in probabilities and index < raw_probabilities.shape[1]:
            probabilities[label] = float(raw_probabilities[0, index])

    total_probability = sum(probabilities.values())
    if total_probability <= 0:
        probabilities = {label: 1 / len(probabilities) for label in probabilities}
    else:
        probabilities = {
            label: probability / total_probability
            for label, probability in probabilities.items()
        }

    return {
        RESULT_PROBABILITY_KEYS[label]: probabilities[label]
        for label in RESULT_PROBABILITY_KEYS
    }


def _feature_frame(feature_row: pd.Series | dict | pd.DataFrame) -> pd.DataFrame:
    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError("feature_row DataFrame must contain exactly one row.")
        frame = feature_row.copy()
    elif isinstance(feature_row, pd.Series):
        frame = feature_row.to_frame().T
    else:
        frame = pd.DataFrame([feature_row])

    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in frame.columns
    ]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Feature row is missing model features: {missing}")

    return frame[MODEL_FEATURES]
