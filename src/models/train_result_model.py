"""Train the match result prediction model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.build_dataset import MODEL_FEATURES, TRAINING_DATASET_PATH
from src.utils.config import MODELS_DIR


RANDOM_STATE = 42
RESULT_MODEL_PATH = MODELS_DIR / "result_model.pkl"
RESULT_MODEL_METADATA_PATH = MODELS_DIR / "result_model_metadata.json"
RESULT_LABELS = [0, 1, 2]


def train_result_model(
    data_path: str | Path = TRAINING_DATASET_PATH,
    model_path: str | Path = RESULT_MODEL_PATH,
    metadata_path: str | Path = RESULT_MODEL_METADATA_PATH,
) -> dict[str, Any]:
    """Train and save a multiclass result prediction model."""

    _ensure_sklearn_available()
    from joblib import dump
    from sklearn.metrics import accuracy_score, log_loss

    dataset = _load_training_dataset(data_path)
    X = dataset[MODEL_FEATURES]
    y = dataset["result_label"].astype(int)
    X_train, X_test, y_train, y_test = _split_train_test(X, y)

    model, model_type = _fit_best_available_model(X_train, y_train)
    probabilities = _predict_proba_for_labels(model, X_test, RESULT_LABELS)
    predictions = probabilities.idxmax(axis=1).astype(int)

    accuracy = float(accuracy_score(y_test, predictions))
    log_loss_value = _safe_log_loss(y_test, probabilities, log_loss)
    class_distribution = {
        str(label): int(count)
        for label, count in y.value_counts().sort_index().items()
    }

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)

    metadata = {
        "model_type": model_type,
        "features": MODEL_FEATURES,
        "accuracy": accuracy,
        "log_loss": log_loss_value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_distribution": class_distribution,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _print_training_report(accuracy, log_loss_value, class_distribution, model_type)
    return metadata


def main() -> None:
    """CLI entry point for result model training."""

    train_result_model()


def _load_training_dataset(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {path}. "
            "Run `python -m src.data.build_dataset` first."
        )

    dataset = pd.read_csv(path)
    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in dataset.columns
    ]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Training dataset is missing model features: {missing}")
    if "result_label" not in dataset.columns:
        raise ValueError("Training dataset is missing required column: result_label")

    dataset = dataset.dropna(subset=[*MODEL_FEATURES, "result_label"]).copy()
    if dataset.empty:
        raise ValueError("Training dataset has no usable rows after dropping nulls.")

    return dataset


def _split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    if len(y) < 2:
        return X, X, y, y

    class_count = y.nunique()
    test_size = max(1, round(len(y) * 0.25))
    stratify = None

    if y.value_counts().min() >= 2:
        test_size = max(test_size, class_count)
        if len(y) - test_size >= class_count:
            stratify = y

    if len(y) - test_size < 1:
        return X, X, y, y

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )


def _fit_best_available_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, str]:
    xgboost_model = _try_fit_xgboost(X_train, y_train)
    if xgboost_model is not None:
        return xgboost_model, "xgboost"

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model, "random_forest"


def _try_fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    try:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return model
    except Exception:
        return None


def _predict_proba_for_labels(
    model: Any,
    X: pd.DataFrame,
    labels: list[int],
) -> pd.DataFrame:
    probabilities = model.predict_proba(X)
    model_classes = list(getattr(model, "classes_", labels))
    probability_frame = pd.DataFrame(0.0, index=X.index, columns=labels)

    for index, label in enumerate(model_classes):
        if label in probability_frame.columns and index < probabilities.shape[1]:
            probability_frame[label] = probabilities[:, index]

    row_sums = probability_frame.sum(axis=1)
    zero_sum_rows = row_sums == 0
    if zero_sum_rows.any():
        probability_frame.loc[zero_sum_rows, labels] = 1 / len(labels)
        row_sums = probability_frame.sum(axis=1)

    return probability_frame.div(row_sums, axis=0)


def _safe_log_loss(
    y_test: pd.Series,
    probabilities: pd.DataFrame,
    log_loss_fn: Any,
) -> float | None:
    try:
        return float(log_loss_fn(y_test, probabilities[RESULT_LABELS], labels=RESULT_LABELS))
    except Exception:
        return None


def _print_training_report(
    accuracy: float,
    log_loss_value: float | None,
    class_distribution: dict[str, int],
    model_type: str,
) -> None:
    print(f"model_type: {model_type}")
    print(f"accuracy: {accuracy:.4f}")
    if log_loss_value is not None:
        print(f"log_loss: {log_loss_value:.4f}")
    else:
        print("log_loss: unavailable")
    print(f"class_distribution: {class_distribution}")


def _ensure_sklearn_available() -> None:
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Training requires scikit-learn and joblib. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc


if __name__ == "__main__":
    main()
