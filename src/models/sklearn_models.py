from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset_utils import DatasetBundle


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    accuracy: float
    loss: float | None
    duration_seconds: float
    params: dict[str, Any]
    y_true: np.ndarray
    predictions: np.ndarray
    model: Any | None = None
    artifact_path: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "duration_seconds": self.duration_seconds,
            "params": self.params,
            "y_true": self.y_true,
            "predictions": self.predictions,
            "artifact_path": self.artifact_path,
        }


def train_logistic_regression(
    bundle: DatasetBundle,
    max_iter: int = 250,
    c_value: float = 1.0,
    random_state: int = 42,
) -> TrainingResult:
    start_time = time.perf_counter()
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    C=c_value,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(bundle.X_train, bundle.y_train)
    predictions = pipeline.predict(bundle.X_test)
    probabilities = pipeline.predict_proba(bundle.X_test)
    duration_seconds = time.perf_counter() - start_time
    accuracy = float(accuracy_score(bundle.y_test, predictions))
    loss = float(log_loss(bundle.y_test, probabilities))
    return TrainingResult(
        model_name="logistic_regression",
        accuracy=accuracy,
        loss=loss,
        duration_seconds=duration_seconds,
        params={
            "max_iter": max_iter,
            "c_value": c_value,
            "solver": "lbfgs",
            "random_state": random_state,
        },
        y_true=bundle.y_test,
        predictions=predictions,
        model=pipeline,
    )


def train_random_forest(
    bundle: DatasetBundle,
    n_estimators: int = 120,
    max_depth: int | None = 12,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> TrainingResult:
    start_time = time.perf_counter()

    # Each Ray task already reserves one CPU, so the model itself stays single-threaded.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(bundle.X_train, bundle.y_train)
    predictions = model.predict(bundle.X_test)
    probabilities = model.predict_proba(bundle.X_test)
    duration_seconds = time.perf_counter() - start_time
    accuracy = float(accuracy_score(bundle.y_test, predictions))
    loss = float(log_loss(bundle.y_test, probabilities))
    return TrainingResult(
        model_name="random_forest",
        accuracy=accuracy,
        loss=loss,
        duration_seconds=duration_seconds,
        params={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state,
            "n_jobs": 1,
        },
        y_true=bundle.y_test,
        predictions=predictions,
        model=model,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    accuracy = float(accuracy_score(y_true, predictions))
    positive_rate = float(np.mean(predictions))
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "positive_prediction_rate": positive_rate,
        "sample_count": int(len(y_true)),
    }
