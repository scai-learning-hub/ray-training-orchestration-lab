from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import joblib
import mlflow

from src.utils.common import resolve_tracking_uri

try:
    import torch
except ImportError:  # pragma: no cover - torch might not be installed during static inspection
    torch = None


EXPERIMENT_NAME = "ray_cpu_training_orchestration_lab"


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    resolved_uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)
    return resolved_uri


@contextmanager
def start_training_run(
    run_name: str,
    tracking_uri: str | None = None,
    experiment_name: str = EXPERIMENT_NAME,
    tags: dict[str, Any] | None = None,
    description: str | None = None,
) -> Iterator[Any | None]:
    active_run = None
    try:
        configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
        active_run = mlflow.start_run(run_name=run_name, description=description)
        if tags:
            mlflow.set_tags({key: str(value) for key, value in tags.items()})
        yield active_run
    except Exception as exc:  # pragma: no cover - external MLflow server issues are environment-specific
        print(f"[mlflow] warning: unable to log run '{run_name}': {exc}")
        yield None
    finally:
        if active_run is not None and mlflow.active_run() is not None:
            mlflow.end_run()


def log_params(params: dict[str, Any]) -> None:
    if mlflow.active_run() is None:
        return

    sanitized_params: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            sanitized_params[key] = json.dumps(value)
        else:
            sanitized_params[key] = value
    if sanitized_params:
        mlflow.log_params(sanitized_params)


def log_metrics(metrics: dict[str, float | int], step: int | None = None) -> None:
    if mlflow.active_run() is None:
        return

    for key, value in metrics.items():
        mlflow.log_metric(key, float(value), step=step)


def log_text_artifact(text: str, artifact_file: str) -> None:
    if mlflow.active_run() is None:
        return

    mlflow.log_text(text, artifact_file)


def log_directory_artifacts(directory_path: str | Path, artifact_path: str | None = None) -> None:
    if mlflow.active_run() is None:
        return

    mlflow.log_artifacts(str(directory_path), artifact_path=artifact_path)


def log_artifact_if_exists(artifact_path: str | Path | None, destination: str | None = None) -> None:
    if mlflow.active_run() is None or artifact_path is None:
        return

    artifact = Path(artifact_path)
    if artifact.exists():
        mlflow.log_artifact(str(artifact), artifact_path=destination)


def log_model_artifact(
    model: Any,
    artifact_path: str,
    file_name: str,
    framework: str = "joblib",
) -> None:
    if mlflow.active_run() is None:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(temp_dir)
        model_path = artifact_dir / file_name
        if framework == "joblib":
            joblib.dump(model, model_path)
        elif framework == "torch":
            if torch is None:
                raise RuntimeError("torch is required to log a torch artifact")
            torch.save(model, model_path)
        else:
            raise ValueError(f"Unsupported framework '{framework}'")
        mlflow.log_artifact(str(model_path), artifact_path=artifact_path)
