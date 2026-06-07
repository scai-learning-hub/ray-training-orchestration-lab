from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may not exist during static inspection
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_OUTPUT_DIR = OUTPUTS_DIR / "datasets"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"
RAY_RESULTS_DIR = OUTPUTS_DIR / "ray_results"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_output_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_tracking_uri(explicit_uri: str | None = None) -> str:
    return explicit_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def humanize_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def perform_cpu_burn(duration_seconds: float, matrix_size: int = 192, seed: int = 42) -> None:
    if duration_seconds <= 0:
        return

    rng = np.random.default_rng(seed)
    left = rng.normal(size=(matrix_size, matrix_size)).astype(np.float32)
    right = rng.normal(size=(matrix_size, matrix_size)).astype(np.float32)
    deadline = time.perf_counter() + duration_seconds
    accumulator = 0.0

    while time.perf_counter() < deadline:
        result = left @ right
        accumulator += float(result[0, 0])
        left = np.roll(left, shift=1, axis=0)

    if accumulator == float("inf"):
        raise RuntimeError("Unexpected infinite accumulator while simulating CPU load")

