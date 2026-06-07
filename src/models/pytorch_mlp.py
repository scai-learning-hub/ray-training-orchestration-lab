from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils.common import MODEL_OUTPUT_DIR, ensure_directory


class LoanDefaultDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class LoanDefaultMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        second_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def build_mlp(
    input_dim: int,
    hidden_dim: int = 128,
    second_hidden_dim: int = 64,
    dropout: float = 0.2,
) -> LoanDefaultMLP:
    return LoanDefaultMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        second_hidden_dim=second_hidden_dim,
        dropout=dropout,
    )


def create_data_loader(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = LoanDefaultDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == targets).float().mean().item()
    return {"accuracy": float(accuracy)}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_targets.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == batch_targets).sum().item())
        total_examples += int(batch_targets.size(0))

    return {
        "loss": float(total_loss / max(total_examples, 1)),
        "accuracy": float(total_correct / max(total_examples, 1)),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        logits = model(batch_features)
        loss = criterion(logits, batch_targets)

        total_loss += loss.item() * batch_targets.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == batch_targets).sum().item())
        total_examples += int(batch_targets.size(0))

    return {
        "loss": float(total_loss / max(total_examples, 1)),
        "accuracy": float(total_correct / max(total_examples, 1)),
    }


def persist_torch_model(model: nn.Module, model_name: str, run_id: str, artifact_dir: str | Path | None = None) -> str:
    output_dir = ensure_directory(artifact_dir or MODEL_OUTPUT_DIR)
    artifact_path = output_dir / f"{run_id}_{model_name}.pt"
    torch.save(model.state_dict(), artifact_path)
    return str(artifact_path)


def train_single_process_mlp(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    hidden_dim: int = 128,
    second_hidden_dim: int = 64,
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    num_epochs: int = 6,
    device: str = "cpu",
    save_artifact: bool = False,
    run_id: str = "mlp_run",
) -> dict[str, Any]:
    torch_device = torch.device(device)
    model = build_mlp(
        input_dim=int(X_train.shape[1]),
        hidden_dim=hidden_dim,
        second_hidden_dim=second_hidden_dim,
        dropout=dropout,
    ).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = create_data_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = create_data_loader(X_valid, y_valid, batch_size=batch_size, shuffle=False)

    history: list[dict[str, float]] = []
    started = perf_counter()
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, torch_device)
        valid_metrics = evaluate_model(model, valid_loader, criterion, torch_device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "valid_loss": valid_metrics["loss"],
                "valid_accuracy": valid_metrics["accuracy"],
            }
        )

    training_time = perf_counter() - started
    artifact_path = persist_torch_model(model, "loan_default_mlp", run_id) if save_artifact else None
    final_metrics = history[-1] if history else {}
    return {
        "history": history,
        "training_time": training_time,
        "accuracy": float(final_metrics.get("valid_accuracy", 0.0)),
        "loss": float(final_metrics.get("valid_loss", 0.0)),
        "artifact_path": artifact_path,
        "state_dict": model.state_dict(),
    }
