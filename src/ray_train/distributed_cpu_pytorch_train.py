from __future__ import annotations

import argparse
import json
import shutil
import socket
import tempfile
import time
from pathlib import Path

import ray
import torch
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader, prepare_model
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset_utils import load_or_generate_bundle
from src.models.pytorch_mlp import LoanDefaultDataset, build_mlp, evaluate_model, train_epoch
from src.tracking.mlflow_utils import (
    log_directory_artifacts,
    log_metrics,
    log_params,
    log_text_artifact,
    start_training_run,
)
from src.utils.common import MODEL_OUTPUT_DIR, ensure_directory, resolve_tracking_uri, set_global_seeds


@ray.remote(num_cpus=0)
class FinalMetricsRecorder:
    def __init__(self) -> None:
        self._latest_metrics: dict[str, object] | None = None

    def update(self, metrics: dict[str, object]) -> None:
        self._latest_metrics = dict(metrics)

    def get_latest(self) -> dict[str, object] | None:
        return None if self._latest_metrics is None else dict(self._latest_metrics)


def _resolve_final_metrics(result, recorder: ray.actor.ActorHandle) -> dict[str, object]:
    if result.metrics:
        return dict(result.metrics)

    recorder_metrics = ray.get(recorder.get_latest.remote())
    if recorder_metrics:
        return recorder_metrics

    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distributed CPU-only PyTorch training with Ray Train.")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address. Use 'local' for a local runtime.")
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic dataset size.")
    parser.add_argument("--features", type=int, default=50, help="Synthetic dataset feature count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Ray Train workers.")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per worker.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="MLP dropout rate.")
    parser.add_argument(
        "--disable-training-worker-resource",
        action="store_true",
        help="Disable the training_worker custom resource requirement for local fallback runs.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Prefer the head node private IP when workers log remotely.",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow tracking for this run.")
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help=(
            "Shared storage URI or filesystem path for Ray Train checkpoints "
            "(for example s3://bucket/path or /mnt/nfs). If omitted, multi-node runs skip checkpoint uploads."
        ),
    )
    return parser.parse_args()


def _init_ray(ray_address: str) -> None:
    if ray_address.lower() == "local":
        ray.init(ignore_reinit_error=True, log_to_driver=True)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)


def train_loop_per_worker(config: dict[str, object]) -> None:
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    hostname = socket.gethostname()
    device = get_device()
    metrics_recorder = config.get("metrics_recorder")
    set_global_seeds(int(config["seed"]) + rank)

    train_dataset = LoanDefaultDataset(config["X_train"], config["y_train"])
    valid_dataset = LoanDefaultDataset(config["X_valid"], config["y_valid"])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    train_loader = prepare_data_loader(train_loader)
    valid_loader = prepare_data_loader(valid_loader)

    model = build_mlp(
        input_dim=int(config["input_dim"]),
        hidden_dim=128,
        second_hidden_dim=64,
        dropout=float(config["dropout"]),
    )
    model = prepare_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    for epoch in range(1, int(config["epochs"]) + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_metrics = evaluate_model(model, valid_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "loss": valid_metrics["loss"],
            "accuracy": valid_metrics["accuracy"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "rank": rank,
            "world_size": world_size,
        }
        print(
            f"rank={rank} world_size={world_size} hostname={hostname} device={device} "
            f"epoch={epoch} loss={valid_metrics['loss']:.4f} accuracy={valid_metrics['accuracy']:.4f}"
        )

        if rank == 0 and metrics_recorder is not None:
            ray.get(metrics_recorder.update.remote(epoch_metrics))

        if epoch == int(config["epochs"]) and rank == 0 and bool(config.get("emit_checkpoint", False)):
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_dir = Path(temp_dir)
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, checkpoint_dir / "model_state.pt")
                (checkpoint_dir / "worker_metadata.json").write_text(
                    json.dumps(
                        {
                            "hostname": hostname,
                            "rank": rank,
                            "world_size": world_size,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
                train.report(epoch_metrics, checkpoint=checkpoint)
        else:
            train.report(epoch_metrics)


def _copy_checkpoint_to_outputs(checkpoint: Checkpoint | None, destination_dir: Path) -> Path | None:
    if checkpoint is None:
        return None

    destination = ensure_directory(destination_dir)
    checkpoint_dir = Path(checkpoint.to_directory())
    for source in checkpoint_dir.iterdir():
        shutil.copy2(source, destination / source.name)
    return destination


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    _init_ray(args.ray_address)

    bundle = load_or_generate_bundle(num_rows=args.rows, total_features=args.features, seed=args.seed)
    tracking_uri = resolve_tracking_uri(args.mlflow_tracking_uri)

    print(
        "This demo shows distributed training of one PyTorch MLP across two CPU workers. "
        "That is different from the Ray Core scheduler demo, where multiple independent models run in parallel."
    )

    resources_per_worker: dict[str, float] = {"CPU": 1}
    if not args.disable_training_worker_resource:
        resources_per_worker["training_worker"] = 1

    emit_checkpoint = bool(args.storage_path) or args.ray_address.lower() == "local"
    if not emit_checkpoint:
        print(
            "No shared Ray Train storage path configured. "
            "This cluster run will skip Ray Train checkpoint uploads and log metrics only."
        )

    run_config_kwargs: dict[str, object] = {"name": "ray_cpu_distributed_train"}
    if args.storage_path:
        run_config_kwargs["storage_path"] = args.storage_path

    final_metrics_recorder = FinalMetricsRecorder.remote()

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "X_train": bundle.X_train,
            "y_train": bundle.y_train,
            "X_valid": bundle.X_test,
            "y_valid": bundle.y_test,
            "input_dim": bundle.input_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "seed": args.seed,
            "emit_checkpoint": emit_checkpoint,
            "metrics_recorder": final_metrics_recorder,
        },
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=False,
            placement_strategy="SPREAD",
            resources_per_worker=resources_per_worker,
        ),
        run_config=RunConfig(**run_config_kwargs),
    )

    started = time.perf_counter()
    result = trainer.fit()
    total_duration = time.perf_counter() - started
    final_metrics = _resolve_final_metrics(result, final_metrics_recorder)

    print("Distributed training completed.")
    print("Result metrics:")
    if result.metrics is None and final_metrics:
        print("Recovered final metrics from in-run reports because Ray Train checkpointing is disabled.")
    print(final_metrics)

    checkpoint_output = _copy_checkpoint_to_outputs(
        result.checkpoint,
        MODEL_OUTPUT_DIR / "distributed_train",
    )

    if not args.disable_mlflow:
        with start_training_run(
            run_name="distributed_cpu_pytorch_train",
            tracking_uri=tracking_uri,
            tags={"runtime": "ray_train", "model_name": "pytorch_mlp"},
        ):
            log_params(
                {
                    "rows": args.rows,
                    "features": args.features,
                    "num_workers": args.num_workers,
                    "use_gpu": False,
                    "placement_strategy": "SPREAD",
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "dropout": args.dropout,
                    "checkpoint_enabled": emit_checkpoint,
                    "requires_training_worker": not args.disable_training_worker_resource,
                    "storage_path": args.storage_path or "not_configured",
                }
            )
            log_metrics(
                {
                    "final_accuracy": float(final_metrics.get("accuracy", 0.0)),
                    "final_loss": float(final_metrics.get("loss", 0.0)),
                    "training_time_seconds": total_duration,
                    "rank": float(final_metrics.get("rank", 0.0)),
                    "world_size": float(final_metrics.get("world_size", args.num_workers)),
                }
            )
            log_text_artifact(json.dumps(final_metrics, indent=2), "distributed_train/metrics.json")
            if checkpoint_output is not None:
                log_directory_artifacts(checkpoint_output, artifact_path="models/distributed_pytorch")

    print("Use the Ray Dashboard to observe Ray Train worker actors, CPU usage, and worker logs during this run.")


if __name__ == "__main__":
    main()
