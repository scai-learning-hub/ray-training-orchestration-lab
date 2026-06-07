from __future__ import annotations

import argparse
import json
import socket

import ray
from ray import tune
from ray.train import RunConfig

from src.data.dataset_utils import load_or_generate_bundle
from src.models.pytorch_mlp import train_single_process_mlp
from src.tracking.mlflow_utils import log_artifact_if_exists, log_metrics, log_params, log_text_artifact, start_training_run
from src.utils.common import resolve_tracking_uri, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ray Tune hyperparameter orchestration for a CPU-only PyTorch MLP.")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address. Use 'local' for a local runtime.")
    parser.add_argument("--rows", type=int, default=40_000, help="Synthetic dataset size for Tune trials.")
    parser.add_argument("--features", type=int, default=50, help="Synthetic dataset feature count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per trial.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of sampled trial configurations.")
    parser.add_argument("--max-concurrent-trials", type=int, default=2, help="Maximum concurrent Ray Tune trials.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Prefer the head node private IP when workers log remotely.",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow tracking for Tune trials.")
    return parser.parse_args()


def _init_ray(ray_address: str) -> None:
    if ray_address.lower() == "local":
        ray.init(ignore_reinit_error=True, log_to_driver=True)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)


def tune_trainable(
    config: dict[str, object],
    X_train,
    y_train,
    X_valid,
    y_valid,
    seed: int,
    tracking_uri: str | None,
    disable_mlflow: bool,
) -> None:
    context = tune.get_context()
    trial_id = context.get_trial_id() if context else "trial"
    hostname = socket.gethostname()
    set_global_seeds(seed + (abs(hash(trial_id)) % 10_000))

    result = train_single_process_mlp(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        hidden_dim=int(config["hidden_dim"]),
        second_hidden_dim=64,
        dropout=float(config["dropout"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        optimizer_name=str(config["optimizer"]),
        num_epochs=int(config["num_epochs"]),
        device="cpu",
        save_artifact=True,
        run_id=trial_id,
    )

    if not disable_mlflow:
        with start_training_run(
            run_name=f"tune_{trial_id}",
            tracking_uri=tracking_uri,
            tags={"runtime": "ray_tune", "trial_id": trial_id, "hostname": hostname},
        ):
            log_params(
                {
                    "model_name": "pytorch_mlp",
                    "learning_rate": config["learning_rate"],
                    "batch_size": config["batch_size"],
                    "hidden_dim": config["hidden_dim"],
                    "dropout": config["dropout"],
                    "optimizer": config["optimizer"],
                    "num_epochs": config["num_epochs"],
                    "hostname": hostname,
                }
            )
            for epoch_metrics in result["history"]:
                log_metrics(
                    {
                        "train_loss": float(epoch_metrics["train_loss"]),
                        "train_accuracy": float(epoch_metrics["train_accuracy"]),
                        "val_loss": float(epoch_metrics["valid_loss"]),
                        "val_accuracy": float(epoch_metrics["valid_accuracy"]),
                    },
                    step=int(epoch_metrics["epoch"]),
                )
            log_metrics(
                {
                    "final_accuracy": float(result["accuracy"]),
                    "final_loss": float(result["loss"]),
                    "training_time_seconds": float(result["training_time"]),
                }
            )
            log_text_artifact(json.dumps(result["history"], indent=2), f"tune_trials/{trial_id}_history.json")
            log_artifact_if_exists(result["artifact_path"], destination="models/tune_trials")

    tune.report(
        val_accuracy=float(result["accuracy"]),
        val_loss=float(result["loss"]),
        training_time=float(result["training_time"]),
        hostname=hostname,
        trial_id=trial_id,
    )


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    _init_ray(args.ray_address)

    bundle = load_or_generate_bundle(num_rows=args.rows, total_features=args.features, seed=args.seed)
    tracking_uri = resolve_tracking_uri(args.mlflow_tracking_uri)

    print(
        "Ray Tune will schedule more experiment configurations than are allowed to run concurrently. "
        "Only two trials run at a time by default, and the next pending trial starts automatically when one finishes."
    )

    trainable = tune.with_resources(
        tune.with_parameters(
            tune_trainable,
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            X_valid=bundle.X_test,
            y_valid=bundle.y_test,
            seed=args.seed,
            tracking_uri=tracking_uri,
            disable_mlflow=args.disable_mlflow,
        ),
        resources={"cpu": 1},
    )

    param_space = {
        "learning_rate": tune.choice([0.001, 0.005, 0.01]),
        "batch_size": tune.choice([32, 64]),
        "hidden_dim": tune.choice([64, 128]),
        "dropout": tune.choice([0.1, 0.3]),
        "optimizer": tune.choice(["adam", "sgd"]),
        "num_epochs": args.epochs,
    }

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="val_accuracy",
            mode="max",
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
        ),
        run_config=RunConfig(name="ray_tune_pytorch_cpu"),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_accuracy", mode="max")

    print("Ray Tune completed.")
    print("Best trial config:")
    print(best_result.config)
    print("Best trial metrics:")
    print(best_result.metrics)
    print("Use the Ray Dashboard and MLflow UI to watch running trials, pending trials, and per-trial metrics.")


if __name__ == "__main__":
    main()