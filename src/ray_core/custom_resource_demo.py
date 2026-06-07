from __future__ import annotations

import argparse
import time
from typing import Any

import ray

from src.data.dataset_utils import load_or_generate_bundle
from src.models.sklearn_models import train_logistic_regression, train_random_forest
from src.ray_core.training_job_tracker import TrainingJobTracker
from src.tracking.mlflow_utils import log_metrics, log_model_artifact, log_params, start_training_run
from src.utils.common import perform_cpu_burn, resolve_tracking_uri, set_global_seeds
from src.utils.system_info import format_task_event, get_hostname, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate Ray custom resource scheduling on labeled CPU workers.")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address. Use 'local' for local testing.")
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic dataset size.")
    parser.add_argument("--features", type=int, default=50, help="Synthetic dataset feature count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Prefer the head node private IP when workers log remotely.",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--simulate-seconds", type=float, default=4.0, help="Extra CPU burn to keep placement visible.")
    return parser.parse_args()


@ray.remote(num_cpus=1)
def labeled_training_job(
    job_id: str,
    model_name: str,
    tracker: Any,
    dataset_config: dict[str, int],
    model_config: dict[str, Any],
    tracking_uri: str | None,
    disable_mlflow: bool,
    simulate_seconds: float,
    resource_label: str,
) -> dict[str, Any]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    hostname = get_hostname()
    ray.get(tracker.update_job.remote(job_id, status="RUNNING", hostname=hostname))

    bundle = load_or_generate_bundle(**dataset_config)
    if model_name == "logistic_regression":
        result = train_logistic_regression(bundle, **model_config)
    elif model_name == "random_forest":
        result = train_random_forest(bundle, **model_config)
    else:
        raise ValueError(f"Unsupported model name '{model_name}'")

    perform_cpu_burn(simulate_seconds, seed=int(model_config.get("random_state", 42)))
    duration = time.perf_counter() - start_perf
    end_time = utc_now_iso()

    if not disable_mlflow:
        with start_training_run(
            run_name=f"{job_id}_{model_name}",
            tracking_uri=tracking_uri,
            tags={"job_id": job_id, "model_name": model_name, "resource_label": resource_label},
        ):
            log_params({**model_config, **dataset_config, "resource_label": resource_label, "hostname": hostname})
            log_metrics(
                {
                    "accuracy": result.accuracy,
                    "loss": result.loss or 0.0,
                    "training_time_seconds": duration,
                }
            )
            log_model_artifact(
                model=result.model,
                artifact_path=f"models/{model_name}",
                file_name=f"{job_id}.joblib",
                framework="joblib",
            )

    ray.get(
        tracker.update_job.remote(
            job_id,
            status="COMPLETED",
            hostname=hostname,
            accuracy=result.accuracy,
            training_time=duration,
        )
    )
    print(
        format_task_event(
            job_id=job_id,
            model_name=f"{model_name}@{resource_label}",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            accuracy=result.accuracy,
        )
    )
    return {
        "job_id": job_id,
        "model_name": model_name,
        "resource_label": resource_label,
        "hostname": hostname,
        "accuracy": result.accuracy,
        "training_time": duration,
    }


def _init_ray(ray_address: str) -> None:
    if ray_address.lower() == "local":
        ray.init(ignore_reinit_error=True, log_to_driver=True)
    else:
        ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    _init_ray(args.ray_address)

    tracker = TrainingJobTracker.remote()
    dataset_config = {"num_rows": args.rows, "total_features": args.features, "seed": args.seed}
    tracking_uri = resolve_tracking_uri(args.mlflow_tracking_uri)

    print("Cluster resources:", ray.cluster_resources())
    print("Node resource labels:")
    for node in ray.nodes():
        print(node["NodeManagerAddress"], node.get("Resources", {}))

    jobs = [
        ("resource-job-01", "logistic_regression", "cpu_worker_1", {"max_iter": 250, "random_state": args.seed}),
        (
            "resource-job-02",
            "random_forest",
            "cpu_worker_2",
            {"n_estimators": 120, "max_depth": 14, "min_samples_split": 2, "random_state": args.seed + 1},
        ),
        (
            "resource-job-03",
            "random_forest",
            "training_worker",
            {"n_estimators": 100, "max_depth": 12, "min_samples_split": 3, "random_state": args.seed + 2},
        ),
        ("resource-job-04", "logistic_regression", "training_worker", {"max_iter": 300, "random_state": args.seed + 3}),
    ]

    refs = []
    for job_id, model_name, resource_label, model_config in jobs:
        ray.get(tracker.start_job.remote(job_id, model_name))
        ref = labeled_training_job.options(resources={resource_label: 1}).remote(
            job_id,
            model_name,
            tracker,
            dataset_config,
            model_config,
            tracking_uri,
            args.disable_mlflow,
            args.simulate_seconds,
            resource_label,
        )
        refs.append(ref)
        print(f"Submitted {job_id} requiring resource '{resource_label}'.")

    results = ray.get(refs)
    print("Placement results:")
    for item in results:
        print(item)

    print("Tracker summary:")
    ray.get(tracker.print_summary.remote())
    print(
        "If a job stays pending, verify the custom resource name matches the worker label exactly. "
        "Ray will not place a task if the requested label is missing or misspelled."
    )


if __name__ == "__main__":
    main()