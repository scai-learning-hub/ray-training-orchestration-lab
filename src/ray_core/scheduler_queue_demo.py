from __future__ import annotations

import argparse
import time
from typing import Any

import ray

from src.data.dataset_utils import load_or_generate_bundle
from src.models.sklearn_models import evaluate_predictions, train_logistic_regression, train_random_forest
from src.ray_core.training_job_tracker import TrainingJobTracker
from src.tracking.mlflow_utils import log_metrics, log_model_artifact, log_params, start_training_run
from src.utils.common import perform_cpu_burn, resolve_tracking_uri, set_global_seeds
from src.utils.system_info import format_task_event, get_hostname, get_process_id, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit more CPU-only Ray training jobs than the cluster can run immediately.")
    parser.add_argument("--ray-address", type=str, default="auto", help="Ray cluster address. Use 'local' for a local runtime.")
    parser.add_argument("--rows", type=int, default=100_000, help="Synthetic dataset size.")
    parser.add_argument("--features", type=int, default=50, help="Synthetic dataset feature count.")
    parser.add_argument("--jobs", type=int, default=12, help="Number of jobs to submit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--simulate-seconds", type=float, default=6.0, help="Additional CPU burn per longer-running job.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Set this to the head node private IP for cluster workers.",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def _log_run_if_enabled(
    *,
    job_id: str,
    model_name: str,
    tracking_uri: str | None,
    disable_mlflow: bool,
    params: dict[str, Any],
    metrics: dict[str, float | int],
    artifact_model: Any | None = None,
    artifact_file_name: str | None = None,
) -> None:
    if disable_mlflow:
        return

    with start_training_run(
        run_name=f"{job_id}_{model_name}",
        tracking_uri=tracking_uri,
        tags={"job_id": job_id, "model_name": model_name, "runtime": "ray_core"},
    ):
        log_params(params)
        log_metrics(metrics)
        if artifact_model is not None and artifact_file_name is not None:
            log_model_artifact(
                model=artifact_model,
                artifact_path=f"models/{model_name}",
                file_name=artifact_file_name,
                framework="joblib",
            )


def _build_dataset_config(args: argparse.Namespace) -> dict[str, int]:
    return {
        "num_rows": args.rows,
        "total_features": args.features,
        "seed": args.seed,
    }


@ray.remote(num_cpus=1)
def train_logistic_regression_task(
    job_id: str,
    tracker: Any,
    dataset_config: dict[str, int],
    model_config: dict[str, Any],
    tracking_uri: str | None,
    disable_mlflow: bool,
    simulate_seconds: float,
) -> dict[str, Any]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    hostname = get_hostname()
    process_id = get_process_id()
    ray.get(tracker.update_job.remote(job_id, status="RUNNING", hostname=hostname))

    try:
        bundle = load_or_generate_bundle(**dataset_config)
        result = train_logistic_regression(bundle, **model_config)
        perform_cpu_burn(simulate_seconds, seed=int(model_config.get("random_state", 42)))
        duration = time.perf_counter() - start_perf
        end_time = utc_now_iso()

        _log_run_if_enabled(
            job_id=job_id,
            model_name=result.model_name,
            tracking_uri=tracking_uri,
            disable_mlflow=disable_mlflow,
            params={**model_config, **dataset_config, "hostname": hostname},
            metrics={
                "accuracy": result.accuracy,
                "loss": result.loss or 0.0,
                "training_time_seconds": duration,
                "process_id": process_id,
            },
            artifact_model=result.model,
            artifact_file_name=f"{job_id}.joblib",
        )

        ray.get(
            tracker.update_job.remote(
                job_id,
                status="TRAINED",
                hostname=hostname,
                accuracy=result.accuracy,
                training_time=duration,
            )
        )
        print(
            format_task_event(
                job_id=job_id,
                model_name=result.model_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                accuracy=result.accuracy,
            )
        )
        payload = result.to_payload()
        payload.update(
            {
                "job_id": job_id,
                "hostname": hostname,
                "process_id": process_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
            }
        )
        return payload
    except Exception as exc:
        duration = time.perf_counter() - start_perf
        ray.get(
            tracker.update_job.remote(
                job_id,
                status="FAILED",
                hostname=hostname,
                training_time=duration,
                error=str(exc),
            )
        )
        raise


@ray.remote(num_cpus=1)
def train_random_forest_task(
    job_id: str,
    tracker: Any,
    dataset_config: dict[str, int],
    model_config: dict[str, Any],
    tracking_uri: str | None,
    disable_mlflow: bool,
    simulate_seconds: float,
) -> dict[str, Any]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    hostname = get_hostname()
    process_id = get_process_id()
    ray.get(tracker.update_job.remote(job_id, status="RUNNING", hostname=hostname))

    try:
        bundle = load_or_generate_bundle(**dataset_config)
        result = train_random_forest(bundle, **model_config)
        perform_cpu_burn(simulate_seconds, seed=int(model_config.get("random_state", 42)))
        duration = time.perf_counter() - start_perf
        end_time = utc_now_iso()

        _log_run_if_enabled(
            job_id=job_id,
            model_name=result.model_name,
            tracking_uri=tracking_uri,
            disable_mlflow=disable_mlflow,
            params={**model_config, **dataset_config, "hostname": hostname},
            metrics={
                "accuracy": result.accuracy,
                "loss": result.loss or 0.0,
                "training_time_seconds": duration,
                "process_id": process_id,
            },
            artifact_model=result.model,
            artifact_file_name=f"{job_id}.joblib",
        )

        ray.get(
            tracker.update_job.remote(
                job_id,
                status="TRAINED",
                hostname=hostname,
                accuracy=result.accuracy,
                training_time=duration,
            )
        )
        print(
            format_task_event(
                job_id=job_id,
                model_name=result.model_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                accuracy=result.accuracy,
            )
        )
        payload = result.to_payload()
        payload.update(
            {
                "job_id": job_id,
                "hostname": hostname,
                "process_id": process_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
            }
        )
        return payload
    except Exception as exc:
        duration = time.perf_counter() - start_perf
        ray.get(
            tracker.update_job.remote(
                job_id,
                status="FAILED",
                hostname=hostname,
                training_time=duration,
                error=str(exc),
            )
        )
        raise


@ray.remote(num_cpus=1)
def evaluate_model_task(job_id: str, tracker: Any, training_result: dict[str, Any]) -> dict[str, Any]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    hostname = get_hostname()
    metrics = evaluate_predictions(
        y_true=training_result["y_true"],
        predictions=training_result["predictions"],
        model_name=str(training_result["model_name"]),
    )
    duration = time.perf_counter() - start_perf
    end_time = utc_now_iso()
    ray.get(
        tracker.update_job.remote(
            job_id,
            status="COMPLETED",
            hostname=hostname,
            accuracy=float(metrics["accuracy"]),
            training_time=float(training_result["duration_seconds"]),
        )
    )
    print(
        format_task_event(
            job_id=job_id,
            model_name=f"{training_result['model_name']}_evaluation",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            accuracy=float(metrics["accuracy"]),
        )
    )
    return metrics


@ray.remote(num_cpus=1)
def simulate_cpu_job(job_id: str, tracker: Any, duration_seconds: float) -> dict[str, Any]:
    start_time = utc_now_iso()
    start_perf = time.perf_counter()
    hostname = get_hostname()
    ray.get(tracker.update_job.remote(job_id, status="RUNNING", hostname=hostname))
    perform_cpu_burn(duration_seconds, seed=abs(hash(job_id)) % 10_000)
    duration = time.perf_counter() - start_perf
    end_time = utc_now_iso()
    ray.get(tracker.update_job.remote(job_id, status="COMPLETED", hostname=hostname, training_time=duration))
    print(
        format_task_event(
            job_id=job_id,
            model_name="synthetic_cpu_hold",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            accuracy=None,
        )
    )
    return {
        "job_id": job_id,
        "model_name": "synthetic_cpu_hold",
        "hostname": hostname,
        "duration_seconds": duration,
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
    dataset_config = _build_dataset_config(args)
    tracking_uri = resolve_tracking_uri(args.mlflow_tracking_uri)

    print("Cluster resources:", ray.cluster_resources())
    print("Available resources before submission:", ray.available_resources())
    print(
        "Submitting more jobs than there are immediately available CPUs demonstrates Ray's queueing behavior. "
        "Only the first set of jobs can run now; the rest remain pending until a worker frees one CPU."
    )

    submitted_refs: dict[Any, tuple[str, str]] = {}
    evaluation_refs: list[Any] = []

    for job_index in range(args.jobs):
        job_id = f"job-{job_index + 1:02d}"
        model_selector = job_index % 4
        if model_selector == 0:
            model_name = "logistic_regression"
            ray.get(tracker.start_job.remote(job_id, model_name))
            ref = train_logistic_regression_task.remote(
                job_id,
                tracker,
                dataset_config,
                {"max_iter": 250, "random_state": args.seed + job_index},
                tracking_uri,
                args.disable_mlflow,
                max(args.simulate_seconds / 2.0, 0.0),
            )
        elif model_selector in {1, 2}:
            model_name = "random_forest"
            ray.get(tracker.start_job.remote(job_id, model_name))
            ref = train_random_forest_task.remote(
                job_id,
                tracker,
                dataset_config,
                {
                    "n_estimators": 80 + (job_index * 10),
                    "max_depth": 10 + model_selector,
                    "min_samples_split": 2 + (job_index % 3),
                    "random_state": args.seed + job_index,
                },
                tracking_uri,
                args.disable_mlflow,
                args.simulate_seconds,
            )
        else:
            model_name = "synthetic_cpu_hold"
            ray.get(tracker.start_job.remote(job_id, model_name))
            ref = simulate_cpu_job.remote(job_id, tracker, args.simulate_seconds)

        submitted_refs[ref] = (job_id, model_name)
        print(f"Submitted {job_id} ({model_name}).")

    completed_count = 0
    while submitted_refs:
        ready_refs, _ = ray.wait(list(submitted_refs.keys()), num_returns=1)
        ready_ref = ready_refs[0]
        job_id, model_name = submitted_refs.pop(ready_ref)
        completed_count += 1
        try:
            training_result = ray.get(ready_ref)
            print(f"Completed {completed_count}/{args.jobs}: {job_id} ({model_name}).")
            print("Available resources after completion:", ray.available_resources())
            if model_name != "synthetic_cpu_hold":
                evaluation_refs.append(evaluate_model_task.remote(job_id, tracker, training_result))
        except Exception as exc:
            print(f"Job {job_id} failed: {exc}")
        ray.get(tracker.print_summary.remote())

    if evaluation_refs:
        evaluation_results = ray.get(evaluation_refs)
        print("Evaluation results:")
        for item in evaluation_results:
            print(item)

    print("Final job tracker state:")
    ray.get(tracker.print_summary.remote())
    print("Use 'ray status' in another terminal while this demo runs to watch pending and running tasks.")


if __name__ == "__main__":
    main()
