from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import ray


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@ray.remote
class TrainingJobTracker:
    """Stateful Ray Actor used by the demos to track training progress across nodes."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, object | None]] = {}

    def start_job(self, job_id: str, model_name: str) -> dict[str, object | None]:
        self.jobs[job_id] = {
            "job_id": job_id,
            "model_name": model_name,
            "status": "SUBMITTED",
            "hostname": None,
            "accuracy": None,
            "training_time": None,
            "error": None,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        return self.jobs[job_id]

    def update_job(
        self,
        job_id: str,
        status: str,
        hostname: str | None = None,
        accuracy: float | None = None,
        training_time: float | None = None,
        error: str | None = None,
    ) -> dict[str, object | None]:
        if job_id not in self.jobs:
            self.start_job(job_id=job_id, model_name="unknown")

        job = self.jobs[job_id]
        job["status"] = status
        job["updated_at"] = utc_now_iso()
        if hostname is not None:
            job["hostname"] = hostname
        if accuracy is not None:
            job["accuracy"] = accuracy
        if training_time is not None:
            job["training_time"] = training_time
        if error is not None:
            job["error"] = error
        return job

    def get_job(self, job_id: str) -> dict[str, object | None] | None:
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[dict[str, object | None]]:
        return [self.jobs[job_id] for job_id in sorted(self.jobs)]

    def print_summary(self) -> list[dict[str, object | None]]:
        rows = self.get_all_jobs()
        counts = Counter(str(row["status"]) for row in rows)
        print("=" * 96)
        print("TrainingJobTracker summary")
        print(f"Total jobs: {len(rows)} | Status counts: {dict(counts)}")
        for row in rows:
            accuracy = row["accuracy"]
            training_time = row["training_time"]
            print(
                f"{row['job_id']:<12} {row['model_name']:<20} {row['status']:<12} "
                f"host={row['hostname'] or 'pending':<18} "
                f"accuracy={accuracy if accuracy is not None else 'n/a':<8} "
                f"time={training_time if training_time is not None else 'n/a':<8} "
                f"error={row['error'] or '-'}"
            )
        print("=" * 96)
        return rows