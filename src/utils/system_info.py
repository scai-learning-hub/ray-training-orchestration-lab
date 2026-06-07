from __future__ import annotations

import os
import socket
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_hostname() -> str:
    return socket.gethostname()


def get_process_id() -> int:
    return os.getpid()


def get_runtime_metadata() -> dict[str, str | int]:
    return {
        "hostname": get_hostname(),
        "process_id": get_process_id(),
        "timestamp": utc_now_iso(),
    }


def format_task_event(
    *,
    job_id: str,
    model_name: str,
    start_time: str,
    end_time: str,
    duration_seconds: float,
    accuracy: float | None,
) -> str:
    return (
        f"job_id={job_id} model_name={model_name} hostname={get_hostname()} "
        f"pid={get_process_id()} start_time={start_time} end_time={end_time} "
        f"training_duration={duration_seconds:.2f}s accuracy="
        f"{accuracy if accuracy is not None else 'n/a'}"
    )
