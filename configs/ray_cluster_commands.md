# Ray Cluster Commands

## Start Head Node

```bash
ray start \
  --head \
  --node-ip-address=<HEAD_PRIVATE_IP> \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --resources='{"head_node": 1}'
```

## Start Worker 1

```bash
ray start \
  --address='<HEAD_PRIVATE_IP>:6379' \
  --node-ip-address=<WORKER_1_PRIVATE_IP> \
  --resources='{"cpu_worker_1": 1, "training_worker": 1}'
```

## Start Worker 2

```bash
ray start \
  --address='<HEAD_PRIVATE_IP>:6379' \
  --node-ip-address=<WORKER_2_PRIVATE_IP> \
  --resources='{"cpu_worker_2": 1, "training_worker": 1}'
```

## Inspect Cluster Health

```bash
ray status
ray list nodes
ray list actors
ray list tasks
```

## Stop Ray Cleanly

```bash
ray stop
```

## Demo Commands

Scheduler queue demo:

```bash
./scripts/run_scheduler_demo.sh
```

Custom resource demo:

```bash
./scripts/run_custom_resource_demo.sh
```

Distributed CPU PyTorch training:

```bash
./scripts/run_distributed_train.sh
```

Ray Tune orchestration:

```bash
./scripts/run_tune.sh
```

## What To Watch In Terminal

- `ray status` while the scheduler demo is running to see resource pressure.
- `ray list tasks` to identify active and recently finished work.
- `ray list actors` to observe `TrainingJobTracker` and Ray Train worker actors.
