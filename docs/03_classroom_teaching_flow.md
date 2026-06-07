# Classroom Teaching Flow

## Phase 1: Cluster Setup

Goal: show how the head node and workers form one Ray cluster.

Steps:

1. Show the three EC2 instances in AWS.
2. Start the Ray head node.
3. Start MLflow.
4. Start worker 1 with `cpu_worker_1`.
5. Start worker 2 with `cpu_worker_2`.
6. Run `ray status`.

Talking points:

- The head coordinates the cluster
- Workers contribute CPUs and run jobs
- Custom resources are labels, not physical hardware types

## Phase 2: Scheduler Queue Demo

Goal: show that Ray queues extra work automatically.

Command:

```bash
./scripts/run_scheduler_demo.sh
```

Talking points:

- Submit more jobs than available CPUs
- First jobs start immediately
- Extra jobs stay pending
- When one finishes, the next pending job starts automatically
- `TrainingJobTracker` keeps state while tasks remain stateless

## Phase 3: Custom Resource Scheduling

Goal: show placement control.

Command:

```bash
./scripts/run_custom_resource_demo.sh
```

Talking points:

- `cpu_worker_1` jobs only run where that label exists
- `cpu_worker_2` jobs only run where that label exists
- `training_worker` is the shared label for distributed training nodes
- A spelling mismatch causes tasks to stay pending

## Phase 4: Distributed CPU PyTorch Training

Goal: explain one model training across multiple workers.

Command:

```bash
./scripts/run_distributed_train.sh
```

Talking points:

- This is not many models in parallel
- This is one model distributed across two workers
- Batches are processed on both workers
- Gradients are synchronized
- One final model checkpoint is produced

## Phase 5: Ray Tune Hyperparameter Orchestration

Goal: explain controlled experiment concurrency.

Command:

```bash
./scripts/run_tune.sh
```

Talking points:

- Many candidate configurations exist
- Only two trials run at the same time
- The next pending trial starts when one finishes
- MLflow tracks the trials

## Phase 6: MLflow Tracking

Goal: compare runs and explain experiment metadata.

Show in MLflow:

- Baseline sklearn runs
- Distributed PyTorch run
- Tune trial runs
- Hyperparameters and metrics
- Model artifacts where available

## Phase 7: Final Architecture Recap

Close with these questions:

1. Where does scheduling happen?
2. Why do some jobs stay pending?
3. What is the difference between a Task and an Actor?
4. What is the difference between parallel jobs and distributed training?
5. Why are custom resources useful?
6. Why does Tune use controlled concurrency?
