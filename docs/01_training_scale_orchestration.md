# Training-Scale Orchestration: Concepts And This Lab

This document is the theory layer for the lab. It explains why training-scale
orchestration matters, the core Ray concepts behind it, and how each piece maps
to the code in this repository. Read this once before running the runtime guide.

## Why Orchestration For Training At Scale

A single machine trains a single model fine. Real training-scale problems look
different:

- Many models must be trained and compared.
- Some models are too big for one process and must be split across workers.
- Hyperparameter search creates many trial runs that compete for limited CPUs.
- Jobs must be routed to the right machines for policy or resource reasons.
- Every run must be tracked so results are reproducible and comparable.

Doing this by hand does not scale. You need three things:

1. A scheduler that places work across machines.
2. A training layer that can run one model across many workers.
3. A tracking layer that records every run.

This lab uses Ray for the first two and MLflow for the third.

## Frameworks In This Lab

| Layer      | Tool        | Role In This Lab                                      |
|------------|-------------|-------------------------------------------------------|
| Cluster    | Ray Core    | Tasks, actors, scheduling, custom resources           |
| Training   | Ray Train   | One PyTorch model trained across two workers          |
| Tuning     | Ray Tune    | Hyperparameter trials with limited concurrency        |
| Tracking   | MLflow      | Parameters, metrics, artifacts for every run          |
| Platform   | AWS EC2     | Three CPU-only Ubuntu nodes in one VPC                |

Everything in this lab is CPU-only on purpose. The goal is to learn orchestration
behavior, not to maximize training speed.

## Core Ray Concepts

### Cluster

A Ray cluster is a group of nodes that share resources and run distributed
workloads together. In this lab the cluster has one head and two workers.

### Head Node

The head node runs cluster coordination, the Ray scheduler, the Ray Dashboard
on port `8265`, and the MLflow server on port `5000`. Demos are launched from
the head.

### Worker Node

A worker node contributes CPUs and memory and executes work assigned by Ray.
Workers do not run the dashboard or MLflow. They only join the cluster and
execute tasks.

### Driver

The driver is the Python process that calls `ray.init()`, defines tasks or
actors, and submits work. In this lab the driver runs on the head.

### Task

A Ray Task is a stateless distributed function created with `@ray.remote`.
Logistic regression, random forest, and evaluation jobs in this lab are tasks.
They run, return a result, and keep no state between calls.

### Actor

A Ray Actor is a stateful distributed service created from a class decorated
with `@ray.remote`. The `TrainingJobTracker` actor in this lab keeps job state
across updates. This is the contrast: tasks are stateless, actors are stateful.

### Scheduler

The scheduler decides where work runs based on available CPUs, memory, and
custom resources. In the scheduler demo, more jobs are submitted than free CPUs,
so the extra jobs stay pending until a worker frees a CPU.

### Custom Resources

Custom resources are labels that restrict where work can run. This lab uses:

- `cpu_worker_1` to target worker 1
- `cpu_worker_2` to target worker 2
- `training_worker` to label nodes allowed to host distributed training workers
- `head_node` to label the head

This enables policy-based placement, not just generic scheduling.

## Ray Train

Ray Train is Ray's distributed training library. It is used when one model
needs multiple workers cooperating on the same training job.

In this lab the PyTorch MLP is trained with `TorchTrainer`:

- `num_workers=2`
- `use_gpu=False`
- `placement_strategy="SPREAD"`
- `resources_per_worker={"CPU": 1, "training_worker": 1}`

`SPREAD` places workers on different nodes when possible so the class can see
distributed workers landing on separate machines.

## Ray Tune

Ray Tune runs many configurations and compares them. In this lab the search
space is:

- `learning_rate`: `[0.001, 0.005, 0.01]`
- `batch_size`: `[32, 64]`
- `hidden_dim`: `[64, 128]`
- `dropout`: `[0.1, 0.3]`
- `optimizer`: `["adam", "sgd"]`

Each configuration is one trial. `max_concurrent_trials=2` means only two trials
run at a time and the rest wait. This is orchestrated experiment concurrency.

## Parallel Training vs Distributed Training

This distinction is central to the lab.

Parallel training:

- Many different models train independently
- Each job is separate
- Ray Core tasks are the right fit
- The scheduler and custom-resource demos show this

Distributed training:

- One model trains across multiple workers
- The dataset is split across workers
- Gradients are synchronized
- One final model is produced
- Ray Train is the right fit
- The distributed training demo shows this

Ray Tune is a third pattern: many independent trials, each one a separate
training run, evaluated and compared.

## How The Lab Maps To The Concepts

| Demo                         | Concept Demonstrated                     | Ray Layer    | MLflow Runs        |
|------------------------------|-----------------------------------------|--------------|--------------------|
| Scheduler queue              | Queueing when jobs exceed free CPUs      | Ray Core     | One per model job   |
| Custom resource placement    | Policy-based routing by node label       | Ray Core     | One per placed job  |
| Distributed PyTorch training | One model across two workers             | Ray Train    | One per run         |
| Ray Tune HPO                 | Many trials with limited concurrency     | Ray Tune     | One per trial       |

## Architecture

```text
                               +--------------------------------------+
                               | Head Node                            |
                               |--------------------------------------|
                               | Ray Head                             |
                               | Ray Scheduler                        |
                               | Ray Dashboard :8265                  |
                               | MLflow Server :5000                  |
                               | Driver Scripts                       |
                               +------------------+-------------------+
                                                  |
                    ----------------------------------------------
                    |                                            |
       +------------v-------------+                 +--------------v-------------+
       | Worker Node 1            |                 | Worker Node 2              |
       |--------------------------|                 |----------------------------|
       | Ray Worker               |                 | Ray Worker                 |
       | cpu_worker_1 = 1         |                 | cpu_worker_2 = 1           |
       | training_worker = 1      |                 | training_worker = 1        |
       +--------------------------+                 +----------------------------+
```

## What This Lab Does Not Cover

Kubernetes, KubeRay, DVC, Dagster, KServe, and GPU training are intentionally
out of scope. This lab focuses only on Ray plus AWS EC2 CPU nodes plus MLflow so
the orchestration concepts are visible without extra infrastructure noise.