# Ray Concepts

## Ray Cluster

A Ray cluster is the group of nodes that share resources and execute distributed workloads together.

## Ray Head Node

The head node runs cluster coordination services, accepts job submissions, exposes the dashboard, and usually acts as the driver location for classroom demos.

## Ray Worker Node

A worker node contributes CPUs and memory to the cluster and executes work assigned by Ray.

## Ray Driver

The driver is the Python process that calls `ray.init()`, defines tasks or actors, and submits work to the cluster.

## Ray Task

A Ray Task is a stateless distributed function created with `@ray.remote`.

In this lab:

- Logistic regression training jobs are Ray Tasks
- Random forest training jobs are Ray Tasks
- Evaluation jobs are Ray Tasks

## Ray Actor

A Ray Actor is a stateful distributed service created from a class decorated with `@ray.remote`.

In this lab:

- `TrainingJobTracker` is a Ray Actor
- It keeps job state across updates
- It is used to explain why Actors are stateful while Tasks are stateless

## Object Store

Ray uses a distributed object store to pass data and results between tasks, actors, and drivers efficiently.

## Scheduler

The scheduler decides where work runs based on available CPUs, memory, placement requests, and custom resources.

In the queue demo:

- Jobs request `num_cpus=1`
- If more jobs are submitted than free CPUs, the extra jobs stay pending
- When a running job finishes, the next pending job starts automatically

## Custom Resources

Custom resources are labels that let you restrict where work can run.

In this lab:

- `cpu_worker_1` targets worker node 1
- `cpu_worker_2` targets worker node 2
- `training_worker` labels nodes that are allowed to host distributed training workers

## Ray Train

Ray Train is Ray's distributed training library. It is used when one model needs multiple workers cooperating on the same training job.

In this lab:

- The PyTorch MLP is trained with `TorchTrainer`
- `num_workers=2`
- `use_gpu=False`
- `placement_strategy="SPREAD"`

## ScalingConfig

`ScalingConfig` defines how many workers the distributed training job gets and how those workers are placed.

Important fields in this lab:

- `num_workers=2`
- `use_gpu=False`
- `placement_strategy="SPREAD"`
- `resources_per_worker={"CPU": 1, "training_worker": 1}`

## Placement Strategy

`SPREAD` tells Ray to distribute workers across nodes when possible. That helps the class see that distributed workers are not all landing on one machine.

## Ray Tune

Ray Tune is the experiment orchestration layer for running many configurations and comparing them.

In this lab the search space includes:

- `learning_rate`: `[0.001, 0.005, 0.01]`
- `batch_size`: `[32, 64]`
- `hidden_dim`: `[64, 128]`
- `dropout`: `[0.1, 0.3]`
- `optimizer`: `["adam", "sgd"]`

## Trial Scheduling

Each Tune run is called a trial.

In this lab:

- `max_concurrent_trials=2`
- Only two trials run at a time
- Remaining trials wait until a slot becomes free
- This is a clean way to explain orchestrated experiment concurrency

## Parallel Training vs Distributed Training

Parallel training jobs:

- Many different models train independently
- Each job is separate
- Ray Core Tasks are a good fit

Distributed training:

- One model trains across multiple workers
- The dataset is split across workers
- Gradients are synchronized
- One final model is produced
- Ray Train is the right abstraction
