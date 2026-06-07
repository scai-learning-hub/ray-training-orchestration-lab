# Architecture

## AI Architect View

This lab uses a simple three-node AWS EC2 architecture to isolate the control plane from the execution plane while keeping the environment small enough for a classroom.

## Components

### Head Node

- Runs the Ray head service
- Hosts the Ray scheduler and cluster metadata services
- Exposes Ray Dashboard on port `8265`
- Runs the MLflow server on port `5000`
- Acts as the driver machine where demo commands are launched

### Worker Nodes

- Run Ray worker services only
- Execute remote Ray tasks
- Participate in distributed Ray Train jobs
- Advertise custom labels such as `cpu_worker_1`, `cpu_worker_2`, and `training_worker`

### Training Jobs

- Logistic regression provides a fast baseline
- Random forest provides a heavier CPU workload that makes scheduling visible
- PyTorch MLP provides a distributed CPU training example

### Tracking And Observation

- MLflow records experiment parameters, metrics, and artifacts
- Ray Dashboard provides cluster, job, task, actor, and log visibility

## Text Diagram

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
                        ----------------------------------------------------------
                        |                                                        |
         +--------------v---------------+                       +----------------v---------------+
         | Worker Node 1                |                       | Worker Node 2                  |
         |------------------------------|                       |-------------------------------|
         | Ray Worker                   |                       | Ray Worker                    |
         | cpu_worker_1 = 1             |                       | cpu_worker_2 = 1              |
         | training_worker = 1          |                       | training_worker = 1           |
         | Executes Task / Train work   |                       | Executes Task / Train work    |
         +------------------------------+                       +-------------------------------+
```

## Runtime Flow

1. The head node starts Ray and MLflow.
2. Workers join using the head node private IP.
3. The driver submits Ray Core tasks for independent model training jobs.
4. The scheduler places tasks on workers based on free CPUs and custom resource labels.
5. The `TrainingJobTracker` actor keeps state while tasks remain stateless.
6. Ray Train launches a distributed PyTorch job across two `training_worker` nodes.
7. Ray Tune launches multiple trial runs with limited concurrency.
8. All results are observed in Ray Dashboard and MLflow.

## Why This Architecture Works For A Classroom

- Small enough to set up in one session
- Large enough to demonstrate real cluster behavior
- Uses CPU only, which avoids GPU setup complexity
- Makes queueing and scheduling behavior easy to visualize
- Keeps the control plane and execution plane conceptually separate

## Optional Extensions

Possible follow-on architecture topics include Kubernetes, KubeRay, DVC, Dagster, and KServe, but they are intentionally out of scope for this repository. This lab focuses only on Ray plus AWS EC2 CPU nodes plus MLflow.
