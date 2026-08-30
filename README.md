# Ray CPU Training Orchestration Platform on AWS EC2

This repository is a classroom lab for running Ray on AWS EC2 CPU instances only. It covers Ray Core task orchestration, scheduler queueing, custom resource placement, Ray Actor state tracking, Ray Train distributed CPU PyTorch training, Ray Tune hyperparameter orchestration, MLflow experiment tracking, and Ray Dashboard observation.

This lab is intended to run on Ubuntu 22.04 EC2 instances on AWS. It is not designed as a Windows runtime lab.

## What Students Learn

- How a Ray cluster works across one head node and multiple worker nodes.
- How Ray schedules CPU-only jobs when more work is submitted than available CPUs.
- How pending jobs automatically become active when CPUs are freed.
- How custom Ray resources control task placement.
- The difference between Ray Tasks and Ray Actors.
- The difference between parallel independent training and distributed training.
- How Ray Train performs one distributed PyTorch training job across CPU workers.
- How Ray Tune manages many experiments with limited concurrency.
- How MLflow tracks metrics, parameters, and model artifacts.
- How to observe nodes, tasks, actors, and logs in Ray Dashboard.

## Architecture

```text
                         +--------------------------------------+
                         | AWS EC2 Head Node                    |
                         | Ubuntu 22.04                         |
                         | Ray Head + Scheduler + Dashboard     |
                         | MLflow Server                        |
                         | Driver Scripts                       |
                         +-------------------+------------------+
                                             |
                    ----------------------------------------------------------
                    |                                                        |
    +---------------v----------------+                     +------------------v---------------+
    | AWS EC2 Worker Node 1          |                     | AWS EC2 Worker Node 2            |
    | Ubuntu 22.04                   |                     | Ubuntu 22.04                     |
    | Ray Worker                     |                     | Ray Worker                       |
    | cpu_worker_1 = 1               |                     | cpu_worker_2 = 1                 |
    | training_worker = 1            |                     | training_worker = 1              |
    +--------------------------------+                     +----------------------------------+

Workloads:
- Ray Core tasks: independent sklearn training jobs
- Ray Actor: TrainingJobTracker state service
- Ray Train: one distributed CPU PyTorch MLP across 2 workers
- Ray Tune: multiple PyTorch MLP trials with max_concurrent_trials=2
```

## Quick Start

Concepts and theory: [docs/01_training_scale_orchestration.md](docs/01_training_scale_orchestration.md)

Runtime guide: [docs/02_runtime_guide.md](docs/02_runtime_guide.md)

Detailed AWS setup: [configs/aws_ec2_setup.md](configs/aws_ec2_setup.md)

Security group rules: [configs/security_group_rules.md](configs/security_group_rules.md)

Ray cluster commands: [configs/ray_cluster_commands.md](configs/ray_cluster_commands.md)

### How The Code Gets Onto EC2

You do not have to make this repository public.

Recommended classroom options:

1. Public Git repository.
This is the easiest for a classroom because every EC2 node can run one `git clone` command.

2. Private Git repository.
This is usually the best real-world choice. Each EC2 node clones the same private repository using SSH deploy keys or a personal access token.

3. Direct copy without Git hosting.
Zip the project on your machine and copy it to each EC2 instance with `scp` or WinSCP.

Recommended path for this lab:

- Use one GitHub or GitLab repository.
- Public is fine for classroom convenience.
- Private is also fine and usually preferable if you do not want the material exposed.
- Make all three EC2 nodes use the same branch, tag, or commit.

### 1. Launch EC2 Instances

- 1 head node: `t3.large`
- 2 worker nodes: `t3.large`
- OS: Ubuntu 22.04 LTS
- Attach the same security group to all nodes

### 2. Install the Project on All Three Nodes

If you are using Git:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip build-essential
git clone https://github.com/scai-learning-hub/ray-training-orchestration-lab.git ray-cpu-training-orchestration-lab
cd ray-cpu-training-orchestration-lab
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

If you are not using Git hosting, copy the project folder to each EC2 VM first, then run:

```bash
cd ray-cpu-training-orchestration-lab
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

### 3. Start the Head Node

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_head.sh <HEAD_PRIVATE_IP>
```

### 4. Start MLflow on the Head Node

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_mlflow.sh
```

### 5. Start the Workers

Worker 1:

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_worker.sh <HEAD_PRIVATE_IP> cpu_worker_1 <WORKER_1_PRIVATE_IP>
```

Worker 2:

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_worker.sh <HEAD_PRIVATE_IP> cpu_worker_2 <WORKER_2_PRIVATE_IP>
```

### 6. Validate the Cluster

```bash
ray status
ray list nodes
```

### 7. Run the Demos From the Head Node

Scheduler queue demo:

```bash
./scripts/run_scheduler_demo.sh
```

Custom resource placement demo:

```bash
./scripts/run_custom_resource_demo.sh
```

Distributed CPU PyTorch training with Ray Train:

```bash
./scripts/run_distributed_train.sh
```

Hyperparameter orchestration with Ray Tune:

```bash
./scripts/run_tune.sh
```

## URLs

- Ray Dashboard: `http://<HEAD_PUBLIC_IP>:8265`
- MLflow UI: `http://<HEAD_PUBLIC_IP>:5000`

## What To Observe

### Ray Dashboard

- Nodes joining the cluster
- Cluster CPU resources
- Running tasks and pending tasks
- `TrainingJobTracker` actor
- Ray Train worker actors
- Ray Tune trials
- Driver logs and worker logs

### MLflow UI

- Experiment: `ray_cpu_training_orchestration_lab`
- Runs for sklearn jobs
- Distributed PyTorch run
- Ray Tune trial runs
- Parameters, metrics, and artifacts

## Recommended Demo Order

1. Open Ray Dashboard before any jobs are submitted.
2. Run the scheduler queue demo and watch pending tasks become active automatically.
3. Run the custom resource demo and show placement labels.
4. Run distributed PyTorch CPU training with Ray Train.
5. Run Ray Tune with `max_concurrent_trials=2`.
6. Open MLflow UI and compare the runs.

## Documentation Map

- Concepts and theory: [docs/01_training_scale_orchestration.md](docs/01_training_scale_orchestration.md)
- Runtime guide: [docs/02_runtime_guide.md](docs/02_runtime_guide.md)
