# AWS Deployment And Run Guide

This is the direct operator guide for deploying and running the lab on AWS EC2.

## Target Topology

- Head node: `t3.large`, Ubuntu 22.04 LTS
- Worker node 1: `t3.large`, Ubuntu 22.04 LTS
- Worker node 2: `t3.large`, Ubuntu 22.04 LTS

All three instances must be in the same VPC and subnet and share the same security group.

## Required Security Group Rules

- SSH `22` from your public IP only
- Ray Dashboard `8265` from your public IP only
- MLflow UI `5000` from your public IP only
- All TCP from the same security group for internal Ray communication

Detailed rule table: [../configs/security_group_rules.md](../configs/security_group_rules.md)

## Step 1: Launch The EC2 Instances

1. Open the EC2 console.
2. Launch three Ubuntu 22.04 instances.
3. Select `t3.large` for each node.
4. Attach the same key pair and same security group.
5. Tag the nodes as `ray-head`, `ray-worker-1`, and `ray-worker-2`.

## Step 2: Decide How The Code Reaches EC2

You do not have to put the repository on a public Git server.

Choose one of these approaches:

### Option A: Public Git Repository

- Easiest for classroom setup
- Every node runs the same `git clone` command
- Best when the lab content can be public

### Option B: Private Git Repository

- Best for real projects or private classroom material
- Every node clones the same private repo
- Use SSH keys or a token-based HTTPS clone

### Option C: Copy The Project Directly

- No Git hosting required
- Zip the repository and copy it to each EC2 VM with `scp` or WinSCP
- Good when you want a quick one-off lab without setting up repo access

Recommended for this lab:

- Public repo if you want the simplest classroom flow
- Private repo if you want controlled access
- In both cases, make sure all three nodes use the same branch, tag, or commit

## Step 3: Install The Lab On Every Node

Run the same commands on the head node and both workers.

If you are using Git:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip build-essential
git clone https://github.com/scai-learning-hub/ray-training-orchestration-lab.git ray-cpu-training-orchestration-lab
cd ray-cpu-training-orchestration-lab
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

If you copied the project manually instead of cloning it:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential
cd ray-cpu-training-orchestration-lab
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

## Step 4: Discover The Private IPs

Run on each node:

```bash
hostname -I | awk '{print $1}'
```

Record these values:

- `HEAD_PRIVATE_IP`
- `WORKER_1_PRIVATE_IP`
- `WORKER_2_PRIVATE_IP`

## Step 5: Start Ray Head On The Head Node

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_head.sh <HEAD_PRIVATE_IP>
```

Equivalent raw command:

```bash
ray start --head --node-ip-address=<HEAD_PRIVATE_IP> --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --resources='{"head_node": 1}'
```

## Step 6: Start MLflow On The Head Node

Open another terminal on the head node and run:

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_mlflow.sh
```

Equivalent raw command:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

## Step 7: Join The Workers

Worker 1:

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_worker.sh <HEAD_PRIVATE_IP> cpu_worker_1 <WORKER_1_PRIVATE_IP>
```

Equivalent raw command:

```bash
ray start --address='<HEAD_PRIVATE_IP>:6379' --node-ip-address=<WORKER_1_PRIVATE_IP> --resources='{"cpu_worker_1": 1, "training_worker": 1}'
```

Worker 2:

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_worker.sh <HEAD_PRIVATE_IP> cpu_worker_2 <WORKER_2_PRIVATE_IP>
```

Equivalent raw command:

```bash
ray start --address='<HEAD_PRIVATE_IP>:6379' --node-ip-address=<WORKER_2_PRIVATE_IP> --resources='{"cpu_worker_2": 1, "training_worker": 1}'
```

## Step 8: Verify The Cluster

On the head node:

```bash
ray status
ray list nodes
ray list actors
```

You should see three nodes in the cluster.

## Step 9: Open The UIs

- Ray Dashboard: `http://<HEAD_PUBLIC_IP>:8265`
- MLflow UI: `http://<HEAD_PUBLIC_IP>:5000`

Open both before starting the demos.

## Step 10: Run The Classroom Demos

### Scheduler Queue Demo

```bash
./scripts/run_scheduler_demo.sh
```

What this proves:

- More jobs are submitted than the cluster can run immediately
- Running jobs consume the CPUs first
- Extra jobs remain pending
- Ray automatically schedules the next pending job when a CPU becomes free

### Custom Resource Scheduling Demo

```bash
./scripts/run_custom_resource_demo.sh
```

What this proves:

- `cpu_worker_1` jobs only run on worker 1
- `cpu_worker_2` jobs only run on worker 2
- `training_worker` can be used as a label for distributed training workers

### Distributed CPU PyTorch Training

```bash
./scripts/run_distributed_train.sh
```

What this proves:

- One PyTorch MLP is trained across two workers
- Dataset batches are split across workers
- Gradients are synchronized
- A single final model is produced

### Ray Tune Hyperparameter Orchestration

```bash
./scripts/run_tune.sh
```

What this proves:

- Many trial configurations exist
- Only two trials run at the same time by default
- The next pending trial starts automatically when a previous one finishes
- All trials log to MLflow

## Step 11: Useful Commands During The Lab

```bash
ray status
ray list nodes
ray list tasks
ray list actors
```

## When You Are Done

Cleanup guide: [06_cleanup.md](06_cleanup.md)
