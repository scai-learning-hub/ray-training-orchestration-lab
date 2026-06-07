# AWS EC2 Setup

## Target Topology

- Head node: `t3.large`, Ubuntu 22.04 LTS
- Worker node 1: `t3.large`, Ubuntu 22.04 LTS
- Worker node 2: `t3.large`, Ubuntu 22.04 LTS

## Launch Steps

1. Open AWS EC2 console.
2. Launch three Ubuntu 22.04 LTS instances in the same VPC and subnet.
3. Attach the same security group to all three instances.
4. Use the same key pair on all instances.
5. Tag the instances clearly: `ray-head`, `ray-worker-1`, `ray-worker-2`.

## How To Put The Project On The EC2 Nodes

You have three reasonable choices:

### Option A: Public Git Repo

- Simplest for classroom labs
- Every node clones the same URL

### Option B: Private Git Repo

- Better if the repo should not be public
- Use SSH keys or token-authenticated HTTPS clone

### Option C: Manual Copy

- Copy the project folder or a zip file to each VM
- Use `scp`, WinSCP, or an S3 download step

Recommended classroom path:

- Use a single Git repository
- Public if convenience matters most
- Private if access control matters more
- Pin all nodes to the same branch, tag, or commit

## Ubuntu 22.04 Commands On Every Node

If you are using Git:

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip build-essential
git clone https://github.com/scai-learning-hub/ray-training-orchestration-lab.git ray-cpu-training-orchestration-lab
cd ray-cpu-training-orchestration-lab
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

If the project was copied manually to the VM already:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential
cd ray-cpu-training-orchestration-lab
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

## Discover Private IPs

Run this on each instance:

```bash
hostname -I | awk '{print $1}'
```

Record:

- `HEAD_PRIVATE_IP`
- `WORKER_1_PRIVATE_IP`
- `WORKER_2_PRIVATE_IP`

## Start Sequence

1. Start Ray head on the head node.
2. Start MLflow on the head node.
3. Join worker 1 with resource label `cpu_worker_1`.
4. Join worker 2 with resource label `cpu_worker_2`.
5. Run the lab demos from the head node.

## Head Node Commands

```bash
cd ray-cpu-training-orchestration-lab
source .venv/bin/activate
./scripts/start_ray_head.sh <HEAD_PRIVATE_IP>
./scripts/start_mlflow.sh
```

Open a second head-node terminal for the MLflow command because it runs in the foreground.

## Worker Commands

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

## Verify Cluster State

On the head node:

```bash
ray status
ray list nodes
```

## URLs

- Ray Dashboard: `http://<HEAD_PUBLIC_IP>:8265`
- MLflow UI: `http://<HEAD_PUBLIC_IP>:5000`
