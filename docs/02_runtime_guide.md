# Runtime Guide

This is the single operational guide for the lab. It covers creating the EC2
nodes, connecting them, starting services, and running the four demos, in
terminal order, with what to watch in Ray and MLflow for each one.

Prerequisites: one AWS account, one key pair, and your current public IP.

## Cluster Topology

- Head node: `t3.large`, Ubuntu 24.04 LTS
- Worker node 1: `t3.large`, Ubuntu 24.04 LTS
- Worker node 2: `t3.large`, Ubuntu 24.04 LTS

All three instances must be in the same VPC, the same subnet, and the same
security group.

Recommended disk: 30 GB root volume per instance so the Python environment plus
CPU-only PyTorch install comfortably fits.

## Security Group Rules

Create one security group and attach it to all three instances.

| Type        | Protocol | Port | Source                       | Purpose                       |
|-------------|----------|------|------------------------------|-------------------------------|
| SSH         | TCP      | 22   | `<your-public-ip>/32`        | Terminal access from your PC  |
| Custom TCP  | TCP      | 8265 | `<your-public-ip>/32`        | Ray Dashboard                 |
| Custom TCP  | TCP      | 5000 | `<your-public-ip>/32`        | MLflow UI                     |
| All TCP     | TCP      | All  | `<this-security-group-id>`   | Internal Ray node-to-node com |

If your home or office public IP changes, update the SSH, 8265, and 5000
sources. Find your current public IP from a browser at `https://checkip.amazonaws.com`.

## Step 1: Launch Three EC2 Instances

1. Open the AWS EC2 console.
2. Launch three Ubuntu 24.04 LTS instances.
3. Choose `t3.large` for each.
4. Set the root volume to 30 GB.
5. Attach the same key pair and the same security group to all three.
6. Tag them as `ray-head`, `ray-worker-1`, and `ray-worker-2`.

After launch, record from the EC2 console:

- `HEAD_PUBLIC_IP`
- `HEAD_PRIVATE_IP`
- `WORKER1_PUBLIC_IP`
- `WORKER1_PRIVATE_IP`
- `WORKER2_PUBLIC_IP`
- `WORKER2_PRIVATE_IP`

## Step 2: Open Five Terminals

You will use five separate terminal windows on your local machine. Each one
SSHes into one node. Only the head runs the demos; the workers only join the
cluster.

| Terminal | Node      | Purpose                              |
|----------|-----------|--------------------------------------|
| 1        | Head      | Setup, start Ray head, run demos     |
| 2        | Worker 1  | Setup, join Ray as `cpu_worker_1`    |
| 3        | Worker 2  | Setup, join Ray as `cpu_worker_2`    |
| 4        | Head      | Run MLflow server                    |
| 5        | Head      | Live Ray tracking                    |

## Step 3: Install The Lab On Every Node

Run the same install block on the head and both workers. Wait for each node to
finish fully before moving on.

Terminal 1 (head):
```bash
ssh -i <your-key>.pem ubuntu@<HEAD_PUBLIC_IP>
git clone https://github.com/scai-learning-hub/ray-training-orchestration-lab.git ray-cpu-training-orchestration-lab
cd ~/ray-cpu-training-orchestration-lab
bash ./scripts/setup_environment.sh
source .venv/bin/activate
```

Terminal 2 (worker 1):
```bash
ssh -i <your-key>.pem ubuntu@<WORKER1_PUBLIC_IP>
git clone https://github.com/scai-learning-hub/ray-training-orchestration-lab.git ray-cpu-training-orchestration-lab
cd ~/ray-cpu-training-orchestration-lab
bash ./scripts/setup_environment.sh
source .venv/bin/activate
```

Terminal 3 (worker 2):
```bash
ssh -i <your-key>.pem ubuntu@<WORKER2_PUBLIC_IP>
git clone https://github.com/scai-learning-hub/ray-training-orchestration-lab.git ray-cpu-training-orchestration-lab
cd ~/ray-cpu-training-orchestration-lab
bash ./scripts/setup_environment.sh
source .venv/bin/activate
```

What this does:
- Installs OS packages
- Creates a Python virtualenv at `.venv`
- Installs Ray, CPU-only PyTorch, scikit-learn, MLflow, and other dependencies
- Creates the output directories

## Step 4: Start Ray Head

Terminal 1 (head):
```bash
bash ./scripts/start_ray_head.sh <HEAD_PRIVATE_IP>
```

What this does: stops any old Ray state and starts the Ray head node with the
dashboard exposed externally.

## Step 5: Join The Workers

Terminal 2 (worker 1):
```bash
bash ./scripts/start_ray_worker.sh <HEAD_PRIVATE_IP> cpu_worker_1 <WORKER1_PRIVATE_IP>
```

Terminal 3 (worker 2):
```bash
bash ./scripts/start_ray_worker.sh <HEAD_PRIVATE_IP> cpu_worker_2 <WORKER2_PRIVATE_IP>
```

What this does: each worker joins the Ray head and advertises its custom
resource labels.

## Step 6: Start MLflow

Terminal 4 (head):
```bash
ssh -i <your-key>.pem ubuntu@<HEAD_PUBLIC_IP>
cd ~/ray-cpu-training-orchestration-lab
source .venv/bin/activate
bash ./scripts/start_mlflow.sh
```

Leave this terminal open. MLflow runs in the foreground.

## Step 7: Start Ray Tracking

Terminal 5 (head):
```bash
ssh -i <your-key>.pem ubuntu@<HEAD_PUBLIC_IP>
cd ~/ray-cpu-training-orchestration-lab
source .venv/bin/activate
watch -n 2 'ray status; echo; ray list nodes'
```

Leave this terminal open. It refreshes cluster state every two seconds.

## Step 8: Verify The Cluster

Terminal 1 (head):
```bash
ray status
ray list nodes
```

You should see:
- 3 alive nodes
- Total CPU: 6
- `training_worker: 2`
- `cpu_worker_1: 1`
- `cpu_worker_2: 1`
- `head_node: 1`

## Step 9: Open The UIs

Open both in your browser before running demos:

- Ray Dashboard: `http://<HEAD_PUBLIC_IP>:8265`
- MLflow UI: `http://<HEAD_PUBLIC_IP>:5000`

Important difference:
- Ray Dashboard shows live cluster activity. Watch it while a demo is running.
- MLflow shows persistent history. Runs remain after demos finish and after Ray
  restarts.

## Step 10: Set MLflow Tracking URI

Terminal 1 (head):
```bash
export MLFLOW_TRACKING_URI=http://<HEAD_PRIVATE_IP>:5000
```

This makes every demo log to the MLflow server running on the head.

## Step 11: Run The Four Demos

Run all four from Terminal 1 (head) only. Run them one at a time and wait for
each to finish before starting the next.

### Demo 1: Scheduler Queue

```bash
bash ./scripts/run_scheduler_demo.sh
```

What it shows: more independent jobs than free CPUs, so Ray queues the extra
jobs and starts them as CPUs free up.

Watch in Ray Dashboard: CPUs fill up, pending tasks appear, then drain.

Watch in MLflow: one run per model-training job, persisted as history.

### Demo 2: Custom Resource Placement

```bash
bash ./scripts/run_custom_resource_demo.sh
```

What it shows: jobs that require specific labels such as `cpu_worker_1`,
`cpu_worker_2`, and `training_worker`, so Ray places them on the matching node.

Watch in Ray Dashboard: tasks land on the intended nodes, not just any free
node.

Watch in MLflow: one run per placed job.

### Demo 3: Distributed PyTorch Training

```bash
bash ./scripts/run_distributed_train.sh
```

What it shows: one PyTorch MLP trained across two `training_worker` workers with
rank 0 and rank 1.

Watch in Ray Dashboard: two training workers reserved together for one
training group.

Watch in MLflow: one run named `distributed_cpu_pytorch_train` with the final
distributed metrics.

### Demo 4: Ray Tune HPO

```bash
bash ./scripts/run_tune.sh
```

What it shows: many hyperparameter trials with only two running concurrently;
the next pending trial starts when one finishes.

Watch in Ray Dashboard: trial workers come and go with controlled concurrency.

Watch in MLflow: one run per trial, each with its own hyperparameters and
metrics, so you can compare trials.

## Step 12: Useful Commands During The Lab

On the head:
```bash
ray status
ray list nodes
ray list tasks
ray list actors
```

## Step 13: Stop Or Tear Down

To stop Ray on a node:
```bash
ray stop
```

To stop MLflow on the head, use Ctrl-C in Terminal 4.

To remove the EC2 nodes entirely, terminate the three instances in the AWS
EC2 console. MLflow history stored on the head is lost when the head is
terminated.

## Quick Mental Model For The Four Demos

| Demo                  | Pattern                          | Ray Layer   | MLflow Runs          |
|-----------------------|----------------------------------|-------------|----------------------|
| Scheduler queue       | Many independent jobs, queueing  | Ray Core    | One per model job    |
| Custom resource       | Many jobs, controlled placement  | Ray Core    | One per placed job   |
| Distributed train     | One model across two workers      | Ray Train   | One per run          |
| Ray Tune              | Many trials, limited concurrency  | Ray Tune    | One per trial        |