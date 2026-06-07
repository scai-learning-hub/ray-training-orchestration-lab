# Dashboard Observation Guide

Open the Ray Dashboard before starting the demos:

`http://<HEAD_PUBLIC_IP>:8265`

## Before Any Jobs

Show:

- Nodes page
- Cluster resources
- Low CPU utilization
- No active tasks

Students should observe:

- Three connected nodes
- Idle cluster state
- Dashboard is the control-room view of the cluster

## During Scheduler Queue Demo

Show:

- Jobs view
- Tasks view
- CPU utilization increase
- Pending tasks alongside running tasks
- Logs for finished and active jobs

Students should observe:

- The first jobs consume available CPUs immediately
- Extra jobs remain pending
- As tasks finish, pending tasks become running
- Queueing is automatic and resource-driven

## During Custom Resource Demo

Show:

- Task placement across nodes
- Node resources page
- Task logs with hostnames

Students should observe:

- Work is placed only on nodes with matching labels
- Wrong labels would keep work pending
- Placement rules are explicit and visible

## During Ray Train Distributed CPU Training

Show:

- Actors view
- Worker logs
- CPU usage across both worker nodes

Students should observe:

- Ray Train launches worker actors
- Both workers participate in the same training job
- The workload is coordinated, not just independently parallel

## During Ray Tune

Show:

- Trial activity in jobs and tasks
- Logs for trial runs
- Resource usage over time

Students should observe:

- Only the configured number of trials run at once
- Remaining trials wait for slots
- New trials start after older trials finish

## Dashboard Areas To Highlight

- Nodes
- Jobs
- Tasks
- Actors
- Logs
- Metrics
- CPU and memory usage
