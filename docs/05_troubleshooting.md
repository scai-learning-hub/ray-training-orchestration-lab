# Troubleshooting

## Worker Cannot Connect To Head

Checks:

- Confirm the head node private IP is correct
- Confirm all nodes are in the same VPC and subnet
- Confirm the security group allows all TCP from the same security group
- Confirm Ray is listening on port `6379`

Useful command:

```bash
ray status
```

## Dashboard Not Opening

Checks:

- Confirm the head node started with `--dashboard-host=0.0.0.0`
- Confirm port `8265` is allowed in the security group
- Confirm the browser is using the head node public IP

## Port Blocked

Checks:

- Security group inbound rules
- Ubuntu firewall rules if `ufw` is enabled

## MLflow Not Opening

Checks:

- Confirm MLflow is running on the head node
- Confirm port `5000` is open from your IP
- Confirm the server was started from the repo root

## Ray Address Issue

Checks:

- Verify `--address='<HEAD_PRIVATE_IP>:6379'`
- Verify the worker uses the head node private IP, not the public IP

## Not Enough CPU

Symptoms:

- Jobs stay pending longer than expected

Checks:

- Run `ray status`
- Verify the number of tasks submitted versus cluster CPUs
- Reduce job count or increase instance size if required

## Object Store Memory Warning

Symptoms:

- Ray prints object store or memory pressure warnings

Checks:

- Reduce dataset size
- Avoid submitting too many large jobs at the same time
- Use the default synthetic dataset size before increasing it

## Dependency Mismatch

Checks:

- Activate `.venv`
- Re-run `./scripts/setup_environment.sh`
- Confirm `pip install -r requirements.txt` completed successfully

## Job Stuck Pending Due To Resources

Checks:

- Confirm the task requests only resources that actually exist in the cluster
- Run `ray status`
- Run `ray list nodes` to inspect resource labels

## Custom Resource Spelling Mismatch

Symptoms:

- Task never starts

Example:

- Worker advertises `cpu_worker_1`
- Task requests `cpu-worker-1`

Fix:

- Make the label text match exactly
