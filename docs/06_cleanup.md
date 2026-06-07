# Cleanup

## Stop Ray On Every Node

Run on the head node and both workers:

```bash
ray stop
```

## Stop MLflow On The Head Node

If MLflow is running in the foreground, stop it with `Ctrl+C`.

If needed:

```bash
pkill -f mlflow
```

## Optional Local File Cleanup

```bash
rm -rf mlartifacts
rm -f mlflow.db
```

## Terminate EC2 Instances

In AWS EC2 console:

1. Select the head and worker instances.
2. Choose `Instance state`.
3. Choose `Terminate instance`.

## Delete Security Group If No Longer Needed

Only do this after the instances are terminated and nothing else uses the group.

## End-Of-Lab Reminder

- Stop Ray on all nodes
- Stop MLflow on the head node
- Terminate all three EC2 instances
- Remove or archive artifacts if needed
