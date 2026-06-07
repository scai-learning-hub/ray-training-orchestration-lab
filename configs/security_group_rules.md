# Security Group Rules

Use one security group for all three EC2 instances so the Ray nodes can communicate internally.

## Inbound Rules

| Type | Protocol | Port Range | Source | Why |
| --- | --- | --- | --- | --- |
| SSH | TCP | 22 | `<YOUR_PUBLIC_IP>/32` | SSH access from the instructor machine |
| Custom TCP | TCP | 8265 | `<YOUR_PUBLIC_IP>/32` | Ray Dashboard UI |
| Custom TCP | TCP | 5000 | `<YOUR_PUBLIC_IP>/32` | MLflow UI |
| All TCP | TCP | 0-65535 | `sg-<THIS_SECURITY_GROUP_ID>` | Internal Ray cluster communication |

## Outbound Rules

| Type | Protocol | Port Range | Destination | Why |
| --- | --- | --- | --- | --- |
| All traffic | All | All | `0.0.0.0/0` | Package install, updates, and cluster traffic |

## Notes

- The `All TCP` rule must reference the same security group, not the public internet.
- Keep ports `8265` and `5000` restricted to the instructor IP.
- If the browser cannot reach the dashboard or MLflow, verify both the security group and the instance-level Ubuntu firewall.
