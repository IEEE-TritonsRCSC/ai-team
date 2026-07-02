# SSL field networking

Field network parameters (GC multicast, vision multicast, robot command
multicast, and the interface to use) are centralized in
[`networking/net_config.py`](../networking/net_config.py) and can be
overridden at runtime via environment variables — no source changes needed
to adapt to a new match subnet:

| Env var | Default | Purpose |
|---|---|---|
| `SSL_GC_MULTICAST_IP` | `224.5.23.1` | GC referee multicast group |
| `SSL_GC_PORT` | `10003` | GC referee multicast port |
| `SSL_GC_TEAM_PORT` | `10008` | Team -> GC rcon TCP port |
| `SSL_GC_IP_OVERRIDE` | unset | Force the GC IP instead of discovering it from referee packets (useful for local testing) |
| `SSL_GC_AUTH_MODE` | `none` | `none` or `signed` (signed handshake not yet implemented — see `networking/gc_client.py`) |
| `SSL_VISION_MULTICAST_IP` | `224.5.23.2` | SSL-Vision multicast group |
| `SSL_VISION_PORT` | `10006` | SSL-Vision multicast port |
| `SSL_VISION_TRACKER_PORT` | `10010` | Optional Vision Tracker port |
| `ROBOT_COMMAND_MULTICAST_IP` | `239.42.42.42` | Our own robot command multicast group |
| `ROBOT_COMMAND_PORT` | `10000` | Robot command multicast base port |
| `SSL_NETWORK_INTERFACE` | unset (OS picks) | Local IP of the NIC to use for multicast joins/sends — set this to the field ethernet's address when the laptop also has Wi-Fi |

## Manual action items at competition

These are **not** handled by this repo and must be done separately once the
Technical Committee announces the match subnet:

- **Robot / gateway IPs (Arduino firmware or radio bridge credentials)**:
  no `.ino` files or flashing scripts exist in this repo. If robots are
  addressed individually (rather than purely via the `ROBOT_COMMAND_MULTICAST_IP`
  broadcast group), their firmware/credentials must be reflashed or
  reconfigured to the TC's assigned subnet separately.
- **`SSL_NETWORK_INTERFACE`**: set this on the field laptop to the IP of
  the ethernet NIC connected to the field router before starting any
  `field-practice` / `field-tournament` run, if the laptop also has Wi-Fi
  active.
- **`SSL_GC_AUTH_MODE`**: confirm with whoever manages the competition GC
  whether it requires the signed rcon handshake. Do not set this to
  `signed` — `networking/gc_client.py` will raise `NotImplementedError`
  until that handshake is built.
- **`sslclient` interface binding**: SSL-Vision reception goes through the
  external `sslclient` library (`networking/socket_utils.py`), whose
  interface-selection API wasn't available to check in this environment —
  verify at the venue whether it needs its own interface override (e.g. via
  `ip route` on the field laptop) if vision packets aren't received on the
  first attempt.
