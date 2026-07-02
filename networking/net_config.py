"""
Single source of truth for SSL field network parameters.

All values default to the current hardcoded constants / SSL standard, but can
be overridden at runtime via environment variables so the field router subnet
and GC IP (assigned per-match by the Technical Committee) can be updated at
the competition without touching source code.
"""

import os


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_opt_str(name: str) -> str | None:
    return os.environ.get(name) or None


# ------------------------------------------------------------------ #
#  SSL Game Controller (referee) — multicast receive                  #
# ------------------------------------------------------------------ #
SSL_GC_MULTICAST_IP = _env_str("SSL_GC_MULTICAST_IP", "224.5.23.1")
SSL_GC_PORT = _env_int("SSL_GC_PORT", 10003)

# ------------------------------------------------------------------ #
#  Team -> GC rcon channel — unicast TCP                              #
# ------------------------------------------------------------------ #
SSL_GC_TEAM_PORT = _env_int("SSL_GC_TEAM_PORT", 10008)
# Optional manual override of the GC IP, e.g. for testing without a live
# multicast referee feed. Normally the GC IP is discovered dynamically from
# the source address of incoming referee multicast packets.
SSL_GC_IP_OVERRIDE = _env_opt_str("SSL_GC_IP_OVERRIDE")
# "none" (default, unauthenticated) or "signed" (ECDSA signed-hello rcon
# handshake) - confirm which mode the venue's GC runs before flipping this.
SSL_GC_AUTH_MODE = _env_str("SSL_GC_AUTH_MODE", "none")

# ------------------------------------------------------------------ #
#  SSL Vision — multicast receive (consumed via the sslclient library) #
# ------------------------------------------------------------------ #
SSL_VISION_MULTICAST_IP = _env_str("SSL_VISION_MULTICAST_IP", "224.5.23.2")
SSL_VISION_PORT = _env_int("SSL_VISION_PORT", 10006)
SSL_VISION_TRACKER_PORT = _env_int("SSL_VISION_TRACKER_PORT", 10010)

# ------------------------------------------------------------------ #
#  Robot command multicast (our own radio bridge, not part of the      #
#  official SSL GC/vision spec, but lives on the same per-match subnet) #
# ------------------------------------------------------------------ #
ROBOT_COMMAND_MULTICAST_IP = _env_str("ROBOT_COMMAND_MULTICAST_IP", "239.42.42.42")
ROBOT_COMMAND_PORT = _env_int("ROBOT_COMMAND_PORT", 10000)

# ------------------------------------------------------------------ #
#  Local simulator loopback (unaffected by field subnet)               #
# ------------------------------------------------------------------ #
LOCALHOST_IP = _env_str("SIM_LOCALHOST_IP", "127.0.0.1")

# ------------------------------------------------------------------ #
#  Network interface selection                                        #
# ------------------------------------------------------------------ #
# Local IP address of the NIC to use for multicast joins/sends (e.g. the
# field ethernet, not Wi-Fi). None means "let the OS pick" (current
# behavior).
SSL_NETWORK_INTERFACE = _env_opt_str("SSL_NETWORK_INTERFACE")
