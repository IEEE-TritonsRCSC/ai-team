"""
GCClient: unicast TCP connection to the ssl-game-controller "team" rcon
channel, used to send keeper ID / substitution intent to the GC.

Per the SSL field network spec, the GC's IP is not known in advance (it is
assigned per-match along with the rest of the field subnet) — it must be
discovered from the source address of incoming referee multicast packets.
Pass a `GCReceiver` instance (or any object exposing `get_gc_ip()`) as
`gc_ip_source`, or set the SSL_GC_IP_OVERRIDE env var for local testing
without a live GC.

Auth handshake status
----------------------
The real ssl-game-controller rcon protocol normally requires an ECDSA
signed-hello handshake before it accepts authenticated team commands. We do
not yet know whether the competition GC will run in this mode. This client
frames and sends messages today (SSL_GC_AUTH_MODE="none"), and reserves a
hook (`_perform_signed_handshake`) to fill in once that's confirmed with
whoever manages the venue's GC config. Do not flip SSL_GC_AUTH_MODE to
"signed" until that handshake is implemented.
"""

import socket
import struct
import threading
import time
from typing import Callable, Optional

from .net_config import SSL_GC_TEAM_PORT, SSL_GC_IP_OVERRIDE, SSL_GC_AUTH_MODE

RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 5.0)  # capped exponential backoff


class GCClient:
    """
    Background-thread TCP client for the Team -> GC rcon channel.

    Usage::

        receiver = GCReceiver()
        receiver.start()
        client = GCClient(gc_ip_source=receiver)
        client.start()
        ...
        client.send_raw(message_bytes)
        ...
        client.stop()
    """

    def __init__(
        self,
        gc_ip_source,
        port: int = SSL_GC_TEAM_PORT,
        ip_override: str | None = SSL_GC_IP_OVERRIDE,
        auth_mode: str = SSL_GC_AUTH_MODE,
        on_connect: Optional[Callable[[], None]] = None,
    ):
        if auth_mode not in ("none", "signed"):
            raise ValueError(f"Unknown SSL_GC_AUTH_MODE: {auth_mode!r}")
        if auth_mode == "signed":
            raise NotImplementedError(
                "Signed rcon handshake is not implemented yet — confirm the "
                "venue's GC auth requirements before enabling SSL_GC_AUTH_MODE=signed."
            )

        self._gc_ip_source = gc_ip_source
        self._port = port
        self._ip_override = ip_override
        self._auth_mode = auth_mode
        self._on_connect = on_connect

        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
                self._connected = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def send_raw(self, message: bytes) -> bool:
        """
        Send a length-delimited message (4-byte big-endian length prefix,
        matching ssl-game-controller rcon framing) to the GC. Returns False
        (and lets the reconnect loop take over) if not currently connected
        or the send fails.
        """
        with self._lock:
            if not self._connected or self._sock is None:
                return False
            try:
                self._sock.sendall(struct.pack('>I', len(message)) + message)
                return True
            except OSError:
                self._connected = False
                return False

    # ---------------------------------------------------------------- #
    #  Internal                                                         #
    # ---------------------------------------------------------------- #

    def _resolve_gc_ip(self) -> str | None:
        if self._ip_override:
            return self._ip_override
        return self._gc_ip_source.get_gc_ip()

    def _perform_signed_handshake(self, sock: socket.socket) -> None:
        """Placeholder for the ECDSA signed-hello rcon handshake."""
        raise NotImplementedError(
            "Signed rcon handshake not yet implemented (see module docstring)."
        )

    def _run_loop(self) -> None:
        backoff_idx = 0
        while self._running:
            gc_ip = self._resolve_gc_ip()
            if gc_ip is None:
                time.sleep(RECONNECT_BACKOFF_S[0])
                continue

            try:
                sock = socket.create_connection((gc_ip, self._port), timeout=2.0)
                if self._auth_mode == "signed":
                    self._perform_signed_handshake(sock)

                with self._lock:
                    self._sock = sock
                    self._connected = True
                backoff_idx = 0

                if self._on_connect:
                    self._on_connect()

                # Block here reading (and discarding, for now) any keepalive
                # / ControllerReply traffic from the GC until the socket
                # drops, at which point the outer loop reconnects.
                sock.settimeout(1.0)
                while self._running:
                    try:
                        data = sock.recv(4096)
                        if not data:
                            break
                    except socket.timeout:
                        continue
                    except OSError:
                        break
            except OSError:
                pass
            finally:
                with self._lock:
                    if self._sock:
                        try:
                            self._sock.close()
                        except OSError:
                            pass
                    self._sock = None
                    self._connected = False

            if not self._running:
                break

            delay = RECONNECT_BACKOFF_S[min(backoff_idx, len(RECONNECT_BACKOFF_S) - 1)]
            backoff_idx += 1
            time.sleep(delay)
