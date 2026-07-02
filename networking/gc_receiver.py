"""
GCReceiver: listens for ssl-game-controller Referee packets over UDP multicast
and converts them to GCState objects.

Setup
-----
The Referee message is defined in ssl_gc_referee_message.proto (part of the
ssl-game-controller repo).  Generate the Python stubs once with:

    pip install grpcio-tools
    python -m grpc_tools.protoc -I. --python_out=networking/proto/ \\
        ssl_gc_referee_message.proto

Or install the pre-built package:

    pip install robocup-ssl-proto

Then ensure `networking/proto/__init__.py` exists.

If the stubs are absent, GCReceiver raises ImportError with a clear message.
MockGCReceiver (at the bottom of this file) is always available without any
proto dependency and is the recommended way to test dispatcher logic offline.

Network defaults
----------------
The ssl-game-controller broadcasts on multicast group 224.5.23.1, port 10003
(the SSL standard).  Override with the GC_MULTICAST_IP / GC_PORT constants or
pass them to GCReceiver.__init__().
"""

import socket
import struct
import threading
from collections import deque
from typing import Callable

from .gc_state import (
    GCState, GCTeamInfo,
    COMMAND_NAMES, STAGE_NAMES,
)
from .net_config import (
    SSL_GC_MULTICAST_IP,
    SSL_GC_PORT,
    SSL_NETWORK_INTERFACE,
)

# ------------------------------------------------------------------ #
#  Network defaults (sourced from net_config.py / env vars)           #
# ------------------------------------------------------------------ #

GC_MULTICAST_IP = SSL_GC_MULTICAST_IP
GC_PORT = SSL_GC_PORT
BUFFER_SIZE = 65536


# ------------------------------------------------------------------ #
#  Proto import                                                       #
# ------------------------------------------------------------------ #

try:
    from .proto.ssl_gc_referee_message_pb2 import Referee as _RefereeProto
    _PROTO_AVAILABLE = True
except ImportError:
    _RefereeProto = None
    _PROTO_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Conversion helpers                                                 #
# ------------------------------------------------------------------ #

def _team_info(proto_team) -> GCTeamInfo:
    """Convert a proto TeamInfo message to a GCTeamInfo dataclass."""
    return GCTeamInfo(
        name=proto_team.name,
        score=proto_team.score,
        red_cards=proto_team.red_cards,
        yellow_cards=proto_team.yellow_cards,
        timeouts=proto_team.timeouts,
        timeout_time=proto_team.timeout_time,
        goalkeeper=proto_team.goalkeeper if proto_team.HasField('goalkeeper') else 0,
        foul_counter=getattr(proto_team, 'foul_counter', 0),
        ball_placement_failures=getattr(proto_team, 'ball_placement_failures', 0),
        can_place_ball=getattr(proto_team, 'can_place_ball', True),
        max_allowed_bots=getattr(proto_team, 'max_allowed_bots', 6),
    )


def _designated_pos(proto_ref) -> tuple[float, float] | None:
    """Return (x, y) in meters, or None if the field is absent."""
    if not proto_ref.HasField('designated_position'):
        return None
    p = proto_ref.designated_position
    # ssl-game-controller broadcasts positions in millimeters; convert to meters.
    return (p.x / 1000.0, p.y / 1000.0)


def _command_name(value: int) -> str:
    return COMMAND_NAMES.get(value, f'UNKNOWN_{value}')


def _stage_name(value: int) -> str:
    return STAGE_NAMES.get(value, f'UNKNOWN_{value}')


def _proto_to_gc_state(ref) -> GCState:
    """Convert a parsed Referee proto object to a GCState."""
    next_cmd = None
    if ref.HasField('next_command'):
        next_cmd = _command_name(ref.next_command)

    return GCState(
        command=_command_name(ref.command),
        stage=_stage_name(ref.stage),
        next_command=next_cmd,
        designated_pos=_designated_pos(ref),
        command_counter=ref.command_counter,
        blue_on_positive_half=getattr(ref, 'blue_team_on_positive_half', False),
        blue=_team_info(ref.blue),
        yellow=_team_info(ref.yellow),
        packet_timestamp_us=ref.packet_timestamp,
        command_timestamp_us=ref.command_timestamp,
    )


# ------------------------------------------------------------------ #
#  Live receiver                                                      #
# ------------------------------------------------------------------ #

class GCReceiver:
    """
    Background-thread UDP listener for ssl-game-controller Referee packets.

    Usage::

        receiver = GCReceiver()
        receiver.start()
        ...
        gc_state = receiver.get_latest()   # None until first packet arrives
        ...
        receiver.stop()

    Callbacks::

        receiver = GCReceiver(on_state=lambda gs: print(gs.command))
        receiver.start()
    """

    def __init__(
        self,
        multicast_ip: str = GC_MULTICAST_IP,
        port: int = GC_PORT,
        on_state: Callable[[GCState], None] | None = None,
        interface: str | None = SSL_NETWORK_INTERFACE,
    ):
        if not _PROTO_AVAILABLE:
            raise ImportError(
                "Protobuf stubs for ssl-game-controller are not installed.\n"
                "Run:  pip install grpcio-tools robocup-ssl-proto\n"
                "Or generate stubs from ssl_gc_referee_message.proto and place\n"
                "the output in networking/proto/.\n"
                "Use MockGCReceiver for offline testing without stubs."
            )

        self._multicast_ip = multicast_ip
        self._port = port
        self._on_state = on_state
        self._interface = interface

        self._latest: GCState | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Source address of the most recently received referee packet.
        # Per the SSL spec, this is how the GC's own IP is discovered for
        # the Team -> GC rcon TCP channel (port 10008), since the GC IP is
        # assigned per-match and not known in advance.
        self._last_gc_addr: tuple[str, int] | None = None

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #

    def start(self) -> None:
        """Open the socket and begin receiving in a daemon thread."""
        self._sock = self._open_socket()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the receive loop to exit and close the socket."""
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> GCState | None:
        """Return the most recently parsed GCState, or None if none yet."""
        with self._lock:
            return self._latest

    def get_gc_ip(self) -> str | None:
        """
        Return the IP address the last referee packet arrived from, i.e. the
        GC's own address. Used to open the Team -> GC rcon TCP connection
        (port 10008) without hardcoding the GC's IP, since it's assigned
        per-match. Returns None until the first packet has been received.
        """
        with self._lock:
            return self._last_gc_addr[0] if self._last_gc_addr else None

    # ---------------------------------------------------------------- #
    #  Internal                                                         #
    # ---------------------------------------------------------------- #

    def _open_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self._port))
        # Join the multicast group on the configured interface (falls back
        # to INADDR_ANY / OS default when none is set), so the correct NIC
        # is used when the machine has both Wi-Fi and field ethernet.
        join_iface = socket.inet_aton(self._interface) if self._interface else struct.pack('I', socket.INADDR_ANY)
        mreq = struct.pack('4s4s', socket.inet_aton(self._multicast_ip), join_iface)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(1.0)
        return sock

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                ref = _RefereeProto()
                ref.ParseFromString(data)
                gc_state = _proto_to_gc_state(ref)
            except Exception as exc:
                print(f'[GCReceiver] Failed to parse packet: {exc}')
                continue

            with self._lock:
                self._latest = gc_state
                self._last_gc_addr = addr

            if self._on_state:
                self._on_state(gc_state)


# ------------------------------------------------------------------ #
#  Mock receiver — no proto dependency, for unit / integration tests  #
# ------------------------------------------------------------------ #

class MockGCReceiver:
    """
    Drop-in replacement for GCReceiver that accepts programmatic state pushes.

    Use this for offline testing of the Dispatcher and all AccessoryAlgo
    modules without needing the ssl-game-controller binary or protobuf stubs.

    Example::

        mock = MockGCReceiver()
        mock.push(GCState(command='STOP', stage='NORMAL_FIRST_HALF', ...))
        assert mock.get_latest().command == 'STOP'

    A pre-built sequence of states can also be replayed automatically::

        mock = MockGCReceiver(sequence=[
            GCState(command='HALT', ...),
            GCState(command='STOP', ...),
            GCState(command='BALL_PLACEMENT_BLUE', ...),
        ])
        mock.start(interval=0.5)   # advance every 0.5 s
    """

    def __init__(self, sequence: list[GCState] | None = None):
        self._latest: GCState | None = None
        self._lock = threading.Lock()
        self._sequence = deque(sequence or [])
        self._timer: threading.Timer | None = None

    # ---------------------------------------------------------------- #
    #  GCReceiver-compatible API                                       #
    # ---------------------------------------------------------------- #

    def start(self, interval: float = 1.0) -> None:
        """Optionally start auto-advancing through the injected sequence."""
        if self._sequence:
            self._schedule_next(interval)

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def get_latest(self) -> GCState | None:
        with self._lock:
            return self._latest

    # ---------------------------------------------------------------- #
    #  Test helpers                                                     #
    # ---------------------------------------------------------------- #

    def push(self, state: GCState) -> None:
        """Immediately make state the latest, as if a packet just arrived."""
        with self._lock:
            self._latest = state

    def push_command(self, command: str, **kwargs) -> GCState:
        """Convenience: push a minimal GCState with only command set."""
        state = GCState(
            command=command,
            stage=kwargs.get('stage', 'NORMAL_FIRST_HALF'),
            next_command=kwargs.get('next_command', None),
            designated_pos=kwargs.get('designated_pos', None),
            command_counter=kwargs.get('command_counter', 0),
            blue_on_positive_half=kwargs.get('blue_on_positive_half', False),
        )
        self.push(state)
        return state

    # ---------------------------------------------------------------- #
    #  Internal                                                         #
    # ---------------------------------------------------------------- #

    def _schedule_next(self, interval: float) -> None:
        def _advance():
            if self._sequence:
                self.push(self._sequence.popleft())
            if self._sequence:
                self._schedule_next(interval)

        self._timer = threading.Timer(interval, _advance)
        self._timer.daemon = True
        self._timer.start()
