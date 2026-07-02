"""
Socket-based networking utilities for simulator and robot communication.

This module provides classes for listening to game data, managing client connections,
and commanding both simulated and physical robots through various network protocols.
"""

import re
import gc
import fcntl
import math
import os
import random
import time
import socket
import tempfile
import threading
from contextlib import contextmanager
from typing import Optional
from .data_utils import GameState, TeamInfo, Deserializer
from .net_config import (
    LOCALHOST_IP,
    ROBOT_COMMAND_MULTICAST_IP,
    ROBOT_COMMAND_PORT,
    SSL_NETWORK_INTERFACE,
)
from .vision_client import SSLVisionClient

try:
    import rcssserver_embedded as embedded_sim
except ImportError:
    embedded_sim = None

# Network constants for listening to simulator data
BUFFER_SIZE = 1536
DEFAULT_SIM_PLAYER_PORT = 6000
DEFAULT_SIM_TRAINER_PORT = 6001
INIT_PATTERN = r"\(init ([lr]) (1[0-1]|[1-9]) \w+\)"
PLAYMODE_REGEX = r"\(hear \d+ referee (\w+)\)"

# Multicast settings for real robots (sourced from net_config.py / env vars,
# so the field subnet can be updated at competition without touching code)
COMMAND_IP = ROBOT_COMMAND_MULTICAST_IP
COMMAND_PORT = ROBOT_COMMAND_PORT

UDP_SIM_ENVIRONMENTS = {"sim-only", "sim-mixed"}
EMBEDDED_SIM_ENVIRONMENTS = {"sim-embedded"}
SIM_ENVIRONMENTS = UDP_SIM_ENVIRONMENTS | EMBEDDED_SIM_ENVIRONMENTS
ROBOT_ENVIRONMENTS = {"sim-mixed", "field-practice", "field-tournament"}


def _uses_udp_simulator(environment: str) -> bool:
    return environment in UDP_SIM_ENVIRONMENTS


def _uses_embedded_simulator(environment: str) -> bool:
    return environment in EMBEDDED_SIM_ENVIRONMENTS


def _uses_simulator(environment: str) -> bool:
    return environment in SIM_ENVIRONMENTS


def _uses_robot_multicast(environment: str) -> bool:
    return environment in ROBOT_ENVIRONMENTS


def _compute_init_pose(side: str, first: bool, goalie: bool):
    """Compute the initial spawn pose for a simulator-side player."""
    x, y = random.uniform(15, 30), random.uniform(-25, 25)
    theta = random.uniform(-180, 180)

    if side == "left":
        x = -x
        if first:
            x, y, theta = (-10, 0, 0.0)
        if goalie:
            x, y, theta = -41.4, 0.0, 0.0
    else:
        if first:
            x, y, theta = (20, 10, 180.0)
        if goalie:
            x, y, theta = (41.4, 0.0, 180.0)

    return (x, y, theta)


# Play modes the embedded engine can recover from with a soft reset
# (PM_PlayOn + move_ball/move_player). These are either active play, the
# kick-off lifecycle, or modes the *env* terminated on while the engine was
# still in play (OOB / keeper-cleared detected by ball position). Any OTHER
# mode is a referee set-piece (free kick, goal kick, corner, catch, foul, …)
# that latches held-ball state inside the native engine which set_play_mode
# does NOT clear — the next step() re-snaps the ball and bricks the sim, so
# those require a full engine rebuild instead. See EmbeddedSimulatorBackend.reset.
_SOFT_RESETTABLE_PLAYMODES = frozenset({
    "PM_PlayOn",
    "PM_BeforeKickOff",
    "PM_KickOff_Left",
    "PM_KickOff_Right",
    "PM_AfterGoal_Left",
    "PM_AfterGoal_Right",
    "PM_Null",
})


@contextmanager
def _embedded_sim_init_lock():
    """Serialize native embedded-server port discovery and socket binding.

    The C++ wrapper finds an unused three-port block before Stadium::init()
    binds it. Those operations are individually correct but not atomic across
    inference subprocesses: concurrent initializers can select overlapping
    blocks and all but one then fail to bind. The lock covers initialization
    only; simulator stepping remains fully parallel afterward.
    """
    lock_path = os.path.join(tempfile.gettempdir(),
                             "rcssserver_embedded_init.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class EmbeddedSimulatorBackend:
    """Shared synchronous simulator backend used by Listener and Commander."""

    def __init__(self, team_infos: list[TeamInfo]):
        if embedded_sim is None:
            raise ImportError(
                "sim-embedded mode requires the rcssserver_embedded module. "
                "Use the rcai conda environment when running this mode."
            )

        self.team_infos = team_infos
        self._lock = threading.Lock()
        self._pending_commands = []
        self.desired_init_poses = []
        self.side_by_team = {}
        self.team_by_side = {}
        self._sim = None
        self._last_state = None
        self._force_hard_reset = False
        self._initialize_simulator()

    def _initialize_simulator(self):
        # When rebuilding mid-run (sticky referee-state recovery in reset), the
        # previous native simulator must be released *before* constructing the
        # replacement: its server socket is still bound, so a fresh init() would
        # otherwise fail with "Address already in use" and come up dead. Dropping
        # the last Python reference runs the C++ destructor (which closes the
        # socket); gc.collect() also breaks any reference cycles holding it.
        if self._sim is not None:
            self._sim = None
            self._last_state = None
            gc.collect()

        # The native wrapper performs free-port discovery followed by socket
        # binding. Keep that pair atomic across embedded inference processes.
        with _embedded_sim_init_lock():
            sim = embedded_sim.EmbeddedSimulator()
            if not sim.init():
                raise RuntimeError(
                    "Embedded simulator initialization failed; native "
                    "rcssserver could not initialize its server sockets."
                )

        left_team, right_team = self.team_infos
        sim.set_team_name(embedded_sim.Side.LEFT, left_team.name)
        sim.set_team_name(embedded_sim.Side.RIGHT, right_team.name)

        self.side_by_team = {
            left_team.name: embedded_sim.Side.LEFT,
            right_team.name: embedded_sim.Side.RIGHT,
        }
        self.team_by_side = {
            embedded_sim.Side.LEFT.name: left_team.name,
            embedded_sim.Side.RIGHT.name: right_team.name,
        }

        desired_init_poses = []
        for team_info, side_name, side_enum in (
            (left_team, "left", embedded_sim.Side.LEFT),
            (right_team, "right", embedded_sim.Side.RIGHT),
        ):
            goalie_0idx = team_info.goalie_id - 1
            for i in range(team_info.n_players):
                goalie = i == goalie_0idx
                init_pose = _compute_init_pose(side_name, i == 0, goalie)
                sim.enable_player(side_enum, i + 1, goalie)
                desired_init_poses.append(
                    (
                        f"(player {team_info.name} {i+1}{' goalie' if goalie else ''})",
                        init_pose,
                    )
                )

        sim.start_match()

        self._sim = sim
        self._pending_commands.clear()
        # A fresh engine holds no ball; clear any tracked catch-glue ownership.
        self._ball_caught_by = None
        self.desired_init_poses[:] = desired_init_poses
        self._last_state = self._sim.snapshot()

    def _iter_player_slots(self):
        """Yield (side_enum, unum, init_pose) for each enabled player slot."""
        pose_index = 0
        for team_info, side_enum in (
            (self.team_infos[0], embedded_sim.Side.LEFT),
            (self.team_infos[1], embedded_sim.Side.RIGHT),
        ):
            for unum in range(1, team_info.n_players + 1):
                _obj_name, init_pose = self.desired_init_poses[pose_index]
                pose_index += 1
                yield side_enum, unum, init_pose

    def _normalize_command(self, command) -> str:
        if isinstance(command, bytes):
            command_text = command.decode()
        else:
            command_text = str(command)
        command_text = command_text.rstrip("\0").strip()
        if command_text and not command_text.startswith("("):
            command_text = f"({command_text})"
        return command_text

    def queue_commands(self, teamname: str, commands: list[bytes]):
        """Buffer commands to apply on the next synchronous simulator step."""
        side = self.side_by_team[teamname]
        queued = []
        for unum, command in enumerate(commands, start=1):
            if command is None:
                continue
            command_text = self._normalize_command(command)
            if not command_text:
                continue
            # Track catch-glue ownership. A `catch` glues the ball to this player;
            # `drop`/`kick` release it. If an episode ENDS while the ball is still
            # caught (e.g. max_steps hits mid-carry), the engine re-snaps the held
            # ball onto the holder on the next episode's first step, tripping the
            # ball_teleport guard and bricking every following episode (all 1-step,
            # playmode stays play_on so the referee-mode recovery never fires).
            # reset() reads this to rebuild the engine and clear the latched hold.
            inner = command_text.strip("()").strip()
            if inner.startswith("catch"):
                self._ball_caught_by = (side, unum)
            elif inner.startswith("drop") or inner.startswith("kick"):
                self._ball_caught_by = None
            queued.append(embedded_sim.PlayerCommand(side, unum, command_text))

        with self._lock:
            self._pending_commands.extend(queued)

    def _playmode_to_string(self, playmode) -> Optional[str]:
        if playmode is None:
            return None

        name = getattr(playmode, "name", None)
        if not name:
            text = str(playmode)
            name = text.split(".")[-1]
        if name.startswith("PM_"):
            name = name[3:]

        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub(r"__+", "_", name).lower()
        if name.endswith("_left"):
            return f"{name[:-5]}_l"
        if name.endswith("_right"):
            return f"{name[:-6]}_r"
        return name

    def _side_to_teamname(self, side) -> Optional[str]:
        side_name = getattr(side, "name", None)
        if not side_name:
            side_name = str(side).split(".")[-1]
        return self.team_by_side.get(side_name)

    def _to_game_state(self, state) -> GameState:
        robot_entries = {team_info.name: [] for team_info in self.team_infos}

        for player in getattr(state, "players", []) or []:
            if not getattr(player, "enabled", False):
                continue

            teamname = self._side_to_teamname(player.side)
            if teamname is None:
                continue

            # The embedded engine reports body_angle in radians; the rest of
            # the stack (UDP deserializer, JAL_env) works in degrees, so convert
            # here to keep the two backends interchangeable.
            pose = (
                float(player.pos.x),
                float(player.pos.y),
                math.degrees(float(player.body_angle)),
            )
            robot_entries[teamname].append((int(player.unum), pose))

        robot_poses = {}
        for teamname, entries in robot_entries.items():
            entries.sort(key=lambda item: item[0])
            robot_poses[teamname] = [{unum: pose} for unum, pose in entries]

        ball = getattr(state, "ball", None)
        ball_pos = None
        if ball is not None and getattr(ball, "pos", None) is not None:
            ball_pos = (float(ball.pos.x), float(ball.pos.y))

        return GameState(
            int(getattr(state, "time", 0)),
            time.time(),
            ball_pos,
            robot_poses,
            self._playmode_to_string(getattr(state, "playmode", None)),
        )

    def watch_game(self) -> GameState:
        """Advance the simulator once if commands are pending; otherwise snapshot."""
        with self._lock:
            if self._pending_commands:
                commands = list(self._pending_commands)
                self._pending_commands.clear()
                self._last_state = self._sim.step(commands)
            elif self._last_state is None:
                self._last_state = self._sim.snapshot()
            else:
                self._last_state = self._sim.snapshot()

            return self._to_game_state(self._last_state)

    def reset(self, ball_pos=None, player_poses_override=None) -> bool:
        """Soft-reset the episode without recreating the embedded server.

        Reinitializing the native simulator every episode causes repeated
        server-side player-type logs and transient socket bind failures inside
        the embedded module, so we keep the simulator instance alive and use the
        trainer-style teleports (`move_player` / `move_ball`) exposed by the
        binding to reposition objects.

        Unlike a player `(move x y)` command — which rcssserver only honours in
        before_kick_off and which left the robot stuck out of bounds during
        play_on — these teleports place objects directly in any play mode. We
        force PM_PlayOn so the placed positions persist (a kick-off transition
        would otherwise reset the ball to centre) and physics runs immediately.

        Args:
            ball_pos: optional (x, y) for the ball. Defaults to centre (0, 0).
            player_poses_override: optional list parallel to `desired_init_poses`
                of (obj_name, (x, y, theta_deg)) tuples. When provided, each
                player is placed at its override pose; otherwise the fixed
                formation init poses are used. theta is in degrees (project
                convention) and converted to radians for the engine.
        """
        with self._lock:
            # If the previous episode left the engine in a referee set-piece mode
            # (free kick, goal kick, corner, catch, foul, …), a PM_PlayOn +
            # move_ball soft reset does NOT clear the latched held-ball state:
            # the next step() re-snaps the ball to the set-piece spot and bricks
            # the sim for every following episode. Detect that here from the last
            # observed play mode and rebuild the engine from a clean slate, which
            # has no latched referee state. Goals/kick-offs/PlayOn-terminated
            # episodes (OOB, keeper-cleared) stay on the cheap soft-reset path.
            prev_pm = getattr(self._last_state, "playmode", None)
            prev_pm_name = getattr(prev_pm, "name", None) or ""
            # Rebuild when the engine left a sticky referee set-piece OR when the
            # ball is still caught (glued to a player) — both latch held-ball state
            # that a PM_PlayOn + move_ball soft reset cannot clear. The caught case
            # stays on play_on, so it must be detected from tracked catch ownership
            # rather than the play mode. _initialize_simulator clears _ball_caught_by.
            if (
                prev_pm_name not in _SOFT_RESETTABLE_PLAYMODES
                or self._ball_caught_by is not None
                or self._force_hard_reset
            ):
                self._force_hard_reset = False
                self._initialize_simulator()

            # PlayOn first so subsequent teleports are not overwritten by a
            # kick-off ball reset, and so the next step advances real physics.
            self._sim.set_play_mode(embedded_sim.PlayMode.PM_PlayOn)

            override_poses = None
            if player_poses_override is not None:
                override_poses = [pose for _obj_name, pose in player_poses_override]

            for idx, (side_enum, unum, init_pose) in enumerate(self._iter_player_slots()):
                if override_poses is not None and idx < len(override_poses):
                    pose = override_poses[idx]
                else:
                    pose = init_pose
                x, y = float(pose[0]), float(pose[1])
                theta_deg = float(pose[2]) if len(pose) > 2 else 0.0
                self._sim.move_player(side_enum, unum, x, y, math.radians(theta_deg))

            bx, by = (0.0, 0.0) if ball_pos is None else (float(ball_pos[0]), float(ball_pos[1]))
            self._sim.move_ball(bx, by, 0.0, 0.0)

            # Refresh the cached state so the next watch_game/snapshot reflects
            # the teleported positions without consuming a simulator step.
            self._pending_commands.clear()
            self._last_state = self._sim.snapshot()
        return True

    def shutdown(self):
        """Release buffered commands and the simulator reference."""
        with self._lock:
            self._pending_commands.clear()
            self._last_state = None
            self._sim = None

class Listener:
    """Listens for game state updates from simulators or cameras."""
    def __init__(self, team_infos: list[TeamInfo], environment: str,
                 desired_init_poses: list, sim_host: str = LOCALHOST_IP,
                 sim_trainer_port: int = DEFAULT_SIM_TRAINER_PORT,
                 embedded_backend: Optional[EmbeddedSimulatorBackend] = None):
        """
        Initialize listener for the specified environment.
        
        Args:
            team_infos: List of team information including names and player counts
            environment: Type of environment to listen to
            desired_init_poses: List of desired initial poses for simulated robots
        """
        self.parser = Deserializer(team_infos)

        if _uses_udp_simulator(environment):
            self.source = "simulator"
            self.addr = (sim_host, int(sim_trainer_port))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.2)    # Non-blocking with timeout
            self.desired_init_poses = desired_init_poses
            self.connect_to_sim(desired_init_poses)
        elif _uses_embedded_simulator(environment):
            self.source = "embedded"
            self.embedded_backend = embedded_backend or EmbeddedSimulatorBackend(team_infos)
            self.desired_init_poses = self.embedded_backend.desired_init_poses
        else:
            self.source = "camera"
            # Use our SSLVisionClient (not the raw sslclient package) so the
            # multicast join honors SSL_NETWORK_INTERFACE instead of
            # sslclient's unreliable gethostbyname(gethostname()) guess —
            # critical when vision is on its own directly-wired NIC.
            self.vision_client = SSLVisionClient()
            self.vision_client.connect()

    def watch_game(self) -> GameState:
        """
        Watch for and return the next game state update.
        
        Returns:
            Current game state or None if no valid data received
        """
        if self.source == "simulator":
            try:
                (data, address) = self.sock.recvfrom(BUFFER_SIZE)
            except (TimeoutError, OSError):
                return None  # socket timeout – no data this tick
            if address == self.addr:
                game_state = self.parser.sim_deserialize(data)
                return game_state
            else:
                return None
        elif self.source == "embedded":
            return self.embedded_backend.watch_game()
        else:
            data = self.vision_client.receive()
            if data.HasField("detection"):
                game_state = self.parser.cam_deserialize(data.detection)
                return game_state
            else:
                return None

    def connect_to_sim(self, desired_init_poses: list):
        """
        Establish connection to simulator and initialize monitoring.
        Args:
            desired_init_poses: List of desired initial poses for simulated robots
        """
        self.sock.bind((LOCALHOST_IP, 0))
        self.sock.sendto(b"(init (version 19))\0", self.addr)
        (data, address) = self.sock.recvfrom(16)
        if data != b"(init ok)\0":
            raise Exception(f"Unexpected response: {data} from {address}")
        else:
            self.addr = address    # Save address for subsequent communication

        # Drain any pending initialization messages from the socket buffer
        try:
            while True:
                self.sock.recvfrom(16)
        except TimeoutError:
            pass

        # Set desired initial poses
        for (obj_name, pose) in desired_init_poses:
            init_command = f"(move {obj_name} {pose[0]} {pose[1]} {pose[2]})\0".encode()
            self.sock.sendto(init_command, self.addr)
            (data, address) = self.sock.recvfrom(16)
            if (self.addr != address or data != b"(ok move)\0"):
                raise Exception(f"Unexpected response: {data} from {address}")
            time.sleep(0.1)   # Allow time for simulator to process
        
        # Enable "eye on" to start receiving data about the game state and start the game
        self.sock.sendto(b"(eye on)\0", self.addr)
        self.sock.sendto(b"(change_mode play_on)\0", self.addr)

        (data, address) = self.sock.recvfrom(16)
        if (self.addr != address or data != b"(ok eye on)\0"):
            raise Exception(f"Unexpected response: {data} from {address}")

        (data, address) = self.sock.recvfrom(16)
        if (self.addr != address or data != b"(ok change_mode)"):
            raise Exception(f"Unexpected response: {data} from {address}")

    def _wait_for_trainer_ok(self, expected_prefix: bytes, timeout_s: float = 1.0) -> bool:
        """Wait until trainer socket receives an expected OK response."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                data, address = self.sock.recvfrom(BUFFER_SIZE)
            except (TimeoutError, socket.timeout):
                continue
            if address != self.addr:
                continue
            if data.startswith(expected_prefix):
                return True
        return False

    def reset_sim(self, desired_init_poses: Optional[list] = None) -> bool:
        """Reset simulator playmode and restore initial poses through trainer port."""
        if self.source == "embedded":
            return self.embedded_backend.reset()

        if self.source != "simulator":
            return False

        poses = desired_init_poses if desired_init_poses is not None else self.desired_init_poses

        try:
            self.sock.sendto(b"(change_mode before_kick_off)\0", self.addr)
            if not self._wait_for_trainer_ok(b"(ok change_mode)"):
                return False

            for obj_name, pose in poses:
                init_command = f"(move {obj_name} {pose[0]} {pose[1]} {pose[2]})\0".encode()
                self.sock.sendto(init_command, self.addr)
                if not self._wait_for_trainer_ok(b"(ok move)"):
                    return False

            self.sock.sendto(b"(change_mode play_on)\0", self.addr)
            return self._wait_for_trainer_ok(b"(ok change_mode)")
        except Exception:
            return False

    def disconnect_from_sim(self):
        """Disconnect from simulator and close socket."""
        if self.source == "embedded":
            self.embedded_backend.shutdown()
            return

        self.sock.sendto(b"(bye)\0", self.addr)
        time.sleep(0.1)
        self.sock.close()

    def restart_game(self):
        """Keep sending change_mode play_on until the server acknowledges it."""
        if self.source == "embedded":
            return

        play_on = False
        while not play_on:
            self.sock.sendto(b"(change_mode play_on)\0", self.addr)
            try:
                (data, address) = self.sock.recvfrom(16)
                if address == self.addr and data == b"(ok change_mode)":
                    play_on = True
            except TimeoutError:
                pass

    def disconnect_from_camera(self):
        """Disconnect from camera vision client."""
        self.vision_client.sock.close()


class Client:
    """Represents a single robot client connection to the simulator."""
    def __init__(self, teamname: str, side: str = "left", 
                 first: bool = False, goalie: bool = False,
                 sim_host: str = LOCALHOST_IP,
                 sim_player_port: int = DEFAULT_SIM_PLAYER_PORT):
        """
        Initialize a client connection for a single robot.
        
        Args:
            teamname: Name of the team this robot belongs to
            side: Which side of field ("left" or "right")
            first: Whether this is the first robot
            goalie: Whether this robot is a goalie
        """
        self.teamname = teamname
        self.init_pose = self.get_init_pose(side, first, goalie)

        self.addr = (sim_host, int(sim_player_port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect_to_sim(goalie)

    def get_init_pose(self, side: str, first: bool, goalie: bool):
        """
        Calculate initial pose for the robot.
        
        Args:
            side: Which side of field ("left" or "right")
            first: Whether this is the first robot
            goalie: Whether this robot is a goalie
            
        Returns:
            Tuple of (x, y, theta) initial pose
        """
        return _compute_init_pose(side, first, goalie)

    def send_command(self, command: bytes):
        """Send a command to the simulator for this robot."""
        self.sock.sendto(command, self.addr)

    def watch_game(self) -> dict:
        (data, address) = self.sock.recvfrom(BUFFER_SIZE)
        if address == self.addr:
            data = data.decode()
            if m := re.search(PLAYMODE_REGEX, data):
                playmode = m.group(1).strip()
                return {"playmode": playmode}
            else:
                return {}
        else:
            return None

    def connect_to_sim(self, goalie: bool):
        """Connect to simulator and initialize robot pose.
        Args:
            goalie: Whether this robot is a goalie
        """
        init_args = f"{self.teamname} (version 19)".encode()
        init_args += b" (goalie)" if goalie else b""

        # Initialize the connection
        time.sleep(0.1)
        self.send_command(b"(init %b)\0" % init_args)
        (data, address) = self.sock.recvfrom(64)
        if m := re.search(INIT_PATTERN, data.decode()):
            self.side = m.group(1)
            self.id = int(m.group(2))
            self.addr = address    # Save the address for later use
        else:
            raise Exception(f"Unexpected response: {data} from {address}")

    def disconnect_from_sim(self):
        """Disconnect from simulator and close socket."""
        self.send_command(b"(bye)\0")
        self.sock.close()


class Commander:
    """Manages command sending to both simulated and physical robots."""
    def __init__(self, team_infos: list[TeamInfo], environment: str,
                 sim_host: str = LOCALHOST_IP,
                 sim_player_port: int = DEFAULT_SIM_PLAYER_PORT):
        """
        Initialize commander for the given teams and environment.
        
        Args:
            team_infos: Information about teams to command
            environment: Type of environment for command routing
        """
        self.team_infos = team_infos
        self.environment = environment
        self.sim_host = sim_host
        self.sim_player_port = int(sim_player_port)
        self.desired_init_poses = []
        self.sample_client = None
        self.embedded_backend = None

        if _uses_udp_simulator(environment):
            self.create_sim_clients()
            time.sleep(0.1)    # Allow time for simulator to set up
        elif _uses_embedded_simulator(environment):
            self.embedded_backend = EmbeddedSimulatorBackend(team_infos)
            self.desired_init_poses = self.embedded_backend.desired_init_poses

        self.socks, self.addrs = {}, {}
        if _uses_robot_multicast(environment):
            for i, team_info in enumerate(team_infos):
                teamname = team_info.name
                port_num = COMMAND_PORT + (i * 1000)
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                if SSL_NETWORK_INTERFACE:
                    # Select the field ethernet (rather than Wi-Fi or
                    # another NIC) for outgoing multicast robot commands.
                    sock.setsockopt(
                        socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                        socket.inet_aton(SSL_NETWORK_INTERFACE),
                    )
                self.socks[teamname] = sock
                self.addrs[teamname] = (COMMAND_IP, port_num)

    def create_sim_clients(self):
        """Create simulator client connections for all teams and robots."""
        self.sim_clients = {}
        for team_info, side in zip(self.team_infos, ["left", "right"]):
            self.sim_clients[team_info.name] = [None] * team_info.n_players
            # Populate clients for each team
            goalie_0idx = team_info.goalie_id - 1    # Convert goalie_id to 0-based index
            for i in range(team_info.n_players):
                teamname = team_info.name
                goalie = (i == goalie_0idx)

                client = Client(teamname, side, i == 0, goalie,
                                sim_host=self.sim_host,
                                sim_player_port=self.sim_player_port)
                self.sim_clients[teamname][i] = client
                if self.sample_client is None:
                    self.sample_client = client    # Save one sample client for reference

                # Store desired initial poses by tuple (ObjName, (x, y, theta))
                obj_name = f"(player {teamname} {i+1}{' goalie' if goalie else ''})"
                self.desired_init_poses.append((obj_name, client.init_pose))

    def send_to_sim(self, teamname: str, commands: list[bytes]):
        """
        Send commands to simulated robots using threading.
        
        Args:
            teamname: Name of team to send commands to
            commands: List of command bytes for each robot
        """
        if self.embedded_backend is not None:
            self.embedded_backend.queue_commands(teamname, commands)
            return

        threads = []
        for (client, command) in zip(self.sim_clients[teamname], commands):
            if command is None:
                continue
            args = (command,)
            thread = threading.Thread(target=client.send_command, args=args)
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

    def send_to_robots(self, teamname: str, command: bytes):
        """
        Send commands to physical robots via multicast.
        
        Args:
            teamname: Name of team to send commands to
            command: Command bytes to send
        """
        sock, addr = self.socks[teamname], self.addrs[teamname]
        sock.sendto(command, addr)
        print(command, addr)

    def reset_sim(self, ball_pos=None, player_poses_override=None) -> bool:
        """Reset simulator by moving players to initial positions.

        Args:
            ball_pos: optional (x, y) ball start (embedded backend only).
            player_poses_override: optional per-episode poses parallel to
                desired_init_poses (embedded backend only).

        Returns:
            True if at least one player reset command was sent successfully.
        """
        if self.embedded_backend is not None:
            return self.embedded_backend.reset(
                ball_pos=ball_pos,
                player_poses_override=player_poses_override,
            )

        if not hasattr(self, "sim_clients"):
            return False

        reset_count = 0

        # Reset each client to initial position instead of full disconnect/reconnect
        for team_info, side in zip(self.team_infos, ["left", "right"]):
            team_clients = self.sim_clients[team_info.name]
            goalie_0idx = team_info.goalie_id - 1

            for i, client in enumerate(team_clients):
                if not client:
                    continue

                is_first = i == 0
                is_goalie = i == goalie_0idx

                try:
                    # Keep role-aware spawn logic consistent with initial setup.
                    init_pose = client.get_init_pose(side, is_first, is_goalie)

                    move_args = f"{init_pose[0]} {init_pose[1]}".encode()
                    turn_args = f"{init_pose[2]}".encode()
                    client.send_command(b"(move %b)\0" % move_args)
                    time.sleep(0.05)
                    client.send_command(b"(turn %b)\0" % turn_args)

                    client.init_pose = init_pose
                    reset_count += 1
                except Exception:
                    # If one client fails, continue resetting the remaining clients.
                    continue

        return reset_count > 0

    def disconnect_from_sim(self):
        """Disconnect all simulator clients."""
        if self.embedded_backend is not None:
            self.embedded_backend.shutdown()
            return

        for team_clients in self.sim_clients.values():
            for client in team_clients:
                time.sleep(0.1)
                client.disconnect_from_sim()

    def stop_robots(self):
        """Send stop commands to all robots."""
        stop_command = b"stop\0"
        for teamname in self.socks.keys():
            # Send stop command twice because OS-level buffering may drop the last packet
            self.send_to_robots(teamname, stop_command)
            self.send_to_robots(teamname, stop_command)
            time.sleep(0.1)
