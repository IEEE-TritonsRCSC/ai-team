"""
GCState: lightweight snapshot of a single ssl-game-controller Referee packet.

Units
-----
designated_pos is stored in **meters** using the SSL field coordinate system
(origin at center, x positive toward the right goal, y positive toward the top
touch line).  The dispatcher converts to field units before passing to
BallPlacement via update_designated_ball_pos().

Command strings match the Referee.Command enum names from
ssl_gc_referee_message.proto exactly, so they can be compared with simple
string equality or set membership.
"""

from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
#  Per-team data embedded in every Referee packet                     #
# ------------------------------------------------------------------ #

@dataclass
class GCTeamInfo:
    name: str = ''
    score: int = 0
    red_cards: int = 0
    yellow_cards: int = 0
    timeouts: int = 4           # remaining timeout slots
    timeout_time: int = 300     # remaining timeout budget in seconds
    goalkeeper: int = 0
    foul_counter: int = 0
    ball_placement_failures: int = 0
    can_place_ball: bool = True
    max_allowed_bots: int = 6


# ------------------------------------------------------------------ #
#  Top-level snapshot                                                  #
# ------------------------------------------------------------------ #

@dataclass
class GCState:
    """
    Immutable snapshot produced by GCReceiver for each Referee packet.

    command / next_command are Referee.Command enum names, e.g.:
        'HALT', 'STOP', 'BALL_PLACEMENT_BLUE', 'DIRECT_FREE_YELLOW', …
    stage is a Referee.Stage enum name, e.g.:
        'NORMAL_FIRST_HALF', 'PENALTY_SHOOTOUT', …
    """
    command: str
    stage: str
    next_command: str | None            # command that will follow ball placement
    designated_pos: tuple[float, float] | None  # (x, y) in meters
    command_counter: int                # increments on every new GC command
    blue_on_positive_half: bool         # True → blue attacks right goal (x > 0)
    blue: GCTeamInfo = field(default_factory=GCTeamInfo)
    yellow: GCTeamInfo = field(default_factory=GCTeamInfo)
    packet_timestamp_us: int = 0        # microseconds (wall-clock)
    command_timestamp_us: int = 0       # microseconds (wall-clock)

    # ---------------------------------------------------------------- #
    #  Convenience helpers                                              #
    # ---------------------------------------------------------------- #

    def is_ball_placement(self) -> bool:
        return self.command in ('BALL_PLACEMENT_BLUE', 'BALL_PLACEMENT_YELLOW')

    def placing_color(self) -> str | None:
        """Return 'blue' or 'yellow' if we are in a ball-placement command."""
        if self.command == 'BALL_PLACEMENT_BLUE':
            return 'blue'
        if self.command == 'BALL_PLACEMENT_YELLOW':
            return 'yellow'
        return None

    def is_timeout(self) -> bool:
        return self.command in ('TIMEOUT_BLUE', 'TIMEOUT_YELLOW')

    def is_prepare_kickoff(self) -> bool:
        return self.command in ('PREPARE_KICKOFF_BLUE', 'PREPARE_KICKOFF_YELLOW')

    def is_prepare_penalty(self) -> bool:
        return self.command in ('PREPARE_PENALTY_BLUE', 'PREPARE_PENALTY_YELLOW')


# ------------------------------------------------------------------ #
#  Enum value → string name tables (mirrors proto enum)              #
# ------------------------------------------------------------------ #

# Used by GCReceiver when converting raw integer values from the wire.
COMMAND_NAMES: dict[int, str] = {
    0:  'HALT',
    1:  'STOP',
    2:  'NORMAL_START',
    3:  'FORCE_START',
    4:  'PREPARE_KICKOFF_YELLOW',
    5:  'PREPARE_KICKOFF_BLUE',
    6:  'PREPARE_PENALTY_YELLOW',
    7:  'PREPARE_PENALTY_BLUE',
    8:  'DIRECT_FREE_YELLOW',
    9:  'DIRECT_FREE_BLUE',
    10: 'INDIRECT_FREE_YELLOW',
    11: 'INDIRECT_FREE_BLUE',
    12: 'TIMEOUT_YELLOW',
    13: 'TIMEOUT_BLUE',
    14: 'GOAL_YELLOW',
    15: 'GOAL_BLUE',
    16: 'BALL_PLACEMENT_YELLOW',
    17: 'BALL_PLACEMENT_BLUE',
}

STAGE_NAMES: dict[int, str] = {
    0:  'NORMAL_FIRST_HALF_PRE',
    1:  'NORMAL_FIRST_HALF',
    2:  'NORMAL_HALF_TIME',
    3:  'NORMAL_SECOND_HALF_PRE',
    4:  'NORMAL_SECOND_HALF',
    5:  'EXTRA_TIME_BREAK',
    6:  'EXTRA_FIRST_HALF_PRE',
    7:  'EXTRA_FIRST_HALF',
    8:  'EXTRA_HALF_TIME',
    9:  'EXTRA_SECOND_HALF_PRE',
    10: 'EXTRA_SECOND_HALF',
    11: 'PENALTY_SHOOTOUT_BREAK',
    12: 'PENALTY_SHOOTOUT',
    13: 'POST_GAME',
}
