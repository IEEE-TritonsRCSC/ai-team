"""
Dispatcher: translates a GCState into the correct AccessoryAlgo and drives it.

The Dispatcher is the bridge between the ssl-game-controller and our per-state
robot behavior modules.  It holds the *current* AccessoryAlgo instance and
re-instantiates it only when the GC command changes, so modules that maintain
internal state across ticks (e.g. Penalty's phase tracker) are not reset on
every cycle.

Usage
-----
    from support.dispatcher import Dispatcher
    from networking.gc_receiver import GCReceiver   # or MockGCReceiver

    dispatcher = Dispatcher(
        team_infos=team_infos,
        our_color='blue',        # 'blue' or 'yellow', set from GC team name
        our_teamname='TritonBots',
    )

    while True:
        game_state = networker.get_game_state()
        gc_state   = gc_receiver.get_latest()       # may be None initially
        actions    = dispatcher.decide_action(game_state, gc_state)
        networker.execute_ai_output(actions, dispatcher.our_teamname)

Coordinate conversion
---------------------
The GC publishes designated_position in meters using the SSL field frame
(origin at center, x toward right goal, y toward top touch line).  Our sim
uses field units where 10 units = 1 m, so the dispatcher multiplies by
field_units_per_meter before calling update_designated_ball_pos().
"""

import importlib
from typing import Type

from networking.data_utils import GameState, TeamInfo
from networking.gc_state import GCState
from ai_interface.utils.algo_utils import field_units_per_meter

# ------------------------------------------------------------------ #
#  Command → module mapping                                           #
# ------------------------------------------------------------------ #

# Maps Referee.Command name → module path inside `support/`.
# NORMAL_START is context-dependent; handled separately in _resolve_module().
_COMMAND_MODULE: dict[str, str] = {
    'HALT':                      'support.Halt',
    'STOP':                      'support.Stop',
    'TIMEOUT_BLUE':              'support.Timeout',
    'TIMEOUT_YELLOW':            'support.Timeout',
    'BALL_PLACEMENT_BLUE':       'support.BallPlacement',
    'BALL_PLACEMENT_YELLOW':     'support.BallPlacement',
    'PREPARE_KICKOFF_BLUE':      'support.PrepareKickoff',
    'PREPARE_KICKOFF_YELLOW':    'support.PrepareKickoff',
    'PREPARE_PENALTY_BLUE':      'support.PreparePenalty',
    'PREPARE_PENALTY_YELLOW':    'support.PreparePenalty',
    'DIRECT_FREE_BLUE':          'support.FreeKick',
    'DIRECT_FREE_YELLOW':        'support.FreeKick',
    'FORCE_START':               'support.FreeKick',
    # NORMAL_START resolved at runtime based on previous command
    'NORMAL_START__KICKOFF':     'support.Kickoff',
    'NORMAL_START__PENALTY':     'support.Penalty',
}

# Commands that are "prepare" phases, used to resolve NORMAL_START
_KICKOFF_PREPARES  = {'PREPARE_KICKOFF_BLUE', 'PREPARE_KICKOFF_YELLOW'}
_PENALTY_PREPARES  = {'PREPARE_PENALTY_BLUE', 'PREPARE_PENALTY_YELLOW'}

# Field conversion constant
_FIELD_LENGTH_M = 9.0


# ------------------------------------------------------------------ #
#  Dispatcher                                                         #
# ------------------------------------------------------------------ #

class Dispatcher:
    """
    Selects and drives the AccessoryAlgo that matches the current GC command.

    Parameters
    ----------
    team_infos:
        Passed through to every AccessoryAlgo constructor.
    our_color:
        'blue' or 'yellow' — which SSL color we are playing as.
    our_teamname:
        The team name string used in GameState.robot_poses (simulator name).
    fallback_command:
        Module to use when gc_state is None (e.g. before first GC packet).
        Defaults to 'HALT' so robots stay still until the GC connects.
    """

    def __init__(
        self,
        team_infos: list[TeamInfo],
        our_color: str,
        our_teamname: str,
        fallback_command: str = 'HALT',
    ):
        self.team_infos = team_infos
        self.our_color = our_color              # 'blue' or 'yellow'
        self.our_teamname = our_teamname
        self._fallback_command = fallback_command

        self._upm = field_units_per_meter(_FIELD_LENGTH_M)
        self._current_command: str | None = None
        self._prev_command: str | None = None   # used to resolve NORMAL_START
        self._algo = None                       # current AccessoryAlgo instance

    # ---------------------------------------------------------------- #
    #  Main entry point                                                 #
    # ---------------------------------------------------------------- #

    def decide_action(
        self,
        game_state: GameState,
        gc_state: GCState | None,
    ) -> list[str]:
        """
        Return one action string per robot on our team.

        Switches to a new AccessoryAlgo whenever the GC command changes.
        """
        command = self._effective_command(gc_state)
        if command != self._current_command:
            self._transition(command, gc_state)

        if gc_state is not None and gc_state.is_ball_placement():
            self._update_ball_placement(gc_state)

        return self._algo.decide_action(game_state, self.our_teamname)

    # ---------------------------------------------------------------- #
    #  State transitions                                               #
    # ---------------------------------------------------------------- #

    def _effective_command(self, gc_state: GCState | None) -> str:
        """Resolve the canonical command key, handling NORMAL_START."""
        if gc_state is None:
            return self._fallback_command

        cmd = gc_state.command

        if cmd == 'NORMAL_START':
            if self._prev_command in _KICKOFF_PREPARES:
                return 'NORMAL_START__KICKOFF'
            if self._prev_command in _PENALTY_PREPARES:
                return 'NORMAL_START__PENALTY'
            # Unexpected NORMAL_START without a prepare phase; treat as FreeKick
            return 'DIRECT_FREE_BLUE'

        return cmd

    def _transition(self, command: str, gc_state: GCState | None) -> None:
        """Instantiate the AccessoryAlgo for the new command."""
        module_path = _COMMAND_MODULE.get(command)
        if module_path is None:
            print(f'[Dispatcher] Unknown command "{command}" — falling back to Halt')
            module_path = 'support.Halt'

        module = importlib.import_module(module_path)
        self._algo = module.AccessoryAlgo(self.team_infos)

        # Wire Penalty / PrepareKickoff attack direction
        if hasattr(self._algo, 'attack_direction'):
            self._algo.attack_direction = self._attack_direction(gc_state)

        # Wire Penalty placing side
        if hasattr(self._algo, 'penalty_against'):
            self._algo.penalty_against = self._penalty_is_against_us(command)

        # Track command history AFTER the transition so _prev_command is the
        # command we're transitioning *from*, not the new one.
        self._prev_command = self._current_command
        self._current_command = command

    # ---------------------------------------------------------------- #
    #  BallPlacement wiring                                            #
    # ---------------------------------------------------------------- #

    def _update_ball_placement(self, gc_state: GCState) -> None:
        """Push designated position and team flags into the BallPlacement algo."""
        algo = self._algo
        if not hasattr(algo, 'update_designated_ball_pos'):
            return

        if gc_state.designated_pos is not None:
            x_m, y_m = gc_state.designated_pos
            algo.update_designated_ball_pos(x_m * self._upm, y_m * self._upm)

        placing_color = gc_state.placing_color()
        algo.is_placing_team = (placing_color == self.our_color)

        # Derive next_command from the GCState field or fall back to 'free_kick'
        if gc_state.next_command in ('FORCE_START',):
            algo.next_command = 'force_start'
        else:
            algo.next_command = 'free_kick'

    # ---------------------------------------------------------------- #
    #  Field geometry helpers                                          #
    # ---------------------------------------------------------------- #

    def _attack_direction(self, gc_state: GCState | None) -> int:
        """
        Return +1 if we attack toward positive x (right goal), -1 otherwise.

        SSL convention: blue_on_positive_half = True means blue defends the
        positive-x goal, so blue *attacks* toward negative x (-1) and yellow
        attacks toward positive x (+1).
        """
        if gc_state is None:
            return 1
        blue_defends_positive = gc_state.blue_on_positive_half
        if self.our_color == 'blue':
            return -1 if blue_defends_positive else 1
        else:
            return 1 if blue_defends_positive else -1

    def _penalty_is_against_us(self, command: str) -> bool:
        """True when the penalty is being taken *against* our team."""
        if command in ('PREPARE_PENALTY_BLUE', 'NORMAL_START__PENALTY'):
            return self.our_color == 'yellow'
        if command in ('PREPARE_PENALTY_YELLOW',):
            return self.our_color == 'blue'
        return False
