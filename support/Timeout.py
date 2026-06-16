# Assignee: Lukas Cao
from networking.data_utils import GameState, TeamInfo


class AccessoryAlgo:
    """
    Timeout game state behavior.

    During a timeout, the game is halted and both teams may freely modify
    their software and hardware (SSL rules §4.4.2). Any robot that was
    physically touched must be removed from the field via the substitution
    procedure. From the robots' perspective this means: stop all motion and
    hold position until the game resumes.

    Timeout budget (tracked externally by the game controller):
      - Regular game : 4 timeouts, 300 s total per team
      - Overtime     : 2 timeouts, 150 s total per team
      - Shoot-out    : no timeouts allowed
    """

    def __init__(self, team_infos: list[TeamInfo]):
        self.team_infos = team_infos
        self._all_dropped = False

    def decide_action(self, game_state: GameState, teamname: str) -> list[str]:
        """
        Return one action string per robot on teamname.

        Robots drop held momentum on the first cycle, then idle.  The game
        controller issues a Stop command when the timeout ends, which
        transitions us out of this state.
        """
        team_robots = game_state.robot_poses[teamname]
        actions = []

        if not self._all_dropped:
            self._all_dropped = True
            return ['drop'] * len(team_robots)

        for _ in team_robots:
            actions.append('dash 0 0')
        return actions
