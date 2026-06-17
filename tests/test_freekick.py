"""FreeKick (Division B) team-behavior unit tests.

No simulator required: AccessoryAlgo is fed fabricated GameState namedtuples and
its geometry helpers are checked directly. Rule distances are in meters; the
field uses 10 units = 1 m, so 0.05 m = 0.5 u, 0.2 m = 2 u, 0.5 m = 5 u.

Run: python -m pytest tests/test_freekick.py -q
  or python tests/test_freekick.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_interface.constants.field_constants import FIELD_X
from ai_interface.constants.player_constants import PLAYER_SIZE
from ai_interface.utils.algo_utils import distance
from networking.data_utils import GameState
from support.dispatcher import Dispatcher
from support.FreeKick import AccessoryAlgo

TEAM = "TritonBots"
OPP = "TeamB"


def _gs(ball, our, opp=None, count=0, timestamp=0.0):
    """Fabricate a GameState. `our`/`opp` are lists of (unum, (x, y, theta))."""
    poses = {TEAM: [{u: p} for u, p in our]}
    if opp is not None:
        poses[OPP] = [{u: p} for u, p in opp]
    return GameState(count, timestamp, tuple(ball), poses)


def _algo(against=False, attack_direction=1):
    algo = AccessoryAlgo([])
    algo.attack_direction = attack_direction
    algo.free_kick_against = against
    return algo


# --------------------------------------------------------------------------- #
#  Dispatcher ownership wiring                                                 #
# --------------------------------------------------------------------------- #

def test_dispatcher_free_kick_ownership():
    blue = Dispatcher(team_infos=[], our_color="blue", our_teamname=TEAM)
    yellow = Dispatcher(team_infos=[], our_color="yellow", our_teamname=TEAM)

    # DIRECT_FREE_BLUE -> blue attacks, yellow defends.
    assert blue._free_kick_is_against_us("DIRECT_FREE_BLUE") is False
    assert yellow._free_kick_is_against_us("DIRECT_FREE_BLUE") is True
    # DIRECT_FREE_YELLOW -> yellow attacks, blue defends.
    assert yellow._free_kick_is_against_us("DIRECT_FREE_YELLOW") is False
    assert blue._free_kick_is_against_us("DIRECT_FREE_YELLOW") is True
    # FORCE_START -> out of scope (None).
    assert blue._free_kick_is_against_us("FORCE_START") is None


# --------------------------------------------------------------------------- #
#  Ball-in-play tracking                                                       #
# --------------------------------------------------------------------------- #

def test_ball_in_play_on_movement():
    algo = _algo(against=False)
    our = [(1, (-40.0, 0.0, 0.0)), (5, (-1.0, 0.0, 0.0))]
    algo.decide_action(_gs((0.0, 0.0), our, opp=[], timestamp=0.0), TEAM)
    assert not algo.ball_in_play
    # Move ball 0.06 m (0.6 u) > 0.05 m threshold.
    algo.decide_action(_gs((0.6, 0.0), our, opp=[], timestamp=0.1), TEAM)
    assert algo.ball_in_play


def test_ball_in_play_on_timeout():
    algo = _algo(against=True)
    our = [(1, (40.0, 0.0, 0.0)), (2, (10.0, 5.0, 0.0))]
    algo.decide_action(_gs((0.0, 0.0), our, opp=[(5, (-1.0, 0.0, 0.0))], timestamp=0.0), TEAM)
    assert not algo.ball_in_play
    # 11 s later, ball unmoved -> Division B timeout puts it in play.
    algo.decide_action(_gs((0.0, 0.0), our, opp=[(5, (-1.0, 0.0, 0.0))], timestamp=11.0), TEAM)
    assert algo.ball_in_play


# --------------------------------------------------------------------------- #
#  Defending constraints                                                       #
# --------------------------------------------------------------------------- #

def test_enforce_ball_clearance_pushes_to_ring():
    algo = _algo(against=True)
    ball = (0.0, 0.0)
    # The 0.5 m ring is measured from the robot's edge, so the center clears
    # defender_min_ball + PLAYER_SIZE (robot radius).
    min_center = algo.defender_min_ball + PLAYER_SIZE
    pushed = algo._enforce_ball_clearance((2.0, 0.0), ball)  # inside the ring
    assert math.isclose(distance(pushed, ball), min_center, rel_tol=1e-6)
    # Already-legal target is left untouched.
    legal = (8.0, 0.0)
    assert algo._enforce_ball_clearance(legal, ball) == legal


def test_wall_targets_respect_ball_distance():
    algo = _algo(against=True)
    ball = (0.0, 0.0)
    _, _, goal_center = __import__(
        "ai_interface.utils.algo_utils", fromlist=["get_goal_params"]
    ).get_goal_params(algo._defended_goal_side())
    for target in algo._wall_targets(ball, goal_center):
        assert distance(target, ball) >= algo.defender_min_ball - 1e-6


def test_defending_actions_one_per_robot():
    algo = _algo(against=True)
    our = [(1, (40.0, 0.0, 0.0)), (2, (20.0, 5.0, 0.0)),
           (3, (20.0, -5.0, 0.0)), (4, (15.0, 8.0, 0.0)),
           (5, (15.0, -8.0, 0.0)), (6, (25.0, 0.0, 0.0))]
    opp = [(5, (0.5, 0.0, 0.0)), (7, (-5.0, 3.0, 0.0)), (8, (-10.0, -4.0, 0.0))]
    actions = algo.decide_action(_gs((0.0, 0.0), our, opp=opp, timestamp=0.0), TEAM)
    assert len(actions) == len(our)
    assert all(isinstance(a, str) for a in actions)


# --------------------------------------------------------------------------- #
#  Opponent defense-area clearance                                             #
# --------------------------------------------------------------------------- #

def test_respect_defense_area_pushes_out():
    algo = _algo(against=False, attack_direction=1)
    # Opponent box (attack right): x in [35, 45], |y| <= 10 (units). Margin ~2.9 u.
    inside = (36.0, 0.0)
    out = algo._respect_defense_area(inside)
    inner_x = FIELD_X[1] - algo.defense_area_depth
    margin = algo.defense_area_min  # min legal clearance to the area itself
    # New x must be at least `margin` outside the box inner edge.
    assert out[0] <= inner_x - margin + 1e-6
    # A target nowhere near the box is unchanged.
    far = (0.0, 0.0)
    assert algo._respect_defense_area(far) == far


# --------------------------------------------------------------------------- #
#  Attacking: shot lane + double-touch avoidance                               #
# --------------------------------------------------------------------------- #

def test_corridor_clear_detects_blocker():
    algo = _algo(against=False)
    ball, goal = (0.0, 0.0), (45.0, 0.0)
    blocker = [(7, (20.0, 0.1, 0.0))]          # on the line, ahead of ball
    assert not algo._corridor_clear(ball, goal, blocker, algo.shot_corridor)
    wide = [(7, (20.0, 9.0, 0.0))]             # far off the line
    assert algo._corridor_clear(ball, goal, wide, algo.shot_corridor)
    behind = [(7, (-5.0, 0.0, 0.0))]           # behind the ball: ignored
    assert algo._corridor_clear(ball, goal, behind, algo.shot_corridor)


def test_kicker_demoted_after_kick():
    algo = _algo(against=False, attack_direction=1)
    robot_poses = [(1, (-40.0, 0.0, 0.0)), (5, (-1.0, 0.0, 0.0)),
                   (2, (-5.0, 3.0, 0.0)), (3, (-5.0, -3.0, 0.0))]
    algo.kicker_id = 5
    algo.ball_kicked = True
    # Once kicked, the chaser is a different robot than the kicker...
    chaser = algo._chaser_id(robot_poses, goalie_id=1, ball=(0.0, 0.0))
    assert chaser is not None and chaser != algo.kicker_id
    # ...and the kicker is sent to its release (retreat) target, never the ball.
    targets = algo._support_targets(robot_poses, goalie_id=1, chaser_id=chaser)
    assert targets[algo.kicker_id] == algo._kicker_release_target()


def test_attacking_actions_one_per_robot():
    algo = _algo(against=False, attack_direction=1)
    our = [(1, (-40.0, 0.0, 0.0)), (2, (-5.0, 5.0, 0.0)),
           (3, (-5.0, -5.0, 0.0)), (4, (-10.0, 8.0, 0.0)),
           (5, (-1.0, 0.0, 0.0)), (6, (-15.0, 0.0, 0.0))]
    opp = [(1, (40.0, 0.0, 0.0)), (7, (10.0, 1.0, 0.0))]
    actions = algo.decide_action(_gs((0.0, 0.0), our, opp=opp, timestamp=0.0), TEAM)
    assert len(actions) == len(our)
    assert all(isinstance(a, str) for a in actions)


def test_force_start_ball_in_play_immediately():
    # FORCE_START: ball must be live from frame 0 — no 0.05 m / 10 s wait.
    algo = _algo(against=None)
    our = [(1, (-40.0, 0.0, 0.0)), (5, (-1.0, 0.0, 0.0))]
    algo.decide_action(_gs((0.0, 0.0), our, opp=[], timestamp=0.0), TEAM)
    assert algo.ball_in_play


def test_force_start_contests_ball():
    # FORCE_START: robots must not freeze — actions should not all be "dash 0 0".
    algo = _algo(against=None)
    our = [(1, (-40.0, 0.0, 0.0)), (2, (-5.0, 3.0, 0.0)),
           (3, (-5.0, -3.0, 0.0)), (4, (-10.0, 5.0, 0.0)),
           (5, (-1.0, 0.0, 0.0)), (6, (-15.0, 0.0, 0.0))]
    opp = [(1, (40.0, 0.0, 0.0)), (7, (10.0, 1.0, 0.0))]
    actions = algo.decide_action(_gs((0.0, 0.0), our, opp=opp, timestamp=0.0), TEAM)
    assert len(actions) == len(our)
    assert all(isinstance(a, str) for a in actions)
    assert not all(a == "dash 0 0" for a in actions)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
