"""Drive a FreeKick scenario against a running rcssserver and log behavior.

Prereq: rcssserver must already be listening on 127.0.0.1:6000/6001, e.g.
    rcssserver server::coach_w_referee=true

Usage:
    python scripts/freekick_sim_check.py DIRECT_FREE_BLUE [frames]

In sim-only mode both teams are driven: TritonBots as `blue`, TeamB as `yellow`.
DIRECT_FREE_BLUE  -> TritonBots attacks, TeamB defends.
DIRECT_FREE_YELLOW-> TeamB attacks, TritonBots defends.

Reports, sampled over time: ball displacement (=> ball-in-play), and for each
team the closest robot's distance to the ball (defenders must stay >= 0.5 m
until the ball is in play).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_interface.utils.algo_utils import distance, robot_id_and_pose
from networking.gc_receiver import MockGCReceiver
from networking.networker import Networker, TeamInfo
from support.dispatcher import Dispatcher

BLUE_TEAM = "TritonBots"
YELLOW_TEAM = "TeamB"
UPM = 10.0  # field units per meter (FIELD_X span 90 / 9 m)


def _closest_to_ball(poses, ball):
    if not poses or ball is None:
        return None
    return min(distance(p[:2], ball[:2]) for _, p in poses)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "DIRECT_FREE_BLUE"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    team_infos = [TeamInfo(BLUE_TEAM, 6), TeamInfo(YELLOW_TEAM, 6)]
    print(f"[harness] connecting to sim, command={command}, frames={frames}")
    networker = Networker(team_infos, "sim-only")

    mock = MockGCReceiver()
    mock.push_command(command)

    dispatchers = {
        BLUE_TEAM: Dispatcher(team_infos, our_color="blue", our_teamname=BLUE_TEAM),
        YELLOW_TEAM: Dispatcher(team_infos, our_color="yellow", our_teamname=YELLOW_TEAM),
    }

    defending = {
        "DIRECT_FREE_BLUE": YELLOW_TEAM,
        "DIRECT_FREE_YELLOW": BLUE_TEAM,
    }.get(command)

    # Optional: teleport ball + kicker into the clear so the attacker can
    # actually take the kick (default spawn crowds 12 robots at center).
    if "teleport" in sys.argv:
        trainer = networker.game_watcher.sock
        addr = networker.game_watcher.addr
        for msg in (b"(move (ball) 10 0)\0",
                    b"(move (player %s 5) 7 0 0)\0" % BLUE_TEAM.encode()):
            trainer.sendto(msg, addr)
        print("[harness] teleported ball->(10,0), TritonBots#5->(7,0)")

    start_ball = None
    sample_actions = {}
    for i in range(frames):
        gs = networker.get_game_state()
        if gs is None:
            continue
        gc = mock.get_latest()

        for name in (BLUE_TEAM, YELLOW_TEAM):
            actions = dispatchers[name].decide_action(gs, gc)
            networker.execute_ai_output(actions, name)
            if name not in sample_actions:
                sample_actions[name] = actions

        ball = gs.ball_pos
        if ball is not None and start_ball is None:
            start_ball = ball[:2]

        if i % 25 == 0 and ball is not None:
            blue = [robot_id_and_pose(r) for r in gs.robot_poses.get(BLUE_TEAM, [])]
            yel = [robot_id_and_pose(r) for r in gs.robot_poses.get(YELLOW_TEAM, [])]
            moved = distance(ball[:2], start_ball) / UPM if start_ball else 0.0
            cb = _closest_to_ball(blue, ball)
            cy = _closest_to_ball(yel, ball)
            print(
                f"[f{i:3d}] ball=({ball[0]:6.1f},{ball[1]:6.1f}) moved={moved:5.2f}m"
                f" | nearest->ball  {BLUE_TEAM}={'-' if cb is None else f'{cb/UPM:4.2f}m'}"
                f"  {YELLOW_TEAM}={'-' if cy is None else f'{cy/UPM:4.2f}m'}"
                f" | in_play(blue_algo)={dispatchers[BLUE_TEAM]._algo.ball_in_play}"
            )

    print("\n[harness] sample first-frame actions:")
    for name, acts in sample_actions.items():
        print(f"  {name}: {acts}")
    print(f"[harness] defending team this scenario: {defending}")
    networker.disconnect_from_sim()


if __name__ == "__main__":
    main()
