"""
Main entry point for the AI Team soccer robot control system.

This module provides the main program for controlling RoboCup soccer teams
in various environments including simulation and physical robot scenarios.
"""

import argparse
import threading
from networking.networker import TeamInfo, GameState, Networker
from networking.gc_receiver import GCReceiver, MockGCReceiver
from support.dispatcher import Dispatcher

UCSD_ROBOCUP_TEAM_NAME = "TritonBots"

parser = argparse.ArgumentParser()
parser.add_argument("--teamname", type=str, default=UCSD_ROBOCUP_TEAM_NAME)
parser.add_argument("--color", choices=["blue", "yellow"], default="blue",
                    help="Which SSL color we are assigned by the referee")
parser.add_argument("--env", choices=[
    "sim-only",        # one or both teams in simulator
    "sim-mixed",       # one or both teams - simulator + physical robots
    "field-practice",  # one or both teams - camera + physical robots
    "field-tournament" # our team only - camera + physical robots
], default="sim-only")
parser.add_argument("--mock-gc", action="store_true",
                    help="Use MockGCReceiver (no GC binary needed); robots halt")


def main():
    args = parser.parse_args()
    team_infos = [TeamInfo(args.teamname, 6), TeamInfo("TeamB", 6)]

    networker = Networker(team_infos, args.env)

    # Game-controller receiver: real or mock
    if args.mock_gc or args.env == "sim-only":
        gc_receiver = MockGCReceiver()
        gc_receiver.start()
        print("[main] Using MockGCReceiver — robots will halt until GC state is pushed.")
    else:
        gc_receiver = GCReceiver()
        gc_receiver.start()
        print("[main] Listening for ssl-game-controller on 224.5.23.1:10003")

    # One Dispatcher per team (each needs its own color / attack direction)
    dispatchers: dict[str, Dispatcher] = {
        team_info.name: Dispatcher(
            team_infos=team_infos,
            our_color=args.color if team_info.name == args.teamname else _opposite(args.color),
            our_teamname=team_info.name,
        )
        for team_info in team_infos
    }

    try:
        while True:
            game_state = networker.get_game_state()
            if game_state is None:
                continue

            gc_state = gc_receiver.get_latest()

            if args.env == "field-tournament":
                _process_team(dispatchers[args.teamname], networker, game_state, gc_state)
            else:
                threads = []
                for team_info in team_infos:
                    t = threading.Thread(
                        target=_process_team,
                        args=(dispatchers[team_info.name], networker, game_state, gc_state),
                    )
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

    except KeyboardInterrupt:
        print("\nShutting down…")
        gc_receiver.stop()
        if args.env in ["sim-only", "sim-mixed"]:
            networker.disconnect_from_sim()


def _process_team(
    dispatcher: Dispatcher,
    networker: Networker,
    game_state: GameState,
    gc_state,
) -> None:
    actions = dispatcher.decide_action(game_state, gc_state)
    networker.execute_ai_output(actions, dispatcher.our_teamname)


def _opposite(color: str) -> str:
    return "yellow" if color == "blue" else "blue"


if __name__ == "__main__":
    main()
