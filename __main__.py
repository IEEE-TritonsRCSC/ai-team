
import argparse
import threading

from networking.networker import TeamInfo, GameState, Networker
from ai_interface.naive import SoccerAI

DEFAULT_TEAM_NAME = "TritonBots"

parser = argparse.ArgumentParser()
parser.add_argument("--teamname", type=str, default=DEFAULT_TEAM_NAME)
parser.add_argument(
    "--env",
    choices=[
        "sim-only",        # both teams in simulator
        "sim-mixed",       # simulator + physical robots
        "field-practice",  # camera + physical robots
        "field-tournament" # only our team, camera + physical robots
    ],
    default="sim-only",
)


def main():
    args = parser.parse_args()

    # 2 players per team: 1 goalkeeper, 1 shooter
    team_infos = [
        TeamInfo(args.teamname, 2),
        TeamInfo("TeamB", 2),
    ]

    soccer_ai = SoccerAI()
    networker = Networker(team_infos, args.env)

    try:
        while True:
            game_state: GameState | None = networker.get_game_state()
            if game_state is None:
                continue

            # Debug print (可以根据需要注释掉)
            print("Current Game State:", game_state)

            if args.env == "field-tournament":
                # Tournament: only control our own team
                process_team(soccer_ai, networker, game_state, args.teamname)
            else:
                # Other envs: control both teams (mirror match)
                threads: list[threading.Thread] = []
                for team_info in team_infos:
                    t = threading.Thread(
                        target=process_team,
                        args=(soccer_ai, networker, game_state, team_info.name),
                        daemon=True,
                    )
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

    except KeyboardInterrupt:
        print("\nShutting down... please wait a few seconds.")
        if args.env in ["sim-only", "sim-mixed"]:
            networker.disconnect_from_sim()


def process_team(
    soccer_ai: SoccerAI,
    networker: Networker,
    game_state: GameState,
    team_name: str,
):
    """
    Run AI for a single team and send commands via Networker.
    """
    ai_output = soccer_ai.decide_action(game_state, team_name)
    translated = soccer_ai.translate_ai_output(ai_output)

    try:
        networker.execute_ai_output(translated, team_name)
    except OSError as e:
        # Handle "No route to host" gracefully so threads don't crash.
        if getattr(e, "errno", None) == 65:
            print(f"[WARN] No route to host when sending to team {team_name}: {e}")
        else:
            raise


if __name__ == "__main__":
    main()
