import argparse
import threading
from networking.networker import TeamInfo, GameState, Networker
from ai_interface.naive import SoccerAI

UCSD_ROBOCUP_TEAM_NAME = "TritonBots"

parser = argparse.ArgumentParser()
parser.add_argument("--teamname", type=str, default=UCSD_ROBOCUP_TEAM_NAME)
parser.add_argument("--env", choices=[
    "sim-only",        # one or both teams in simulator
    "sim-mixed",       # one or both teams - simulator + physical robots  
    "field-practice",  # one or both teams - camera + physical robots
    "field-tournament" # our team only - camera + physical robots
], default="sim-only")


def main():
    args = parser.parse_args()
    team_infos = [TeamInfo(args.teamname, 6), TeamInfo("TeamB", 6)]
    soccer_ai = SoccerAI()

    networker = Networker(team_infos, args.env)

    try:
        while True:
            game_state = networker.get_game_state()
            if game_state is None:
                continue
            print("Current Game State:", game_state)

            if args.env == "field-tournament":
                # In tournament mode, we only control our own team
                process_team(soccer_ai, networker, game_state, args.teamname)
            else:
                # Process both teams with threading
                threads = []
                for team_info in team_infos:
                    t_args = (soccer_ai, networker, game_state, team_info.name)
                    thread = threading.Thread(target=process_team, args=t_args)
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()


    except KeyboardInterrupt:
        print("\nShutting down...please patiently wait for a few seconds.")
        if args.env in ["sim-only", "sim-mixed"]:
            networker.disconnect_from_sim()


def process_team(soccer_ai: SoccerAI, networker: Networker,
                 game_state: GameState, team_name: str):
    ai_output = soccer_ai.decide_action(game_state, team_name)
    translated = soccer_ai.translate_ai_output(ai_output)
    networker.execute_ai_output(translated, team_name)


if __name__ == "__main__":
    main()
