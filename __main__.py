"""
Main entry point for the AI Team soccer robot control system.

This module provides the main program for controlling RoboCup soccer teams
in various environments including simulation and physical robot scenarios.
"""

import json
import argparse
import threading
from networking.networker import TeamInfo, GameState, Networker
from ai_interface.naive import SoccerAI

UCSD_ROBOCUP_TEAM_NAME = "TritonBots"

parser = argparse.ArgumentParser()

parser.add_argument("--team_config", type=str, default="team_config.json",
    help="Path to team configuration JSON file"
)

parser.add_argument("--env", choices=[
    "sim-only",        # one or both teams in simulator
    "sim-mixed",       # one or both teams - simulator + physical robots  
    "field-practice",  # one or both teams - camera + physical robots
    "field-tournament" # our team only - camera + physical robots
], default="sim-only")


def main():
    """
    Main execution function for the soccer AI system.
    
    Parses command-line arguments, initializes team information and AI components,
    and runs the main game loop that processes game states and executes AI decisions.
    """
    args = parser.parse_args()
    team_infos = load_team_config(args.team_config)
    soccer_ai = SoccerAI(team_infos)
    networker = Networker(team_infos, args.env)

    try:
        while True:
            game_state = networker.get_game_state()
            if game_state is None:
                continue
            print("Current Game State:", game_state)

            if args.env == "field-tournament":
                # In tournament mode, we only control our own team
                process_team(soccer_ai, networker, game_state, UCSD_ROBOCUP_TEAM_NAME)
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


def load_team_config(file_path: str) -> list[TeamInfo]:
    """
    Load team configuration from a JSON file.
    
    Args:
        file_path: Path to the JSON configuration file
    Returns:
        List of TeamInfo objects representing team configurations
    Raises:
        ValueError: If the configuration file is malformed
    """
    with open(file_path, 'r') as f:
        config = json.load(f)["teams"]
    
    if len(config) != 2:
        raise ValueError("Team configuration must contain exactly two teams.")
    
    team1_info, team2_info = config
    if len(team1_info) != 3 or len(team2_info) != 3:
        raise ValueError("Each team configuration must have " + \
                         "name, n_players, and goalie_id.")
    
    return [TeamInfo(*team1_info), TeamInfo(*team2_info)]


def process_team(soccer_ai: SoccerAI, networker: Networker,
                 game_state: GameState, team_name: str):
    """
    Process AI decisions for a specific team.
    
    Args:
        soccer_ai: The AI instance that makes decisions
        networker: The networking component for command execution
        game_state: Current state of the game
        team_name: Name of the team to process
    """
    ai_output = soccer_ai.decide_action(game_state, team_name)
    translated = soccer_ai.translate_ai_output(ai_output)
    networker.execute_ai_output(translated, team_name)


if __name__ == "__main__":
    main()
