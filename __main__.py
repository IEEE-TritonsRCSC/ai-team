"""
Main entry point for the AI Team soccer robot control system.

This module provides the main program for controlling RoboCup soccer teams
in various environments including simulation and physical robot scenarios.
"""

import json
import argparse
import threading
import time
from queue import Queue, Full, Empty
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
parser.add_argument("--demo", type=str, default=None, help="Run a demo: enter AI demo file name")
parser.add_argument("--estimate", choices=["ball", "player"], dest="estimate_params", default=None,)


def main():
    """
    Main execution function for the soccer AI system.
    
    Parses command-line arguments, initializes team information and AI components,
    and runs the main game loop that processes game states and executes AI decisions.
    """
    args = parser.parse_args()
    
    if args.estimate_params is not None:
        from ai_interface.utils.param_estimator import ParamEstimatorAI
        team_infos = load_team_config("ai_interface/utils/estimator_config.json")
        soccer_ai = ParamEstimatorAI(team_infos, mode=args.estimate_params)
    else:
        team_infos = load_team_config(args.team_config)
        if args.demo is None:
            soccer_ai = SoccerAI(team_infos)
            print("Running default SoccerAI.")
        else:
            demo_name = args.demo
            try:
                class_name = ''.join(p.capitalize() for p in demo_name.split('_')) + 'AI'
                soccer_ai_class = getattr(__import__("ai_interface.demos." + demo_name, fromlist=[demo_name]), class_name)
                print(f"Running demo: {demo_name}.")
            except (ImportError, AttributeError) as e:
                raise RuntimeError(f"Error: Could not load demo '{demo_name}'. Falling back to default SoccerAI.") from e
            soccer_ai = soccer_ai_class(team_infos)
    networker = Networker(team_infos, args.env)
    game_state_queue: Queue[GameState] = Queue(maxsize=1)
    stop_event = threading.Event()
    client_thread = None
    if args.env in ["sim-only", "sim-mixed"]:
        sample_client = networker.commander.sample_client
        if sample_client is not None:
            client_thread = threading.Thread(
                target=_client_watch_worker,
                args=(networker, sample_client, stop_event),
                daemon=True,
            )
            client_thread.start()
    state_thread = threading.Thread(
        target=_game_state_worker,
        args=(networker, game_state_queue, stop_event),
        daemon=True,
    )
    state_thread.start()

    try:
        while True:
            if stop_event.is_set():
                break
            try:
                game_state = game_state_queue.get(timeout=0.5)
            except Empty:
                continue
            print('Game State:', game_state)

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
    finally:
        stop_event.set()
        networker.shutdown()
        state_thread.join(timeout=1)
        if args.estimate_params is not None:
            soccer_ai.estimate()
        if client_thread is not None:
            client_thread.join(timeout=1)


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
        raise ValueError("Each team configuration must have name, n_players, and goalie_id.")
    
    return [TeamInfo(*team1_info), TeamInfo(*team2_info)]


def _game_state_worker(networker: Networker, state_queue: Queue,
                       stop_event: threading.Event) -> None:
    """
    Continuously fetch game states and keep only the most recent one.
    """
    while not stop_event.is_set():
        try:
            game_state = networker.get_game_state()
        except TimeoutError:
            stop_event.set()
            break
        if game_state is None:
            continue
        try:
            state_queue.put(game_state, block=False)
        except Full:
            try:
                state_queue.get(block=False)
            except Empty:
                pass
            try:
                state_queue.put(game_state, block=False)
            except Full:
                pass


def _client_watch_worker(networker: Networker, client,
                         stop_event: threading.Event) -> None:
    """
    Continuously receive simulator client updates in a dedicated thread.
    """
    while not stop_event.is_set():
        client_data = client.watch_game()
        if client_data is not None:
            networker.update_client_data(client_data)

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
