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
from ai_interface.intercept_demo import InterceptDemoAI

UCSD_ROBOCUP_TEAM_NAME = "TritonBots"

parser = argparse.ArgumentParser()

parser.add_argument("--team_config", type=str, default="team_config_competition.json",
    help="Path to team configuration JSON file"
)

parser.add_argument("--env", choices=[
    "sim-only",        # one or both teams in simulator
    "sim-embedded",    # one or both teams in the synchronous embedded simulator
    "sim-mixed",       # one or both teams - simulator + physical robots
    "field-practice",  # one or both teams - camera + physical robots
    "field-tournament" # our team only - camera + physical robots
], default="field-tournament")
parser.add_argument("--estimate", choices=["ball", "player"], dest="estimate_params", default=None,)
parser.add_argument("--sim_host", type=str, default="127.0.0.1",
    help="Simulator host for sim-only/sim-mixed")
parser.add_argument("--sim_player_port", type=int, default=6000,
    help="Simulator player port")
parser.add_argument("--sim_trainer_port", type=int, default=6001,
    help="Simulator trainer/monitor port")

# Competition / PPO JAL args
parser.add_argument("--model_path", type=str, default="final_models/stage3_complete.pt",
    help="Path to PPO JAL checkpoint for robot 1")
parser.add_argument("--stage", type=str, default="stage4_hardcoded_support_2v2",
    help="Curriculum stage key to load env settings from")
parser.add_argument("--config", type=str, default="configs/ppo_jal_curriculum_config.json",
    help="Path to curriculum config JSON")
parser.add_argument("--our_side", choices=["left", "right"], default="left",
    help="Which side TritonBots defends: left=defend left goal, attack right; right=flipped")
parser.add_argument("--ppo_param_noise_std", type=float, default=0.3,
    help="Gaussian noise std on PPO params to prevent bang-bang turn stall")
parser.add_argument("--our_color", choices=["blue", "yellow"], default="blue",
    help="Our team's SSL color assigned by the Game Controller")


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
        competition_ai = None
    elif args.env == "field-tournament":
        from ai_interface.competition_ai import CompetitionAI
        team_infos = load_team_config(args.team_config)
        competition_ai = CompetitionAI(team_infos, args)
        soccer_ai = None
    else:
        team_infos = load_team_config(args.team_config)
        team_infos = _apply_side_order(team_infos, args.env, getattr(args, "our_side", "left"))
        soccer_ai = SoccerAI(team_infos)
        competition_ai = None

    networker = Networker(
        team_infos,
        args.env,
        sim_host=args.sim_host,
        sim_player_port=args.sim_player_port,
        sim_trainer_port=args.sim_trainer_port,
    )

    if competition_ai is not None:
        competition_ai.setup(networker)

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

            if competition_ai is not None:
                competition_ai.step(game_state)
            elif args.env == "field-tournament":
                process_team(soccer_ai, networker, game_state, UCSD_ROBOCUP_TEAM_NAME)
            else:
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
        if competition_ai is not None and hasattr(competition_ai, 'shutdown'):
            competition_ai.shutdown()
        networker.shutdown()
        state_thread.join(timeout=1)
        if args.estimate_params is not None:
            soccer_ai.estimate()
        if client_thread is not None:
            client_thread.join(timeout=1)


def _apply_side_order(team_infos: list, env: str, our_side: str) -> list:
    """
    Reorder team_infos so TritonBots spawns on the requested simulator side.

    The simulator backends (socket_utils.py) assign spawn side purely by list
    order (team_infos[0] = left, team_infos[1] = right) — they have no concept
    of --our_side. Reordering here is the only way to make --our_side actually
    move TritonBots's spawn position in sim; it's a no-op for field modes,
    which have no team-order concept (camera + physical placement instead).
    """
    if env not in ("sim-only", "sim-embedded", "sim-mixed"):
        return team_infos
    if len(team_infos) != 2:
        return team_infos
    triton_idx = next(
        (i for i, t in enumerate(team_infos) if t.name == UCSD_ROBOCUP_TEAM_NAME), None
    )
    if triton_idx is None:
        return team_infos
    wants_first = (our_side == "left")
    is_first = (triton_idx == 0)
    if wants_first != is_first:
        return [team_infos[1], team_infos[0]]
    return team_infos


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
