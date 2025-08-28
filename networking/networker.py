from .data_utils import GameState, Serializer
from .socket_utils import TeamInfo, Listener, Commander

class Networker:
    def __init__(self, team_infos: list[TeamInfo], environment: str):
        self.environment = environment
        self.serializer = Serializer()
        self.commander = Commander(team_infos, environment)
        self.game_watcher = Listener(environment)

    def get_game_state(self) -> GameState:
        return self.game_watcher.watch_game()

    def execute_ai_output(self, output: list[str], team_name: str):
        if self.environment in ["sim-only", "sim-mixed"]:
            messages = self.serializer.sim_serialize(output)
            self.commander.send_to_sim(team_name, messages)

        if self.environment != "sim-only":
            messages = self.serializer.robot_serialize(output)
            self.commander.send_to_robots(team_name, messages)

    def disconnect_from_sim(self):
        self.commander.disconnect_from_sim()
        self.game_watcher.disconnect_from_sim()
