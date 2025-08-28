from networking.data_utils import GameState

class SoccerAI:
    def __init__(self):
        pass

    def decide_action(self, game_state: GameState, teamname: str):
        actions = []
        kick = game_state.count % 2 == 0
        for robot in game_state.robot_poses[teamname]:
            unum = int(list(robot.keys())[0])
            if unum == 1:
                actions.append("kick 100 0" if kick else "dash 100")
            else:
                actions.append("dash 20")
        return actions

    def translate_ai_output(self, ai_output) -> list[str]:
        return ai_output
