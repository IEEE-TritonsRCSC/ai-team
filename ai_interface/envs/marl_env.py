from typing import Dict
from .sim_env import SimulatorEnv

class MultiAgentSoccerEnv:
    def __init__(self, envs: Dict[int, SimulatorEnv]):
        self.envs = envs
    
    def reset(self):
        return {aid: env.reset() for aid, env in self.envs.items()}
    
    def step(self, actions: Dict[int, Dict]):
        obs, rewards, dones, infos = {}, {}, {}, {}

        for aid, env in self.envs.items():
            obs[aid], rewards[aid], dones[aid], infos[aid] = env.step(actions[aid])
        
        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos