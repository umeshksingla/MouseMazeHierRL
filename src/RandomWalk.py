"""
RandomWalk
"""

import numpy as np
import pickle

import utils
import parameters as p
from BaseModel import BaseModel
from MM_Traj_Utils import NewMaze, Make2ndMarkov


class RandomWalk(BaseModel):

    def __init__(self, file_suffix='_RandomWalkTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.S = 128

    def generate_exploration_episode(self, MAX_LENGTH):
        ma = NewMaze()
        tf = Make2ndMarkov(ma, rs=np.random.seed(), n=MAX_LENGTH)
        self.episode_state_traj = sum(utils.convert_traj_to_episodes(tf), [])
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)
        _, episode_state_trajs, episode_maze_trajs, LL = self.generate_exploration_episode(MAX_LENGTH)
        stats = {
            "agentId": agentId,
            "episodes_states": episode_state_trajs,
            "episodes_positions": episode_maze_trajs,
            "MAX_LENGTH": MAX_LENGTH,
        }
        return success, stats


# Driver Code
if __name__ == '__main__':
    from sample_agent import run, load
    param_sets = [{}]*1
    runids = run(RandomWalk(), param_sets, '/Users/us3519/mouse-maze/figs/may28', '500000', analyze=False)
    print(runids)
