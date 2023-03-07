"""
Biased walk model by original author
"""

import numpy as np
import pickle

import utils
import parameters as p
from BaseModel import BaseModel
from MM_Traj_Utils import NewMaze, Make2ndMarkov


class BiasedWalk4(BaseModel):

    def __init__(self, file_suffix='_BiasedWalk4Trajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.S = 128
        with open(p.OUTDATA_PATH + 'decision_biases_unrewarded.pkl', 'rb') as f:
            self.bi_data = pickle.load(f)
        print(self.bi_data)

    def generate_exploration_episode(self, MAX_LENGTH):
        ma = NewMaze()
        mean_animal_biases = np.mean(self.bi_data, axis=0)
        print("mean_animal_biases", mean_animal_biases)
        tf = Make2ndMarkov(ma, n=MAX_LENGTH, bi=mean_animal_biases)
        self.episode_state_traj = sum(utils.convert_traj_to_episodes(tf), [])
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)
        Q = np.zeros((self.S, self.A))
        _, episode_state_trajs, episode_maze_trajs, LL = self.generate_exploration_episode(MAX_LENGTH)
        stats = {
            "agentId": agentId,
            "episodes_states": episode_state_trajs,
            "episodes_positions": episode_maze_trajs,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
        }
        return success, stats

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


# Driver Code
if __name__ == '__main__':
    from sample_agent import run, load
    param_sets = [{'rew': True}]
    runids = run(BiasedWalk4(), param_sets, '/Users/usingla/mouse-maze/figs', '35000')
    print(runids)
    base_path = '/Users/usingla/mouse-maze/figs/'
    load([
        ('BiasedWalk4', runids)
    ], base_path)