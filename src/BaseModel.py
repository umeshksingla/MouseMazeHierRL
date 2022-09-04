"""
Use this base class to define your model: from extracting data to loading states
and actions, to simulating agents, etc.
This is only supposed to provide a skeleton, feel free to override any
function.

For an example, refer to TDLambdaXSteps_model.py file that inherits from
this class. Refer to TDlambda20.ipynb for an example usage.
"""

import os
import numpy as np
import pickle
from multiprocessing import Pool

from MM_Traj_Utils import LoadTrajFromPath, NewMaze, StepType2
from parameters import INVALID_STATE, RWD_STATE, WATERPORT_NODE, LVL_BY_NODE, HOME_NODE, NODE_LVL, ALL_MAZE_NODES
from collections import defaultdict
from utils import break_simulated_traj_into_episodes


class BaseModel:
    def __init__(self, file_suffix='BaseModel'):
        """
        :param file_suffix: name used for saving model in a file
        """
        self.S = 129  # Number of states, including 127 maze nodes plus home and waterport states
        self.A = 3    # Number of max actions for a state
        self.file_suffix = file_suffix
        self.nodemap = self.get_SAnodemap()
        self.base_nodemap = self.get_SAnodemap_orig()
        self.nodemap_direction_dict = self.get_nodemap_direction_dict()
        self.terminal_nodes = {HOME_NODE, RWD_STATE}

    def get_initial_state(self):
        a=list(range(self.S))
        a.remove(28)
        a.remove(57)
        a.remove(115)
        a.remove(HOME_NODE)
        a.remove(WATERPORT_NODE)
        a.remove(RWD_STATE)
        return np.random.choice(a)    # Random initial state

    def extract_trajectory_data(self, mice_subject_list, orig_data_dir='../outdata/', save_dir=None):
        """
        save_dir: path to the directory where you want to save the pickled
        data object.
        """
        trajectory_data = []
        for mouseId, nickname in enumerate(mice_subject_list):
            trajectory_data.append(self.__get_trajectory_data_by_nickname__(orig_data_dir, nickname))
        if save_dir:
            with open(os.path.join(save_dir, f'{self.file_suffix}.p'), 'wb') as f:
                pickle.dump(trajectory_data, f)
        return trajectory_data

    def __get_trajectory_data_by_nickname__(self, orig_data_dir, nickname):
        """
        Returns ALL trajectory data for a mouse
        """
        print(f"Returning all the trajectories for {nickname}.")
        tf = LoadTrajFromPath(os.path.join(orig_data_dir, nickname + '-tf'))
        trajectory_data = []
        for boutId, bout in enumerate(tf.no):
            trajectory_data.append(bout[:, 0].tolist())
        return trajectory_data

    @staticmethod
    def __load_trajectories__(data):
        # TrajS   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)

        N = len(data)
        B = max([len(n) for n in data])
        BL = max([len(b) for n in data for b in n])

        TrajS = np.ones((N, B, BL)) * INVALID_STATE

        # over mouse
        for n in np.arange(len(data)):
            # over each of their bouts
            for b in np.arange(len(data[n])):
                # over each step of the bout
                for s in np.arange(len(data[n][b])):
                    TrajS[n, b, s] = data[n][b][s]
        return TrajS.astype(int)

    def load_trajectories_from_file(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return self.__load_trajectories__(data)

    def load_trajectories_from_object(self, trajectory_data):
        return self.__load_trajectories__(trajectory_data)

    def load_TrajA(self, TrajS, nodemap):
        # TrajA   : 3D matrix of (number of mice, number of bouts, number of steps in each bout)
        # TrajA   : Matrix entries are action indices (1, 2 or 3) taken to transition from t to t+1 in TrajS
        #           extra space in the matrix is filled with an invalid action, 0.
        #           Action values of 1 is a transition from a deep node, s to shallow node sprime
        #           Action values 2 and 3 are transitions from a shallow node, s to deeper nodes, sprime
        N, B, BL = TrajS.shape
        TrajA = np.zeros((N, B, BL)).astype(int)
        for n in np.arange(N):
            for b in np.arange(B):
                for bl in np.arange(BL - 1):
                    if TrajS[n, b, bl + 1] == INVALID_STATE or TrajS[n, b, bl + 1] == RWD_STATE:
                        break
                    TrajA[n, b, bl] = np.where(
                        nodemap[TrajS[n, b, bl], :] == TrajS[n, b, bl + 1]
                    )[0][0] + 1
        return TrajA

    def get_SAnodemap_orig(self):
        """
        Creates a mapping based on the maze layout where current states and actions
        are linked to the 3 possible future states (the states that would be the result
        of taking action A in state S).

        Action 0 returns to a shallower node, action 1 goes to deeper node with smaller index,
        and action 2 goes to deeper node with higher index. Actions 1 and 2 *do not* map to left and right.

        Returns: SAnodemap, a 2D array of (current state, action) to future state mappings
                 Also saves SAnodemap in the main_dir as 'nodemap.p'
        Return type: ndarray[(S, A), int], the int is the next state after taking action A in state S
        """
        SAnodemap = np.ones((self.S, self.A), dtype=int) * INVALID_STATE
        for node in np.arange(self.S - 1):
            # Shallow level node available from current node is accessed via action 0 (i.e. action 0 is the return)
            if node==0:
                SAnodemap[node, 0] = HOME_NODE
            elif node % 2 == 0:  # when current node number is even, action zero leads to the shallower node whose number is half of the current node minus two
                SAnodemap[node, 0] = (node - 2) / 2
            elif node % 2 == 1:  # when current node number is odd, action zero leads to the shallower node whose number is half of the current node minus one
                SAnodemap[node, 0] = (node - 1) / 2

            if node not in NODE_LVL[6]:  # level 6 nodes are not set here, which means they lead to invalid states unders actions 1 and 2
                # Deeper level nodes available from current node are accessed via action 1 and 2
                SAnodemap[node, 1] = node * 2 + 1 # action 1 leads to odd deeper node
                SAnodemap[node, 2] = node * 2 + 2 # action 2 leads to even deeper node

        # Nodes available from home node
        SAnodemap[HOME_NODE, 0] = INVALID_STATE
        SAnodemap[HOME_NODE, 1] = 0
        SAnodemap[HOME_NODE, 2] = INVALID_STATE

        # Nodes at RWD_STATE
        SAnodemap[RWD_STATE, 0] = INVALID_STATE
        SAnodemap[RWD_STATE, 1] = INVALID_STATE
        SAnodemap[RWD_STATE, 2] = INVALID_STATE

        # Arbitrarily set action 1 to lead to rwdstate,
        # but agent automatically goes from RWD_STATE to WATERPORT_NODE without any action, i.e., this is not
        # being used
        SAnodemap[WATERPORT_NODE, 1] = RWD_STATE
        return SAnodemap

    def get_SAnodemap(self):
        # print(f"Constructing the nodemap in {BaseModel.__name__} model ..")
        ma = NewMaze()
        SAnodemap = np.ones((self.S, self.A), dtype=int) * INVALID_STATE
        for i in range(0, self.S):
            if i not in ALL_MAZE_NODES:
                continue
            SAnodemap[i, 0] = ma.pa[i]
            if i not in NODE_LVL[6]:
                for j in [2 * i + 1, 2 * i + 2]:
                    a = StepType2(i, j, ma) + 1
                    # print(i, '=>', j, "action", a)
                    SAnodemap[i, a] = j
                # print("==")
        SAnodemap[0, 0] = HOME_NODE
        # Nodes available from home node
        SAnodemap[HOME_NODE, 0] = INVALID_STATE
        SAnodemap[HOME_NODE, 1] = 0
        SAnodemap[HOME_NODE, 2] = INVALID_STATE

        # Nodes at WATERPORT_NODE, see original def of nodemap in get_SAnodemap_orig
        SAnodemap[WATERPORT_NODE, 1] = RWD_STATE
        SAnodemap = SAnodemap.astype(int)
        # print(SAnodemap)
        return SAnodemap

    @staticmethod
    def get_action_direction_mapping_orig(state):
        """ mapping of direction to actions in the original nodemap for each state in the order
             **Note: each output "directions" list has to be length 3 **
        """

        if state == HOME_NODE:
            return ['', 'east', '']
        if state == RWD_STATE:
            return ['', '', '']

        node_lvl = LVL_BY_NODE[state]
        if node_lvl == 6:
            if state % 2 == 0:
                directions = ['west', '', '']      # e.g. 84, 108, 72
            else:
                directions = ['east', '', '']     # e.g. 99, 111, 83
        elif node_lvl % 2 == 0:
            # i.e. level is 0, 2, 4
            if state % 2 == 0:
                directions = ['west', 'north', 'south']    # e.g. 6, 28, 16
            else:
                directions = ['east', 'north', 'south']  # e.g. 5, 27, 21
        else:
            # i.e. level is 1, 3, 5
            if state % 2 == 0:
                directions = ['north', 'west', 'east']    # e.g. 10, 42, 2
            else:
                directions = ['south', 'west', 'east']  # e.g. 13, 1, 35
        return directions

    def get_nodemap_direction_dict(self):
        from collections import defaultdict
        directions = defaultdict(dict)

        # Get directions for each node-action using the global direction mapping at a node
        for i in range(self.S):
            global_direction_mapping = self.get_action_direction_mapping_orig(i)
            directions[i][self.base_nodemap[i][0]] = global_direction_mapping[0]
            directions[i][self.base_nodemap[i][1]] = global_direction_mapping[1]
            directions[i][self.base_nodemap[i][2]] = global_direction_mapping[2]
            # print(self.base_nodemap[i], self.nodemap[i], directions[i])

        # Now Get action for each node-direction using the new nodemap i.e. inverse
        nodemap_dict = defaultdict(dict)
        for s, l in enumerate(self.nodemap):
            for node, direction in directions[s].items():
                if direction != '':
                    nodemap_dict[s][direction] = np.where(self.nodemap[s] == node)[0].tolist()[0]
        return nodemap_dict

    def get_action_direction_mapping(self, state):
        # print(self.nodemap_direction_dict[state])
        return self.nodemap_direction_dict[state]

    def get_action_probabilities(self, state, beta, V):
        raise NotImplementedError(
            "You need to define your own get_action_probabilities function."
            " Base model doesn't have any.")

    def is_valid_prob(self, action_prob):
        """
        Check for invalid action probability set
        """
        for i in action_prob:
            if np.isnan(i):
                raise Exception(f'Invalid action probabilities {action_prob}')
        if np.sum(action_prob) < 0.999:
            raise Exception(f'Invalid action probabilities, failed summing to 1: {action_prob}')

    def take_action(self, s: int, a: int) -> int:
        return int(self.nodemap[s, a])

    def choose_action(self):
        raise NotImplementedError(
            "You need to define your own get_action_probabilities function."
            " Base model doesn't have any.")

    def is_valid_state_value(self, v):
        if np.isnan(v):
            raise Exception(f'Warning invalid state-value: {v}')
        elif np.isinf(v):
            raise Exception(f'Warning infinite state-value: {v}')
        elif abs(v) >= 1e5:
            print('Warning state value exceeded upper bound. Might approach infinity.')
            v = np.sign(v) * 1e5
            return v
        return v

    def simulate(self, agentId, sub_fits):
        """
        Simulate the agent with given set of parameters sub_fits.
        """
        raise NotImplementedError(
            "You need to define your own simulate function."
            " Base model doesn't have any.")

    def simulate_multiple(self, sub_fits, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        """
        This function calls `simulate` in multiple processes
        :param sub_fits:
            Subject fits, dictionary with agentIds as keys and value is a dict
            of parameters that are going to be used in the model
            for example 0: {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.5}.
           i.e. {AgentId: {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.5}}
        Example usage:
            success, stats = agentObj.simulate_multiple({0: {"alpha": 0.1, "gamma": 0.9, "epsilson": 0.5}})
        """
        tasks = []
        for agentId in sub_fits:
            print("agentId=", agentId, "params=", sub_fits[agentId])
            tasks.append((agentId, sub_fits[agentId], MAX_LENGTH, N_BOUTS_TO_GENERATE))
        with Pool(4) as p:  # running in parallel in 4 processes
            simulation_results = p.starmap(self.simulate, tasks)
        return dict([(a[0], simulation_results[i]) for i, a in enumerate(tasks)])

    def get_maze_state_values(self, V):
        """
        Get state values to plot against the nodes on the maze
        """
        raise NotImplementedError(
            "You need to define your own state_values function. Base model "
            "doesn't have any. If your states are the maze nodes, then it could"
            " just be as simple as `return V`.")

    def get_valid_actions(self, s):
        """
        Get valid actions available at state s
        """
        return np.where(self.nodemap[s] != INVALID_STATE)[0].tolist()

    def get_valid_next_states(self, s):
        """
        Get valid next states available at state s
        """
        return self.nodemap[s][self.nodemap[s] != INVALID_STATE]

    @staticmethod
    def test_traj(traj):
        for i, j in zip(traj, traj[1:]):
            assert i != j
        return

    def test_episodes(self, episode_state_traj):
        try:
            for i, t in enumerate(episode_state_traj):
                self.test_traj(t)
        except:
            raise Exception(f"Corrupt traj {i} with adjacent similar nodes found: {t}")

    def wrap(self, episode_state_traj):
        episode_state_trajs = break_simulated_traj_into_episodes(episode_state_traj)
        self.test_episodes(episode_state_trajs)
        episode_state_trajs = list(filter(lambda e: len(e) >= 3, episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs  # in pure exploration, both are same
        return episode_state_trajs, episode_maze_trajs
