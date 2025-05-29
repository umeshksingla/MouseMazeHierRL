"""
Our model: version 3
"""
import random

import numpy as np

import parameters as p
from V2_model import V2
import utils
from parameters import STRAIGHT, BENT_STRAIGHT, OPP_STRAIGHT, OPP_BENT_STRAIGHT, node_subquadrant_dict, \
    node_subquadrant_label_dict, subquadrant_label_dict
from V1_model import V1
from utils import get_outward_pref_order, get_parent_node, connect_path_node, \
    get_all_end_nodes_from_level4_node, get_all_subq_from_current_subq


class V3(V2):
    """
    outward preference
    backward preference

    At L6: Choose to stay in same subq, q or go out and go to the chosen one with higher pref.

    BUT trying to reduce the params
    """

    def __init__(self, file_suffix='_V3Trajectories'):
        V2.__init__(self, file_suffix=file_suffix)

    def within_subq_biases(self):
        n = self.s
        assert n in p.LVL_6_NODES
        level4_parent = get_parent_node(get_parent_node(n))
        nodes = set(get_all_end_nodes_from_level4_node(level4_parent))
        nodes.remove(n)
        nodes = list(nodes)
        next_node = random.choice(nodes)
        return next_node

    def within_q_biases(self):
        current_subq = node_subquadrant_dict[self.s]
        subqs = set(get_all_subq_from_current_subq(current_subq))
        subqs.remove(current_subq)
        subqs = list(subqs)
        next_subq = random.choice(subqs)
        return next_subq

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)

        self.BACKWARD_GOBACK_PROB = self.FORWARD_GOBACK_PROB = self.STRAIGHT_BACK_PROB = params['back_prob']
        self.OUTWARD_PREFERENCE_PROB = params['outward_pref']

        self.staySQp = params['staySQ']
        self.stayQp = params['stayQ']

        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        Q = np.zeros((self.S, self.A))  # Initialize state values
        _, episode_state_trajs, episode_maze_trajs, _ = self.generate_exploration_episode(MAX_LENGTH)
        all_episodes_state_trajs.extend(episode_state_trajs)
        all_episodes_pos_trajs.extend(episode_maze_trajs)
        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": 0.0,
            "MAX_LENGTH": MAX_LENGTH,
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
        }
        return success, stats


# Driver Code
if __name__ == '__main__':
    from sample_agent import run

    param_sets = [
        # {'back_prob': 0.2, "staySQ": 0.65, "stayQ": 0.9},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85, "opposite": 0.34, "diagonal_outer": 0.33},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85},
        {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85},
        {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85},
        {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85},
        {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85},
        {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.67, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.69, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.71, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.75, "stayQ": 0.85},
    ]

    runids = run(V3(), param_sets, '/Users/usingla/mouse-maze/figs', '30004')
    print(runids)
