"""
Our model: version 1
"""

import os
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy

from parameters import *
from BaseModel import BaseModel
from EpsilonDirectionGreedy_model import EpsilonDirectionGreedy
from EpsilonGreedy_model import EpsilonGreedy
from utils import calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level, get_outward_pref_order
from MM_Traj_Utils import StepType2, NewMaze
import evaluation_metrics as em
from utils import get_parent_node, connect_path_node


def get_all_end_nodes_from_level4_node(level4_node):
    """Returns all 4 end nodes in that particular subquadrant"""
    assert level4_node in LVL_4_NODES
    p1, p2 = level4_node * 2 + 1, level4_node * 2 + 2
    n1, n2, n3, n4 = p1 * 2 + 1, p1 * 2 + 2, p2 * 2 + 1, p2 * 2 + 2
    return n1, n2, n3, n4


def get_all_subq_from_current_subq(subq):
    """Returns all 4 subquadrants in that particular quadrant"""
    for sq in subquadrant_sets:
        if subq in subquadrant_sets[sq]:
            return subquadrant_sets[sq]


class BiasedModelV1(BaseModel):

    def __init__(self, file_suffix='_V1Trajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        self.S = 128  # total states

        # self.curr_directions = None
        self.episode_state_traj = []
        print("self.nodemap_direction_dict")
        print(self.nodemap_direction_dict)
        self.s = 0  # Start from 0
        self.prev_s = HOME_NODE

    @staticmethod
    def lowest_level_transition_biases(label):
        return sorted({
            STRAIGHT: {OPP_STRAIGHT: 0.65,
                       BENT_STRAIGHT: 0.3,
                       OPP_BENT_STRAIGHT: 0.05},

            OPP_STRAIGHT: {STRAIGHT: 0.65,
                           BENT_STRAIGHT: 0.3,
                           OPP_BENT_STRAIGHT: 0.05},

            BENT_STRAIGHT: {STRAIGHT: 0.3,
                            OPP_STRAIGHT: 0.05,
                            OPP_BENT_STRAIGHT: 0.65},

            OPP_BENT_STRAIGHT: {STRAIGHT: 0.3,
                                OPP_STRAIGHT: 0.05,
                                BENT_STRAIGHT: 0.65}
        }[label].items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def subq_level_transition_biases(label):
        return sorted({
            STRAIGHT: {OPP_STRAIGHT: 0.6,
                       BENT_STRAIGHT: 0.25,
                       OPP_BENT_STRAIGHT: 0.15},

            OPP_STRAIGHT: {STRAIGHT: 0.6,
                           BENT_STRAIGHT: 0.25,
                           OPP_BENT_STRAIGHT: 0.15},

            BENT_STRAIGHT: {STRAIGHT: 0.25,
                            OPP_STRAIGHT: 0.15,
                            OPP_BENT_STRAIGHT: 0.6},

            OPP_BENT_STRAIGHT: {STRAIGHT: 0.25,
                                OPP_STRAIGHT: 0.15,
                                BENT_STRAIGHT: 0.6}
        }[label].items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def sample_key_from_dict(pref_order):
        pref_order = list(pref_order.items())
        next_node = random.choices([k[0] for k in pref_order], weights=[k[1] for k in pref_order], k=1)[0]
        return next_node

    @staticmethod
    def sample_key_from_dict_unweighted(pref_order):
        pref_order = list(pref_order.items())
        next_node = random.choices([k[0] for k in pref_order], k=1)[0]
        return next_node

    def within_subq_biases(self):
        # pref to go to opposite node in same subq

        n = self.s
        assert n in LVL_6_NODES
        label = node_subquadrant_label_dict[n]
        level4_parent = get_parent_node(get_parent_node(n))
        n1, n2, n3, n4 = get_all_end_nodes_from_level4_node(level4_parent)
        labels_nodes = {node_subquadrant_label_dict[n1]: n1, node_subquadrant_label_dict[n2]: n2,
                        node_subquadrant_label_dict[n3]: n3, node_subquadrant_label_dict[n4]: n4}
        pref_order = {labels_nodes[l]: p for l, p in self.lowest_level_transition_biases(label)}
        # print("at", n, "got within subq pref_order", pref_order)
        next_node = self.sample_key_from_dict(pref_order)
        return next_node

    def within_q_biases(self):
        # pref to go to opposite subq

        current_subq = node_subquadrant_dict[self.s]
        current_subq_label = subquadrant_label_dict[current_subq]

        n1, n2, n3, n4 = get_all_subq_from_current_subq(current_subq)
        labels_subqs = {subquadrant_label_dict[n1]: n1, subquadrant_label_dict[n2]: n2,
                        subquadrant_label_dict[n3]: n3, subquadrant_label_dict[n4]: n4}
        pref_order = {labels_subqs[l]: p for l, p in self.subq_level_transition_biases(current_subq_label)}
        # print("at", self.s, "got within q pref_order", pref_order)
        next_subq = self.sample_key_from_dict(pref_order)
        return next_subq

    def get_out_of_the_maze(self, destination_parent_level):
        while LVL_BY_NODE[self.s] != destination_parent_level:
            self.prev_s = self.s
            self.s = self.take_action(self.s, 0)  # go to parent node
            self.episode_state_traj.append(self.s)
        return

    def make_it_go_to_target_node(self, target_node):
        p = connect_path_node(self.s, target_node)[1:]
        # print("connect", self.s, target_node, p)

        for n in p:
            if np.random.random() <= self.STRAIGHT_BACK_PROB:        # TODO: have a probability going in random or back
                self.episode_state_traj.append(self.prev_s)     # go back
                self.s = self.episode_state_traj[-1]
                self.prev_s = self.episode_state_traj[-2]
                break

            self.episode_state_traj.append(n)
            self.s = self.episode_state_traj[-1]
            self.prev_s = self.episode_state_traj[-2]
        return

    def forward_biases(self):

        # print("forward", self.s, self.prev_s)

        n = self.s
        l_child, h_child = 2 * n + 1, 2 * n + 2
        back_child = get_parent_node(n)
        assert self.prev_s == back_child

        if n in LVL_0_NODES:    # 50-50
            pref_order = {l_child: 0.5, h_child: 0.5}
        elif n in LVL_1_NODES:  # 50-50
            pref_order = {l_child: 0.5, h_child: 0.5}
        elif (n in LVL_2_NODES) or (n in LVL_3_NODES) or (n in LVL_4_NODES) or (n in LVL_5_NODES): # tendency to go outwards
            pref_order = get_outward_pref_order(n)
            pref_order = {pref_order[0]: self.NODE_PREFERRED_PROB, pref_order[1]: 1-self.NODE_PREFERRED_PROB}
        else:
            raise Exception(f"Error in forward biases. At level 6 probably. n = {n}")

        # Add back prob
        pref_order = {k: v * (1-self.FORWARD_GOBACK_PROB) for k, v in pref_order.items()}
        pref_order[back_child] = self.FORWARD_GOBACK_PROB
        # print("at", n, "got fwd pref_order", pref_order)
        next_node = self.sample_key_from_dict(pref_order)
        return next_node

    def backward_biases(self):

        valid_next_s = self.get_valid_next_states(self.s)
        # print("valid_next_s", valid_next_s, "prev_s", self.prev_s)

        pref_order = dict.fromkeys(valid_next_s, 0.5 * (1-self.BACKWARD_GOBACK_PROB))
        pref_order[self.prev_s] = self.BACKWARD_GOBACK_PROB   # i.e. go back

        # print("at", self.s, "got bkwd pref_order", pref_order)
        next_node = self.sample_key_from_dict(pref_order)
        return next_node

    def choose_action(self):

        if self.s == HOME_NODE:
            return 0

        while self.s in LVL_6_NODES:    # as long as it is at a level 6 node
            # TODO: change how to decide when to go out and how much
            prob = np.random.random()
            if prob <= 0.33:
                # print("going somewhere else in this subQ", prob)
                target_node = self.within_subq_biases()
                # print("got target in same subQ", target_node, "self.s prev_s", self.s, self.prev_s)
                self.make_it_go_to_target_node(target_node)
            elif prob <= 0.66:
                # print("going out of this subQ but staying in the same Q", prob)
                target_node = self.within_q_biases()  # returns a subq node  # TODO: think on picking a random subq anywhere in maze as well
                self.make_it_go_to_target_node(target_node)
            else:
                # print("going out of this Q", prob)
                self.get_out_of_the_maze(1)

        # if self.s in LVL_6_NODES:
        #     return get_parent_node(self.s)

        # print("levels", LVL_BY_NODE[self.s], LVL_BY_NODE[self.prev_s], LVL_BY_NODE[self.s] != LVL_BY_NODE[self.prev_s] + 1)
        if LVL_BY_NODE[self.s] == LVL_BY_NODE[self.prev_s] + 1:     # if moving down the tree, forward biases apply with less prob to moving up
            next_node = self.forward_biases()
        else:
            next_node = self.backward_biases()                      # if moving up the tree, backward biases apply with less prob to moving down
        # assert next_node in self.nodemap[self.s, :]
        return next_node

    def generate_exploration_episode(self, MAX_LENGTH):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        # self.nodemap[0][0] = -1  # No action to go to HOME_NODE

        print("Starting at", self.s, "with prev at", self.prev_s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # Act
            s_next = self.choose_action()   # NOTE: there's a side effect to some versions where they change s
            # print("traj till now", self.episode_state_traj)
            # print(f"moving from {self.s} => {s_next}")
            self.prev_s = self.s
            self.s = s_next

            if len(self.episode_state_traj) % 1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)

        self.BACKWARD_GOBACK_PROB = self.FORWARD_GOBACK_PROB = self.STRAIGHT_BACK_PROB = params['back_prob']
        self.NODE_PREFERRED_PROB = params['node_preferred_prob']

        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        Q = np.zeros((self.S, self.A))  # Initialize state values
        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, _ = self.generate_exploration_episode(MAX_LENGTH)
            all_episodes_state_trajs.extend(episode_state_trajs)
            all_episodes_pos_trajs.extend(episode_maze_trajs)
        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": 0.0,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes_state_trajs),
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
            # "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency": calculate_normalized_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency_by_level": calculate_normalized_visit_frequency_by_level(all_episodes_state_trajs)
        }
        return success, stats

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


# Driver Code
if __name__ == '__main__':
    from sample_agent import run
    param_sets = {
        1: {'back_prob': 0.2},
        # 2: {'back_prob': 0.2},
        # 2: {'back_prob': 0.3},
        # 3: {'back_prob': 0.2},
    }
    run(BiasedModelV1(), param_sets, '/Users/usingla/mouse-maze/figs', '20000_opp')


