"""
Our model: version 2
"""

import numpy as np

import parameters as p
from parameters import STRAIGHT, BENT_STRAIGHT, OPP_STRAIGHT, OPP_BENT_STRAIGHT, node_subquadrant_dict, \
    node_subquadrant_label_dict, subquadrant_label_dict
from V1_model import V1
from utils import get_outward_pref_order, get_parent_node, connect_path_node, \
    get_all_end_nodes_from_level4_node, get_all_subq_from_current_subq


class V2(V1):
    """
    outward preference
    backward preference

    At L6: Choose to stay in same subq, q or go out and go to the chosen one with higher pref.
    """

    def __init__(self, file_suffix='_V2Trajectories'):
        V1.__init__(self, file_suffix=file_suffix)

        # self.S = 128  # total states
        #
        # # self.curr_directions = None
        # self.episode_state_traj = []
        # self.s = 0  # Start from 0
        # self.prev_s = HOME_NODE

    # @staticmethod
    # def lowest_level_transition_biases(label):
    #     return sorted({
    #         STRAIGHT: {OPP_STRAIGHT: 0.65,
    #                    BENT_STRAIGHT: 0.3,
    #                    OPP_BENT_STRAIGHT: 0.05},
    #
    #         OPP_STRAIGHT: {STRAIGHT: 0.65,
    #                        BENT_STRAIGHT: 0.3,
    #                        OPP_BENT_STRAIGHT: 0.05},
    #
    #         BENT_STRAIGHT: {OPP_BENT_STRAIGHT: 0.65,
    #                         STRAIGHT: 0.3,
    #                         OPP_STRAIGHT: 0.05},
    #
    #         OPP_BENT_STRAIGHT: {BENT_STRAIGHT: 0.65,
    #                             STRAIGHT: 0.3,
    #                             OPP_STRAIGHT: 0.05}
    #     }[label].items(), key=lambda x: x[1], reverse=True)
    #
    # @staticmethod
    # def subq_level_transition_biases(label):
    #     return sorted({
    #         STRAIGHT: {OPP_STRAIGHT: 0.6,
    #                    BENT_STRAIGHT: 0.25,
    #                    OPP_BENT_STRAIGHT: 0.15},
    #
    #         OPP_STRAIGHT: {STRAIGHT: 0.6,
    #                        BENT_STRAIGHT: 0.25,
    #                        OPP_BENT_STRAIGHT: 0.15},
    #
    #         BENT_STRAIGHT: {OPP_BENT_STRAIGHT: 0.6,
    #                         STRAIGHT: 0.25,
    #                         OPP_STRAIGHT: 0.15},
    #
    #         OPP_BENT_STRAIGHT: {BENT_STRAIGHT: 0.6,
    #                             STRAIGHT: 0.25,
    #                             OPP_STRAIGHT: 0.15}
    #     }[label].items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def transition_prefs(opposite_prob, diagonally_outer_prob, diagonally_inner_prob):
        return {
            STRAIGHT: {
                OPP_STRAIGHT: opposite_prob,
                BENT_STRAIGHT: diagonally_outer_prob,
                OPP_BENT_STRAIGHT: diagonally_inner_prob
            },

            OPP_STRAIGHT: {
                STRAIGHT: opposite_prob,
                BENT_STRAIGHT: diagonally_outer_prob,
                OPP_BENT_STRAIGHT: diagonally_inner_prob
            },

            BENT_STRAIGHT: {
                OPP_BENT_STRAIGHT: opposite_prob,
                STRAIGHT: diagonally_outer_prob,
                OPP_STRAIGHT: diagonally_inner_prob,
            },

            OPP_BENT_STRAIGHT: {
                BENT_STRAIGHT: opposite_prob,
                STRAIGHT: diagonally_outer_prob,
                OPP_STRAIGHT: diagonally_inner_prob,
            }
        }

    def endnode_level_transition_biases(self, label):
        return sorted(
            self.transition_prefs(self.opposite_prob, self.diagonally_outer_prob, self.diagonally_inner_prob)[label].items(),
            key=lambda x: x[1],
            reverse=True)

    def subq_level_transition_biases(self, label):
        return sorted(
            # self.transition_prefs(self.opposite_prob, self.diagonally_outer_prob, self.diagonally_inner_prob)[label].items(),
            self.transition_prefs(0.34, 0.33, 0.33)[
                label].items(),
            key=lambda x: x[1],
            reverse=True)

    def within_subq_biases(self):
        # pref to go to opposite node in same subq

        n = self.s
        assert n in p.LVL_6_NODES
        label = node_subquadrant_label_dict[n]
        level4_parent = get_parent_node(get_parent_node(n))
        n1, n2, n3, n4 = get_all_end_nodes_from_level4_node(level4_parent)
        labels_nodes = {node_subquadrant_label_dict[n1]: n1, node_subquadrant_label_dict[n2]: n2,
                        node_subquadrant_label_dict[n3]: n3, node_subquadrant_label_dict[n4]: n4}
        pref_order = {labels_nodes[l]: pr for l, pr in self.endnode_level_transition_biases(label)}
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
        pref_order = {labels_subqs[l]: pr for l, pr in self.subq_level_transition_biases(current_subq_label)}
        # print("at", self.s, "got within q pref_order", pref_order)
        next_subq = self.sample_key_from_dict(pref_order)
        return next_subq

    def get_out_of_the_maze(self, destination_parent_level):
        while p.LVL_BY_NODE[self.s] != destination_parent_level:
            self.prev_s = self.s
            self.s = self.take_action(self.s, 0)  # go to parent node
            self.episode_state_traj.append(self.s)
        return

    def make_it_go_to(self, target_node):
        path = connect_path_node(self.s, target_node)[1:]
        # print("connect", self.s, target_node, p)
        for n in path:
            if np.random.random() <= self.STRAIGHT_BACK_PROB:        # TODO: have a probability going in random or back
                self.episode_state_traj.append(self.prev_s)     # go back
                self.s = self.episode_state_traj[-1]
                self.prev_s = self.episode_state_traj[-2]
                break
            self.episode_state_traj.append(n)
            self.s = self.episode_state_traj[-1]
            self.prev_s = self.episode_state_traj[-2]
        return

    def choose_action(self):

        if self.s == p.HOME_NODE:
            return 0

        while self.s in p.LVL_6_NODES:    # as long as it is at a level 6 node
            # TODO: change how to decide when to go out and how much
            prob = np.random.random()
            # if prob <= 0.2:
            #     self.make_it_go_to(get_parent_node(self.s))
            if prob <= self.staySQp:
                # print("going somewhere else in this subQ", prob)
                target_node = self.within_subq_biases()
                # print("got target in same subQ", target_node, "self.s prev_s", self.s, self.prev_s)
                self.make_it_go_to(target_node)
            elif prob <= self.stayQp:
                # print("going out of this subQ but staying in the same Q", prob)
                target_node = self.within_q_biases()  # returns a subq node  # TODO: think on picking a random subq anywhere in maze as well
                self.make_it_go_to(target_node)
                # self.get_out_of_the_maze(2)
            else:
                # print("going out of this Q", prob)
                self.get_out_of_the_maze(1)

        # print("levels", LVL_BY_NODE[self.s], LVL_BY_NODE[self.prev_s], LVL_BY_NODE[self.s] != LVL_BY_NODE[self.prev_s] + 1)
        if p.LVL_BY_NODE[self.s] == p.LVL_BY_NODE[self.prev_s] + 1:     # if moving down the tree, forward biases apply with less prob to moving up
            next_node = self.forward_biases()
            # If I make it go alternate as below, then I am removing a parameter but basically
            # hard coding it inside the model
            # action = (3 - prev_action) % 3
        else:
            next_node = self.backward_biases()                      # if moving up the tree, backward biases apply with less prob to moving down where it just came from
        # assert next_node in self.nodemap[self.s, :]
        return next_node

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)

        self.BACKWARD_GOBACK_PROB = self.FORWARD_GOBACK_PROB = self.STRAIGHT_BACK_PROB = params['back_prob']
        self.OUTWARD_PREFERENCE_PROB = params['outward_pref']

        self.opposite_prob = params['opposite']
        self.diagonally_outer_prob = self.diagonally_inner_prob = (1-params['opposite'])/2
        # self.diagonally_inner_prob = 1-self.opposite_prob-self.diagonally_outer_prob

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
    from sample_agent import run, load
    # param_sets = [
    #     {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.65, "stayQ": 0.8},
    #     {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.65, "stayQ": 0.82},
    #     {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.65, "stayQ": 0.84},
    #     {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.65, "stayQ": 0.86},
    #     {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.65, "stayQ": 0.9},
    # ]

    param_sets = [
        # {'back_prob': 0.2, 'outward_pref': 0.75, "staySQ": 0.8, "stayQ": 0.85},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85, "opposite": 0.65, "diagonal_outer": 0.33}
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85, "opposite": 0.65},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.85, "opposite": 0.65},
        {'back_prob': 0.22, 'outward_pref': 0.7, "staySQ": 0.65, "stayQ": 0.9, "opposite": 0.65, 'model': 'Custom'},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.75, "stayQ": 0.8, "opposite": 0.33, 'model': 'Custom'},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.7, "stayQ": 0.8, "opposite": 0.33, 'model': 'Custom'},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65,  "opposite": 0.42},
        # {'back_prob': 0.2, 'outward_pref': 0.7, "staySQ": 0.65,  "opposite": 0.42},

        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.65, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.67, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.69, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.71, "stayQ": 0.85},
        # {'back_prob': 0.2, 'node_preferred_prob': 0.75, "staySQ": 0.75, "stayQ": 0.85},
    ]

    runids = run(V2(), param_sets, '/Users/usingla/mouse-maze/figs', '39999')
    print(runids)
    base_path = '/Users/usingla/mouse-maze/figs/'
    load([
        ('V2', runids)
    ], base_path)


