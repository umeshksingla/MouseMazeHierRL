"""
EZgreedy and Custom fused
"""
import os
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock


import parameters as p
import utils
from BaseModel import BaseModel
from parameters import STRAIGHT, BENT_STRAIGHT, OPP_STRAIGHT, OPP_BENT_STRAIGHT, node_subquadrant_dict, \
    node_subquadrant_label_dict, subquadrant_label_dict
from utils import get_outward_pref_order, get_parent_node, connect_path_node, nodes2cell, \
    get_all_end_nodes_from_level4_node, get_all_subq_from_current_subq, get_opp_child, get_children, \
    get_parent_node_x_level_up, D
from maze_spatial_mapping import CELL_XY, NODE_CELL_MAPPING


class EZCustom(BaseModel):

    def __init__(self, file_suffix='_EZCustomTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.sampled_durations = []
        self.duration = 0   # self.sample_duration()
        self.S = 128    # total states

        self.episode_state_traj = []
        # self.visited_corners = []
        # self.prev_s = None
        self.s = p.HOME_NODE

    def __random_action__(self, state):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        actions = self.get_valid_actions(state)
        return np.random.choice(actions)

    # @staticmethod
    # def transition_prefs(opposite_prob, diagonally_outer_prob, diagonally_inner_prob):
    #     return {
    #         STRAIGHT: {
    #             OPP_STRAIGHT: opposite_prob,
    #             BENT_STRAIGHT: diagonally_outer_prob,
    #             OPP_BENT_STRAIGHT: diagonally_inner_prob
    #         },
    #
    #         OPP_STRAIGHT: {
    #             STRAIGHT: opposite_prob,
    #             BENT_STRAIGHT: diagonally_outer_prob,
    #             OPP_BENT_STRAIGHT: diagonally_inner_prob
    #         },
    #
    #         BENT_STRAIGHT: {
    #             OPP_BENT_STRAIGHT: opposite_prob,
    #             STRAIGHT: diagonally_outer_prob,
    #             OPP_STRAIGHT: diagonally_inner_prob,
    #         },
    #
    #         OPP_BENT_STRAIGHT: {
    #             BENT_STRAIGHT: opposite_prob,
    #             STRAIGHT: diagonally_outer_prob,
    #             OPP_STRAIGHT: diagonally_inner_prob,
    #         }
    #     }
    #
    # def endnode_level_transition_biases(self, label):
    #     return sorted(
    #         self.transition_prefs(self.opposite_prob, self.diagonally_outer_prob, self.diagonally_inner_prob)[label].items(),
    #         key=lambda x: x[1],
    #         reverse=True)
    #
    # def subq_level_transition_biases(self, label):
    #     return sorted(
    #         # self.transition_prefs(self.opposite_prob, self.diagonally_outer_prob, self.diagonally_inner_prob)[label].items(),
    #         self.transition_prefs(0.4, 0.3, 0.3)[label].items(),
    #         key=lambda x: x[1],
    #         reverse=True)
    #
    # @staticmethod
    # def sample_key_from_dict(pref_order):
    #     pref_order = list(pref_order.items())
    #     next_node = random.choices([k[0] for k in pref_order], weights=[k[1] for k in pref_order], k=1)[0]
    #     return next_node

    # def within_subq_biases(self):
    #     # pref to go to opposite node in same subq
    #
    #     n = self.s
    #     assert n in p.LVL_6_NODES
    #     label = node_subquadrant_label_dict[n]
    #     level4_parent = get_parent_node(get_parent_node(n))
    #     n1, n2, n3, n4 = get_all_end_nodes_from_level4_node(level4_parent)
    #     labels_nodes = {node_subquadrant_label_dict[n1]: n1, node_subquadrant_label_dict[n2]: n2,
    #                     node_subquadrant_label_dict[n3]: n3, node_subquadrant_label_dict[n4]: n4}
    #     pref_order = {labels_nodes[l]: pr for l, pr in self.endnode_level_transition_biases(label)}
    #     # print("at", n, "got within subq pref_order", pref_order)
    #     next_node = self.sample_key_from_dict(pref_order)
    #     return next_node

    # def within_q_biases(self):
    #     # pref to go to opposite subq
    #
    #     current_subq = node_subquadrant_dict[self.s]
    #     current_subq_label = subquadrant_label_dict[current_subq]
    #
    #     n1, n2, n3, n4 = get_all_subq_from_current_subq(current_subq)
    #     labels_subqs = {subquadrant_label_dict[n1]: n1, subquadrant_label_dict[n2]: n2,
    #                     subquadrant_label_dict[n3]: n3, subquadrant_label_dict[n4]: n4}
    #     pref_order = {labels_subqs[l]: pr for l, pr in self.subq_level_transition_biases(current_subq_label)}
    #     # print("at", self.s, "got within q pref_order", pref_order)
    #     next_subq = self.sample_key_from_dict(pref_order)
    #     return next_subq

    def make_it_go_to(self, target_node):
        path = connect_path_node(self.s, target_node)[1:]
        # print("connect", self.s, target_node, p)
        for n in path:
            # if np.random.random() <= self.STRAIGHT_BACK_PROB:        # TODO: have a probability going in random or back
            #     self.episode_state_traj.append(self.prev_s)     # go back
            #     self.s = self.episode_state_traj[-1]
            #     # self.prev_s = self.episode_state_traj[-2]
            #     break
            self.episode_state_traj.append(n)
            self.s = self.episode_state_traj[-1]
            self.prev_s = self.episode_state_traj[-2]
        return

    def choose_action(self, Q, *args, **kwargs):

        # if self.s == p.HOME_NODE:
        #     return 0
        # if p.LVL_BY_NODE[self.s] == p.LVL_BY_NODE[self.prev_s] + 1:     # if moving down the tree, forward biases apply with less prob to moving up
        #     next_node = self.forward_biases()
        #     # If I make it go alternate as below, then I am removing a parameter but basically
        #     # hard coding it inside the model
        #     # action = (3 - prev_action) % 3
        # else:
        #     next_node = self.backward_biases()                      # if moving up the tree, backward biases apply with less prob to moving down where it just came from
        # # assert next_node in self.nodemap[self.s, :]
        # return next_node

        if self.s == p.HOME_NODE:
            self.duration = 0
            return 1

        # if self.s in p.LVL_6_NODES:
        #     self.visited_corners.append(self.s)

        assert self.duration >= 0
        prev_action = kwargs['prev_action']

        def options_at_level_6(d: int):
            # d = random.choice([2, 3, 4])
            assert d >= 1
            choices = {
                '1': [
                    get_parent_node_x_level_up(self.s, x=1)
                ],
                '2': [
                    get_parent_node_x_level_up(self.s, x=2),
                    get_opp_child(self.s)
                ],
                '3': [
                    get_opp_child(self.s),  # 2

                    get_parent_node_x_level_up(self.s, x=3),
                    get_opp_child(get_parent_node(self.s))
                ],
                '4': [
                    get_opp_child(self.s),  # 2

                    get_parent_node_x_level_up(self.s, x=4),
                    *get_children(get_opp_child(get_parent_node(self.s))),
                    get_opp_child(get_parent_node_x_level_up(self.s, x=2))
                ],
                '5': [
                    get_opp_child(self.s),  # 2
                    * get_children(get_opp_child(get_parent_node(self.s))),  # 4

                    get_parent_node_x_level_up(self.s, x=5),
                    get_opp_child(get_parent_node_x_level_up(self.s, x=3)),
                    *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=2)))
                ],
                '6': [
                      # get_parent_node_x_level_up(self.s, x=5),

                    get_opp_child(self.s),    # 2
                    *get_children(get_opp_child(get_parent_node(self.s))),   # 4

                    get_parent_node_x_level_up(self.s, x=6),
                    get_opp_child(get_parent_node_x_level_up(self.s, x=4)),
                    *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=3))),
                    *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=2)))[0]),
                    *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=2)))[1])
                ],
                '>=7': [
                    p.HOME_NODE,
                    *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=4))),
                    get_opp_child(get_parent_node_x_level_up(self.s, x=5)),
                    *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=3)))[0]),
                    *get_children(get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=3)))[1]),
                    *get_children(get_opp_child(get_parent_node_x_level_up(self.s, x=5)))
                ]
            }

            if d <= 6:
                options = choices[str(d)]
            else:
                options = choices['>=7']

            # print(self.s, options)
            # inv_weights = np.array([D[self.s][o] for o in options])
            # weights = -1 * inv_weights + np.max(inv_weights) + 1
            # norm_weights = weights / np.linalg.norm(weights, ord=1)
            return random.choices(options)[0]  # equivalent to uniformly sampling from set of available actions

        if self.duration == 0 or prev_action not in self.get_valid_actions(self.s):
            # while self.s in p.LVL_6_NODES:  # as long as it is at a level 6 node
            #     n_step = random.choices(['1', '2', '4', '5', '6', '7', '8'], weights=[0.5, 0.2, 0.16, 0.036, 0.054, 0.045, 0.005])[0]
            #     if n_step == '1':
            #         self.make_it_go_to(get_parent_node(self.s))
            #     elif n_step == '2':
            #         self.make_it_go_to(get_opp_children(self.s))
            #     elif n_step == '4':
            #         self.make_it_go_to(random.choice(get_children(get_opp_children(get_parent_node(self.s)))))
            #     elif n_step == '5':   # more like 4+out
            #         self.make_it_go_to(get_opp_children(get_parent_node(get_parent_node(self.s))))
            #     elif n_step == '6':
            #         self.make_it_go_to(random.choice(get_children(get_opp_children(get_parent_node(get_parent_node(get_parent_node(self.s)))))))
            #     elif n_step == '7':   # more like 5+out
            #         self.make_it_go_to(1)
            #     elif n_step == '8':   # more like 6+out
            #         self.make_it_go_to(0)
            #     else:
            #         raise Exception(f'wrong sampled step {n_step}')
            #     self.sampled_durations.append(int(n_step))


            # while self.s in p.LVL_6_NODES:  # as long as it is at a level 6 node
                # n_step = random.choices(['1', '2', '4', '4_out', '6', '5_out', '6_out'],
                #                         weights=[0.5, 0.2, 0.16, 0.036, 0.054, 0.045, 0.005])[0]
                # if n_step == '1':  # takes it to level 5
                #     self.make_it_go_to(get_parent_node(self.s))
                # elif n_step == '2':  # takes it to level 6
                #     self.make_it_go_to(get_opp_children(self.s))
                # elif n_step == '4':  # takes it to level 6
                #     self.make_it_go_to(random.choice(get_children(get_opp_children(get_parent_node(self.s)))))
                # elif n_step == '4_out':  # takes it to level 4
                #     self.make_it_go_to(get_opp_children(get_parent_node(get_parent_node(self.s))))
                # elif n_step == '6':  # takes it to level 4
                #     self.make_it_go_to(random.choice(
                #         get_children(get_opp_children(get_parent_node(get_parent_node(get_parent_node(self.s)))))))
                # elif n_step == '5_out':  # takes it to level 1
                #     self.make_it_go_to(1)
                # elif n_step == '6_out':  # takes it to level 0
                #     self.make_it_go_to(0)
                # else:
                #     raise Exception(f'wrong sampled step {n_step}')
                # self.sampled_durations.append(int(n_step))

            # while self.s in p.LVL_6_NODES:  # as long as it is at a level 6 node
            #     prob = np.random.random()
            #     if prob <= 0.5:
            #         l5 = get_parent_node(self.s)
            #         self.make_it_go_to(l5)
            #     elif prob <= 0.7:
            #         opp_node = utils.get_opp_children(self.s)
            #         self.make_it_go_to(opp_node)
            #     # elif prob <= 0.7:
            #     #     l4 = get_parent_node(get_parent_node(self.s))
            #     #     self.make_it_go_to(l4)
            #     # elif prob <= 0.6:
            #     #     opp_node = utils.get_opp_children(self.s)
            #     #     self.make_it_go_to(opp_node)
            #     elif prob <= self.staySQp:
            #     # if prob <= self.staySQp:
            #         # print("going somewhere else in this subQ", prob)
            #         target_node = self.within_subq_biases()
            #         # print("got target in same subQ", target_node, "self.s prev_s", self.s, self.prev_s)
            #         self.make_it_go_to(target_node)
            #     elif prob <= self.stayQp:
            #         # print("going out of this subQ but staying in the same Q", prob)
            #         target_node = self.within_q_biases()  # returns a subq node  # TODO: think on picking a random subq anywhere in maze as well
            #         self.make_it_go_to(target_node)
            #         # self.get_out_of_the_maze(2)
            #     else:
            #         # print("going out of this Q", prob)
            #         self.get_out_of_the_maze(1)
            if np.random.random() <= self.epsilon:
                self.duration = self.sample_duration()
            else:
                self.duration = 1

            if self.s in p.LVL_6_NODES:
                self.make_it_go_to(options_at_level_6(self.duration))   # composite actions
                self.duration = 1
                action = None
            else:
                action = self.__random_action__(self.s)
            self.sampled_durations.append(self.duration)
        else:
            # if (self.prev_s in p.LVL_6_NODES) and (self.s in p.LVL_5_NODES):
            #     assert utils.get_parent_node(self.prev_s) == self.s
            #     opp_c = utils.get_the_other_children(self.s, self.prev_s)
            #     action = np.where(self.nodemap[self.s] == opp_c)[0][0]
            #     self.duration = 1

            # if self.memory_l5 != 'absent' and (self.prev_s in p.LVL_6_NODES) and (self.s in p.LVL_5_NODES):
            #     assert prev_action == 0
            #     # From L6 to opposite L6 component
            #     opp_c = utils.get_the_other_children(self.s, self.prev_s)
            #     # Memory component if already visited opposite
            #     x = {'strong': 1, 'weak': 2, 'mix': random.randint(1, 2)}[self.memory_l5]  # implement weak memory with randint if desired
            #     if self.visited_corners.count(opp_c) >= x:  # if already visited opposite node x times, go up the tree
            #         action = prev_action
            #         self.visited_corners.clear()
            #     else:  # else go to the opposite node
            #         action = np.where(self.nodemap[self.s] == opp_c)[0][0]

            # else:
            #     action = (3 - prev_action) % 3
            action = (3 - prev_action) % 3
        self.duration -= 1
        return action

    def sample_duration(self):
        d = np.random.zipf(a=self.mu)
        return d

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[p.WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE

        a = None   # Take action 1 at HOME NODE
        print("Starting at", self.s)
        self.episode_state_traj.append(self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != p.RWD_STATE   # since it's pure exploration

            # acting
            # print("---------- s=", self.s)
            a = self.choose_action(Q, prev_action=a)
            if a is None:
                continue
            else:
                s_next = self.take_action(self.s, a)     # Take action
                # print("a=", a)
                # self.prev_s = self.s
                self.s = s_next
                # Record current state
                self.episode_state_traj.append(self.s)

            if len(self.episode_state_traj)%1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(self.episode_state_traj)
        # print(episode_maze_trajs)

        # plot durations sampled
        d_values = np.array(self.sampled_durations)
        unique, counts = np.unique(d_values, return_counts=True)
        # print("unique, counts raw", unique, counts)
        n, bins, patches = plt.hist(d_values[d_values<30], bins=150, density=True, facecolor='blue', alpha=0.5)
        plt.title(f'mu={self.mu}')
        plt.savefig(f'../../figs/duration-mu={self.mu}_e={self.epsilon}.png')
        # plt.show()
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1

        # self.enable_alternate_action = True     # params["enable_alternate_action"]
        # self.memory_l5 = params["memory_l5"]      # 'strong', 'weak', 'absent'

        # self.BACKWARD_GOBACK_PROB = self.FORWARD_GOBACK_PROB = self.STRAIGHT_BACK_PROB = params['back_prob']
        # self.opposite_prob = params['opposite']
        # self.diagonally_outer_prob = self.diagonally_inner_prob = (1-params['opposite'])/2
        self.mu = params["mu"]
        self.epsilon = params["epsilon"]
        # self.staySQp = params['staySQ']
        # self.stayQp = params['stayQ']

        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[p.HOME_NODE, :] = 0
        if self.S == 129:
            Q[p.RWD_STATE, :] = 0
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []

        _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH, Q)
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

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


if __name__ == '__main__':
    from sample_agent import run, load
    param_sets = [
        ############  epsilon 0.3 mu 1.5-1.55   ############
        # 1.55 BiasedWalk4-50173-EZCustom-844821-EZCustom-281200-EZCustom-782845
        # 1.5 BiasedWalk4-50173-EZCustom-219055-EZCustom-890374-EZCustom-668768

        # epsilon 0.5 mu 2 BiasedWalk4-50173-EZCustom-191698-EZCustom-394050-EZCustom-52176
        {"epsilon": 1, "mu": 2, 'model': 'EZCustom'},
    ]*3
    runids = run(EZCustom(), param_sets, '/Users/usingla/mouse-maze/figs', '39995',
                 analyze=False)
    print(runids)
    # base_path = '/Users/usingla/mouse-maze/figs/'
    # load([
    #     ('BiasedWalk4', [50173]),
    #     ('EZCustom', runids),
    # ], base_path)

"""
↑ epsilon affects pbf(-) psf(-) psa(+) efficiency(not much) fractiontime (-), revisits (not much when >= 0.35) -  812279

    subq level transitions - self.transition_prefs(0.4, 0.3, 0.3)[label].items(),
    {"epsilon": 0.35, "mu": 2, 'back_prob': 0.0, "staySQ": 0.8, "stayQ": 0.95, 'opposite': 0.0, 'model': 'EZCustom'},

        if prob <= 0.55:
            l5 = get_parent_node(self.s)
            self.make_it_go_to(l5)
        elif prob <= 0.7:
            opp_node = utils.get_opp_children(self.s)
            self.make_it_go_to(opp_node)
        elif prob <= self.staySQp:
            target_node = self.within_subq_biases()
            self.make_it_go_to(target_node)
        elif prob <= self.stayQp:
            target_node = self.within_q_biases()  # returns a subq node  # TODO: think on picking a random subq anywhere in maze as well
            self.make_it_go_to(target_node)
        else:
            self.get_out_of_the_maze(1)

    Works decent but not the fraction time spent and only slightly forward bias, BAD markov
    # BEST - 567390

    subq level transitions - self.transition_prefs(0.34, 0.33, 0.33)[label].items(),
    {"epsilon": 0.4, "mu": 2, 'back_prob': 0.0, "staySQ": 0.8, "stayQ": 0.95, 'opposite': 0.0, 'model': 'EZCustom'},
    
    Works decent but not the fraction time spent and BAD forward bias, bad markov, OKAY OI
    # BEST - 948043
    
    
    WINNER!!!! subq level transitions - self.transition_prefs(0.4, 0.3, 0.3)[label].items(),
    {"epsilon": 0.4, "mu": 2, 'back_prob': 0.0, "staySQ": 0.8, "stayQ": 0.95, 'opposite': 0.0, 'model': 'EZCustom'}
        if prob <= 0.5:
            l5 = get_parent_node(self.s)
            self.make_it_go_to(l5)
        elif prob <= 0.7:
            opp_node = utils.get_opp_children(self.s)
            self.make_it_go_to(opp_node)
        elif prob <= self.staySQp:
            target_node = self.within_subq_biases()
            self.make_it_go_to(target_node)
        elif prob <= self.stayQp:
            target_node = self.within_q_biases()  # returns a subq node  # TODO: think on picking a random subq anywhere in maze as well
            self.make_it_go_to(target_node)
        else:
            self.get_out_of_the_maze(1)
    Works decent but not the fraction time spent and  forward bias,  AA,  markov,  OI
    # BEST [606273, 733545, 554307, 538228, 649370, 191695]  EZCustom-606273-EZCustom-733545-EZCustom-554307-EZCustom-538228-EZCustom-649370-EZCustom-191695
    
    ??????????? subq level transitions - self.transition_prefs(0.34, 0.33, 0.33)[label].items(),
    {"epsilon": 0.4, "mu": 2, 'back_prob': 0.0, "staySQ": 0.8, "stayQ": 0.95, 'opposite': 0.0, 'model': 'EZCustom'}
    Works decent but not the fraction time spent and BAD forward bias, GOOD AA, REALLY GOOD markov, BAD OI
    # BEST - 633937 OR [770501, 782738, 23618]
    
    {'epsilon'/ 0.35, 'mu'/ 2, 'back_prob'/ 0.0, 'staySQ'/ 0.8, 'stayQ'/ 0.95, 'opposite'/ 0.0, 'model'/ 'EZCustom'}
    Works decent but not the fraction time spent and only slightly worse forward bias and aa bias, GOOD markov 
    # BEST - 45938
    
    ??????????? subq level transitions - self.transition_prefs(0.34, 0.33, 0.33)[label].items(),
    {"epsilon": 0.4, "mu": 2, 'back_prob': 0.0, "staySQ": 0.8, "stayQ": 1, 'opposite': 0.0, 'model': 'EZCustom'}
    Works decent but OKAY fraction time spent and BAD forward bias, OKAY AA, GOOD markov, BAD OI
    # BEST - [581728, 674422, 515576]




"""


