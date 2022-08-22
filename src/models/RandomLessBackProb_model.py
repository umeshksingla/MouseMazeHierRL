"""

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
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level
import evaluation_metrics as em


class RandomLessBackProb(EpsilonGreedy):
    """
    Random policy with low probability to take back action, controlled by parameter back_action_prob.
    """

    def __init__(self, file_suffix='_DFSv21Trajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        self.S = 128  # total states
        self.episode_state_traj = []
        self.back_action_prob = None
        self.s = 0  # Start from 0

    def choose_action(self, prev_s):
        """
        Algo:
        Choose a random direction at 0 and follow it for "duration" steps
        """
        print("at ", self.s, "coming from", prev_s)
        if self.s == HOME_NODE:
            return 1, 1.0
        valid_actions = self.get_valid_actions(self.s)
        print("valid_actions", valid_actions)
        possible_s_next = [self.take_action(self.s, a) for a in valid_actions]
        print("possible_s_next", possible_s_next)
        if self.s in LVL_6_NODES:   # only 1 valid action at end nodes
            action = np.random.choice(valid_actions)
        else:
            # reduce probability of action that takes it to prev state
            back_action = possible_s_next.index(prev_s)
            # take a random action now with these probabilities
            action_probs = [(1.0-self.back_action_prob)/2] * 3
            action_probs[back_action] = self.back_action_prob
            print("back_action p", back_action, action_probs)
            action = np.random.choice(valid_actions, p=action_probs)
            print("action chosen", action)

        assert action in self.get_valid_actions(self.s)
        return action, 1.0

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        # self.nodemap[0][0] = -1  # No action to go to HOME_NODE
        print("FINAL self.nodemap", self.nodemap)
        # print("FINAL self.nodemap_direction_dict", self.nodemap_direction_dict)

        # e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        print("Starting at", self.s)
        prev_s = HOME_NODE
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a, a_prob = self.choose_action(prev_s)   # NOTE: there's a side-affect to some versions where they change s
            s_next = self.take_action(self.s, a)     # Take action

            print("action: ", a, f": {self.s} => {s_next}")

            prev_s = self.s
            self.s = s_next

            if len(self.episode_state_traj) % 1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print(self.episode_state_traj)
        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs = break_simulated_traj_into_episodes(self.episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same
        print(episode_maze_trajs)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        print("params", params)
        self.back_action_prob = params["back_prob"]
        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[HOME_NODE, :] = 0
        if self.S == 129:
            Q[RWD_STATE, :] = 0
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        LL = 0.0
        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(MAX_LENGTH, Q)
            all_episodes_state_trajs.extend(episode_state_trajs)
            all_episodes_pos_trajs.extend(episode_maze_trajs)
            LL += episode_ll
        stats = {
            "agentId": agentId,
            "episodes_states": all_episodes_state_trajs,
            "episodes_positions": all_episodes_pos_trajs,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes_state_trajs),
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
            "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency": calculate_normalized_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency_by_level": calculate_normalized_visit_frequency_by_level(
                all_episodes_state_trajs)
        }
        return success, stats