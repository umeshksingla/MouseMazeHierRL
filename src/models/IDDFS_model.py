"""
IDDFS
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


class IDDFS(EpsilonGreedy):

    def __init__(self, file_suffix='_IDDFSTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        self.S = 128  # total states

        self.curr_directions = None
        # self.prev_level4_node = None
        # self.visited_corners = set()

        self.sampled_durations = []
        # self.duration = self.sample_duration()
        # self.remember_corners = None
        self.episode_state_traj = []

        # self.directions_in_memory = DirectionMemory()

        self.s = 0  # Start from 0

    def get_out_of_the_maze(self, destination_parent_level=3):
        # It can be at any level when a z gets over
        # assert LVL_BY_NODE[self.s] == 6
        print("trying to get out", self.s, destination_parent_level)
        while LVL_BY_NODE[self.s] != destination_parent_level:
            print("before", self.s)
            self.s = self.take_action(self.s, 0)  # go to parent node
            print("after", self.s)
            self.episode_state_traj.append(self.s)
        return

    def choose_action(self):
        """
        Algo:
        Choose a random direction at 0 and follow it for "duration" steps
        """

        if self.s == HOME_NODE:
            self.duration = 0
            self.curr_directions = None
            return 1, 1.0

        if (self.duration == 0) or (not self.is_any_curr_direction_valid()):
            self.get_out_of_the_maze(destination_parent_level=0)
            self.duration = self.sample_duration()
            prev_direction = self.curr_directions
            print("prev_direction", prev_direction)
            self.sample_directions()     # Sample a new direction here
            print("node", self.s, "and node level", LVL_BY_NODE[self.s], ": for sampling duration and direction")
            print("sampled new direction and duration:", "new directions", self.curr_directions)
            action, _ = self.get_action_based_on_current_direction()
            self.duration -= 1
        else:
            action, chosen_dir = self.get_action_based_on_current_direction()
            self.duration -= 1
            print("prev direction used")
        assert action in self.get_valid_actions(self.s)
        return action, 1.0

    def DLS(self, src, parent, max_depth):
        """Depth limited search"""
        # print("at", src)
        if max_depth == 0:
            self.episode_state_traj.append(src)
            return
        for a in self.get_valid_actions(src):
            s_next = self.take_action(src, a)
            if s_next == parent:
                continue
            self.episode_state_traj.append(src)
            self.DLS(s_next, src, max_depth-1)
        self.episode_state_traj.append(src)
        return

    def DFS_limited_memory(self, visited, src, parent, max_depth):
        """DFS with limited memory"""
        # print("at", src)
        if max_depth == 0:
            self.episode_state_traj.append(src)
            return
        if visited[src]:
            return
        visited[src] = 1
        for a in self.get_valid_actions(src):
            s_next = self.take_action(src, a)
            # if s_next == parent:
            #     continue
            self.episode_state_traj.append(src)
            self.DFS_limited_memory(visited, s_next, src, max_depth-1)
        self.episode_state_traj.append(src)
        return

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        self.nodemap[0][0] = -1  # No action to go to HOME_NODE
        # print("FINAL self.nodemap", self.nodemap)
        # print("FINAL self.nodemap_direction_dict", self.nodemap_direction_dict)

        # e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # for depth in range(1, 7):
            #     self.DLS(self.s, -1, depth)
            #     print("self.s", self.s)
            #     self.episode_state_traj.append(HOME_NODE)

            visited = np.zeros(self.S)
            self.DFS_limited_memory(visited, self.s, -1, 6)

            break

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
            # "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency": calculate_normalized_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency_by_level": calculate_normalized_visit_frequency_by_level(
                all_episodes_state_trajs)
        }
        return success, stats