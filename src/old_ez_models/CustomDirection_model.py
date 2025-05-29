"""
Custom Direction model making the agent choose a direction at 0 and following it all the way
and coming back to 0. Few more slight variations of it.
"""

import numpy as np

from parameters import *
from BaseModel import BaseModel
from old_ez_models.EpsilonDirectionGreedy_model import EpsilonDirectionGreedy
from utils import calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level
import evaluation_metrics as em


# def get_opposite_direction(d):
#     return {
#         'north': 'south',
#         'south': 'north',
#         'east': 'west',
#         'west': 'east',
#     }[d]
#
#
# class DirectionMemory:
#     def __init__(self):
#         self.store = defaultdict(list)
#         self.last_x_directions = 1
#
#     def remember(self, node, direction):
#         self.store[node].append(direction)
#         self.store[node] = self.store[node][-self.last_x_directions:]
#
#     def last(self, node):
#         return [get_opposite_direction(x) for x in self.store[node]]
#
#     def __str__(self):
#         return str(self.store)


class CustomDirection(EpsilonDirectionGreedy):

    def __init__(self, file_suffix='_CustomDirectionTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        # self.x = None
        self.is_strict = None
        self.version = None
        self.z_type = None
        self.S = 128  # total states

        self.curr_directions = None
        # self.prev_level4_node = None
        # self.visited_corners = set()

        self.sampled_durations = []
        # self.duration = self.sample_duration()
        # self.remember_corners = None
        self.episode_state_traj = []

        # self.directions_in_memory = DirectionMemory()

        self.s = HOME_NODE  # Start from HOME

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

    def choose_action_1(self):
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

    def choose_action_2(self):
        """
        Algo:
        Choose a random direction at 0 and at level 2 and come back to 0.
        """

        if self.s == HOME_NODE:
            self.duration = 0
            self.curr_directions = None
            return 1, 1.0

        if LVL_BY_NODE[self.s] == 2:
            self.duration = self.sample_duration()
            prev_direction = self.curr_directions
            print("prev_direction", prev_direction)
            self.sample_directions()     # Sample a new direction here
            print("node", self.s, "and node level", LVL_BY_NODE[self.s], ": for sampling duration and direction")
            print("sampled new direction and duration:", "new directions", self.curr_directions)

        if (self.duration == 0) or (not self.is_any_curr_direction_valid()):
            self.get_out_of_the_maze(destination_parent_level=0)     # I have got stuck, Go to level 3 - form of "planning"?
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

    def sample_duration(self):
        if self.z_type == 'fixed':
            return 7
        if self.z_type == 'zipf':
            d = 1+np.random.zipf(a=2)
            print("duration chosen", d)
            self.sampled_durations.append(d)
            return d
        raise Exception("duration sampling type not specified")

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        self.is_strict = params["is_strict"]  # is_strict about choosing among primary and secondary direction
        self.version = params["version"]  # switch between algos to choose action
        self.z_type = params["z_type"]  # switch between algos to choose action
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
            "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency": calculate_normalized_visit_frequency(all_episodes_state_trajs),
            "normalized_visit_frequency_by_level": calculate_normalized_visit_frequency_by_level(
                all_episodes_state_trajs)
        }
        return success, stats

