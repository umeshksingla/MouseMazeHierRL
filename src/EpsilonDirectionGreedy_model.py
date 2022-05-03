"""
EpsilonDirectionGreedy model from Dabney et al 2020.
"""
import os
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy

from parameters import *
from BaseModel import BaseModel
from EpsilonGreedy_model import EpsilonGreedy
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency, calculate_normalized_visit_frequency, \
    calculate_normalized_visit_frequency_by_level
import evaluation_metrics as em


def get_opposite_direction(d):
    return {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east',
    }[d]


class DirectionMemory:
    def __init__(self):
        self.store = defaultdict(list)
        self.last_x_directions = 1

    def remember(self, node, direction):
        self.store[node].append(direction)
        self.store[node] = self.store[node][-self.last_x_directions:]

    def last(self, node):
        return [get_opposite_direction(x) for x in self.store[node]]

    def __str__(self):
        return str(self.store)


class EpsilonDirectionGreedy(EpsilonGreedy):

    def __init__(self, file_suffix='_EpsilonDirectionGreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

        self.x = None
        self.epsilon = None
        self.is_strict = None
        self.version = None
        self.S = 128  # total states

        self.curr_directions = None
        self.prev_level4_node = None
        self.visited_corners = set()

        self.sampled_durations = []
        self.duration = self.sample_duration()
        self.remember_corners = None
        self.episode_state_traj = []

        self.directions_in_memory = DirectionMemory()

        self.s = HOME_NODE  # Start from HOME

    @staticmethod
    def get_interval(target_angle):
        direction_intervals = [(i - 90, i) for i in range(90, 450, 90)]
        for i in direction_intervals:
            if i[0] <= target_angle <= i[1]:
                return i

    def get_valid_directions(self, target_angle):
        """
        returns primary and secondary direction
        At present, directions are defined globally.
        """
        vertical_dir = 'north' if 0 <= target_angle <= 180 else 'south'
        horizontal_dir = 'west' if 90 <= target_angle <= 270 else 'east'
        target_interval = self.get_interval(target_angle)
        mid = abs(target_interval[1]+target_interval[0])/2
        if target_interval in [(0, 90), (180, 270)]:
            if target_angle >= mid:
                return vertical_dir, horizontal_dir
            else:
                return horizontal_dir, vertical_dir
        else:
            if target_angle >= mid:
                return horizontal_dir, vertical_dir
            else:
                return vertical_dir, horizontal_dir

    def is_any_curr_direction_valid(self):
        if not self.curr_directions:
            return False
        state_valid_directions = self.get_action_direction_mapping(self.s)
        dir1, dir2 = self.curr_directions
        # print("state", self.s, "current", self.curr_directions, "- superset of valid at this state", state_valid_directions)
        return (dir1 in state_valid_directions) or (dir2 in state_valid_directions)

    def get_action_based_on_current_direction(self):
        state_valid_directions = self.get_action_direction_mapping(self.s)
        dir1, dir2 = self.curr_directions
        print("self.curr_directions", self.curr_directions, "state_valid_directions", state_valid_directions, self.s)
        if self.is_strict:
            if dir1 in state_valid_directions:
                chosen_dir = dir1
            elif dir2 in state_valid_directions:
                chosen_dir = dir2
            else:
                raise f"what just happened {state_valid_directions} and {self.curr_directions}"
            print("priority direction chosen", chosen_dir)
        else:
            chosen_dir = random.choice(tuple({dir1, dir2}.intersection(state_valid_directions.keys())))
            print("random direction chosen", chosen_dir)
        action = state_valid_directions[chosen_dir]
        return action, chosen_dir

    def get_out_of_the_maze(self, destination_parent_level=3):
        # It will be stuck in L6 coz a direction that led an agent in cannot get it out
        assert LVL_BY_NODE[self.s] == 6
        while LVL_BY_NODE[self.s] != destination_parent_level:
            print("before", self.s)
            self.s = self.take_action(self.s, 0)  # go to parent node
            print("after", self.s)
            self.episode_state_traj.append(self.s)
        return

    def choose_action(self):
        if self.version == 1:
            return self.choose_action_1()
        if self.version == 2:
            return self.choose_action_2()
        if self.version == 3:
            return self.choose_action_3()

    def choose_action_3(self):
        """
        Version 3:

        if duration is still left and possible to go in the current sampled direction:
            do it (either priority or random out of prev)   # 2 variants
            duration -= 1
        else:
            sample a new duration and direction and follow
        """
        print("Version 3")

        if self.s == HOME_NODE:
            # self.duration = 1
            # self.sample_directions()
            self.duration = 0
            self.curr_directions = None
            self.directions_in_memory.remember(0, 'east')
            return 1, 1.0

        if self.remember_corners:
            if LVL_BY_NODE[self.s] == 4 and self.prev_level4_node != self.s:
                # entering a new level 4 node, start tracking level 6 from now on
                self.prev_level4_node = self.s
                self.visited_corners = set()
                print(self.s, "tracking new self.prev_level4_node", self.prev_level4_node)
            elif LVL_BY_NODE[self.s] == 6 and self.prev_level4_node is not None:
                # track this level 6 corner
                self.visited_corners.add(self.s)
                print(f"Added {self.s} to self.visited_corners", self.visited_corners)
            if len(self.visited_corners) == 4:
                # visited all 4 end nodes in this sub-quarter, get out and sample new direction
                print(f"All corners visited {self.visited_corners}, resetting and getting out")
                self.prev_level4_node = None
                self.visited_corners = set()
                self.get_out_of_the_maze()
                self.duration = 0
                self.curr_directions = None
                print("Got out of the maze: new state = ", self.s)

        # last_x_nodes_in_memory = self.episode_state_traj[-self.x:]
        # print("last_x_nodes_in_memory", last_x_nodes_in_memory)
        # direction_memory_copy = deepcopy(self.last_x_nodes_direction_in_memory)
        # clear memory for prev directions
        # self.last_x_nodes_direction_in_memory = defaultdict(list)
        # for n in last_x_nodes_in_memory:
        #     if n not in ALL_MAZE_NODES: continue
        #     self.last_x_nodes_direction_in_memory[n] = direction_memory_copy[n]
        print("self.directions_in_memory", self.directions_in_memory)
        print("self.duration", self.duration)

        if (self.duration == 0) or (not self.is_any_curr_direction_valid()):
            if np.random.random() <= self.epsilon:  # inherent randomness
                # if self.planning:
                #     self.get_out_of_the_maze()     # I have got stuck, Go to level 3 - form of "planning"?
                self.duration = self.sample_duration()
                prev_direction = self.curr_directions
                print("prev_direction", prev_direction)
                while True:
                    self.sample_directions()     # Sample a new direction here
                    # print("node", self.s, "and node level", LVL_BY_NODE[self.s], ": for sampling duration and direction")
                    print("sampled new direction and duration:", "new directions", self.curr_directions)
                    action, chosen_dir = self.get_action_based_on_current_direction()
                    potential_next_state = self.take_action(self.s, action)
                    last_followed_direction = self.directions_in_memory.last(self.s)
                    print("last came from", self.s, last_followed_direction)
                    if self.s in LVL_6_NODES:
                        break
                    if chosen_dir not in last_followed_direction:
                        # use this direction
                        break
                self.directions_in_memory.remember(potential_next_state, chosen_dir)
                self.duration -= 1
            else:
                action = self.__random_action__(self.s)
                print("random", self.s, action)
        else:
            action, chosen_dir = self.get_action_based_on_current_direction()
            potential_next_state = self.take_action(self.s, action)
            self.directions_in_memory.remember(potential_next_state, chosen_dir)
            self.duration -= 1
            print("prev direction used")
        assert action in self.get_valid_actions(self.s)
        return action, 1.0

    def choose_action_2(self):
        """
        Version 2

        with epsilon probability: sample a random direction and follow
        else
            if one can follow an action using prev direction:
                do it (either priority or random out of prev) # 2 variants
            else:
                sample a direction and follow

        """
        print("Version 2")
        if self.s == HOME_NODE:
            # self.sample_directions()
            self.duration = 0
            self.curr_directions = None
            return 1, 1.0

        if self.remember_corners:
            if LVL_BY_NODE[self.s] == 4 and self.prev_level4_node != self.s:
                # entering a new level 4 node
                self.prev_level4_node = self.s
                self.visited_corners = set()
                print(self.s, "tracking new self.prev_level4_node", self.prev_level4_node)
            if LVL_BY_NODE[self.s] == 6 and self.prev_level4_node is not None:
                # track this level 6 corner
                self.visited_corners.add(self.s)
                print(f"Added {self.s} to self.visited_corners", self.visited_corners)
            if len(self.visited_corners) == 4:
                # visited all 4 end nodes in this sub-quarter, get out and sample new direction
                print(f"All corners visited {self.visited_corners}, resetting and getting out")
                self.prev_level4_node = None
                self.visited_corners = set()
                self.get_out_of_the_maze()
                self.sample_directions()
                print("Got out of the maze: new state = ", self.s)

        if np.random.random() <= self.epsilon:
            print("random direction")
            self.sample_directions()
            action = self.get_action_based_on_current_direction()
        else:
            if self.is_any_curr_direction_valid():
                action = self.get_action_based_on_current_direction()
                print("prev direction used")
            else:
                # if self.planning:
                #     self.get_out_of_the_maze()     # I have got stuck, go to level 3 - "planning"?
                self.sample_directions()     # Sample a new direction here
                print(self.s, "node level for sampling direction", LVL_BY_NODE[self.s])
                print("sampled new direction")
                action = self.get_action_based_on_current_direction()
        return action, 1.0

    def choose_action_1(self):
        """
        Version 1: Note how this is independent of epsilon and is completely random in direction

        if one can follow an action using prev direction:
            do it (either priority or random out of prev) # 2 variants
        else
            sample a new direction and follow
        """
        print("Version 1")
        if self.s == HOME_NODE:
            # self.sample_directions()
            self.duration = 0
            self.curr_directions = None
            return 1, 1.0

        if self.remember_corners:
            if LVL_BY_NODE[self.s] == 4 and self.prev_level4_node != self.s:
                # entering a new level 4 node
                self.prev_level4_node = self.s
                self.visited_corners = set()
                print(self.s, "tracking new self.prev_level4_node", self.prev_level4_node)
            if LVL_BY_NODE[self.s] == 6 and self.prev_level4_node is not None:
                # track this level 6 corner
                self.visited_corners.add(self.s)
                print(f"Added {self.s} to self.visited_corners", self.visited_corners)
            if len(self.visited_corners) == 4:
                # visited all 4 end nodes in this sub-quarter, get out and sample new direction
                print(f"All corners visited {self.visited_corners}, resetting and getting out")
                self.prev_level4_node = None
                self.visited_corners = set()
                self.get_out_of_the_maze()
                self.sample_directions()
                print("Got out of the maze: new state = ", self.s)

        if self.is_any_curr_direction_valid():
            action = self.get_action_based_on_current_direction()
            print("prev direction used")
        else:
            # if self.planning:
            #     self.get_out_of_the_maze()     # I have got stuck, Go to level 3 - form of "planning"?
            self.sample_directions()     # Sample a new direction here
            print(self.s, "node level for sampling direction", LVL_BY_NODE[self.s])
            print("sampled new direction")
            action = self.get_action_based_on_current_direction()
        assert action in self.get_valid_actions(self.s)
        return action, 1.0

    def sample_directions(self):

        def is_angle_valid(a):
            return a % 45 != 0

        def sample_angle():
            # return np.random.randint(360)
            return np.random.choice([22.5+45*i for i in range(8)])

        prev_direction = self.curr_directions
        angle = sample_angle()
        self.curr_directions = self.get_valid_directions(angle)
        # print("before loop angle chosen", angle, self.curr_directions)
        while (not is_angle_valid(angle)) or (not self.is_any_curr_direction_valid()):
            angle = sample_angle()
            self.curr_directions = self.get_valid_directions(angle)
        print("angle chosen", angle, self.curr_directions)
        return

    def sample_duration(self):
        return 1
        d = 1+np.random.zipf(a=2)
        print("duration chosen", d)
        self.sampled_durations.append(d)
        return d

    def generate_exploration_episode(self, MAX_LENGTH, Q):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        print("FINAL self.nodemap", self.nodemap)
        print("FINAL self.nodemap_direction_dict", self.nodemap_direction_dict)

        # e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        print("Starting at", self.s)
        while len(self.episode_state_traj) <= MAX_LENGTH:
            assert self.s != RWD_STATE   # since it's pure exploration

            # Record current state
            self.episode_state_traj.append(self.s)

            # acting
            a, a_prob = self.choose_action()   # NOTE: there's a side-affect to some versions where they change s
            s_next = self.take_action(self.s, a)     # Take action
            print("action: ", a, f": {self.s} => {s_next}")

            # td_error = 0.0 + gamma * np.max([Q[s_next, a_i] for a_i in self.get_valid_actions(s_next)]) - Q[s, a]   # R = 0
            # e[s, a] += 1
            # for n in np.arange(self.S):
            #     Q[n, :] += alpha * td_error * e[n, :]
            #     e[n, :] = gamma * lamda * e[n, :]
            # Q[s, a] = self.is_valid_state_value(Q[s, a])
            self.s = s_next

            if len(self.episode_state_traj) % 1000 == 0:
                print("current state", self.s, "step", len(self.episode_state_traj))

        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs = break_simulated_traj_into_episodes(self.episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same
        print(episode_maze_trajs)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        # alpha = params["alpha"]         # learning rate
        # gamma = params["gamma"]         # discount factor
        # lamda = params["lamda"]         # eligibility trace-decay
        self.epsilon = params["epsilon"]     # epsilon
        self.is_strict = params["is_strict"]  # is_strict about choosing among primary and secondary direction
        self.version = params["version"]     # switch between algos to choose action
        self.x = params["x"]  # switch between algos to choose action

        # self.planning = params.get("planning", False)  # if "plan" to move out of the maze from level 6
        self.remember_corners = params["remember_corners"]   # if agent remembers to track corner nodes
        initial_v = 1   # params["V"]

        print("params", params)
        Q = np.zeros((self.S, self.A)) * initial_v  # Initialize state values
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


if __name__ == '__main__':
    pass
