"""
Epsilon3Greedy model, with 3 actions at nodes. Contrast it with Epsilon2Greedy.
"""
import os
import numpy as np
import random

from parameters import *
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived
from BaseModel import BaseModel
from utils import break_simulated_traj_into_episodes


def info(title):
    print(*title)
    print('>>> module name:', __name__, 'parent process id:', os.getppid(),
          'process id:', os.getpid())


class Epsilon3Greedy(BaseModel):

    def __init__(self, file_suffix='_Epsilon3GreedyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

    def get_action_probabilities(self, state, beta, V):
        raise Exception("wasn't supposed to be called")

    def __greedy_action__(self, state, V):
        """
        epsilon greedy
        :param state:
        :param V:
        :return: greedy action index based on SA_nodemap
        """
        possible_action_values = [
            V[int(future_state)] for future_state in self.nodemap[state, :]
            if future_state != INVALID_STATE
        ]
        if len(list(set(possible_action_values))) == 1:
            # i.e. if all choices have same value, choose random
            greedy_action = random.choice(np.arange(len(possible_action_values)))
        else:
            greedy_action = np.argmax(possible_action_values)
        return greedy_action

    def __random_action__(self):
        """
        Random action from one of left, right or back.
        :return: random action index
        """
        return np.random.choice(range(3))

    def choose_action(self, s, V, epsilon, *args, **kwargs):
        if s in LVL_6_NODES:
            return 0, 1.0
        else:
            random_action = np.random.choice(range(3))
            greedy_action = self.__greedy_action__(s, V)
            action = np.random.choice([random_action, greedy_action], p=[epsilon, 1-epsilon])
            return action, 0.0

    def generate_exploration_episode(self, alpha, gamma, lamda, epsilon, MAX_LENGTH, V):
        print(self.nodemap)
        s = 0   # Start from 0 in exploration mode
        episode_traj = []
        LL = 0.0
        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        e = np.zeros(self.S)  # eligibility trace vector for all states
        while True:
            assert s != RWD_STATE
            episode_traj.append(s)  # Record current state
            if s in self.terminal_nodes:
                print(f"reached {s}, entering again")
                s = 0   # Start from 0 when you hit home in exploration mode
                episode_traj.append(s)  # Add new initial state s to it
                e = np.zeros(self.S)

            a, a_prob = self.choose_action(s, V, epsilon)  # Choose action
            s_next = self.take_action(s, a)  # Take action
            # LL += np.log(a_prob)    # Update log likelihood
            # print("s, s_next, a, action_prob", s, s_next, a, action_prob)

            R = 0   # Zero reward

            # Update state values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
            for n in np.arange(self.S):
                V[n] += alpha * td_error * e[n]
                e[n] = gamma * lamda * e[n]

            V[s] = self.is_valid_state_value(V[s])

            if len(episode_traj) > MAX_LENGTH:
                print('Max trajectory length reached. Ending this trajectory.')
                break

            s = s_next
            if len(episode_traj)%100 == 0:
                print("current state", s, "step", len(episode_traj))

        episodes = break_simulated_traj_into_episodes(episode_traj)
        return True, episodes, LL

    def generate_episode(self, alpha, gamma, lamda, epsilon, MAX_LENGTH, V):
        raise Exception("Nope!")
        s = self.get_initial_state()
        episode_traj = []
        LL = 0.0
        first_reward = -1
        e = np.zeros(self.S)  # eligibility trace vector for all states
        while True:
            episode_traj.append(s)  # Record current state
            if s in self.terminal_nodes:
                print(f"reached {s}, entering again")
                s = self.get_initial_state()
                e = np.zeros(self.S)

            if s != WATERPORT_NODE:
                a, a_prob = self.choose_action(s, V, epsilon)  # Choose action
                s_next = self.take_action(s, a)  # Take action
                # LL += np.log(a_prob)    # Update log likelihood
                # print("s, s_next, a, action_prob", s, s_next, a, action_prob)
            else:
                s_next = RWD_STATE

            R = 1 if s == WATERPORT_NODE else 0  # Observe reward

            # Update state values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
            for n in np.arange(self.S):
                V[n] += alpha * td_error * e[n]
                e[n] = gamma * lamda * e[n]

            V[s] = self.is_valid_state_value(V[s])

            if s == WATERPORT_NODE:
                print('Reward Reached!')
                if first_reward == -1:
                    first_reward = len(episode_traj)
                    print("First reward:", len(episode_traj))

            if len(episode_traj) > MAX_LENGTH + first_reward:
                print('Max trajectory length reached. Ending this trajectory.')
                break

            s = s_next

            if len(episode_traj)%100 == 0:
                print("current state", s, "step", len(episode_traj))

        episodes = break_simulated_traj_into_episodes(episode_traj)
        return True, episodes, LL

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        alpha = params["alpha"]         # learning rate
        gamma = params["gamma"]         # discount factor
        lamda = params["lamda"]         # eligibility trace-decay
        epsilon = params["epsilon"]     # epsilon
        initial_v = params["V"]

        print("alpha, gamma, lamda, epsilon, V, agentId",
              alpha, gamma, lamda, epsilon, initial_v, agentId)

        V = {'zero': np.zeros(self.S), 'one': np.ones(self.S)}[initial_v]  # Initialize state values
        V[HOME_NODE] = 0
        V[RWD_STATE] = 0
        all_episodes = []
        LL = 0.0
        while len(all_episodes) < N_BOUTS_TO_GENERATE:
            _, episodes, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda, epsilon, MAX_LENGTH, V)
            all_episodes.extend(episodes)
            LL += episode_ll
        stats = {
            "agentId": agentId,
            "episodes": all_episodes,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes),
            "V": V,
        }
        return success, stats

    def get_maze_state_values(self, V):
        """
        Get state values to plot against the nodes on the maze
        """
        return V


if __name__ == '__main__':
    pass
