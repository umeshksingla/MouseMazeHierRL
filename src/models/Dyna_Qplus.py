"""
Dyna Q+ agent.
"""
import os
import numpy as np
import random

from parameters import *
from BaseModel import BaseModel
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency
import evaluation_metrics as em


def info(title):
    print(*title)
    print('>>> module name:', __name__, 'parent process id:', os.getppid(),
          'process id:', os.getpid())


class DynaQPlus(BaseModel):

    def __init__(self, file_suffix='_DynaQPlusTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)
        self.back_action = True # if there is a back action, Note: back_action=False is not verified yet, so keep it True for now.

    def get_action_probabilities(self, state, beta, V):
        raise Exception("wasn't supposed to be called")

    def get_valid_actions(self, state):
        """
        Get valid actions available at the "state".
        Note: back_action=False is not verified yet.
        """
        if state == HOME_NODE:
            return [1]
        if state in LVL_6_NODES:
            return [0]
        else:
            if self.back_action:
                return [0, 1, 2]
            else:
                return [1, 2]

    def __greedy_action__(self, state, Q):
        """
        Choose a valid action in state greedily based on Q-values
        :param state:
        :param Q:
        :return: greedy action index based on SA_nodemap
        """
        action_values = dict([(a_i, round(Q[state, a_i], 10)) for a_i in self.get_valid_actions(state)])
        # print("action_values", action_values, action_values.keys())
        if len(set(list(action_values.values()))) == 1:
            # i.e. if all choices have same value, choose random
            greedy_action = random.choice(list(action_values.keys()))
        else:
            greedy_action = max(action_values.items(), key=lambda x: x[1])[0]
        return greedy_action

    def __random_action__(self, state):
        """
        Random action from the actions available in this state.
        :return: random action index
        """
        return np.random.choice(self.get_valid_actions(state))

    def choose_action(self, s, Q, epsilon, *args, **kwargs):
        random_action = self.__random_action__(s)
        greedy_action = self.__greedy_action__(s, Q)
        action = np.random.choice([random_action, greedy_action], p=[epsilon, 1-epsilon])
        if not epsilon:
            assert action == greedy_action
        # print(f"greedy_action {greedy_action}, random_action {random_action} in state {s}: chosen {action}")
        return action, 1.0

    def generate_exploration_episode(self, alpha, gamma, lamda, epsilon, k, n_plan, bonus_in_planning, T, M, MAX_LENGTH, Q):
        """
        T contains the time steps a state hasn't been tried in T[n] time steps.
        """

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        # print(" === nodemap start ===\n", self.nodemap, "\n === nodemap end ===")

        episode_state_traj = []
        LL = 0.0    # loglikelihood (broken at the moment)
        e = np.zeros((self.S, self.A))  # eligibility trace vector for all states

        s = HOME_NODE  # Start from HOME
        print("Starting at", s)
        while len(episode_state_traj) <= MAX_LENGTH:
            assert s != RWD_STATE   # since it's pure exploration

            # Record current state
            episode_state_traj.append(s)

            # acting
            if bonus_in_planning:   # using bonus in planning
                a, a_prob = self.choose_action(s, Q, epsilon)
            else:   # bonus in action selection
                a, a_prob = self.choose_action(s, Q + k * np.sqrt(T), epsilon)
            s_next = self.take_action(s, a)     # Take action

            # direct RL; learning (updating state values)
            td_error = 0.0 + gamma * np.max([Q[s_next, a_i] for a_i in self.get_valid_actions(s_next)]) - Q[s, a]   # R = 0
            e[s, a] += 1
            for n in np.arange(self.S):
                Q[n, :] += alpha * td_error * e[n, :]
                e[n, :] = gamma * lamda * e[n, :]
                T[n, :] += 1
            T[s, a] = 0    # coz we just tried (s, a)

            Q[s, a] = self.is_valid_state_value(Q[s, a])

            # model learning
            M[s, a] = int(s_next)

            # planning
            assert (np.count_nonzero(Q) == 0 if not bonus_in_planning else True)
            while n_plan:
                s_random = np.random.choice(np.arange(self.S))
                if s_random in self.terminal_nodes:
                    continue
                a_random = np.random.choice(self.get_valid_actions(s_random))   # any random action in S, not necessarily previously taken in S
                s_next_random = int(M[s_random, a_random])
                # print("s_random, a_random, s_next_random", s_random, a_random, s_next_random)
                if bonus_in_planning:
                    r_random = 0.0 + k * np.sqrt(T[s_random, a_random])     # use exploration bonus while planning as well
                else:
                    r_random = 0.0
                Q[s_random, a_random] += alpha * (r_random + gamma * np.max(Q[s_next_random, :]) - Q[s_random, a_random])
                n_plan -= 1

            s = s_next
            if len(episode_state_traj)%1000 == 0:
                print("current state", s, "step", len(episode_state_traj))
                # print("Q", Q)
                # print("T", T)
                # print("M", M)
            # print("===========================")

        print('Max trajectory length reached. Ending this trajectory.')

        episode_state_trajs = break_simulated_traj_into_episodes(episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same

        return True, episode_state_trajs, episode_maze_trajs, LL

    def generate_episode(self):
        raise Exception("Nope!")

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        alpha = params["alpha"]         # learning rate
        gamma = params["gamma"]         # discount factor
        lamda = params["lamda"]         # eligibility trace-decay
        epsilon = params["epsilon"]     # epsilon
        k = params["k"]                 # bonus factor
        n_plan = params["n_plan"]       # number of planning steps
        bonus_in_planning = params["bonus_in_planning"]     # True if exploration bonus to be considered in planning step only, False if during action selection only

        print("alpha, gamma, lamda, epsilon, k, n_plan, bonus_in_planning, agentId",
              alpha, gamma, lamda, epsilon, k, n_plan, bonus_in_planning, agentId)

        Q = np.zeros((self.S, self.A))  # Initialize state values
        Q[HOME_NODE, :] = 0
        Q[RWD_STATE, :] = 0
        T = np.zeros(Q.shape)
        M = np.zeros(Q.shape)
        for n in np.arange(self.S):
            M[n, :] = int(n)     # initial model is that an action would lead back to the same state with a reward of zero until observed

        # print("model", M)
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        LL = 0.0
        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda,
                                                                                                       epsilon,  k, n_plan,
                                                                                                       bonus_in_planning, T, M,
                                                                                                       MAX_LENGTH, Q)
            all_episodes_state_trajs.extend(episode_state_trajs)
            all_episodes_pos_trajs.extend(episode_maze_trajs)
            LL += episode_ll
        print("Q", Q, "T", T)
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
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs)
        }
        return success, stats

    def get_maze_state_values(self, V):
        """
        Get state values to plot against the nodes on the maze.
        Do any transformations here if you need to on V matrix.
        """
        return V

    def get_maze_state_values_from_action_values(self, Q):
        """
        Get state values to plot against the nodes on the maze
        """
        return np.array([np.max([Q[n, a_i] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])


if __name__ == '__main__':

    # sample dynaq+ agent run
    from plot_utils import plot_exploration_efficiency, plot_maze_stats
    param_sets = {
        0: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.0, "n_plan": 50000, "bonus_in_planning": True},
    }
    simulation_results = DynaQPlus().simulate_multiple(param_sets, MAX_LENGTH=20000, N_BOUTS_TO_GENERATE=1)
    success, stats = simulation_results[0]
    episodes = stats["episodes_positions"]
    V = stats["V"]
    plot_exploration_efficiency(episodes, re=False, title=param_sets[0], display=True)
    plot_maze_stats(V, interpolate_cell_values=True,display=True, figtitle=f'state values\n{param_sets[0]}')
    print("finished.")
