"""
Bayesian QL agent.

Implemented with https://github.com/amarildolikmeta/bayesianQLearning as the
reference,

"""

import numpy as np
import scipy.stats
import math
from scipy import special
from collections import defaultdict

from parameters import *
from Dyna_Qplus import DynaQPlus
from utils import break_simulated_traj_into_episodes, calculate_visit_frequency
import evaluation_metrics as em
from models.bayesian_utils import argmaxrand_dict


class BayesianQL(DynaQPlus):

    def __init__(self, file_suffix='_BayesianQLTrajectories'):
        DynaQPlus.__init__(self, file_suffix=file_suffix)
        self.S = 128

    def generate_episode(self):
        raise Exception("Nope!")

    def choose_action(self, state, action_selection_method):
        if action_selection_method == 'myopic':
            return self.Myopic_VPI_action_selection(self.NG, state), 1.0
        elif action_selection_method == 'q_sampling':
            return self.Q_sampling_action_selection(self.NG, state), 1.0
        else:
            raise Exception("Valid values for argument action_selection_method are: 'myopic', 'q_sampling'.")

    def update_q_values(self, state, action, reward, next_state, discount_factor):
        self.moment_updating(state, action, reward, next_state, discount_factor)

    def moment_updating(self, state, action, reward, next_state, discount_factor):
        mean = self.NG[state][action][0]
        lamb = self.NG[state][action][1]
        alpha = self.NG[state][action][2]
        beta = self.NG[state][action][3]

        # assuming that the agent will follow the apparently optimal policy,
        # then R(next_state) is distributed as R(next_state,a'), where a' is the
        # action with the highest expected value at next_state.
        # (Dearden 1998, Section 3.3)
        next_pos_action_means_dict = dict([(a, self.NG[next_state, a, 0]) for a in self.get_valid_actions(next_state)])
        next_action = argmaxrand_dict(next_pos_action_means_dict)

        mean_next = self.NG[next_state][next_action][0]
        lamb_next = self.NG[next_state][next_action][1]
        alpha_next = self.NG[next_state][next_action][2]
        beta_next = self.NG[next_state][next_action][3]

        m1 = reward + discount_factor * mean_next
        m2 = reward ** 2 + \
             2 * discount_factor * reward * mean_next + \
             (discount_factor ** 2) * (((lamb_next+1)/lamb_next)*(beta_next/(alpha_next-1)) + mean_next ** 2)

        # update the distribution
        n = 1
        self.NG[state][action][0] = (lamb * mean + n*m1) / (lamb + 1)
        self.NG[state][action][1] = lamb + n
        self.NG[state][action][2] = alpha + 0.5 * n
        self.NG[state][action][3] = beta + \
                                    0.5 * n * (m2 - m1 ** 2) + \
                                    (n * lamb * (m1 - mean) ** 2) / (2 * (lamb + n))

    def generate_exploration_episode(self, discount_factor, action_selection_method, MAX_LENGTH):

        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        print(" === nodemap start ===\n", self.nodemap, "\n === nodemap end ===")

        episode_state_traj = []
        variances = defaultdict(list)

        s = HOME_NODE  # Start from HOME
        while len(episode_state_traj) <= MAX_LENGTH:
            assert s != RWD_STATE   # since it's pure exploration

            # Record current state
            episode_state_traj.append(s)

            a, a_prob = self.choose_action(s, action_selection_method)   # Choose action
            s_next = self.take_action(s, a)  # Take action
            reward = 0.0

            # update q values
            print("before")
            self.print_q_values(s)
            print("action taken", a, "s_next", s_next)
            self.update_q_values(s, a, reward, s_next, discount_factor)
            print("after")
            self.print_q_values(s)
            print()

            s = s_next
            if len(episode_state_traj) % 1000 == 0:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>  current state", s, "step", len(episode_state_traj), "<<<<<<<<<<<<<<<<<<<<<<<<<")

            # calculating and storing variances for action 0 for each state to help plot
            for n in np.arange(self.S):
                a = 0
                variances[n].append(self.NG[n, a, 3] / ((self.NG[n, a, 2] - 1) * self.NG[n, a, 1]))

        print('Max trajectory length reached. Ending this trajectory.')

        episode_state_trajs = break_simulated_traj_into_episodes(episode_state_traj)
        episode_state_trajs = list(filter(lambda e: len(e), episode_state_trajs))  # remove empty or short episodes
        episode_maze_trajs = episode_state_trajs    # in pure exploration, both are same

        # Just checking variances for few states and see how they change
        # import matplotlib.pyplot as plt
        # random_state = np.random.choice(range(63, self.S))
        # plt.plot(variances[random_state], label=f'variances[{random_state}]')
        # random_state = np.random.choice(range(31, 63))
        # plt.plot(variances[random_state], label=f'variances[{random_state}]')
        # random_state = np.random.choice(range(15, 31))
        # plt.plot(variances[random_state], label=f'variances[{random_state}]')
        # random_state = np.random.choice(range(6, 15))
        # plt.plot(variances[random_state], label=f'variances[{random_state}]')
        # random_state = np.random.choice(range(0, 6))
        # plt.plot(variances[random_state], label=f'variances[{random_state}]')
        # plt.legend()
        # plt.xlabel('time')
        # plt.ylabel('variance')
        # plt.show()
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    @staticmethod
    def get_cumulative_distribution(mean, lamb, alpha, beta, x):
        rv = scipy.stats.t(2 * alpha)
        return rv.cdf((x - mean) * math.sqrt((lamb * alpha) / beta))

    def Q_sampling_action_selection(self, NG, state):

        # Sample one value for each action
        valid_actions = self.get_valid_actions(state)
        samples = defaultdict(float)
        for i in valid_actions:
            mean = NG[state][i][0]
            lamb = NG[state][i][1]
            alpha = NG[state][i][2]
            beta = NG[state][i][3]

            scale = np.sqrt(beta / (alpha * lamb))
            # samples[i] = np.random.standard_t(2 * alpha) * scale + mean
            # samples[i] = self.get_cumulative_distribution(mean, lamb, alpha, beta, x)
            samples[i] = scipy.stats.t.rvs(df=2 * alpha, loc=mean, scale=scale)
        print(samples)
        return argmaxrand_dict(samples)

    def get_best_two_actions(self, A):
        A = sorted(A, key=lambda x: x[1], reverse=True)
        return A[0][0], A[1][0]

    def Myopic_VPI_action_selection(self, NG, state):

        def get_c(mean, lamb, alpha, beta):
            c = math.sqrt(beta) / ((alpha - 0.5) * math.sqrt(2 * lamb) * special.beta(alpha, 0.5))
            c = c * math.pow(1 + (mean ** 2 / (2 * alpha)), 0.5 - alpha)
            return c

        vpi_weight = 0.1

        valid_actions = self.get_valid_actions(state)
        ranking = defaultdict(float)

        means = [(i, NG[state][i][0]) for i in valid_actions]
        if len(valid_actions) == 1:
            best_action = valid_actions[0]
            second_best = best_action
        else:
            best_action, second_best = self.get_best_two_actions(means)   # get best and second best action

        mean1 = NG[state][best_action][0]
        mean2 = NG[state][second_best][0]

        for i in valid_actions:
            mean = NG[state][i][0]
            lamb = NG[state][i][1]
            alpha = NG[state][i][2]
            beta = NG[state][i][3]
            c = get_c(mean, lamb, alpha, beta)
            if i == best_action:    # mean == mean1 in this case
                ranking[i] = vpi_weight * (c + (mean2 - mean1) * self.get_cumulative_distribution(mean1, lamb, alpha, beta, mean2)) + mean1
            else:
                ranking[i] = vpi_weight * (c + (mean - mean1) * (1 - self.get_cumulative_distribution(mean, lamb, alpha, beta, mean1))) + mean

            # print("mean mean1 mean2", mean, mean1, mean2, ": action", i, ":", ranking[i])
        return argmaxrand_dict(ranking)

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        gamma = params["gamma"]         # discount factor
        action_selection_method = params["action_selection_method"]

        initial_alpha = params["initial_alpha"]
        initial_beta = params["initial_beta"]
        initial_lambda = params["initial_lambda"]

        print("gamma, agentId", gamma, agentId)

        # initialize the distributions
        self.NG = np.zeros(shape=(self.S, self.A), dtype=(float, 4))
        for state in range(self.S):
            for action in range(self.A):
                self.NG[state][action][0] = 0.  # mu
                self.NG[state][action][1] = initial_lambda + np.random.rand()  # lambda
                self.NG[state][action][2] = initial_alpha + np.random.rand()  # alpha > 1 ensures the normal-gamma dist is well defined
                self.NG[state][action][3] = initial_beta + np.random.rand()  # high beta to increase the variance of the prior distribution to explore more

        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        LL = 0.0
        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(gamma, action_selection_method, MAX_LENGTH)
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
            "Q": None,
            "V": self.get_maze_state_values_from_action_values(),
            "exploration_efficiency": em.exploration_efficiency(all_episodes_state_trajs, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs)
        }
        return success, stats

    def get_maze_state_values_from_action_values(self):
        """
        Get state values to plot against the nodes on the maze
        """
        print("Variance for each state-action")
        for n in np.arange(self.S):
            for a_i in self.get_valid_actions(n):
                var = self.NG[n, a_i, 3] / ((self.NG[n, a_i, 2]-1) * self.NG[n, a_i, 1])
                print(f"state {n}: action {a_i} -> {var}")
            print()
        return np.array([np.max([self.NG[n, a_i, 0] for a_i in self.get_valid_actions(n)]) for n in np.arange(self.S)])

    def print_q_values(self, state):
        print("state", state)
        for a_i in self.get_valid_actions(state):
            mean = self.NG[state, a_i, 0]
            var = self.NG[state, a_i, 3] / ((self.NG[state, a_i, 2]-1) * self.NG[state, a_i, 1])
            print("action", a_i, ":", "mean", mean, "var", var)
        return


if __name__ == '__main__':

    # sample agent run
    from plot_utils import plot_exploration_efficiency, plot_maze_stats

    param_sets = {
        0: {"gamma": 0.99, "action_selection_method": 'q_sampling',
            "initial_alpha": 1.5, "initial_lambda": 3, "initial_beta": 0.5},
    }
    simulation_results = BayesianQL().simulate_multiple(param_sets, MAX_LENGTH=20000, N_BOUTS_TO_GENERATE=1)
    success, stats = simulation_results[0]
    episodes = stats["episodes_positions"]
    V = stats["V"]
    plot_exploration_efficiency(episodes, re=False, title=param_sets[0], display=True)
    plot_maze_stats(V, interpolate_cell_values=True, display=True, figtitle=f'state values\n{param_sets[0]}')
    print("finished.")
