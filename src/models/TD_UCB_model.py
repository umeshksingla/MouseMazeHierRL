"""
TD + UCB model where UCB doesn't change state values but only affects action
selection.
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


class TD_UCBpolicy(BaseModel):

    def __init__(self, file_suffix='_TD_UCBpolicyTrajectories'):
        BaseModel.__init__(self, file_suffix=file_suffix)

    def get_action_probabilities(self, state, beta, V):
        raise Exception("wasn't supposed to be called")

    def choose_action(self, s, V_ucb, *args, **kwargs):
        if s in LVL_6_NODES:
            return 0, 1.0
        else:
            possible_action_values = [
                round(V_ucb[int(future_state)], 10) for future_state in self.nodemap[s, :]
                if future_state != INVALID_STATE
            ]
            print(s, ":", possible_action_values)

            if len(set(possible_action_values)) == 1:
                # i.e. if all choices have same value, choose random
                a = random.choice(np.arange(len(possible_action_values)))
            else:
                a = np.argmax(possible_action_values)
            return a, 1.0

    def generate_exploration_episode(self, alpha, gamma, lamda, c, t, N, MAX_LENGTH, V):
        print(self.nodemap)
        s = 0   # Start from 0 in exploration mode
        episode_traj = []
        LL = 0.0
        self.nodemap[WATERPORT_NODE][1] = -1  # No action to go to RWD_STATE
        e = np.zeros(self.S)  # eligibility trace vector for all states
        # N[s] += 1
        # t += 1
        while True:
            assert s != RWD_STATE
            episode_traj.append(s)  # Record current state
            N[s] += 1
            t += 1
            if s in self.terminal_nodes and t > 1:
                print(f"reached {s}, entering again")
                s = 0   # Start from 0 when you hit home in exploration mode
                episode_traj.append(s)  # Add new initial state s to it
                N[s] += 1
                t += 1
                e = np.zeros(self.S)
            # print(V.shape, t.shape, N.shape)
            # print(V, c * np.sqrt(np.log(t)/N))
            # a, a_prob = self.choose_action(s, V + c * np.sqrt(np.log(t)/N))  # Choose action
            a, a_prob = self.choose_action(s, c * np.sqrt(np.log(t) / N))  # Choose action
            s_next = self.take_action(s, a)  # Take action
            # LL += np.log(a_prob)    # Update log likelihood
            # print("s, s_next, a, action_prob", s, s_next, a, action_prob)

            R = 0   # Zero reward

            # # Update state values
            # td_error = R + gamma * V[s_next] - V[s]
            # e[s] += 1
            # for n in np.arange(self.S):
            #     V[n] += alpha * td_error * e[n]
            #     e[n] = gamma * lamda * e[n]
            #
            # V[s] = self.is_valid_state_value(V[s])

            assert np.count_nonzero(V) == 0

            if len(episode_traj) > MAX_LENGTH:
                print('Max trajectory length reached. Ending this trajectory.')
                break

            s = s_next
            if len(episode_traj)%100 == 0:
                print("current state", s, "step", len(episode_traj))

        # episodes = break_simulated_traj_into_episodes(episode_traj)
        # return True, episodes, LL

        # if s != HOME_NODE:
        #     self.make_it_go_to(HOME_NODE)

        # print(self.episode_state_traj
        print('Max trajectory length reached. Ending this trajectory.')
        episode_state_trajs, episode_maze_trajs = self.wrap(episode_traj)
        return True, episode_state_trajs, episode_maze_trajs, 0.0

    def generate_episode(self, alpha, gamma, lamda, c, MAX_LENGTH, V):
        raise Exception("Nope!")

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        alpha = params["alpha"]         # learning rate
        gamma = params["gamma"]         # discount factor
        lamda = params["lamda"]         # eligibility trace-decay
        c = params["c"]                 # ucb exploration

        print("alpha, gamma, lamda, c, agentId",
              alpha, gamma, lamda, c, agentId)

        V = np.zeros(self.S)  # Initialize state values
        V[HOME_NODE] = 0
        V[RWD_STATE] = 0
        t = 0
        N = np.ones(self.S)
        all_episodes = []
        # LL = 0.0
        # while len(all_episodes) < N_BOUTS_TO_GENERATE:
        #     _, episodes, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda, c, t, N, MAX_LENGTH, V)
        #     all_episodes.extend(episodes)
        #     LL += episode_ll
        # stats = {
        #     "agentId": agentId,
        #     "episodes": all_episodes,
        #     "LL": LL,
        #     "MAX_LENGTH": MAX_LENGTH,
        #     "count_total": len(all_episodes),
        #     "V": V,
        # }

        _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda, c, t, N, MAX_LENGTH, V)

        stats = {
            "agentId": agentId,
            "episodes_states": episode_state_trajs,
            "episodes_positions": episode_maze_trajs,
            "LL": 0.0,
            "MAX_LENGTH": MAX_LENGTH,
            "Q": None,
            "V": V,
        }
        return success, stats

    def get_maze_state_values(self, V):
        """
        Get state values to plot against the nodes on the maze
        """
        return V


if __name__ == '__main__':
    from sample_agent import run

    param_sets = [{
        'alpha': 0.0, 'gamma': 0.0, 'lamda': 0.0, 'c': 1, 'rew': False,
    }]
    runids = run(TD_UCBpolicy(), param_sets, '/Users/us3519/mouse-maze/figs', '20000', analyze=True)
