"""
SR model: Successor representation RL agent
"""

from multiprocessing import Pool
import os
import numpy as np
from numpy import arange

from BaseModel import BaseModel
from parameters import RWD_NODE, WATER_PORT_STATE, HOME_NODE, LVL_6_NODES, ALL_MAZE_NODES, INVALID_STATE, \
    ALL_VISITABLE_NODES, TIME_EACH_MOVE
from plot_utils import plot_trajectory, plot_maze_stats, plot_exploration_efficiency
from utils import calculate_visit_frequency


def info(title):
    print(*title)
    print('>>> module name:', __name__, 'parent process id:', os.getppid(),
          'process id:', os.getpid())

class SR(BaseModel):
    """RL model with Successor Representation"""

    def __init__(self):
        super().__init__(self)
        self.terminal_nodes = {HOME_NODE}
        self.T = self.get_transition_matrix()

    def get_transition_matrix(self):
        n_states=len(ALL_VISITABLE_NODES)
        T = np.zeros((n_states, n_states))
        for state in arange(n_states):
            mask_valid_next_states = np.logical_and(self.nodemap[state] != INVALID_STATE, self.nodemap[state] != WATER_PORT_STATE)
            n_possibilities = float(sum(mask_valid_next_states))

            for next_state in self.nodemap[state][mask_valid_next_states]:
                T[state, next_state] = float(1 / n_possibilities)
        return T

    def get_action_probabilities(self, state, beta, V):
        """
        Softmax policy to select action, a at current state, s
        :param state:
        :param beta:
        :param V:
        :return: list of action_probabilities for three possible actions [return to shallower node, lower_idx_node, higher_idx_node]
        """
        if state in LVL_6_NODES:
            action_prob = [1, 0, 0]  # return is the only possible action at the end node
        else:  # this part implements softmax
            betaV = [np.exp(beta * V[int(future_state)]) for future_state in self.nodemap[state, :]]
            # print(V[self.nodemap[state, :]], betaV, state)
            action_prob = []
            n_infs = np.sum(np.isinf(betaV))
            if n_infs!=0:
                print("WARNING: Overflow in softmax computation. There were %d infs" % n_infs)
            for action in np.arange(self.A):
                if np.isinf(betaV[action]):  # can happen when using high beta values
                    action_prob.append(1./n_infs)
                elif np.isnan(betaV[action]):  # TODO: When would this happen?
                    action_prob.append(0)
                else:
                    if n_infs==0:
                        action_prob.append(betaV[action] / np.nansum(betaV))
                    else:
                        action_prob.append(0)

            # Check for invalid probabilities
            for i in action_prob:
                if np.isnan(i):
                    print('Invalid action probabilities ', action_prob, betaV, state)

            if np.sum(action_prob) <= 0.999 or np.sum(action_prob) >= 1.001:
                print('Invalid action probabilities, failed summing to 1: ',
                      action_prob, betaV, state)

        return action_prob


    def get_initial_state(self) -> int:
        return 0


    def generate_episode(self, beta, gamma, max_length, V, time_from_last_rwd=90, time_each_move=TIME_EACH_MOVE):
        """
        Generate an episode using the given agent parameters
        :param beta: inverse temperature parameter. It controls randomness of softmax
        :param gamma: discount factor
        :param max_length: max number of actions before aborting episode
        :param V (ndarray): state values
        :param time_each_move: the amount of time in seconds the mouse takes to do each move
        :return:
        valid_episode (bool): False only when episode is aborted because of not reaching a terminal state after
        `max_length` actions,
        episode_state_traj (list),
        episode_maze_traj (list),
        value_hist (list)
        """

        def check_value_function(V):
            # TODO: When would the cases below happen? Add explanation
            if np.isnan(V[s]):
                print('Warning invalid state-value: ')
            elif np.isinf(V[s]):
                print('Warning infinite state-value: ', V)
            elif abs(V[s]) >= 1e5:
                print('Warning state value exceeded upper bound. Might approach infinity.')
                V[s] = np.sign(V[s]) * 1e5
            return V

        s = self.get_initial_state()
        episode_state_traj = [s]  # initialize episode trajectory list with the initial state
        episode_maze_traj = [s]  # initialize episode trajectory list with the initial state
        # time_from_last_rwd = 90  # TODO: change this so that the agent don't receive a reward after every visit home
        M = np.linalg.inv(np.eye(self.T.shape[0]) - gamma * self.T)  # matrix M with expected future occupancies in each line
        tau = .05

        time_from_last_rwd += time_each_move
        rwd = np.zeros(self.S-1)  # S-1, bc I'm not using waterport state
        rwd[RWD_NODE] = 1  # value of RWD_NODE state is set to 1 when it reaches HOME
        rwd[HOME_NODE] = 0  # value of HOME_NODE is reset to 0 when it reaches HOME
        V = self.calculate_value(M, rwd)
        value_hist = [V]

        while s not in self.terminal_nodes and len(episode_state_traj) < max_length:

            # update states and get rewards
            if s == WATER_PORT_STATE:
                s_next = RWD_NODE
                # R = 1
                rwd[RWD_NODE] = 0  # reset the drive to go to waterport
                rwd[HOME_NODE] = 15
                time_from_last_rwd = 0
                # print(rwd)
            elif (s == RWD_NODE) & (time_from_last_rwd >= 90):
                s_next = WATER_PORT_STATE
                # R = 0
            else:
                action_prob = self.get_action_probabilities(s, beta, V)
                a = np.random.choice(range(self.A), 1, p=action_prob)[0]  # Choose action
                s_next = int(self.nodemap[s, a])           # Take action
                time_from_last_rwd += time_each_move #time
                # R = 0

            episode_state_traj.append(s_next)  # Record next state
            if s_next in ALL_VISITABLE_NODES and s!=WATER_PORT_STATE:  # second condition avoids repetition of the RWD_NODE state in the record
                episode_maze_traj.append(s_next)  # Record next state

            # Update state-values
            # rwd[RWD_NODE] = rwd[RWD_NODE] + tau*(1 - rwd[RWD_NODE])
            # rwd[HOME_NODE] = rwd[HOME_NODE] + tau*(1 - rwd[HOME_NODE])
            V = self.calculate_value(M, rwd)
            value_hist.append(V)

            V = check_value_function(V)
            if s_next == WATER_PORT_STATE:
                print('Reward consumed. Trial ', len(episode_maze_traj))
                # print(rwd)
            elif s_next==HOME_NODE:
                print('Home reached. Trial ', len(episode_maze_traj))

            s = s_next

        if s in self.terminal_nodes:
            valid_episode = True
        else:
            print('Trajectory too long. Episode was aborted. Another attempt will be made')
            valid_episode = False

        return valid_episode, episode_state_traj, episode_maze_traj, value_hist, time_from_last_rwd

    def calculate_value(self, M, rwd):
        # pad because value function array includes values for all states, including home node and waterport_state
        return np.pad(M @ rwd, (0, 1))

    def simulate(self, agent_id, agent_params, max_length=200, n_bouts_to_generate=1, debug=False):
        """
        Simulate the agent.
        Example usage: success, stats = agentObj.simulate(0, [0.3, 3, 0.89, 0.3])
        :param agent_id (int):
        :param agent_params: Subject fits, dictionary with agentIds as keys and value is a list of parameters that are going to be used in the model [alpha, beta, gamma, lambda]
           I.e. {AgentId1: [alpha1, beta1, gamma1, lambda1], AgentId2: [alpha2, beta2, gamma2, lambda2]}
        :param max_length:
        :param n_bouts_to_generate: number of episodes to be generated. Default=1
        :return: (success, stats).
            stats is a dict with keys "agentId", "episodes", "LL", "MAX_LENGTH", "count_total", "V"
            success is currently always set to 1 TODO: why is that even here? Remove it? Or else improve this.
        """
        MAX_EPISODE_ATTEMPTS = 500   # max attempts at generating a bout episode
        success = 1

        # learning rate, softmax exploration - exploitation, discount factor, eligibility trace decay lambda
        alpha, beta, gamma, lamda = agent_params
        V = np.zeros(self.S)  # initialize all state values at zero
        time_from_last_rwd = 90 # starting at 90, so that the reward is immediately available when the agent first
        # starts the task
        if debug:
            info([">>> Simulating:", agent_id, agent_params, max_length, n_bouts_to_generate])
            print("alpha %.1f, beta %.1f, gamma %.2f, lambda %.1f, agentId %.0f" % (alpha, beta, gamma, lamda, agent_id))

        episodes_state_trajs = []
        episodes_pos_trajs = []
        episodes_value_hists = []
        count_valid, count_total = 0, 1
        while len(episodes_state_trajs) < n_bouts_to_generate:  # while haven't generated enough valid episodes
            if debug: print(len(episodes_state_trajs))
            V_backup = np.copy(V) # Back-up a copy of state-values to use in case the next episode has to be discarded
            time_from_last_rwd_backup = time_from_last_rwd
            # Begin generating episode
            episode_attempt = 0
            valid_episode = False
            while not valid_episode:
                episode_attempt += 1
                # if debug: print("  Start of episode generation attempt %d =" % episode_attempt)
                valid_episode, episode_state_traj, episode_maze_traj, value_hist, time_from_last_rwd = \
                    self.generate_episode(beta, gamma, max_length, V, time_from_last_rwd)
                # if debug: print("  End of episode generation attempt %d =" % episode_attempt)
                if valid_episode:
                    episodes_state_trajs.append(episode_state_traj)
                    episodes_pos_trajs.append(episode_maze_traj)
                    episodes_value_hists.append(value_hist)
                    V = np.copy(value_hist[-1])
                else:  # reestablish V with the value function of the latest valid episode
                    # this makes prevents invalid episodes from affecting the value function
                    V = np.copy(V_backup)
                    time_from_last_rwd = time_from_last_rwd_backup

                if episode_attempt > MAX_EPISODE_ATTEMPTS:
                    raise Exception("Did not manage to generate valid episodes. Please check if there is"
                                    " a bug in the code")
        stats = {
            "agentId": agent_id,
            "episodes_states": episodes_state_trajs,
            "episodes_positions": episodes_pos_trajs,
            "MAX_LENGTH": max_length,
            "count_total": count_total,
            "value_hists": episodes_value_hists,
            "V": V
        }
        return success, stats

    def simulate_multiple(self, agent_params, n=1, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        """
        This function calls `simulate` in multiple processes
        :param agent_params: dictionary with agent_ids as keys and value is a list of parameters to
        be used in the model [alpha, beta, gamma, lambda]. I.e. {agent_id: [alpha, beta, gamma, lambda]}
        Example usage: success, stats = agentObj.simulate_multiple(dict([(0, [0.3, 3, 0.89, 0.3])]))
        """
        print("agent_params", agent_params)
        tasks = []
        for agent_id in agent_params:
            tasks.append((agent_id, agent_params[agent_id], MAX_LENGTH, N_BOUTS_TO_GENERATE))
        with Pool(4) as p:  # running in parallel in 4 processes
            simulation_results = p.starmap(self.simulate, tasks)
        return simulation_results


if __name__ == '__main__':
    agnt = SR()
    np.random.seed(40)
    ALPHA, BETA, GAMMA, LAMBDA = [0.3, 12, 0.998, 0.8]
    success, stats = agnt.simulate(0, [ALPHA, BETA, GAMMA, LAMBDA], max_length=10000, n_bouts_to_generate=20, debug=True)
    plot_trajectory(stats["episodes_positions"], episode_idx=0)

    episode_i=5
    trial=0
    plot_maze_stats(stats["value_hists"][episode_i][trial], colorbar_label="state value V",
                    figtitle=f"Values at trial {trial} of episode {episode_i}")

    # plot_exploration_efficiency(stats["episodes_positions"])  #TODO: yields an error because agent does not explore
    # todo: the other end nodes. Fix so that it just plots the corresponding curve

    plot_maze_stats(calculate_visit_frequency(stats["episodes_positions"]),
                    colorbar_label="visit freq", figtitle='Node occupancies - all episodes')

    plot_maze_stats(calculate_visit_frequency(stats["episodes_positions"]),
                    colorbar_label="visit freq", figtitle='Node occupancies - all episodes', vmax=100)

    # plot occupancies with error bars # TODO: check how it was done in TDlambda_Shuangquan.ipynb
