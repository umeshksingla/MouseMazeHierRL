"""
TDLambda model
"""

from multiprocessing import Pool

from BaseModel import BaseModel
import os
import numpy as np

from parameters import WATERPORT_NODE, RWD_STATE, HOME_NODE, LVL_6_NODES
from plot_utils import plot_trajectory, plot_maze_stats


def info(title):
    print(*title)
    print('>>> module name:', __name__, 'parent process id:', os.getppid(),
          'process id:', os.getpid())

class TDLambda(BaseModel):
    """TD lambda model"""

    def __init__(self):
        super().__init__(self)
        self.terminal_nodes = {HOME_NODE, RWD_STATE}

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
            action_prob = []
            for action in np.arange(self.A):
                if np.isinf(betaV[action]):  # TODO: When would this happen?
                    action_prob.append(1)
                elif np.isnan(betaV[action]):  # TODO: When would this happen?
                    action_prob.append(0)
                else:
                    action_prob.append(betaV[action] / np.nansum(betaV))

            # Check for invalid probabilities
            for i in action_prob:
                if np.isnan(i):
                    print('Invalid action probabilities ', action_prob, betaV, state)

            if np.sum(action_prob) < 0.999:
                print('Invalid action probabilities, failed summing to 1: ',
                      action_prob, betaV, state)

        return action_prob


    def get_initial_state(self) -> int:
        return 0


    def generate_episode(self, alpha, beta, gamma, lamda, max_length, V):
        """
        Generate an episode using the given agent parameters
        :param alpha: learning rate parameter
        :param beta: inverse temperature parameter. It controls randomness of softmax
        :param gamma: discount factor
        :param lamda: eligibility trace decay parameter
        :param max_length:
        :param V (ndarray): state values
        :return: valid_episode, episode_traj, LL
        """
        et = np.zeros(self.S)  # eligibility trace vector for all states
        s = self.get_initial_state()
        episode_traj = [s]  # initialize episode trajectory list with the initial state
        value_hist = [V]
        LL = 0.0

        while s not in self.terminal_nodes and len(episode_traj) < max_length:

            if s != WATERPORT_NODE:
                action_prob = self.get_action_probabilities(s, beta, V)
                a = np.random.choice(range(self.A), p=action_prob)  # Choose action
                s_next = int(self.nodemap[s, a])           # Take action
                LL += np.log(action_prob[a])
                R = 0 # No reward
                # print("s, s_next, a, action_prob", s, s_next, a, action_prob)
            else:
                s_next = RWD_STATE
                R = 1 # Observe reward

            episode_traj.append(s_next)  # Record next state

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            et[s] += 1
            for node in np.arange(self.S - 1):  # RWD_STATE is never eligible (i.e. et=0), hence no need to include it
                V[node] += alpha * td_error * et[node]
                et[node] = gamma * lamda * et[node]

            value_hist.append(V)

            # TODO: When would these happen? Add explanation
            if np.isnan(V[s]):
                print('Warning invalid state-value: ', s, s_next, V[s], V[s_next], alpha, beta, gamma, R)
            elif np.isinf(V[s]):
                print('Warning infinite state-value: ', V)
            elif abs(V[s]) >= 1e5:
                print('Warning state value exceeded upper bound. Might approach infinity.')
                V[s] = np.sign(V[s]) * 1e5

            if s_next == RWD_STATE:
                print('Reward reached.')
            elif s_next==HOME_NODE:
                print('Home reached.')

            s = s_next

        if s in self.terminal_nodes:
            valid_episode = True
        else:
            print('Trajectory too long. Episode was aborted.')
            valid_episode = False

        return valid_episode, episode_traj, value_hist, LL


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
        if debug:
            info([">>> Simulating:", agent_id, agent_params, max_length, n_bouts_to_generate])

        MAX_EPISODE_ATTEMPTS = 500   # max attempts at generating a bout episode
        success = 1

        alpha = agent_params[0]    # learning rate
        beta = agent_params[1]     # softmax exploration - exploitation
        gamma = agent_params[2]    # discount factor
        lamda = agent_params[3]    # eligibility trace decay lambda

        if debug:
            print("alpha %.1f, beta %.1f, gamma %.2f, lambda %.1f, agentId %.0f" % (alpha, beta, gamma, lamda, agent_id))

        V = np.zeros(self.S)  # initialize all state values at zero

        episodes_trajs = []
        episodes_value_hists = []
        count_valid, count_total = 0, 1
        LL = 0.0
        while len(episodes_trajs) < n_bouts_to_generate:  # while haven't generated enough valid episodes
            if debug: print(len(episodes_trajs))
            # Back-up a copy of state-values to use in case the next episode has to be discarded
            V_backup = np.copy(V)

            # Begin generating episode
            episode_attempt = 0
            valid_episode = False
            while not valid_episode:
                episode_attempt += 1
                if debug: print("  Start of episode generation attempt %d =" % episode_attempt)
                valid_episode, episode_traj, value_hist, episode_ll = self.generate_episode(alpha, beta, gamma, lamda, max_length, V)
                if debug: print("  End of episode generation attempt %d =" % episode_attempt)
                if valid_episode:
                    episodes_trajs.append(episode_traj)
                    episodes_value_hists.append(value_hist)
                    V = np.copy(value_hist[-1])
                    LL += episode_ll
                else:  # reestablish V with the value function of the latest valid episode
                    # this makes invalid episodes have no effect on the value function
                    V = np.copy(V_backup)

                if episode_attempt > MAX_EPISODE_ATTEMPTS:
                    raise Exception("Did not manage to generate valid episodes. Please check if there is"
                                    " a bug in the code")
        stats = {
            "agentId": agent_id,
            "episodes": episodes_trajs,
            "LL": LL,
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
    agnt = TDLambda()
    success, stats = agnt.simulate(0, [0.3, 3, 0.89, 0.8], max_length=2000, n_bouts_to_generate=15, debug=True)
    plot_trajectory([stats["episodes"][0]], 'all')

    episode_i=8
    trial=0
    plot_maze_stats(stats["value_hists"][episode_i][trial], colorbar_label="state value V",
                    figtitle=f"Values at trial {trial} of episode {episode_i}")
