"""

"""
import numpy as np

from utils import calculate_visit_frequency
import evaluation_metrics as em
from Dyna_Qplus import DynaQPlus


class TDLambdaOptimisticInitialization(DynaQPlus):

    def __init__(self, file_suffix='_TDLambdaOptimisticInitializationTrajectories'):
        DynaQPlus.__init__(self, file_suffix=file_suffix)

    def generate_episode(self):
        raise Exception("Nope!")

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        print("Simulating agent with id", agentId)
        success = 1
        alpha = params["alpha"]         # learning rate
        gamma = params["gamma"]         # discount factor
        lamda = params["lamda"]         # eligibility trace-decay
        epsilon = params["epsilon"]     # epsilon
        k = 0                           # bonus factor
        n_plan = 0                      # number of planning steps
        self.back_action = True         # if there is a back action

        print("alpha, gamma, lamda, epsilon, k, n_plan, back_action, agentId",
              alpha, gamma, lamda, epsilon, k, n_plan, self.back_action, agentId)

        Q = np.ones((self.S, self.A)) * 1/(1-gamma)  # Initialize state values
        all_episodes = []
        LL = 0.0
        while len(all_episodes) < N_BOUTS_TO_GENERATE:
            _, episodes, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda, epsilon, k, n_plan,
                                                                        np.zeros(Q.shape), np.zeros(Q.shape), MAX_LENGTH,
                                                                        Q)  #TODO: missing parameter M
            all_episodes.extend(episodes)
            LL += episode_ll
        print("Q", Q)
        print(all_episodes)
        stats = {
            "agentId": agentId,
            "episodes": all_episodes,
            "LL": LL,
            "MAX_LENGTH": MAX_LENGTH,
            "count_total": len(all_episodes),
            "Q": Q,
            "V": self.get_maze_state_values_from_action_values(Q),
            "exploration_efficiency": em.exploration_efficiency(all_episodes, re=False),
            "visit_frequency": calculate_visit_frequency(all_episodes)
        }
        return success, stats


if __name__ == '__main__':
    pass
