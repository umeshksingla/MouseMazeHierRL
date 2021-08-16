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
        bonus_in_planning = False       # check dynaQ+ for this

        print("alpha, gamma, lamda, epsilon, k, n_plan, back_action, agentId",
              alpha, gamma, lamda, epsilon, k, n_plan, self.back_action, agentId)

        Q = np.ones((self.S, self.A)) * 1/(1-gamma)  # Initialize state values
        all_episodes_state_trajs = []
        all_episodes_pos_trajs = []
        LL = 0.0
        while len(all_episodes_state_trajs) < N_BOUTS_TO_GENERATE:
            _, episode_state_trajs, episode_maze_trajs, episode_ll = self.generate_exploration_episode(alpha, gamma, lamda, epsilon, k, n_plan, bonus_in_planning, np.zeros(Q.shape), np.zeros(Q.shape), MAX_LENGTH, Q)
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
            "visit_frequency": calculate_visit_frequency(all_episodes_state_trajs)
        }
        return success, stats


if __name__ == '__main__':
    pass
