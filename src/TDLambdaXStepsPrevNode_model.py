"""
TDLambdaXStepsPrevNode model:
Take only the last X steps before a reward as training data and each state is
a combination of current node and prev node (in an attempt to include direction)
"""

from parameters import *
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived
from MM_Traj_Utils import *


def info(title):
    print(*title)
    print('>>> module name:', __name__, 'parent process id:', os.getppid(),
          'process id:', os.getpid())


class TDLambdaXStepsPrevNodeRewardReceived(TDLambdaXStepsRewardReceived):
    """TD lambda model to fit trajectories starting X (parameter) steps before the reward was received
    This agent incorporates the current and the previous node in its state representation 
    """  # TODO: check if this description is accurate

    def __init__(self, X = 20, file_suffix='_XStepsRewardReceivedTrajectories'):
        TDLambdaXStepsRewardReceived.__init__(self, X, file_suffix)
        self.nodes = 129
        self.S = (self.nodes+1) * (self.nodes+1) # +1 for invalid state

        self.h, self.inv_h = dict(), dict()
        self.construct_node_tuples_to_number_map()

        self.home_terminal_state = self.get_number_from_node_tuple((HOME_NODE, INVALID_STATE))
        self.reward_terminal_state = self.get_number_from_node_tuple((RWD_NODE, WATER_PORT_STATE))
        self.RewardTupleState = self.get_number_from_node_tuple((57, RWD_NODE))
        self.terminal_nodes = {self.home_terminal_state, self.reward_terminal_state}

    def add_LoS(self, s, running_episode_traj, action_prob):
        """
        If at level 5, check if it came from opposite level 5 or level 3
        If it came from level 3, then the nodes in direct line-of-sight is 80%
        more probable. Otherwise, no change.
        """
        if len(running_episode_traj) < 3:
            return action_prob

        node_tuple = self.get_node_tuple_from_number(s)
        current_node = node_tuple[1]
        # print("LoS", s, node_tuple, current_node)

        updated_action_prob = action_prob.copy()
        if current_node in NODE_LVL[5]:
            node_2_step_before = self.get_node_tuple_from_number(running_episode_traj[-3])[1]
            # print("node_2_step_before", node_2_step_before, running_episode_traj[-5:])
            if node_2_step_before in NODE_LVL[3]:   # apply LoS
                # (lv*2+2)*2+1, (lv*2+2)*2+2 for right
                # (lv*2+1)*2+1, (lv*2+1)*2+2 for left
                # get direction
                if (node_2_step_before*2+2)*2+1 == current_node or (node_2_step_before*2+2)*2+2 == current_node:
                    los_action = 2
                    non_los_action = 1
                elif (node_2_step_before*2+1)*2+1 == current_node or (node_2_step_before*2+1)*2+2 == current_node:
                    los_action = 1
                    non_los_action = 2
                else:
                    raise Exception("Invalid node.")

                updated_action_prob[0] = 0.0
                updated_action_prob[los_action] = 0.8
                updated_action_prob[non_los_action] = 0.2
                print("LoS Applied", node_2_step_before, node_tuple, action_prob, updated_action_prob)
                assert np.isclose(sum(action_prob), sum(updated_action_prob))

        return updated_action_prob

    def get_action_probabilities(self, state, beta, V, episode_traj=None, *args, **kwargs):
        # Use softmax policy to select action, a at current state, s

        if episode_traj is None:
            episode_traj = []

        curr = self.get_node_tuple_from_number(state)[1]
        if curr in LVL_6_NODES:
            action_prob = [1, 0, 0]
        else:
            betaV = [np.exp(beta * V[self.get_number_from_node_tuple((curr, val))])
                     for val in self.nodemap[curr, :]]
            action_prob = []
            for action in np.arange(self.A):
                if np.isinf(betaV[action]):  # TODO: ?
                    action_prob.append(1)
                elif np.isnan(betaV[action]):
                    action_prob.append(0)
                else:
                    action_prob.append(betaV[action] / np.nansum(betaV))

            updated_action_prob = self.add_LoS(state, episode_traj, action_prob)
            action_prob = updated_action_prob

            # Check for invalid probabilities
            for i in action_prob:
                if np.isnan(i):
                    raise Exception('Invalid action probabilities ', action_prob, betaV, state)

            if np.sum(action_prob) < 0.999:
                raise Exception('Invalid action probabilities, failed summing to 1: ',
                      action_prob, betaV, state)

        return action_prob

    def get_initial_state(self) -> int:
        from_ = np.random.randint(0, HomeNode)
        to_ = np.random.choice([c for c in self.nodemap[from_] if c not in [INVALID_STATE, RWD_NODE, WATER_PORT_STATE]])
        print("from_, to_", from_, to_)
        return self.get_number_from_node_tuple((from_, to_))
        # a=list(range(self.S))
        # a.remove(28)
        # a.remove(57)
        # a.remove(115)
        # a.remove(RewardNode)
        # return np.random.choice(a)    # Random initial state

    def construct_node_tuples_to_number_map(self):
        c = 0
        for i in range(-1, self.nodes):
            for j in range(-1, self.nodes):
                self.h[c] = (i, j)
                self.inv_h[(i, j)] = c
                c += 1
        # print(self.h, self.inv_h, c)
        return

    def get_node_tuple_from_number(self, c: int) -> tuple:
        return self.h[c]

    def get_number_from_node_tuple(self, t: tuple) -> int:
        return self.inv_h[t]

    def convert_nodes_to_state_traj(self, t):
        return [self.get_number_from_node_tuple((i, j)) for i, j in zip(t, t[1:])]

    def convert_state_traj_to_node(self, l):
        # print(l)
        return [self.get_node_tuple_from_number(n)[0] for n in l] +\
               [self.get_node_tuple_from_number(l[-1])[1]]

    def take_action(self, s: int, a: int) -> int:
        """
        Function that returns the next state when given the current state and the action taken.
        This function also takes care of the (prev, curr) state representation
        :param s: state where the agent is at
        :param a: action taken
        :return s_: state at next time step
        """
        prev, curr = self.get_node_tuple_from_number(s)
        s_ = self.get_number_from_node_tuple((curr, self.nodemap[curr, a]))
        return s_

    def choose_action(self, s, beta, V, episode_traj):
        action_prob = self.get_action_probabilities(s, beta, V, episode_traj)
        action = np.random.choice(range(self.A), 1, p=action_prob)[0]
        return action, action_prob[action]

    def generate_episode(self, alpha, beta, gamma, lamda, MAX_LENGTH, V):
        s = self.get_initial_state()
        LL = 0.0
        first_reward = -1
        episode_traj = []
        e = np.zeros(self.S)
        while True:
            episode_traj.append(s)  # Record current state

            if s in self.terminal_nodes:
                print("entering again", s, self.get_node_tuple_from_number(s))
                s = self.get_initial_state()
                e = np.zeros(self.S)

            if s != self.RewardTupleState:
                a, a_prob = self.choose_action(s, beta, V, episode_traj)  # Choose action
                s_next = self.take_action(s, a)  # Take action
                LL += np.log(a_prob)    # Update log likelihood
                # print("s, s_next, a, action_prob", self.get_node_tuple_from_number(s), self.get_node_tuple_from_number(s_next), a, action_prob)
            else:
                s_next = self.reward_terminal_state

            R = 1 if s == self.RewardTupleState else 0   # Observe reward

            assert isinstance(s, int)
            assert isinstance(s_next, int)

            # Update state-values
            td_error = R + gamma * V[s_next] - V[s]
            e[s] += 1
            for n in np.arange(self.S+1):
                V[n] += alpha * td_error * e[n]
                e[n] = gamma * lamda * e[n]

            # print("V reward", V[self.RewardTupleState])

            V[s] = self.is_valid_state_value(V[s])

            if s == self.RewardTupleState:
                print('Reward Reached!')
                if first_reward == -1:
                    first_reward = len(episode_traj)
                    print("First reward:", len(episode_traj))

            if len(episode_traj) > MAX_LENGTH + first_reward:
                print('Trajectory too long. Aborting episode.')
                break

            s = s_next

            # new_sum = np.nansum(V)
            # diff = abs(prev_sum-new_sum)
            if len(episode_traj)%100 == 0:
                print("current state", self.get_node_tuple_from_number(s)[1], "step", len(episode_traj))
            #     print("current diff", diff)
            #
            # if diff <= 0.0000001:
            #     print("State values have converged.", "current diff", diff, "step", len(episode_traj))
            #     break
            # prev_sum = new_sum

        # print(episode_traj)
        maze_episode_traj = self.convert_state_traj_to_node(episode_traj)

        # split at HomeNode or RewardNode
        # print(maze_episode_traj)
        episodes = []
        epi = []
        for i in maze_episode_traj:
            if i == HomeNode:
                epi.append(i)
                if len(epi) > 2:
                    episodes.append(epi)
                epi = []
            elif i == RewardNode:
                epi.append(i)
                if len(epi) > 2:
                    episodes.append(epi)
                epi = []
            else:
                epi.append(i)
        if epi:
            episodes.append(epi)

        # if len(maze_episode_traj) >= 5:
        #     return True, maze_episode_traj
        # else:
        #     return False, maze_episode_traj
        return True, episodes, LL

    def simulate(self, agentId, params, MAX_LENGTH=25, N_BOUTS_TO_GENERATE=1):
        """
        Check base class for doc string.
        """
        info([">>> Simulating:", agentId, params, MAX_LENGTH, N_BOUTS_TO_GENERATE])

        success = 1

        alpha = params["alpha"]     # learning rate
        beta = params["beta"]       # softmax exploration - exploitation
        gamma = params["gamma"]     # discount factor
        lamda = params["lamda"]     # eligibility trace

        print("alpha, beta, gamma, lamda, agentId",
              alpha, beta, gamma, lamda, agentId)

        V = np.zeros(self.S+1)
        V[self.home_terminal_state] = 0     # setting action-values of maze entry to 0
        V[self.reward_terminal_state] = 0

        all_episodes = []
        LL = 0.0
        while len(all_episodes) < N_BOUTS_TO_GENERATE:
            # Begin generating a bout
            _, episodes, episode_ll = self.generate_episode(alpha, beta, gamma, lamda, MAX_LENGTH, V)
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
        print("V=", V)
        print("len(V)=", len(V))
        return success, stats

    def get_maze_state_values(self, V):
        """
        Get state values to plot against the nodes on the maze
        """
        state_values = np.zeros(128)
        for n in range(128):
            possible_states = self.nodemap[n, :]
            print(n, possible_states)
            possible_states = list(filter(
                lambda
                    p: p != INVALID_STATE and p != WATER_PORT_STATE and p != HOME_NODE,
                possible_states))
            pos_state_values = [V[self.get_number_from_node_tuple((p, n))] for
                                p in possible_states]
            print(n, possible_states, pos_state_values)
            state_values[n] = np.nanmean(pos_state_values)
        return state_values


def test1():
    a = TDLambdaXStepsPrevNodeRewardReceived()
    t = [10, 21, 44, 90, 44, 89, 44, 90, 44, 21, 43]
    l = [a.inv_h[(i, j)] for i, j in zip(t, t[1:])]
    new = [a.h[n][0] for n in l] + [a.h[l[-1]][1]]
    assert new == t


if __name__ == '__main__':
    test1()
