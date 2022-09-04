import utils
import parameters as p
from MM_Maze_Utils import NewMaze
from MM_Traj_Utils import StepType2
import numpy as np


def construct_actions_node_mapping(ma=NewMaze(6)):
    """
    Construct actions dictionary based on:
    0: going forward straight
    1: going left
    2: going right
    3: going back

    A state is a combination of current and prev node.
    """
    n_actions = 4
    actions_dict = [[[-1]*n_actions for j in p.ALL_VISITABLE_NODES] for i in p.ALL_VISITABLE_NODES] # 4 for four actions as above, -1 for invalid action

    for n in p.LVL_5_NODES:
        c1, c2 = utils.get_children(n)

        actions_dict[n][c1][3] = n
        actions_dict[n][c2][3] = n

        # actions_dict[f'{n}-{c1}'] = {3: n}
        # actions_dict[f'{n}-{c2}'] = {3: n}

    for level in [4, 3, 2, 1, 0]:
        for n in p.NODE_LVL[level]:
            c1, c2 = utils.get_children(n)
            c11, c12 = utils.get_children(c1)
            c21, c22 = utils.get_children(c2)

            actions_dict[n][c1][StepType2(c1, c11, ma) + 1] = c11
            actions_dict[n][c1][StepType2(c1, c12, ma) + 1] = c12
            actions_dict[n][c1][3] = n

            # actions_dict[f'{n}-{c1}'] = {
            #     StepType2(c1, c11, ma) + 1: c11,
            #     StepType2(c1, c12, ma) + 1: c12,
            #     3: n
            # }

            actions_dict[n][c2][StepType2(c2, c21, ma) + 1] = c21
            actions_dict[n][c2][StepType2(c2, c22, ma) + 1] = c22
            actions_dict[n][c2][3] = n

            # actions_dict[f'{n}-{c2}'] = {
            #     StepType2(c2, c21, ma) + 1: c21,
            #     StepType2(c2, c22, ma) + 1: c22,
            #     3: n
            # }

    for level in [4, 3, 2, 1, 0]:
        if level in [3, 1]:
            start_a = 1
        elif level in [4, 2, 0]:
            start_a = 2
        else:
            raise Exception('wrong level in action dict')
        for lvl_up_node in p.NODE_LVL[level]:
            n1, n2, n3, n4 = 2 * (2 * lvl_up_node + 1) + 1, 2 * (2 * lvl_up_node + 1) + 2, 2 * (2 * lvl_up_node + 2) + 1, 2 * (2 * lvl_up_node + 2) + 2
            a = start_a
            for n in [n1, n2, n4, n3]:
                parent = utils.get_parent_node(n)

                actions_dict[n][parent][0] = utils.get_opp_children(parent, n)
                actions_dict[n][parent][a] = lvl_up_node
                actions_dict[n][parent][3] = n

                # actions_dict[f'{n}-{parent}'] = {
                #     0: utils.get_opp_children(parent, n),
                #     a: lvl_up_node,
                #     3: n
                # }
                a = 3 - a

    actions_dict[1][0][0] = 2
    actions_dict[1][0][2] = p.HOME_NODE
    actions_dict[1][0][3] = 1

    actions_dict[2][0][0] = 1
    actions_dict[2][0][1] = p.HOME_NODE
    actions_dict[2][0][3] = 2

    actions_dict[p.HOME_NODE][0][StepType2(0, 1, ma)+1] = 1
    actions_dict[p.HOME_NODE][0][StepType2(0, 2, ma)] = 2
    actions_dict[p.HOME_NODE][0][3] = 127

    actions_dict[0][p.HOME_NODE][3] = 0

    # actions_dict['1-0'] = {0: 2, 2: 127, 3: 1}
    # actions_dict['2-0'] = {0: 1, 1: 127, 3: 2}
    # actions_dict['127-0'] = {StepType2(0, 1, ma)+1: 1, StepType2(0, 2, ma)+1: 2, 3: 127}
    # actions_dict['0-127'] = {3: 0}

    return np.array(actions_dict)


class State:
    def __init__(self, prev_n, n):
        # self.s = self.state_from_nodes(prev_n, n)
        self.action_node_arr = actions_node_matrix[prev_n][n]
        self.curr_n = n
        self.prev_n = prev_n

    @staticmethod
    def state_from_nodes(prev_node, current_node):
        return f'{prev_node}-{current_node}'

    def __str__(self):
        return f's{self.prev_n}->{self.curr_n}'


actions_node_matrix = construct_actions_node_mapping()


# print(actions_dict)
# print(actions_dict.shape)


# action_matrix = [[[-1, -1, -1, -1]]*len(p.ALL_VISITABLE_NODES) for i in p.ALL_VISITABLE_NODES]
# from BaseModel import BaseModel
#
# b = BaseModel()
# nodemap = b.get_SAnodemap()
#
# for prev_n in [27]:
#     print(actions_dict[prev_n])
#     for n in nodemap[prev_n]:
#         print(prev_n, n, actions_dict[prev_n][n])
        # action_matrix = State(prev_n, n).node_action_arr


# print(action_matrix)


