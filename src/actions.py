import utils
import parameters as p
from MM_Maze_Utils import NewMaze
from MM_Traj_Utils import StepType2


def construct_actions_dict(ma=NewMaze(6)):
    """
    Construct actions dictionary based on:
    0: going forward straight
    1: going left
    2: going right
    3: going back

    A state is a combination of current and prev node.
    """
    actions_dict = {}
    for n in p.LVL_5_NODES:
        c1, c2 = utils.get_children(n)
        actions_dict[f'{n}-{c1}'] = {3: n}
        actions_dict[f'{n}-{c2}'] = {3: n}

    for level in [4, 3, 2, 1, 0]:
        for n in p.NODE_LVL[level]:
            c1, c2 = utils.get_children(n)
            c11, c12 = utils.get_children(c1)
            actions_dict[f'{n}-{c1}'] = {
                StepType2(c1, c11, ma) + 1: c11,
                StepType2(c1, c12, ma) + 1: c12,
                3: n
            }
            c21, c22 = utils.get_children(c2)
            actions_dict[f'{n}-{c2}'] = {
                StepType2(c2, c21, ma) + 1: c21,
                StepType2(c2, c22, ma) + 1: c22,
                3: n
            }

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
                actions_dict[f'{n}-{parent}'] = {
                    0: utils.get_opp_children(parent, n),
                    a: lvl_up_node,
                    3: n
                }
                a = 3 - a

    actions_dict['1-0'] = {0: 2, 2: 127, 3: 1}
    actions_dict['2-0'] = {0: 1, 1: 127, 3: 2}
    actions_dict['127-0'] = {StepType2(0, 1, ma)+1: 1, StepType2(0, 2, ma)+1: 2, 3: 127}
    actions_dict['0-127'] = {3: 0}
    return actions_dict


def next_state_from_action_dict(action_dict, a, current_s):
    current_node = current_s.split('-')[1]
    next_node = action_dict[current_s][a]
    return f'{current_node}-{next_node}'


def state_from_nodes(prev_node, current_node):
    return f'{prev_node}-{current_node}'


actions_dict = construct_actions_dict()


class State:
    def __init__(self, prev_n, n):
        self.s = state_from_nodes(prev_n, n)
        self.action_node_map = actions_dict[self.s]
        self.node_action_map = {v: k for k, v in self.action_node_map.items()}
        self.curr_n = n
        self.prev_n = prev_n

    def __str__(self):
        return "s" + self.s