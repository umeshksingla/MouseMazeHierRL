from MM_Traj_Utils import LoadTrajFromPath
import parameters as p
import numpy as np
import utils
from regression_utils.actions import State, actions_node_matrix
from regression_utils.maze_spatial_mapping import NODE_CELL_MAPPING, CELL_XY
from scipy.special import logsumexp, log_softmax


def levelup_path(n, go_up):
    path = [n]
    while go_up:
        if n % 2 == 0:
            n -= 1
        n = n // 2
        go_up -= 1
        path.append(n)
    return tuple(path[::-1])


def get_all_querypaths(func_path):
    valid_query_paths = {}
    for n in p.LVL_6_NODES:
        path = func_path(n, go_up=3)
        if path in valid_query_paths:
            raise Exception(f'duplicate path for node {n} from func {func_path}')
        valid_query_paths[path] = 1
    return valid_query_paths


level3_query_paths = get_all_querypaths(levelup_path)

def _count_query_paths_fixed_len(tf, query_paths):

    query_path_counts = dict.fromkeys(query_paths, 0)
    fixed_len = len(query_paths[0])
    assert sum([fixed_len - len(p) for p in query_paths]) == 0  # all query path lengths have to be same

    for i, bout in enumerate(tf.no):
        path = [node for node, _ in bout]
        for j in range(len(path)):
            curr_path = tuple(path[j:j + fixed_len])
            if len(curr_path) < fixed_len:
                break
            if curr_path in query_paths:
                query_path_counts[curr_path] += 1
    return query_path_counts


def softmax_dict(d):
    x = np.array(list(d.values()))
    den_x = logsumexp(x, keepdims=True)
    return {k: np.exp(v - den_x)[0] for k, v in d.items()}





class NodeMaze:
    def __init__(self):
        # self.basemodel = BaseModel()
        self.num_features = 3
        # self.S = p.ALL_VISITABLE_NODES
        # self.nodemap = self.basemodel.nodemap
        self.S_wo_endnodes = range(63)
        self.node_features = np.ones((len(self.S_wo_endnodes), self.num_features), dtype=int) * -1

        # outer feature
        self.outer_nodes_dict = dict()
        for n in self.S_wo_endnodes:
            pref_order = utils.get_outward_pref_order(n, 0.7, 0.1)
            self.node_features[n, 0] = max(pref_order, key=lambda key: pref_order[key])
            self.outer_nodes_dict[self.node_features[n, 0]] = True

        self.action_node_matrix = actions_node_matrix
            # print(n, self.node_features[n], pref_order)
        # print(self.outer_nodes_dict)

        # LoS feature
        # print(b5tf.ce[77:78])

    def get_feature(self, curr_s, last_lr_action, next_n):
        outer = next_n in self.outer_nodes_dict
        alternating = False if curr_s.node_action_map[next_n] not in [1, 2] else last_lr_action != curr_s.node_action_map[next_n]
        straight = curr_s.action_node_map.get(0, None) == next_n
        out_maze = p.LVL_BY_NODE[next_n] == p.LVL_BY_NODE[curr_s.curr_n]-1
        return [outer, alternating, straight, out_maze]

    def build_logical_spatial_features(self):
        for i, b in enumerate(tf.no[77:78]):
            traj = b[:, 0]
            last_lr_action = None
            for j in range(2, len(traj)-1):
                prev_prev_n = traj[j-2]
                prev_n = traj[j-1]
                n = traj[j]
                actual_next_n = traj[j+1]
                prev_s = State(prev_prev_n, prev_n)
                curr_s = State(prev_n, n)
                last_lr_action = prev_s.node_action_map[n] if prev_s.node_action_map[n] in [1, 2] else last_lr_action
                print(prev_prev_n, prev_n, n, actual_next_n, prev_s, curr_s, last_lr_action, prev_s.node_action_map, curr_s.node_action_map)
                for next_n in curr_s.node_action_map:
                    print("=>", next_n, self.get_feature(curr_s, last_lr_action, next_n))

    def get_feature1(self, choice_node, tcell):
        choice_cell = get_cell(choice_node)
        return np.array([ma.di[(get_cell(s), choice_cell)] for s in tcell[::-1]])

    def get_feature2(self, choice_node, s_cell_coors_arr):
        """
        TODO: euclidean distance is not really the end solution but simple to try
        """
        choice_cell_coors = CELL_XY[NODE_CELL_MAPPING[choice_node]]
        d = [np.linalg.norm(choice_cell_coors - s_cell_coors) for s_cell_coors in s_cell_coors_arr]
        return np.array(d)

    def get_distance_LL(self, episodes, c):
        n_time_steps = len(c)
        assert n_time_steps >= 2
        LL = 0.0
        for b in episodes:
            for step in range(n_time_steps, len(b)):
                ts = b[step-n_time_steps:step]
                actual_next_n = b[step]
                prev_n = ts[-2]
                n = ts[-1]
                V = np.zeros(n_time_steps)
                d = np.zeros(n_time_steps)
                actual_next_a = None
                next_possible_action_nodes = self.action_node_matrix[prev_n][n]
                # print(prev_n, n, actual_next_n, next_possible_action_nodes)
                for a in range(len(next_possible_action_nodes)):
                    next_n = next_possible_action_nodes[a]
                    # print(a, next_n)
                    if next_n == actual_next_n:
                        actual_next_a = a
                    if next_n == p.INVALID_STATE:
                        assert actual_next_a != a
                        V[a] = -1e10
                        continue
                    choice_cell_coors = CELL_XY[NODE_CELL_MAPPING[next_n]]
                    for i in range(step-n_time_steps+1, step):
                        d[step - i] = np.linalg.norm(choice_cell_coors - CELL_XY[NODE_CELL_MAPPING[b[i]]])
                    # print(d, c)
                    V[a] = d.dot(c)
                # print(ts, c, self.action_node_matrix[prev_n][n], V)
                assert actual_next_a is not None
                scores = log_softmax(V)     # softmax_dict(scores)
                LL += scores[actual_next_a]
        return LL


class Regression:
    def __init__(self):
        tf = LoadTrajFromPath(p.OUTDATA_PATH + 'B5-tf')
        eps = utils.convert_traj_to_episodes(tf)
        self.nodemaze = NodeMaze()
        # step = 0.8
        # for c1 in np.arange(0, 1, step):
        #     for c2 in np.arange(0, 1, step):
        #         for c3 in np.arange(0, 1, step):
        #             for c4 in np.arange(0, 1, step):
        #                 coef = np.array([c1, c2, c3, c4])
        #                 # print(coef, end="\t ")
        #                 ll = self.nodemaze.get_distance_LL(eps[77:78], coef)
        #                 print(ll)
        coef = np.array([0.05, 0.22, 0.26, -0.1])
        ll = self.nodemaze.get_distance_LL(eps[1:200], coef)
        print(ll)


# Regression()
