import matplotlib.pyplot as plt

from MM_Traj_Utils import LoadTrajFromPath
from MM_Maze_Utils import NewMaze, StepType2
import parameters as p
import numpy as np
import utils
from BaseModel import BaseModel
from actions import State
from scipy.special import softmax, logsumexp


b5tf = LoadTrajFromPath(p.OUTDATA_PATH + 'B5-tf')


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

def log_softmax(d):
    x = np.array(list(d.values()))
    den_x = logsumexp(x, keepdims=True)
    return {k: np.log(np.exp(v - den_x))[0] for k, v in d.items()}


class NodeMaze:
    def __init__(self):
        self.ma = NewMaze(6)
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
        for i, b in enumerate(b5tf.no[77:78]):
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
        choice_cell = self.ma.ru[choice_node][-1]
        return np.array([self.ma.di[(self.ma.ru[s][-1], choice_cell)] for s in tcell[::-1]])

    def get_feature2(self, choice_node, tcell):

        choice_cell = self.ma.ru[choice_node][-1]
        choice_cell_coors = np.array([self.ma.xc[choice_cell], self.ma.yc[choice_cell]])
        d = []
        for s in tcell[::-1]:
            s_cell = self.ma.ru[s][-1]
            s_cell_coors = np.array([self.ma.xc[s_cell], self.ma.yc[s_cell]])
            euc_d = np.linalg.norm(choice_cell_coors - s_cell_coors)
            # print(choice_cell_coors, s_cell_coors, euc_d)
            d.append(euc_d)
        return np.array(d)

    def get_distance_LL(self, episodes, coef):
        n_time_steps = len(coef)
        assert n_time_steps >= 2
        LL = 0.0
        for i, b in enumerate(episodes):
            traj = b + [p.HOME_NODE]
            for j in range(n_time_steps, len(traj)-1):
                tcell = traj[j-n_time_steps:j]
                prev_n = tcell[-2]
                n = tcell[-1]
                actual_next_n = traj[j]
                curr_s = State(prev_n, n)
                scores = {next_n: self.get_feature2(next_n, tcell).dot(coef) for next_n in curr_s.node_action_map}
                log_scores = log_softmax(scores)
                # print(tcell, ": ", scores, log_scores, actual_next_n, log_scores[actual_next_n])
                LL += log_scores[actual_next_n]
        return LL


class Regression:
    def __init__(self):
        self.nodemaze = NodeMaze()
        c1_ll_dict = {}
        step = 0.5
        for c1 in np.arange(0, 1, step):
            for c2 in np.arange(0, 1, step):
                for c3 in np.arange(0, 1, step):
                    for c4 in np.arange(0, 1, step):
                        coef = np.array([c1, c2, c3, c4])
                        print(coef, end="\t ")
                        ll = self.nodemaze.get_distance_LL(b5tf.no, coef)
                        print(ll)
                        c1_ll_dict = {c1: ll}
        plt.plot(c1_ll_dict.keys(), c1_ll_dict.values())
        plt.show()

