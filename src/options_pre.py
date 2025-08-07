import networkx as nx
import matplotlib.pyplot as plt
from utils import get_children, get_parent_node
from evaluation_metrics import nodes2cell
import parameters as p
import pickle
import json


def generate_options(s, d):
    dur = d
    curr = [[(None, [s])]]
    while dur:
        # print(curr)
        curr_level_poss = []
        for each in curr[-1]:
            prev_p, nodes = each
            for e in nodes:
                possibilities = list(filter(lambda x: x is not None, [*get_children(e), get_parent_node(e)]))
                if prev_p is not None:
                    possibilities.remove(prev_p)
                # if None in possibilities:
                #     possibilities.remove(None)
                curr_level_poss.append((e, possibilities))
        curr.append(curr_level_poss)
        dur -= 1
    # print(curr)
    # print([(prev_p, nodes) for each in curr for prev_p, nodes in each])
    curr = dict([(prev_p, nodes) for each in curr for prev_p, nodes in each])
    # print(curr)
    del curr[None]

    G = nx.DiGraph()
    G.add_nodes_from(curr.keys())
    for k, v in curr.items():
        G.add_edges_from(([(k, t) for t in v]))

    # print(G.edges())
    # nx.draw(G, with_labels=True)
    # plt.show()

    options = []
    for node in G:
        if G.out_degree(node) == 0:  # it's a leaf
            options.append(tuple(nx.shortest_path(G, s, node)))
    # print(options)
    return options


def filter_straight(options):
    straight_options = []
    _, xy_options = nodes2cell(options, zigzag=False)

    for each_xy, each_o in zip(xy_options, options):
        # print(each_o, each_xy.__str__())
        diff_dir_x = None
        diff_dir_y = None
        straight_o = True
        for c1, c2 in zip(each_xy, each_xy[1:]):
            diff = c2-c1
            if diff[0] == 0:    # x direction
                pass
            else:
                if diff_dir_x is None:
                    diff_dir_x = diff[0]
                else:
                    if diff[0] != diff_dir_x:   # going in opp direction to initial
                        straight_o = False
                        break
            if diff[1] == 0:    # y direction
                pass
            else:
                if diff_dir_y is None:
                    diff_dir_y = diff[1]
                else:
                    if diff[1] != diff_dir_y:  # going in opp direction to initial
                        straight_o = False
                        break
        if straight_o:
            straight_options.append(each_o)
    # print(options)
    # print(straight_options)
    return straight_options


def construct_options():
    options_dict = dict()
    max_option_length = 10
    for n in p.ALL_MAZE_NODES:
        print(n)
        options_dict[n] = dict.fromkeys(range(1, max_option_length), None)
        for l in range(1, max_option_length):
            all_options = generate_options(n, l)
            # options_dict[n][l] = all_options
            options_dict[n][l] = filter_straight(all_options)

        with open('straight_options_dict_tuple.pkl', 'wb') as f:
            pickle.dump(options_dict, f)

    print(options_dict)
    return


# construct_options()


with open('all_options_dict_tuple.pkl', 'rb') as f:
    all_options_dict = pickle.load(f)
with open('straight_options_dict_tuple.pkl', 'rb') as f:
    straight_options_dict = pickle.load(f)
with open('all_options_dict.json', 'r') as f:
    home_options_dict = {}
    for l, o_list in json.load(f)["127"].items():
        home_options_dict[int(l)] = [tuple(o) for o in o_list]


straight_options_dict[127] = home_options_dict
all_options_dict[127] = home_options_dict

# print("all options", all_options_dict[100][4])
# print("all options", straight_options_dict[100][4])
# print("all options", all_options_dict[4][3])
# print("straight options", straight_options_dict[4][3])
# print("straight options", straight_options_dict[127][3])
