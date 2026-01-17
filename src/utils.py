from MM_Traj_Utils import add_node_times_to_tf, add_reward_times_to_tf, NewMaze, Traj, LoadTrajFromPath
from parameters import FRAME_RATE, WATERPORT_NODE, HOME_NODE, RWD_STATE, ALL_MAZE_NODES, ALL_VISITABLE_NODES, \
    TIME_EACH_MOVE, INVALID_STATE, NODE_LVL, LVL_BY_NODE, HALF_UP, HALF_LEFT, ROW_3, ROW_1, COL_3, COL_1
import parameters as p
import numpy as np
from numpy import array, arange

from collections import defaultdict


def get_node_visit_times(tf, node_id):
    """
    Get lists of node visit times
    :param tf: trajectory object
    :return: times_to_node_visits
    """
    frs_to_node_visits = []
    n_bouts = len(tf.no)
    for bout in range(n_bouts):
        node_visits_frs = tf.no[bout][tf.no[bout][:, 0] == node_id]
        if len(node_visits_frs) > 0:
            bout_init_fr = tf.fr[bout, 0]  # first frame of the bout
            frs_to_node_visits.append(bout_init_fr + node_visits_frs[:, 1])

    if len(frs_to_node_visits)>0:
        frs_to_node_visits = np.concatenate(frs_to_node_visits)

        times_to_node_visits = np.array([frame_to_node_visit / FRAME_RATE
                                              for frame_to_node_visit in frs_to_node_visits])
    else:
        times_to_node_visits=np.array([])
    return times_to_node_visits


def get_SAnodemap():
    ### THIS IS OLD AND PROBABLY BROKEN
    """
    Creates a mapping based on the maze layout where current states are linked to the next 3 future states

    Returns: SAnodemap, a 2D array of current state to future state mappings
    Return type: array[(S, A), int]
    """
    SAnodemap = np.ones((S, A), dtype=int) * INVALID_STATE
    for node in np.arange(S - 1):
        # Shallow level node available from current node
        if node % 2 == 0:
            SAnodemap[node, 0] = (node - 2) / 2
        elif node % 2 == 1:
            SAnodemap[node, 0] = (node - 1) / 2
        if SAnodemap[node, 0] == INVALID_STATE:
            SAnodemap[node, 0] = HOME_NODE

        if node not in NODE_LVL[6]:
            # Deeper level nodes available from current node
            SAnodemap[node, 1] = node * 2 + 1
            SAnodemap[node, 2] = node * 2 + 2

    SAnodemap[HOME_NODE, 0] = INVALID_STATE
    SAnodemap[HOME_NODE, 1] = 0
    SAnodemap[HOME_NODE, 2] = INVALID_STATE
    SAnodemap[RWD_STATE, :] = INVALID_STATE
    SAnodemap[WATERPORT_NODE, 1] = RWD_STATE

    return SAnodemap


def nodes2cell(state_hist_all, zigzag=True):
    '''
    param state_hist_all:  [[nodes], [], ...]
    param zigzag: if the resulting cell traj needs to be a bit noisy or straight
    '''
    state_hist_cell = []
    state_hist_xy = []
    ma=NewMaze(6)
    for epID, epi in enumerate(state_hist_all):
        cells = []
        if not len(epi):
            continue
        cells.extend([ma.ru[epi[0]][-1]])
        for i in range(1, len(epi)):
            node = epi[i]
            if node == HOME_NODE:
                assert i == len(epi)-1
                break
            if node == RWD_STATE: continue
            # if (i == 0 and node == 0) or node > epi[i-1]:
            if node > epi[i-1]:
                # if going to a deeper node
                cells.extend(ma.ru[node])
            elif node < epi[i-1]:
                # if going to a shallower node
                reverse_path = list(reversed(ma.ru[epi[i-1]]))
                reverse_path = reverse_path + [ma.ru[node][-1]]
                cells.extend(reverse_path[1:])
        if epi[-1] == HOME_NODE:
            home_path = list(reversed(ma.ru[0]))
            cells.extend(home_path[1:])  # cells from node 0 to maze exit

        cells = [0, 1, 2, 3, 4, 5, 6] + cells
        state_hist_cell.append(cells)

        a = np.zeros((len(cells),2))
        if zigzag:
            a[:,0] = ma.xc[cells] + np.random.choice([-1,1],len(ma.xc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.xc[cells]))/2
            a[:,1] = ma.yc[cells] + np.random.choice([-1,1],len(ma.yc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.yc[cells]))/2
        else:
            a[:,0] = ma.xc[cells]
            a[:,1] = ma.yc[cells]
        state_hist_xy.append(a)

    return state_hist_cell, state_hist_xy


def convert_episodes_to_traj_class(episodes_pos_trajs, episodes_state_trajs=None, time_each_move=TIME_EACH_MOVE):
    """
    Convert list of lists to Traj class with tf.no containing episode information.
    At the moment, simply using the index as time. This is so that simulated episodes
    can use some of the functions provided original authors that operate on Traj
    class instances.

    :param episodes_pos_trajs:  format [[], [], ..]; contains only positions in the maze that are visitable,
    i.e. from ALL_VISITABLE_NODES
    :param episodes_state_trajs: format [[], [], ..]; contains any state the agent can go to, including the RWD_STATE
    :param time_each_move:
    :return: tf: Traj
    """
    tf = Traj(fr=[],ce=None,ke=None,no=[],re=[])
    start = 0
    end = 0
    frames_per_move=int(FRAME_RATE*time_each_move)
    # frames_per_move =1
    for bout, episode_traj in enumerate(episodes_pos_trajs):
        tf.no.append(np.stack([np.array(episode_traj), arange(len(episode_traj))*frames_per_move], axis=1).astype(int))

        end = start + len(episode_traj)*frames_per_move
        tf.fr.append([start, end])
        start = end+1*frames_per_move  # currently assuming the agent always stays the same amount of time at home

        if episodes_state_trajs is not None:  # use state trajectory to find when rwd was delivered
            rwd_idxs = np.where(array(episodes_state_trajs[bout]) == RWD_STATE)[0]  # received reward
            rwd_times = []
            for i, rwd_idx in enumerate(rwd_idxs):
                wp_visit_idx = rwd_idx - 1 - 2 * i
                rwd_times.append(frames_per_move*(wp_visit_idx + .1)) # adding a small 0.1 constant to make the
                # reward delivery happen slightly after the waterport visit; the -2*i is to account for the extra
                # entries (128 and 116) that are in the list when a reward is received
            rwd_times = array(rwd_times)
        else:  # assume every waterport visit leads to reward delivery
            rwd_times = tf.no[bout][tf.no[bout][:, 0] == WATERPORT_NODE, 1] + frames_per_move * .1
        tmp_re = np.zeros((len(rwd_times), 2))
        tmp_re[:, 0] = rwd_times  # tmp[:, 1] is left as zeros
        tf.re.append(tmp_re)

    tf.fr = np.array(tf.fr)

    return tf


def convert_traj_to_episodes(tf):
    """
    Convert a Traj to list of lists of visited nodes.

    tf: Traj
    Returns episodes: [[], [], ..]
    """
    episodes = []
    for e in tf.no:
        episodes.append(e[:, 0].tolist())
    return episodes


def break_simulated_traj_into_episodes(maze_episode_traj):
    """
    Break the simulated trajectory list at home node
    and return the list of episodes
    param maze_episode_traj: list of nodes
    returns:
    episodes: list of lists (of nodes)
    """
    episodes = []
    epi = []
    for i in maze_episode_traj:
        if i == HOME_NODE:
            epi.append(i)
            episodes.append(epi)
            epi = []
        else:
            epi.append(i)
    if epi:     # last episode
        episodes.append(epi)

    # print("Home visit counts in simulated episodes:", len(episodes))
    return episodes


def test_traj(traj):
    for i, j in zip(traj, traj[1:]):
        assert i != j
        if i == 52 and j == 41:
            return
        assert ((i in get_children(j)) or (i == get_parent_node(j)) or (i in [HOME_NODE, RWD_STATE]) or (j in [HOME_NODE, RWD_STATE])), f"i={i} j={j}"
    return


# def test_traj(traj):
#     for i, j in zip(traj, traj[1:]):
#         assert i != j, f"i={i} j={j}"
#         assert ((i in get_children(j)) or (i == get_parent_node(j))), f"i={i} j={j}"
#         assert (i in [HOME_NODE, RWD_STATE]), f"i={i} j={j}"
#         assert (j in [HOME_NODE, RWD_STATE]), f"i={i} j={j}"
#     return


def test_episodes(episode_state_traj):
    for i, t in enumerate(episode_state_traj):
        try:
            test_traj(t)
        except:
            raise Exception(f"Corrupt trajectory {i} found: {t}")


def wrap(episode_state_traj):
    episode_state_trajs = break_simulated_traj_into_episodes(episode_state_traj)
    test_episodes(episode_state_trajs)
    episode_state_trajs = list(filter(lambda e: len(e) >= 3, episode_state_trajs))  # remove empty or short episodes
    episode_maze_trajs = episode_state_trajs  # in pure exploration, both are same
    return episode_state_trajs, episode_maze_trajs


def calculate_visit_frequency(episodes):
    """
    :param episodes: episodes: [[], [], ...]
    :return: 127-length list of number of times each node was visited in all the
    input episodes
    """
    node_visit_freq = np.zeros(len(ALL_VISITABLE_NODES))
    for episode in episodes:
        for node in episode:
            node_visit_freq[node] += 1
    return node_visit_freq


def calculate_normalized_visit_frequency(episodes):
    """
    :param episodes: episodes: [[], [], ...]
    :return: 127-length list of number of times each node was visited in all the
    input episodes
    """
    node_visit_freq = np.zeros(len(ALL_VISITABLE_NODES))
    for episode in episodes:
        for node in episode:
            node_visit_freq[node] += 1
    node_visit_freq = node_visit_freq / np.sum([len(e) for e in episodes])
    return node_visit_freq


def calculate_normalized_visit_frequency_by_level(episodes):
    """
    :param episodes: episodes: [[], [], ...]
    :return: 7-length list of number of times each level was visited in all the
    input episodes

    Idea: something along the line of how much time animals spend in certain regions of maze.

    """
    node_level_visit_freq = np.zeros(len(NODE_LVL))
    for episode in episodes:
        for node in episode:
            if node not in ALL_MAZE_NODES: continue
            node_level_visit_freq[LVL_BY_NODE[node]] += 1
    node_level_visit_freq = node_level_visit_freq / np.sum([len(e) for e in episodes])
    # for i in range(len(node_level_visit_freq)):
    #     node_level_visit_freq[i] = node_level_visit_freq[i] / (2**i)
    return node_level_visit_freq


def get_parent_node(n):
    if n == p.HOME_NODE:
        return None
    if n in p.LVL_0_NODES:
        return p.HOME_NODE
    return (n + (n % 2) - 1) // 2


def get_parent_node_x_level_up(n, x):
    while x:
        n = get_parent_node(n)
        x -= 1
    return n


def get_children(n):
    if n == p.HOME_NODE:
        return (0, )
    if n in p.LVL_6_NODES:
        return tuple()
    return 2*n+1, 2*n+2


def get_the_other_children(parent, current_child):
    assert current_child != 0
    assert current_child != p.HOME_NODE
    c1, c2 = get_children(parent)
    return c1 if current_child == c2 else c2


def get_opp_child(current_child):
    assert current_child != 0
    assert current_child != p.HOME_NODE
    c1, c2 = get_children(get_parent_node(current_child))
    return c1 if current_child == c2 else c2


def home_path_node(n):
    """
    Returns the node path that leads from node n to the start of the maze
    Includes both start and end nodes
    """
    ret = []
    while n >= 0:
        ret.append(n)
        if n == 0:
            return ret
        n = get_parent_node(n)
    return ret


def connect_path_node(n1, n2):
    """
    Returns the shortest path that connects nodes n1 and n2
    Includes both start and end nodes
    """
    r1 = home_path_node(n1)
    r2 = home_path_node(n2)[::-1]   # reversed
    for i in r1:
        if i in r2:
            return r1[:r1.index(i)]+r2[r2.index(i):]


def get_all_end_nodes_from_level4_node(level4_node):
    """Returns all 4 end nodes in that particular subquadrant"""
    assert level4_node in p.LVL_4_NODES
    p1, p2 = level4_node * 2 + 1, level4_node * 2 + 2
    n1, n2, n3, n4 = p1 * 2 + 1, p1 * 2 + 2, p2 * 2 + 1, p2 * 2 + 2
    return n1, n2, n3, n4


def get_all_subq_from_current_subq(subq):
    """Returns all 4 subquadrants in that particular quadrant"""
    for sq in p.subquadrant_sets:
        if subq in p.subquadrant_sets[sq]:
            return p.subquadrant_sets[sq]


def get_outward_pref_order(turn_node, pref_prob, back_prob):
    le = LVL_BY_NODE[turn_node]
    l_child, h_child = 2 * turn_node + 1, 2 * turn_node + 2
    back_child = get_parent_node(turn_node)
    if le == 0 or le == 1:
        pref_order = [l_child, h_child]
    elif le == 2:
        pref_order = [l_child, h_child] if turn_node in HALF_UP else [h_child, l_child]
    elif le == 3:
        pref_order = [l_child, h_child] if turn_node in HALF_LEFT else [h_child, l_child]
    elif le == 4:
        pref_order = [l_child, h_child] if ((turn_node in ROW_1) or (turn_node in ROW_3)) else [h_child, l_child]
    elif le == 5:
        pref_order = [l_child, h_child] if ((turn_node in COL_1) or (turn_node in COL_3)) else [h_child, l_child]
    else:
        raise Exception(f"Node of an illegal level specified: {le}")
    pref_order = {
        pref_order[0]: pref_prob * (1-back_prob),
        pref_order[1]: (1 - pref_prob)*(1-back_prob),
        back_child: back_prob,
    }
    return pref_order


def get_part_trajs_from_tf(tf, phase):
    if phase == 'all':
        trajs = tf.no
    elif phase == 'first_half':
        trajs = tf.no[:len(tf.no) // 2]
    elif phase == 'second_half':
        trajs = tf.no[len(tf.no) // 2:]
    elif phase == 'first_third':
        trajs = tf.no[:len(tf.no) // 3]
    elif phase == 'second_third':
        trajs = tf.no[len(tf.no) // 3:2 * len(tf.no) // 3]
    elif phase == 'third_third':
        trajs = tf.no[2 * len(tf.no) // 3:]
    elif isinstance(phase, int):
        trajs = tf.no[:phase]
    else:
        raise Exception(f'wrong phase specified: {phase}')
    return trajs


def get_revisits(tf, le, phase='all'):
    """
    Revisits to a node in terms of number of nodes visited
    """
    revisits = defaultdict(list)
    trajs = get_part_trajs_from_tf(tf, phase)
    for t in trajs:
        node_seq = t[:, 0]
        for node in p.NODE_LVL[le]:
            if node in p.SUBQUADRANT_dict[28]:
                continue
            visits_node = np.where(node_seq == node)[0]
            if len(visits_node) >= 2:
                revisits[node] += list(np.diff(visits_node) - 1)
    return revisits


def get_end_nodes_revisits(tf, le, phase='all'):
    """
    Revisits to a node in terms of number of end-nodes visited
    """
    revisits = defaultdict(list)
    trajs = get_part_trajs_from_tf(tf, phase)
    for t in trajs:
        orig_node_seq = t[:, 0]
        for node in p.NODE_LVL[le]:
            node_seq = np.array(list(filter(lambda x: x == node or x in p.LVL_6_NODES, orig_node_seq)))
            visits_node = np.where(node_seq == node)[0]
            if len(visits_node) >= 2:
                revisits[node] += list(np.diff(visits_node) - 1)
    return revisits


def get_unique_node_revisits(tf, le, phase='all'):
    """
    Revisits to a node in terms of number of unique nodes visited
    """

    def split_seq(seq, sep):
        start = 0
        while start < len(seq):
            try:
                stop = start + seq[start:].index(sep)
                yield seq[start:stop]
                start = stop + 1
            except ValueError:
                yield seq[start:]
                break

    revisits = dict()
    trajs = get_part_trajs_from_tf(tf, phase)
    for t in trajs:
        orig_node_seq = list(t[:, 0])
        # print(orig_node_seq)
        for node in p.NODE_LVL[le]:
            paths = list(split_seq(orig_node_seq, node))[1:-1]
            # print(node, paths)
            if len(paths) >= 2:
                if node not in revisits:
                    revisits[node] = []
                [revisits[node].append(len(np.unique(i))) for i in paths]
    # print(revisits)
    return revisits


def locate_first_k_endnodes(bout, k, predicate_to_skip=lambda x: False):
    assert k == 1 or k == 2     # mark it OUTSIDE as soon as it moves outside the current subquad
    visits = []
    first_subq = None
    for i, n in enumerate(bout):
        if n in p.LVL_6_NODES:
            if predicate_to_skip(n):
                # print(n, "here", p.node_subquadrant_dict[n])
                continue
            curr_subq = p.node_subquadrant_dict[n]
            # if first_subq and curr_subq != first_subq:
            #     visits.append((i, n, p.node_quadrant_dict[n], curr_subq, 'OUTSIDE'))
            #     break
            visits.append((i, n, p.node_quadrant_dict[n], curr_subq, p.node_subquadrant_label_dict[n]))
            # if not first_subq:
            #     first_subq = curr_subq
            k -= 1
        if not k: break
    return visits


def split_trajectories_at_first_reward(tf, k):
    """
    k: "until how many bouts are immediately after" the first reward
    """
    for b, bout in enumerate(tf.no):
        a = np.where(bout[:, 0] == WATERPORT_NODE)[0]
        if a.size:
            first_reward_bout_before_reward = tf.no[b][:a[0], :]
            first_reward_bout_after_reward = tf.no[b][a[0] + 1:, :]
            before_first_reward_tf = convert_episodes_to_traj_class([_[:, 0] for _ in tf.no[:b]] + [first_reward_bout_before_reward[:, 0]])
            imm_after_first_rew_tf = convert_episodes_to_traj_class([first_reward_bout_after_reward[:, 0]] + [_[:, 0] for _ in tf.no[b:b+k]])
            long_after_first_rew_tf = convert_episodes_to_traj_class([_[:, 0] for _ in tf.no[b+k:]])
            all_after_first_reward_tf = convert_episodes_to_traj_class([first_reward_bout_after_reward[:, 0]] + [_[:, 0] for _ in tf.no[b:]])
            break
    return {'before': before_first_reward_tf, 'imm_after': imm_after_first_rew_tf, 'long_after': long_after_first_rew_tf, 'all_after': all_after_first_reward_tf, 'all': tf}


def split_trajectories_k_parts(tf, k):

    cl0 = np.concatenate([list(b[:, 0]) for b in tf.no]).flat
    idx = len(cl0) // k  # index for first k nodes
    return convert_episodes_to_traj_class(wrap(cl0[:idx])[0]), convert_episodes_to_traj_class(wrap(cl0[idx:])[0])


def histo(X, bins=50, range=None, density=None, weights=None):
    hist, bin_edges = np.histogram(X, bins=bins, weights=weights, range=range, density=density)
    return hist, bin_edges

