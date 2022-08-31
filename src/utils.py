from MM_Traj_Utils import add_node_times_to_tf, add_reward_times_to_tf, NewMaze, Traj
from parameters import FRAME_RATE, WATERPORT_NODE, HOME_NODE, RWD_STATE, ALL_MAZE_NODES, ALL_VISITABLE_NODES, \
    TIME_EACH_MOVE, INVALID_STATE, NODE_LVL, LVL_BY_NODE, HALF_UP, HALF_DOWN, QUAD1, QUAD2, QUAD3, QUAD4, \
    HALF_LEFT, HALF_RIGHT, ROW_3, ROW_1, COL_3, COL_1
import parameters as p
import numpy as np
from numpy import array, arange

from collections import defaultdict


def get_all_night_nodes_and_times(tf):
    """
    Get the nodes the animal visited across all night and the corresponding times
    :returns: ndarray (n_nodes_traversed, 2) nodes and the time instant the animal was there
    """
    tf_new = add_node_times_to_tf(tf)
    return np.vstack(tf_new.node_times)


def get_re_nodes_and_times(tf):
    """
    Get the reward times the animal got the corresponding times
    :returns: ndarray (n_nodes_traversed, 2) nodes and the time instant the animal was there
    """
    tf_new = add_reward_times_to_tf(tf)
    return np.vstack(tf_new.re_times)


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


def get_wp_visit_times_and_rwd_times(tf):
    """
    Get lists of waterport visit times and reward delivery times
    :param tf: trajectory object
    :return: times_to_waterport_visits, times_to_rwd
    """
    times_to_waterport_visits = get_node_visit_times(tf, WATERPORT_NODE)

    # calculate reward deliveries

    frames_to_rwd = np.array([s_e_frs_of_each_rwd[0] + tf.fr[bout_idx, 0] for bout_idx, frs_rwd in enumerate(tf.re)
                              for s_e_frs_of_each_rwd in frs_rwd])
    times_to_rwd = np.array([frame_to_rwd / FRAME_RATE for frame_to_rwd in frames_to_rwd])

    return times_to_waterport_visits, times_to_rwd


def create_list_waterport_visits_in_between_rwds(times_to_waterport_visits, times_to_rwd, include_wp_visits_after_last_rwd=False):
    """ Creates list of all waterport visits in between each of the rewards in `times_to_rwd`. Includes visits before
    first reward, but does not include visits after the last reward.
    :param times_to_waterport_visits: list of times of visits to the waterport node
    :param times_to_rwd: list of times of reward delivery
    :param include_wp_visits_after_last_rwd: Use True to include waterport visits after last reward delivery. Default is False.
    :return: a list of lists. Each list has the waterport visits in between rewards. The first element has the visits
    before any reward delivery.
    """
    # TODO: this function would probably have been more readable if I had done an external loop of the rwd visits instead of waterport visits.
    #  Maybe change someday. But seems to be working now as it is. Good luck understanding it! :-|
    all_waterport_visits = []
    waterport_visits_in_between_rwds = []
    rwd_i = 0
    for wp_idx, waterport_visit_time in enumerate(times_to_waterport_visits):  # create list of all waterport visits in between rewards
        # print('  wp ', waterport_visit_time)
        if rwd_i < len(times_to_rwd):  # if there are still reward deliveries after current waterport visit
            if waterport_visit_time < times_to_rwd[rwd_i]:  # get waterport visits before each rwd delivery
                waterport_visits_in_between_rwds.append(waterport_visit_time)
                if wp_idx == len(times_to_waterport_visits) - 1:  # if it is the last item in the list
                    # print("last wp visit")
                    # print('wp ', waterport_visits_in_between_rwds)
                    all_waterport_visits.append(waterport_visits_in_between_rwds)
            else:  # reached waterport visit that is already after currently considered rwd delivery
                # print('rwd ', times_to_rwd[rwd_i])
                # print('wps ', waterport_visits_in_between_rwds)
                rwd_i += 1
                all_waterport_visits.append(waterport_visits_in_between_rwds)
                waterport_visits_in_between_rwds = []
                waterport_visits_in_between_rwds.append(waterport_visit_time)  # add current waterport visit to the list of visits after next rwd
                if (include_wp_visits_after_last_rwd or waterport_visit_time < times_to_rwd[-1]) \
                        and (wp_idx == len(times_to_waterport_visits) - 1):  # if it is the last item in the list
                    # print("last wp visit")
                    # print('wps ', waterport_visits_in_between_rwds)
                    all_waterport_visits.append(waterport_visits_in_between_rwds)
        elif include_wp_visits_after_last_rwd:  # waterport visits after last reward visit
            waterport_visits_in_between_rwds.append(waterport_visit_time)  # add to the list of visits after the last rwd
            if wp_idx == len(times_to_waterport_visits)-1:  # if it is the last item in the list
                # print("last wp visit")
                # print('wps ', waterport_visits_in_between_rwds)
                all_waterport_visits.append(waterport_visits_in_between_rwds)

    return all_waterport_visits


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


def nodes2cell(state_hist_all):
    '''
    param state_hist_all:
    '''
    state_hist_cell = []
    state_hist_xy = {}
    ma=NewMaze(6)
    for epID, epi in enumerate(state_hist_all):
        cells = []
        if not len(epi):
            continue
        for i in range(len(epi)-1):
            node = epi[i]
            assert node != HOME_NODE
            if node == RWD_STATE: continue
            if (i == 0 and node == 0) or node > epi[i-1]:
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
        state_hist_cell.append(cells)
        state_hist_xy[epID] = np.zeros((len(cells),2))
        state_hist_xy[epID][:,0] = ma.xc[cells] + np.random.choice([-1,1],len(ma.xc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.xc[cells]))/2
        state_hist_xy[epID][:,1] = ma.yc[cells] + np.random.choice([-1,1],len(ma.yc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.yc[cells]))/2
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
    Convert a Traj to list of lists with tf.no containing episode information.
    At the moment, simply using the index as time. This is so that simulated episodes
    can use some of the functions provided original authors that operate on Traj
    class instances.

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

    print("Home visit counts in simulated episodes:", len(episodes))
    return episodes


def get_reward_times(episodes):
    """
    Steps taken to reach the reward which is assumed to be the last node of an
    episode

    TODO: refactor this to accommodate continuous agent OR use original author's
    implementation
    """
    visit_reward_node = []
    time_reward_node = []
    for i, traj in enumerate(episodes):
        if traj.count(WATERPORT_NODE):
            visit_reward_node.append(i)
            time_reward_node.append(len(traj))
    return visit_reward_node, time_reward_node


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
    """
    node_level_visit_freq = np.zeros(len(NODE_LVL))
    for episode in episodes:
        for node in episode:
            if node not in ALL_MAZE_NODES: continue
            node_level_visit_freq[LVL_BY_NODE[node]] += 1
    node_level_visit_freq = node_level_visit_freq / np.sum([len(e) for e in episodes])
    return node_level_visit_freq


def get_parent_node(n):
    return (n + (n % 2) - 1) // 2


def get_children(n):
    return 2*n+1, 2*n+2


def get_opp_children(n, c):
    c1, c2 = get_children(n)
    return c1 if c == c2 else c2


def home_path_node(n):
    """
    Returns the node path that leads from node n to the start of the maze
    Includes both start and end nodes
    """
    ret = []
    while n >= 0:
        ret.append(n)
        n = get_parent_node(n)
    return ret


def connect_path_node(n1, n2):
    """
    Returns the shortest path that connects nodes n1 and n2
    Includes both start and end nodes
    """
    r1 = home_path_node(n1)
    r2 = home_path_node(n2)[::-1] # reversed
    for i in r1:
        if i in r2:
            return r1[:r1.index(i)]+r2[r2.index(i):]


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

