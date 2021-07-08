from MM_Traj_Utils import add_node_times_to_tf, add_node_times_to_tf_re, NewMaze, Traj
from parameters import FRAME_RATE, RWD_NODE, HOME_NODE, WATER_PORT_STATE, ALL_MAZE_NODES, ALL_VISITABLE_NODES
import numpy as np

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
    Get the nodes the animal visited across all night and the corresponding times
    :returns: ndarray (n_nodes_traversed, 2) nodes and the time instant the animal was there
    """
    tf_new = add_node_times_to_tf_re(tf)
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

    frs_to_node_visits = np.concatenate(frs_to_node_visits)

    times_to_node_visits = np.array([frame_to_node_visit / FRAME_RATE
                                          for frame_to_node_visit in frs_to_node_visits])
    return times_to_node_visits


def get_wp_visit_times_and_rwd_times(tf):
    """
    Get lists of waterport visit times and reward delivery times
    :param tf: trajectory object
    :return: times_to_waterport_visits, times_to_rwd
    """
    times_to_waterport_visits = get_node_visit_times(tf, RWD_NODE)

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
    SAnodemap[WATER_PORT_STATE, :] = INVALID_STATE
    SAnodemap[RWD_NODE, 1] = WATER_PORT_STATE

    return SAnodemap


def nodes2cell(state_hist_all):
    '''
    simulated trajectories, state_hist_all: {mouseID: [[TrajID x TrajSize]]}
    '''
    state_hist_cell = []
    state_hist_xy = {}
    ma=NewMaze(6)
    for epID, epi in enumerate(state_hist_all):
        cells = []
        if not epi:
            continue
        for id,node in enumerate(epi):
            if id != 0 and node != HOME_NODE and node != WATER_PORT_STATE:
                if node > epi[id-1]:
                    # if going to a deeper node
                    cells.extend(ma.ru[node])
                elif node < epi[id-1]:
                    # if going to a shallower node
                    reverse_path = list(reversed(ma.ru[epi[id-1]]))
                    reverse_path = reverse_path + [ma.ru[node][-1]]
                    cells.extend(reverse_path[1:])
        if node==HOME_NODE:
            home_path = list(reversed(ma.ru[0]))
            cells.extend(home_path[1:])  # cells from node 0 to maze exit
        state_hist_cell.append(cells)
        state_hist_xy[epID] = np.zeros((len(cells),2))
        state_hist_xy[epID][:,0] = ma.xc[cells] + np.random.choice([-1,1],len(ma.xc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.xc[cells]))/2
        state_hist_xy[epID][:,1] = ma.yc[cells] + np.random.choice([-1,1],len(ma.yc[cells]),p=[0.5,0.5])*np.random.rand(len(ma.yc[cells]))/2
    return state_hist_cell, state_hist_xy


def convert_episodes_to_traj_class(episodes):
    """
    Convert list of lists to Traj class with tf.no containing episode information.
    At the moment, simply using the index as time. This is so that simulated episodes
    can use some of the functions provided original authors that operate on Traj
    class instances.

    episodes: [[], [], ..]
    Returns tf: Traj
    """

    tf = Traj(fr=None,ce=None,ke=None,no=[],re=None)
    for e in episodes:
        tf.no.append(np.stack([np.array(e), np.arange(len(e))], axis=1))
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
    Break the simulated trajectory list at either home node or reward node
    and return the list of episodes
    param maze_episode_traj: list of nodes
    returns:
    episodes: list of lists (of nodes)
    """
    episodes = []
    epi = []
    counts = defaultdict(int)
    for i in maze_episode_traj:
        if i == HOME_NODE:
            epi.append(i)
            if len(epi) > 2:
                episodes.append(epi)
            epi = []
            counts[HOME_NODE] += 1
        elif i == WATER_PORT_STATE:
            # epi.append(i)
            if len(epi) > 2:
                episodes.append(epi)
            epi = []
            counts[WATER_PORT_STATE] += 1
        # elif i == WATER_PORT_STATE:
        #     continue
        else:
            epi.append(i)
    if epi:
        episodes.append(epi)
    assert WATER_PORT_STATE not in counts
    print("Home and WP node visit counts in simulated episodes:", counts)
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
        if traj.count(RWD_NODE):
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
