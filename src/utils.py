from parameters import FRAME_RATE, RWD_NODE
import numpy as np


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
             Also saves SAnodemap in the main_dir as 'nodemap.p'
    Return type: ndarray[(S, A), int]
    """
    SAnodemap = np.ones((self.S, self.A), dtype=int) * InvalidState
    for node in np.arange(self.S - 1):
        # Shallow level node available from current node
        if node % 2 == 0:
            SAnodemap[node, 0] = (node - 2) / 2
        elif node % 2 == 1:
            SAnodemap[node, 0] = (node - 1) / 2
        if SAnodemap[node, 0] == InvalidState:
            SAnodemap[node, 0] = HomeNode

        if node not in lv6_nodes:
            # Deeper level nodes available from current node
            SAnodemap[node, 1] = node * 2 + 1
            SAnodemap[node, 2] = node * 2 + 2

    # Nodes available from entry point
    SAnodemap[HomeNode, 0] = InvalidState
    SAnodemap[HomeNode, 1] = 0
    SAnodemap[HomeNode, 2] = InvalidState

    # with open(os.path.join(self.main_dir, 'nodemap.p'),'wb') as f:
    #     pickle.dump(SAnodemap, f)
    return SAnodemap

def load_environment():
    '''
    :return: dictionary of frequently used variables, config{N: , S: , A: , nodemap: , ....}
    '''
    config = {}
    config['N'] = 10 # number of rewarded mice
    config['S'] = 128  # number of states/nodes in the maze
    config['A'] = 3  # number of actions available at each state
    config['RewardNodeMag'] = 1  # reward magnitude at the reward port (node 116)
    config['InvalidState'] = -1  # padding to indicate invalid nodes on nodemap and invalid states in trajectories
    config['StartNode'] = 0  # node at which all episodes begin
    config['HomeNode'] = 127  # node at maze entrance
    config['RewardNode'] = 116  # node where liquid reward is located
    config['NumRuns'] = 100  # number of times to repeat model predictions when running them forward on trajectories
    config['nodemap'] = get_SAnodemap()  # mapping from each node to its three neighboring nodes with -1 as padding
                                                              # for invalid nodes
    # Some lists of nicknames for mice
    config['RewNames'] = ['B1', 'B2', 'B3', 'B4', 'C1', 'C3', 'C6', 'C7', 'C8', 'C9']
    config['UnrewNames'] = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
    config['AllNames'] = config['RewNames'] + config['UnrewNames']
    config['UnrewNamesSub'] = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']  # excluding D6 which barely entered the maze

    # Define cell numbers of end/leaf nodes
    lvl6_nodes = list(range(63, 127))
    lvl5_nodes = list(range(31, 63))
    lvl4_nodes = list(range(15, 31))
    lvl3_nodes = list(range(7, 15))
    lvl2_nodes = list(range(3, 7))
    lvl1_nodes = list(range(1, 3))
    lvl0_nodes = list(range(0, 1))
    config['lvl_dict'] = {0: lvl0_nodes, 1: lvl1_nodes, 2: lvl2_nodes, 3: lvl3_nodes, 4: lvl4_nodes, 5: lvl5_nodes, 6: lvl6_nodes}

    return config