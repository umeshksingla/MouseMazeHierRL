from parameters import FRAME_RATE, RWD_NODE
import numpy as np


def get_wp_visit_times_and_rwd_times(tf):
    frs_to_waterport_visits = []
    n_bouts = len(tf.no)
    for bout in range(n_bouts):
        rwd_node_visits_frs = tf.no[bout][tf.no[bout][:, 0] == RWD_NODE]
        if len(rwd_node_visits_frs) > 0:
            bout_init_fr = tf.fr[bout, 0]  # first frame of the bout
            frs_to_waterport_visits.append(bout_init_fr + rwd_node_visits_frs[:, 1])

    frs_to_waterport_visits = np.concatenate(frs_to_waterport_visits)

    times_to_waterport_visits = np.array([frame_to_waterport_visit / FRAME_RATE
                                          for frame_to_waterport_visit in frs_to_waterport_visits])

    # calculate reward deliveries

    frames_to_rwd = np.array([s_e_frs_of_each_rwd[0] + tf.fr[bout_idx, 0] for bout_idx, frs_rwd in enumerate(tf.re)
                              for s_e_frs_of_each_rwd in frs_rwd])
    times_to_rwd = np.array([frame_to_rwd / FRAME_RATE for frame_to_rwd in frames_to_rwd])

    return times_to_waterport_visits, times_to_rwd


def create_list_waterport_visits_in_between_rwds(times_to_waterport_visits, times_to_rwd):
    """ Creates list of all waterport visits in between each of the rewards in `times_to_rwd`
    :param times_to_waterport_visits: list of times of visits to the waterport node
    :param times_to_rwd: list of times of reward delivery
    :return: a list of lists. Each list has the waterport visits in between rewards. The first element has the visits
    before any reward delivery
    """
    all_waterport_visits = []
    waterport_visits_in_between_rwds = []
    rwd_i = 0
    for wp_idx, waterport_visit_time in enumerate(times_to_waterport_visits):  # create list of all waterport visits in between rewards
        # print('wp ', waterport_visit_time)
        if rwd_i < len(times_to_rwd):
            if waterport_visit_time < times_to_rwd[rwd_i]:  # get waterport visits before each rwd delivery
                waterport_visits_in_between_rwds.append(waterport_visit_time)
            else:  # reached waterport visit that is already after currently considered rwd delivery
                # print('wp ', waterport_visits_in_between_rwds)
                # print('rwd ', times_to_rwd[rwd_i])
                rwd_i += 1
                all_waterport_visits.append(waterport_visits_in_between_rwds)
                waterport_visits_in_between_rwds = []
                waterport_visits_in_between_rwds.append(waterport_visit_time)  # add to the list of visits after next rwd
        else:
            waterport_visits_in_between_rwds.append(waterport_visit_time)  # add to the list of visits after the last rwd
            if wp_idx == len(times_to_waterport_visits)-1:
                # print("end")
                # print('wp ', waterport_visits_in_between_rwds)
                all_waterport_visits.append(waterport_visits_in_between_rwds)

    return all_waterport_visits
