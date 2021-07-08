"""

"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from parameters import NODE_LVL
from MM_Traj_Utils import NewMaze, NewNodes4, SplitModeClips
from utils import nodes2cell, convert_episodes_to_traj_class


def exploration_efficiency_sequential(episodes):
    """
    Counts total and distinct end nodes

    See also `exploration_efficiency` for implementation as was used in Rosenberg et al. (2021).
    :param episodes:
    :return:
    """
    step = 0
    steps_taken = dict([(2**i, np.nan) for i in range(0, 15)])
    nodes_explored = defaultdict(int)
    for each in episodes:
        for n in each:
            if n in NODE_LVL[6]:
                step += 1
                nodes_explored[n] += 1
                if step in steps_taken:
                    steps_taken[step] = len(nodes_explored)
    return steps_taken


def exploration_efficiency(episodes, re):
    """
    Averages new and distinct nodes over various window sizes. Based on method from Rosenberg et al. (2021).
    episodes: [[], [], ...]
    re = True for rewarded animals, False for unrewarded
    """
    leave, drink, explore = 0, 1, 2
    ma = NewMaze(6)
    tf = convert_episodes_to_traj_class(episodes)
    cl = SplitModeClips(tf, ma, re=re)  # find the clips; no drink mode for unrewarded animals
    ti = np.array([tf.no[c[0]][c[1] + c[2], 1] - tf.no[c[0]][c[1], 1] for c in cl])  # duration in frames of each clip
    nn = np.array([np.sum(cl[np.where(cl[:, 3] == leave)][:, 2]),
                   np.sum(cl[np.where(cl[:, 3] == drink)][:, 2]),
                   np.sum(cl[np.where(cl[:, 3] == explore)][:, 2])])  # number of node steps in each mode
    nf = np.array([np.sum(ti[np.where(cl[:, 3] == leave)]),
                   np.sum(ti[np.where(cl[:, 3] == drink)]),
                   np.sum(ti[np.where(cl[:, 3] == explore)])])  # number of frames in each mode
    tr = np.zeros((3, 3))  # number of transitions between the 3 modes
    for i in range(1, len(cl)):
        tr[cl[i - 1, 3], cl[i, 3]] += 1
    ce = cl[np.where(cl[:, 3] == explore)]  # clips of exploration
    ne = np.concatenate([tf.no[c[0]][c[1]:c[1] + c[2], 0] for c in ce])  # nodes excluding the last state in each clip
    le = 6  # end nodes only
    ln = list(range(2 ** le - 1, 2 ** (le + 1) - 1))  # list of node numbers in level le
    ns = ne[np.isin(ne, ln)]  # restricted to desired nodes
    _, c, n = NewNodes4(ns, nf[2] / len(ns))  # compute new nodes vs all nodes for exploration mode only
    steps_taken = dict(zip(c, n))
    # print(steps_taken)
    return steps_taken


def rotational_velocity(traj, d=3):
    """
    The rotational velocity is a rolling measure of the angle between
    consecutive points in a trajectory separated by δ time steps.
    This is then normalised by δ and the mean is taken across the
    entirety of a trajectory.
    Reference: William John de Cothi, 2020
    """
    angles_sum = 0.0
    for t in range(len(traj) - d):
        angles_sum += np.arctan2(
            traj[t + d][0] - traj[t][0],
            traj[t + d][1] - traj[t][1]
        )
    normalization_constant = d*(len(traj) - d) if len(traj) > d else 0.5
    vel = angles_sum/normalization_constant
    return vel


def diffusivity(traj, d=3):
    """
    The diffusivity is a rolling measure of the squared Euclidean distance
    travelled between consecutive points in a trajectory separated by δ time
    steps. This is then normalised by δ and the mean is taken across
    the trajectory.
    Reference: William John de Cothi, 2020
    """
    sum = 0.0
    for t in range(len(traj) - d):
        sum += np.power(traj[t + d][0] - traj[t][0], 2) +\
               np.power(traj[t + d][1] - traj[t][1], 2)
    normalization_constant = d*(len(traj) - d) if len(traj) > d else 0.5
    value = sum/normalization_constant
    return value


def tortuosity(traj):
    """
    The tortuosity is a measure of the bendiness of a trajectory and is equal to
    the total path distance travelled divided by the Euclidean distance travelled
    Reference: William John de Cothi, 2020
    """

    if len(traj) <= 1:
        return 1.0

    numerator = len(traj)
    denominator = np.sqrt(
        np.power(traj[-1][0]-traj[0][0], 2) +
        np.power(traj[-1][1]-traj[0][1], 2)
    )
    value = numerator/denominator
    return value


def get_feature_vectors(episodes):
    """Rotational velocity, diffusivity and tortuosity"""
    _, xy_trajectories = nodes2cell(episodes)
    trajs = list(xy_trajectories.values())
    trajs = list(filter(lambda t: len(t) >= 4, trajs))
    features = np.ones((len(trajs), 3))
    features[:, 0] = list(map(lambda t: rotational_velocity(t, 3), trajs))
    features[:, 1] = list(map(lambda t: diffusivity(t, 3), trajs))
    features[:, 2] = list(map(tortuosity, trajs))
    return features
