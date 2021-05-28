"""

"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from parameters import NODE_LVL
from utils import nodes2cell


def exploration_efficiency(episodes):
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

    # print(nodes_explored)
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
    # print(traj)
    # print(len(traj))
    angles_sum = 0.0
    for t in range(len(traj) - d):
        angles_sum += np.arctan2(
            traj[t + d][0] - traj[t][0],
            traj[t + d][1] - traj[t][1]
        )
        # print(traj[t][0], traj[t+d][0], angles_sum)
    normalization_constant = d*(len(traj) - d) if len(traj) > d else 0.5
    vel = angles_sum/normalization_constant
    # print(vel)
    return vel


def diffusivity(traj, d=3):
    """
    The diffusivity is a rolling measure of the squared Euclidean distance
    travelled between consecutive points in a trajectory separated by δ time
    steps. This is then normalised by δ and the mean is taken across
    the trajectory.
    Reference: William John de Cothi, 2020
    """
    # print(traj)
    # print(len(traj))
    sum = 0.0
    for t in range(len(traj) - d):
        sum += np.power(traj[t + d][0] - traj[t][0], 2) +\
               np.power(traj[t + d][1] - traj[t][1], 2)
    normalization_constant = d*(len(traj) - d) if len(traj) > d else 0.5
    value = sum/normalization_constant
    # print(vel)
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
    # print(vel)
    return value


def get_feature_vectors(episodes):
    _, xy_trajectories = nodes2cell(episodes)
    trajs = list(xy_trajectories.values())
    trajs = list(filter(lambda t: len(t) >= 4, trajs))
    features = np.ones((len(trajs), 3))
    features[:, 0] = list(map(lambda t: rotational_velocity(t, 3), trajs))
    features[:, 1] = list(map(lambda t: diffusivity(t, 3), trajs))
    features[:, 2] = list(map(tortuosity, trajs))
    return features
