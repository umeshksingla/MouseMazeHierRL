"""

"""
import sys
from collections import defaultdict, OrderedDict
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pickle

from parameters import NODE_LVL, OUTDATA_PATH
from MM_Traj_Utils import NewMaze, NewNodes4, SplitModeClips, LoadTrajFromPath, Entropy
from MM_Plot_Utils import hist
from MM_Models import MarkovFit2, MarkovFit3, TranslLevelsLR
from utils import nodes2cell, convert_episodes_to_traj_class, convert_traj_to_episodes, locate_first_k_endnodes
from decision_bias_analysis_tools import ComputeFourBiasClips2
from parameters import EXPLORE, RewNames, UnrewNamesSub
import parameters as p

outdata_path = '../outdata/'
if outdata_path not in sys.path:
    sys.path.append(outdata_path)


def exploration_efficiency_sequential(episodes):
    """
    Counts total and distinct end nodes

    See also `exploration_efficiency` for implementation as was used in Rosenberg et al. (2021).
    :param episodes (format [[], [], ...]): list of episode trajectories (which are list of nodes)
    :return: steps_taken
    """
    step = 0
    steps_taken = dict([(2**i, np.nan) for i in range(0, 15)])
    end_nodes_explored = defaultdict(int)
    for episode in episodes:
        for node in episode:
            if node in NODE_LVL[6]:
                step += 1
                end_nodes_explored[node] += 1
                if step in steps_taken:  # step is a power of 2
                    steps_taken[step] = len(end_nodes_explored)
    return steps_taken


def exploration_efficiency(tf, re, le=6):
    """
    Averages new and distinct nodes over various window sizes.
    Based on method from Rosenberg et al. (2021).

    :param episodes: (format [[], [], ...]) list of episode trajectories (which are list of nodes)
    :param re = True for rewarded animals, False for unrewarded
    :param le = level of nodes for which efficiency is to be calculated (it will discard all other level nodes)

    :return: steps_taken (dict of total_nodes_visited -> distinct_nodes_visited
    for various window sizes). Example, {10: 2, 50: 15} means on average it
    visited 2 distinct nodes when it visited a total of 10 nodes. And to cover
    15 distinct nodes it has to visit 50 nodes on average
    """
    print(f"calculating exp efficiency wrt level {le}...")
    leave, drink, explore = 0, 1, 2
    ma = NewMaze(6)
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
    ln = list(range(2 ** le - 1, 2 ** (le + 1) - 1))  # list of node numbers in level le
    ns = ne[np.isin(ne, ln)]  # restricted to desired nodes
    _, c, n = NewNodes4(ns, nf[2] / len(ns))  # compute new nodes vs all nodes for exploration mode only
    steps_taken = dict(zip(c, n))
    print(steps_taken)
    return c, n


def rotational_velocity(traj, d=3):
    """
    The rotational velocity is a rolling measure of the angle between
    consecutive points in a trajectory separated by d time steps.
    This is then normalised by d and the mean is taken across the
    entirety of a trajectory. Reference: William John de Cothi, 2020
    """
    angles_sum = 0.0
    for t in range(len(traj) - d):
        x1, x2 = traj[t][0], traj[t + d][0]
        y1, y2 = traj[t][1], traj[t + d][1]
        angles_sum += np.arctan2(y2-y1, x2-x1)
    normalization_constant = d*(len(traj) - d)
    vel = angles_sum/normalization_constant
    return vel


def angular_diffusivity(traj, d=3):
    """
    Reference: William John de Cothi, 2022
    """
    angles = []
    for t in range(len(traj) - d):
        x1, x2 = traj[t][0], traj[t + d][0]
        y1, y2 = traj[t][1], traj[t + d][1]
        angles.append(np.arctan2(y2-y1, x2-x1))
    sin_D = np.mean(np.sin(angles))
    cod_D = np.mean(np.cos(angles))
    return sin_D


def diffusivity(traj, d=3):
    """
    The diffusivity is a rolling measure of the squared Euclidean distance
    travelled between consecutive points in a trajectory separated by d time
    steps. This is then normalised by d and the mean is taken across
    the trajectory. Reference: William John de Cothi, 2020
    """
    sum = 0.0
    for t in range(len(traj) - d):
        x1, x2 = traj[t][0], traj[t + d][0]
        y1, y2 = traj[t][1], traj[t + d][1]
        sum += np.power(x2-x1, 2) + np.power(y2-y1, 2)
    normalization_constant = d*(len(traj) - d)
    value = sum/normalization_constant
    return value


def tortuosity(traj):
    """
    The tortuosity is a measure of the bendiness of a trajectory and is equal to
    the total path distance travelled divided by the Euclidean distance travelled.
    Reference: William John de Cothi, 2020
    """

    numerator = len(traj)
    denominator = np.sqrt(
        np.power(traj[-1][0]-traj[0][0], 2) +
        np.power(traj[-1][1]-traj[0][1], 2)
    )
    value = numerator/denominator
    return value


def get_feature_vectors(episodes):
    """Rotational velocity, diffusivity and tortuosity"""
    trajs = list(filter(lambda t: len(t) >= 4, episodes))
    _, xy_trajectories = nodes2cell(trajs)
    trajs = list(xy_trajectories.values())
    features = np.ones((len(trajs), 3))
    features[:, 0] = list(map(lambda t: rotational_velocity(t, 5), trajs))
    features[:, 1] = list(map(lambda t: diffusivity(t, 5), trajs))
    features[:, 2] = list(map(tortuosity, trajs))
    return features


def get_direct_paths(episodes, node, mode):
    '''
    Extracting starting points of direct paths to / from node of interest
    :param episodes: episodes (format [[], [], ...]): list of episode trajectories (which are list of nodes)
    :param node: node of interest to analyze
    :param mode: 'source' to look at direct paths coming from node or 'target' to look at direct paths going to node
    :return: list of starting point nodes of direct paths either to / from node of interest
    '''
    tailend_nodelist = []
    for traj in episodes:
        nodepos = np.where(np.array(traj) == node)[0]

        for id, pos in enumerate(nodepos):
            clippedtraj = []
            trajsegment = []
            if mode == 'source':
                # Extracting path starting from current node occurence
                trajsegment = traj[nodepos[id]:]
            elif mode == 'target':
                # Extracting path ending at current node occurence
                trajsegment = traj[nodepos[id]:None:-1]
            trajsegment.pop(0)  # removing analysis node from segment

            for i in trajsegment:
                if i not in clippedtraj:
                    clippedtraj.extend([i])
                else:
                    break
            if clippedtraj: tailend_nodelist.extend([clippedtraj[-1]])
    return tailend_nodelist


def get_dfs_ee(le=6):
    """
    Returns DFS exploration efficiency at level le
    """
    n = 2 ** le
    return zip(*dict([(i, i) for i in range(n+1)]).items())


def get_random_ee(le=6):
    """
    Returns exploration efficiency for a random agent at level le
    Note: these are pre-calculated values by simulating a random agent for 20k steps
    """
    if le == 6: d = {2.0: 1.6216659827656956, 3.0: 2.0535384615384618, 6.0: 3.065270935960591, 10.0: 4.123203285420945, 18.0: 6.037037037037037, 32.0: 8.842105263157896, 56.0: 13.310344827586206, 100.0: 19.708333333333332, 180.0: 31.14814814814815, 320.0: 43.86666666666667, 560.0: 52.875, 1000.0: 59.5, 1800.0: 62.5, 3200.0: 64.0, 4875.0: 64.0}
    if le == 5: d = {2.0: 1.2088091353996737, 3.0: 1.3919249592169658, 6.0: 1.8784665579119086, 10.0: 2.401360544217687, 18.0: 3.2401960784313726, 32.0: 4.545851528384279, 56.0: 6.435114503816794, 100.0: 9.452054794520548, 180.0: 14.325, 320.0: 19.772727272727273, 560.0: 25.923076923076923, 1000.0: 30.714285714285715, 1800.0: 31.75, 3200.0: 32.0, 5600.0: 32.0, 7356.0: 32.0}
    if le == 4: d = {2.0: 1.201167728237792, 3.0: 1.3853503184713376, 6.0: 1.877388535031847, 10.0: 2.422872340425532, 18.0: 3.2440191387559807, 32.0: 4.512820512820513, 56.0: 6.388059701492537, 100.0: 8.891891891891891, 180.0: 12.35, 320.0: 14.818181818181818, 560.0: 15.0, 1000.0: 16.0, 1800.0: 16.0, 3200.0: 16.0, 3768.0: 16.0}
    if le == 3: d = {2.0: 1.2224510813594234, 3.0: 1.3848531684698608, 6.0: 1.888544891640867, 10.0: 2.4536082474226806, 18.0: 3.2803738317757007, 32.0: 4.4, 56.0: 5.794117647058823, 100.0: 7.105263157894737, 180.0: 7.5, 320.0: 8.0, 560.0: 8.0, 1000.0: 8.0, 1800.0: 8.0, 1942.0: 8.0}
    if le == 2: d = {2.0: 1.1629955947136563, 3.0: 1.4105960264900663, 6.0: 1.8675496688741722, 10.0: 2.1666666666666665, 18.0: 2.86, 32.0: 3.357142857142857, 56.0: 3.8125, 100.0: 4.0, 180.0: 4.0, 320.0: 4.0, 560.0: 4.0, 908.0: 4.0}
    return zip(*d.items())


def get_unrewarded_ee(le=6):
    """
    Returns unrewarded animals' pre-computed exploration efficiency
    """
    with open(OUTDATA_PATH + f'ee_unrewarded_le={le}.pkl', 'rb') as f:
        d = pickle.load(f)
    return d


def get_rewarded_ee(le=6):
    """
    Returns any rewarded animal's exploration efficiency before the first reward
    """
    raise NotImplementedError
    tf = LoadTrajFromPath(OUTDATA_PATH + 'B1-tf')
    rew_epi = convert_traj_to_episodes(tf)
    rew_epi_exp = rew_epi[:17]
    rew_epi_exp.append(rew_epi[17][:354])
    return exploration_efficiency(rew_epi_exp, re=False, le=le)     # need to change episodes to tf


def fit_LevyWalk(nicknamelist, plottitle, start_bout=None, end_bout=None, log_scale=False):
    '''
    Fit real or predicted mouse trajectories to a Levy Walk model: c * (x ** - mu) where a 1 < mu <= 3 constitutes a Levy Walk
    source for analysis: Murano, J., Mitsuishi, M. & Moriyama (2018), https://doi.org/10.1007/s10015-018-0457-7
    :param nicknamelist: list of mouse nicknames
    :param plottitle: title for agent or agent group the fitting is being done for
    :param start_bout: bout to begin analysis for all mice, if not leave as None
    :param end_bout: bout to end analysis for all mice, if not leave as None
    :param log_scale: True or False to plot on a double log scale
    :return:
    '''
    def TwoStep():
        '''
        Classifies transitions between every other connecting node on the maze as a
        left: 0 or right: 1 turn. All other types and invalid turns are -1.
        :return: 128-by-128 array [source node, destination node]
                 eg. two_st[0,3] is a left turn (= 0) from node 0 to node 3
        '''
        m = NewMaze()
        stem_vertT = []
        [stem_vertT.extend(vl) for vl in list(NODE_LVL.values())[0:6:2]]
        stem_horzT = []
        [stem_horzT.extend(vl) for vl in list(NODE_LVL.values())[1:5:2]]
        stem_horzT.extend([127])

        two_st = np.full((len(m.ru) + 1, len(m.ru) + 1), -1, dtype=int)

        for snode in stem_vertT:
            # snode is the stem of a vertical T-junction
            dnode = np.arange(snode + (snode + 1) * 3, snode + (snode + 1) * 3 + 4)
            two_st[snode, dnode] = [0, 1, 1, 0]

        for snode in stem_horzT[:-1]:
            # snode is the stem of a horizontal T-junction
            dnode = np.arange(snode + (snode + 1) * 3, snode + (snode + 1) * 3 + 4)
            two_st[snode, dnode] = [0, 1, 1, 0]
        two_st[127, [1, 2]] = [0, 1]

        return two_st

    def get_turn(source, dest, two_st):
        '''
        Identify turn type connecting the source and destination node
        :param source: node to begin a turn from eg. node 0
        :param dest: node to end a turn at eg. node 3 (corresponding to the source node 0)
        :param two_st: 128-by-128 array [source node, destination node] with left: 0 or right: 1 or -1 for all others
        :return: either 'L' or 'R' for left and right turns respectively. Or -1 for invalid turns
        '''
        turns = ['L', 'R']
        turn = -1
        if dest > source:
            # step is inwards
            turn = turns[two_st[source, dest]] if two_st[source, dest] != -1 else -1
        elif source > dest:
            # step is outwards
            turn = turns[1 - two_st[dest, source]] if two_st[dest, source] != -1 else -1
        return turn

    def get_turnseq(nickname, start_bout=None, end_bout=None):
        '''
        Convert trajectory to a sequence of turns
        :param nickname: mouse nickname as a str
        :param start_bout: bout to begin analysis for each mouse
        :param end_bout: bout to end analysis for each mouse
        :return: list of turn types ('L', 'R', -1) for the specified bout duration and mouse
        '''
        tf = LoadTrajFromPath(outdata_path + nickname + '-tf')
        turnseq = []
        two_st = TwoStep()

        for bout in tf.no[start_bout:end_bout]:
            nodelist = bout[:, 0]
            bout_turnseq = []
            for source_node, dest_node in zip(nodelist[:-3], nodelist[2:]):
                turn = get_turn(source_node, dest_node, two_st)
                bout_turnseq.extend([turn])
            turnseq.append(bout_turnseq)
        return turnseq

    def get_indv_turnsegment(turnseq):
        '''
        Analyze trajectory of an individual mouse and calculate frequency of alternating (TA) and repeating (TR) path segments
        :param turnseq: list of turn types ('L', 'R', -1) for the specified bout duration and mouse
        :return: frequency array of TA or TR path segments ranging from length 1 to a maximum length of 15
        '''

        def get_turntype(turn_pair):
            '''
            Identifying whether a sequence of two turns are alternating (TA) or repeating (TR)
            :param turn_pair:
            :return: 'TA' or 'TR' indicating the type of path segment for Levy Walk analysis
            '''
            flag = 'INVALID'
            if turn_pair == ['L', 'R'] or turn_pair == ['R', 'L']:
                flag = 'TA'
            elif turn_pair == ['L', 'L'] or turn_pair == ['R', 'R']:
                flag = 'TR'
            return flag

        def update_seglengths(prev_flag, curr_flag, TA, TR, seg_length):
            '''
            Updates TA and TR frequency arrays before agent changes to a different path segment type
            :param prev_flag: previous path segment type ('TA','TR','INVALID')
            :param curr_flag: current path segment type ('TA','TR','INVALID')
            :param TA: frequency array of TA path segments ranging from length 1 to a maximum length of 15
            :param TR: frequency array of TR path segments ranging from length 1 to a maximum length of 15
            :param seg_length: segment length of the current path segment type
            :return: TA, TR, seg_length
            '''
            if prev_flag == 'TA':
                TA[seg_length - 1] += 1
            elif prev_flag == 'TR':
                TR[seg_length - 1] += 1
            seg_length = 1 if curr_flag != 'INVALID' else 0
            return TA, TR, seg_length

        TA = np.zeros(15, dtype=int)
        TR = np.zeros(15, dtype=int)

        for boutseq in turnseq:
            prev_flag = get_turntype(boutseq[0:2])
            seg_length = 0
            for id, turn in enumerate(boutseq[:-1]):
                curr_flag = get_turntype(boutseq[id:id + 2])
                if prev_flag == 'INVALID' and curr_flag != 'INVALID':
                    seg_length += 1
                elif prev_flag != 'INVALID' and curr_flag == 'INVALID':
                    TA, TR, seg_length = update_seglengths(prev_flag, curr_flag, TA, TR, seg_length)
                elif curr_flag == prev_flag and (prev_flag != 'INVALID'):
                    seg_length += 1
                elif curr_flag != prev_flag and (prev_flag != 'INVALID'):
                    TA, TR, seg_length = update_seglengths(prev_flag, curr_flag, TA, TR, seg_length)
                prev_flag = curr_flag
        return TA, TR

    def get_turnsegment(nicknamelist, start_bout, end_bout):
        '''
        Analyze trajectories of all mice and calculate average frequency of alternating (TA) and repeating (TR) path segments
        :param nicknamelist: list of mouse nicknames
        :param start_bout: bout to begin analysis for all mice
        :param end_bout: bout to end analysis for all mice
        :return: frequency array of average TA or TR path segments ranging from length 1 to a maximum length of 15
        '''
        TA_group = np.zeros(15, dtype=int)
        TR_group = np.zeros(15, dtype=int)

        for nickname in nicknamelist:
            turnseq = get_turnseq(nickname, start_bout, end_bout)
            TA, TR = get_indv_turnsegment(turnseq)
            TA_group = TA_group + TA
            TR_group = TR_group + TR
        TA_group = TA_group / len(nicknamelist)
        TR_group = TR_group / len(nicknamelist)

        return TA_group, TR_group

    def Levy_powerlaw(x, c, m):
        '''

        :param x: length of path segments eg. 'LRLR' is an alternating path segment of length 3 ('LR', 'RL', 'LR')
        :param c: constant
        :param m: mu
        :return: cumulative relative frequency of path segment lengths (y values) predicted by the Levy Walk model
        '''
        return c * (x ** m)

    def plot_freq(turn_seqdata, xlabel, plottitle, mode, log_scale=False):
        '''
        Line plot of mouse data path segment frequencies with a fitted Levy Walk model
        :param turn_seqdata: TA or TR frequency arrays
        :param xlabel: label for x-axis
        :param plottitle: title for agent or agent group the fitting is being done for
        :param mode: relative frequency, 'rel_freq' or cumulative frequency, 'cum_freq'
        :param log_scale: True or False to plot on a double log scale
        :return:
        '''

        def calc_ydata(turn_seqdata, total, mode):
            if mode == 'rel_freq':
                ydata = turn_seqdata * 100 / total
                ylabel = 'Relative Frequency (%)'
            elif mode == 'cum_freq':
                ydata = np.cumsum(turn_seqdata[::-1])[::-1]
                ydata = ydata * 100 / np.max(ydata)
                ylabel = 'Cumulative Relative Frequency (%)'
            return ydata, ylabel

        plt.figure()
        total = np.sum(turn_seqdata)
        ydata, ylabel = calc_ydata(turn_seqdata, total, mode)
        x = np.arange(len(turn_seqdata))
        [c, mu], pcov = curve_fit(Levy_powerlaw, x + 1, ydata, maxfev=10000)

        # R^2 calculation copied from:
        # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
        residuals = ydata - Levy_powerlaw(x + 1, c, mu)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(c, mu, r_squared)

        plt.plot(x, ydata, '.', markersize=12, label='Real data')
        plt.plot(x, Levy_powerlaw(x + 1, c, mu), 'r', label='Fitted power law')
        if log_scale:
            plt.yscale('log')
            plt.xscale('log')
            xlabel = 'Log ' + xlabel
            ylabel = 'Log ' + ylabel
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('%s, fits: c=%.2f, mu=%.2f, $R^2$=%.3f' % (plottitle, c, -1 * mu, r_squared))
        plt.legend()

    TA_results, TR_results = get_turnsegment(nicknamelist, start_bout, end_bout)
    plot_freq(TA_results, xlabel='TA segment length', plottitle=plottitle, mode='cum_freq', log_scale=log_scale)
    plot_freq(TR_results, xlabel='TR segment length', plottitle=plottitle, mode='cum_freq', log_scale=log_scale)


def get_decision_biases(tfs, re):
    # note tfs is a list of trajs
    # 4-bias from trajectories tfs during exploration, mean ± SD across nodes for each animal
    print('Four biases during exploration only, mean and std dev across all nodes')
    print('     SF             SA             BF             BS')
    # old Bf/Ba/Lf/Lo | bottom (B), left (L), or right (R); forward (f) out (o) alternating (a)
    ma = NewMaze(6)
    bi = []
    bis = []
    for i, tf in enumerate(tfs):
        cl = SplitModeClips(tf, ma, re)  # find the clips
        be, bes = ComputeFourBiasClips2(tf, ma, cl, mode=EXPLORE)  # bias using exploration only
        print('%2d' %i + ': {:5.2f} ± {:5.2f}  {:5.2f} ± {:5.2f}  {:5.2f} ± {:5.2f}  {:5.2f} ± {:5.2f}'.
              format(be[0], bes[0], be[1], bes[1], be[2], bes[2], be[3], bes[3]))
        bi += [be]
        bis += [bes]
    bi = np.array(bi)
    bis = np.array(bis)
    return bi, bis


def markov_fit_non_pooling(episodes, re):
    # Markov Fits - No Pooling
    # all animals, T-junctions, Explore, variable and fixed, test and train

    # takes some time to run

    tju = True
    exp = True
    seg = 5
    ma = NewMaze(6)

    tf = convert_episodes_to_traj_class(episodes)
    rew = re

    var = True
    hv5, cv5 = MarkovFit2(tf, ma, var, tju, exp, rew, seg, train=False)  # evaluate on testing set
    hv5tr, cv5tr = MarkovFit2(tf, ma, var, tju, exp, rew, seg, train=True)  # evaluate on training set

    var = False
    hf5, cf5 = MarkovFit2(tf, ma, var, tju, exp, rew, seg, train=False)  # evaluate on testing set
    hf5tr, cf5tr = MarkovFit2(tf, ma, var, tju, exp, rew, seg, train=True)  # evaluate on training set

    return [hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr]


def markov_fit_pooling(episodes, re):
    # Markov Fits - Pooling
    # all animals, T-junctions, Explore, variable and fixed, test and train

    # takes some time to run
    tju = True
    exp = True
    seg = 3
    ma = NewMaze(6)

    tf = convert_episodes_to_traj_class(episodes)
    rew = re

    tra = TranslLevelsLR(ma)

    var = True
    hv5, cv5 = MarkovFit3(tf, ma, var, tju, exp, rew, seg, False, tra)  # var test
    hv5tr, cv5tr = MarkovFit3(tf, ma, var, tju, exp, rew, seg, True, tra)  # var train

    var = False
    hf5, cf5 = MarkovFit3(tf, ma, var, tju, exp, rew, seg, False, tra)  # fix test
    hf5tr, cf5tr = MarkovFit3(tf, ma, var, tju, exp, rew, seg, True, tra)  # fix train

    return [hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr]


def min_cross_entropy(episodes, re, pooling):
    if not pooling:
        [hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr] = markov_fit_non_pooling(episodes, re)
    else:
        [hf5, hv5, hf5tr, hv5tr], [cf5, cv5, cf5tr, cv5tr] = markov_fit_pooling(episodes, re)

    ef = min(cf5)
    ev = min(cv5)
    return ef, ev


def outside_inside_ratio(tf, re):
    """
    Systematic node preferences: Visitations to nodes on the periphery vs visitations to inner most nodes.
    """

    def get_ratio_for_each(tf, ma):
        le = 6
        explore = 2
        ln = list(NODE_LVL[le].keys())
        cl0 = SplitModeClips(tf, ma, re=re)  # find the clips
        cn = np.cumsum(cl0[:, 2])  # cumulative number of nodes visited
        ha = np.where(cn > cn[-1] / 2)[0][0]  # index for half the number of nodes
        cl1 = cl0[:ha]
        cl2 = cl0[ha:]  # two clip arrays for first and second half of nodes
        ns = []
        for j, cl in enumerate([cl1, cl2]):
            ce = cl[np.where(cl[:, 3] == explore)]  # clips of exploration
            ne = np.concatenate(
                [tf.no[c[0]][c[1]:c[1] + c[2], 0] for c in ce])  # nodes excluding the last state in each clip
            ns += [ne[np.isin(ne, ln)]]
        _, Vis, _, _ = hist([ns[0], ns[1]], bins=np.arange(2 ** le - 1, 2 ** (le + 1)) - 0.5)
        Num = [np.sum(Vis[0]), np.sum(Vis[1])]
        Ent = [Entropy(Vis[0]), Entropy(Vis[1])]
        # print('Nodes:   1st = {:5.0f}, 2nd = {:5.0f}'.format(Num[0], Num[1]))
        # print('Entropy: 1st = {:4.3f}, 2nd = {:4.3f}'.format(Ent[0], Ent[1]))
        return [np.array(Vis)], [Ent], [Num]

    # Compute distribution of end nodes visited in first and second half of experiment
    # k = len(RewNames)
    ma = NewMaze(6)
    vi, en, nu = get_ratio_for_each(tf, ma)

    Vi = np.array(vi)  # Vi[i,j,k]=visits animal i, half j, node k
    En = np.array(en)  # En[i,j]=entropy of node visits animal i, half j
    Nu = np.array(nu)  # Nu[i,j]=total node visits animal i, half j

    # Visits to various end nodes, average across animals, by group, by half
    # Vu = np.sum(Vi, axis=(0)) / np.sum(Nu, axis=(0)).reshape(-1, 1)   # avg across animals only

    # Visits to various end nodes, average across animals
    # Vu = np.sum(Vi, axis=(0, 1)) / np.sum(Nu)

    # distribution of fraction of visits to end nodes, mean and SD across animals
    Vu = np.sum(Vi, axis=(1)) / np.sum(Nu, axis=1).reshape(-1, 1)
    Vua = np.mean(Vu, axis=0)
    Vus = np.std(Vu, axis=0)

    # Original paper
    inner = np.array([75, 76, 77, 78, 87, 88, 89, 90, 99, 100, 101, 102, 111, 112, 113, 114]) - 63
    outer = np.array([63, 65, 71, 73, 95, 97, 103, 105, 106, 109, 110, 121, 122, 125, 126, 124, 94, 92, 86, 84, 83, 80, 79, 68, 67, 64]) - 63

    # # LoS nodes
    # outer = {63, 65, 71, 73, 95, 97, 103, 105, 68, 70, 76, 78, 100, 102, 108, 110, 79, 81, 87, 89, 111, 113, 119, 121, 84, 86, 92, 94, 116, 118, 124, 126}
    # inner = set(parameters.LVL_6_NODES.keys()).difference(outer)
    # assert ((len(outer)+len(inner)) == 64)
    # assert (len(outer.intersection(inner)) == 0)
    # outer = np.array(list(outer)) - 63
    # inner = np.array(list(inner)) - 63

    # # LoS nodes without reward node and its neighbor
    # outer = {63, 65, 71, 73, 95, 97, 103, 105, 68, 70, 76, 78, 100, 102, 108, 110, 79, 81, 87, 89, 111, 113, 119, 121, 84, 86, 92, 94, 124, 126}
    # inner = set(parameters.LVL_6_NODES.keys()).difference(outer).difference({116, 118})
    # assert ((len(outer)+len(inner)) == 62)
    # assert (len(outer.intersection(inner)) == 0)
    # outer = np.array(list(outer)) - 63
    # inner = np.array(list(inner)) - 63

    ratio = np.mean(Vua[outer]) / np.mean(Vua[inner])
    print('Agent: Ratio of visits to outer vs inner leaf nodes = {:.3f}'.format(ratio))
    return ratio


def percent_turns(tf, node_seq, n1, n2, individual_plots=False):
    """
    percent turns taken at a node towards n1 and n2, when coming from HOME.
    """
    a = []
    for t in tf.no:
        if len(t[:, 0]) < len(node_seq)+1: continue
        if t[:len(node_seq), 0].tolist() == node_seq and (t[len(node_seq), 0] in [n1, n2]):
            a.append(t[len(node_seq), 0])

    # if individual_plots:
    #     fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    #     ax[0].plot(a, '*')
    #     ax[0].set_xlabel('bout')
    #     ax[0].set_ylabel(f'Go {l_node} or Go {h_node}')
    #     ax[1].set_title(f'#Left {a.count(l_node)} #Right {a.count(h_node)}, at node {turn_node} when coming from {node_seq[-2]}')

    c1 = a.count(n1)
    c2 = a.count(n2)
    total = c1+c2
    if total == 0:
        return {n1: 0, n2: 0}, total

    # if individual_plots:
    #     ax[1].bar([1, 2], [l_count, h_count])
    #     ax[1].set_xlabel(f'node {l_node} or {h_node} from node {turn_node}')
    #     ax[1].set_ylabel('Percent times')

    return {n1: c1, n2: c2}, total


def first_endnode_label(tf):
    """

    """
    first_visit_label_counts = defaultdict(int)
    k = 1
    for it, b in enumerate(tf.no):
        first_k_endnode_visits = locate_first_k_endnodes(
            b[:, 0], k, lambda x: p.node_subquadrant_dict[x] == 14)  # skip reward subq since it seems to be biased
        if not first_k_endnode_visits: continue
        v = first_k_endnode_visits[0]
        ind, node, quad, subquad, label = v
        first_visit_label_counts[label] += 1
    factor = 100.0 / sum(first_visit_label_counts.values())
    first_visit_label_fracs = {p.full_labels[k]: round(v * factor, 2) for k, v in first_visit_label_counts.items()}
    return OrderedDict(sorted(first_visit_label_fracs.items(), reverse=True))


def second_endnode_label(tf):
    """
    Where do they go after hitting the first end-node?
    > Only same sub-quadrant allowed, same node NOT allowed

    Used: All bouts just after hitting first end node as soon as it hits another end node.

    """
    label_transition_counts = defaultdict(lambda: defaultdict(int))
    k = 2
    for it, b in enumerate(tf.no):
        first_k_endnode_visits = locate_first_k_endnodes(b[:, 0], k, lambda x: p.node_subquadrant_dict[x] == 28)
        if len(first_k_endnode_visits) != k: continue
        v1, v2 = first_k_endnode_visits
        ind1, node1, quad1, subquad1, label1 = v1
        ind2, node2, quad2, subquad2, label2 = v2
        if node1 == node2: continue     # then, SKIP
        if subquad1 != subquad2: continue
        # print(v1, v2)
        label_transition_counts[label1][label2] += 1
    print(label_transition_counts)
    for i, c in label_transition_counts.items():
        factor = 100.0 / sum(c.values())
        label_transition_counts[i] = {k: round(v * factor, 2) for k, v in c.items()}  # normalize
    return label_transition_counts


# results = []
# k = len(RewNames)

# plt.rcParams.update({
#     'axes.labelsize': 9,
#     'axes.titlesize': 9,
#     'xtick.labelsize': 9,
#     'ytick.labelsize': 9,
#     'legend.fontsize': 5
# })

# for sub in RewNames+UnrewNamesSub:
#     ratio = outside_inside_ratio(convert_traj_to_episodes(LoadTrajFromPath(outdata_path + sub + "-tf")))
#     print(f'{sub}\t{ratio}')
#     results.append(ratio)

# print('rewarded', np.mean(results[:k]), np.std(results[:k]))
# print('unrewarded', np.mean(results[k:]), np.std(results[k:]))

# plt.clf()
# plt.boxplot([results[:k], results[k:]])
# plt.title('Original periphery-inner definition')
# plt.ylabel('outside/inside ratio')
# plt.ylim([0.5, 3.5])
# plt.errorbar(results[k:], 'b')
# plt.show()

# exploration_efficiency([
#     [25, 12, 5, 11, 24, 49, 99, 49, 100, 49, 24, 50, 24, 11, 5, 12, 25, 52, 25, 52, 105, 52, 25, 51, 25, 51, 25, 51,
#      104, 51, 103, 51, 104, 51, 25, 12, 5, 11, 23, 47, 96, 47, 23, 48, 97, 48, 23, 11, 5, 12, 25, 52, 105, 52, 105, 52,
#      25, 52, 25, 51, 25, 52, 25, 52, 25, 52, 25, 52, 105, 52, 25, 51, 104, 51, 25, 51, 104, 51, 25, 12, 26, 54, 26, 54,
#      110, 54, 26, 53, 108, 53, 26, 53, 107, 53, 26, 12, 25, 51, 104, 51, 25, 52, 105, 52, 106, 52, 25, 51, 104, 51, 103,
#      51, 104, 51, 103, 51, 25, 12, 26, 53, 107, 53, 26, 12, 5, 2, 6, 14]
# ], False, 4)
