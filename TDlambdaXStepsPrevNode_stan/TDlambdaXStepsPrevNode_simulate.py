"""
Post-training analysis on predicted trajectories, generating trajectories
and playing around.

Intended as a custom script file, and not part of the main source code.

"""

import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import random
import numpy as np
import re
from multiprocessing import Pool
from functools import reduce
from sklearn.manifold import TSNE


import sys
module_path = '../src'
if module_path not in sys.path:
    sys.path.append(module_path)

from TDLambdaXStepsPrevNode_model import TDLambdaXStepsPrevNodeRewardReceived
from plot_utils import plot_trajectory, plot_maze_stats
from parameters import HomeNode, RewardNode, InvalidState, WaterPortNode
import evaluation_metrics as em
from BaseModel import BaseModel


def get_reward_times(episodes_mouse):
    visit_home_node = []
    visit_reward_node = []
    time_reward_node = []
    for i, traj in enumerate(episodes_mouse):
        # print(i, ":", traj[:5], '...', traj[-5:])
        if traj.count(HomeNode):
            visit_home_node.append(i)
        if traj.count(RewardNode):
            visit_reward_node.append(i)
            time_reward_node.append(len(traj))
    return visit_home_node, visit_reward_node, time_reward_node


def plot_visits(visit_home_node, visit_reward_node, save_file_path, title_params):
    # print("visit_home_node", visit_home_node)
    plt.plot(visit_home_node, [1]*len(visit_home_node), 'y.', label='home')
    # print("visit_reward_node", visit_reward_node)
    plt.plot(visit_reward_node, [2]*len(visit_reward_node), 'b.', label='reward')
    plt.legend()
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("time")
    plt.savefig(os.path.join(save_file_path, f'home_and_reward_visits.png'))
    # plt.show()
    plt.clf()
    plt.close()
    return


def plot_reward_path_lengths(time_reward_node, save_file_path, title_params, dots=True):
    # print("time_reward_node", time_reward_node)
    if dots:
        plt.plot(time_reward_node, 'b.', label='Steps to reward')
    else:
        plt.plot(time_reward_node, 'b-', label='Steps to reward')
    plt.legend()
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("reward")
    plt.ylabel("number of steps")
    plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_dots.png'))
    # plt.show()
    plt.clf()
    plt.close()
    plt.bar(range(len(time_reward_node)), time_reward_node, label='Steps to reward')
    plt.legend()
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("reward")
    plt.ylabel("number of steps")
    plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_bars.png'))
    # plt.show()
    plt.clf()
    plt.close()
    return


def plot_bout_lengths(episodes_mouse, save_file_path, title_params):
    print("bout_lengths")
    plt.bar(range(len(episodes_mouse)), [len(e) for e in episodes_mouse],
            label='Bout length; from home to home')
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("time")
    plt.ylabel("number of steps")
    plt.legend()
    plt.savefig(os.path.join(save_file_path, f'bout_lengths_bars.png'))
    # plt.show()
    plt.clf()
    plt.close()


def plot_trajs(episodes_mouse, save_file_path, title_params):
    for i, traj in enumerate(episodes_mouse):
        print(i, ":", traj[:5], '...', traj[-5:])
        if len(episodes_mouse) >= 100 and i >=10 and i%5 != 0:
            # save every 5th graph when lots of episodes
            continue
        plot_trajectory([traj], 'all',
                        save_file_name=os.path.join(save_file_path, f'traj_{i}.png'),
                        display=False,
                        figtitle=f'Traj {i} \n alpha, beta, gamma, lambda = {title_params}')
        plt.clf()
        plt.close()
    return


def analyse_episodes(model, episodes_mouse, save_file_path, params):
    print("#Episodes", len(episodes_mouse))
    title_params = [round(p, 3) for p in params]

    visit_home_node, visit_reward_node, time_reward_node = get_reward_times(episodes_mouse)
    new_end_nodes_found = em.exploration_efficiency(episodes_mouse)
    plot_visits(visit_home_node, visit_reward_node, save_file_path, title_params)
    plot_reward_path_lengths(time_reward_node, save_file_path, title_params)
    plot_bout_lengths(episodes_mouse, save_file_path, title_params)
    plot_exploration_efficiency(new_end_nodes_found, save_file_path, title_params)

    # try fitting an exp decay on reward times
    # x = np.arange(len(actual_reward_times))
    # print(x)
    # y = np.max(actual_reward_times)*np.exp(-x/2) + np.min(actual_reward_times)
    # plt.plot(x, y)
    # plt.show()

    plot_trajs(episodes_mouse, save_file_path, title_params)
    return


def analyse_state_values(model, V, save_file_path, title_params):
    state_values = np.zeros(128)
    state_values_1 = np.zeros(128)
    for n in range(128):
        possible_states = model.get_SAnodemap()[n, :]
        print(n, possible_states)
        possible_states = list(filter(
            lambda p: p != InvalidState and p != WaterPortNode and p!= HomeNode,
            possible_states))
        pos_state_values = [V[model.get_number_from_node_tuple((p, n))] for p in possible_states]
        print(n, possible_states, pos_state_values)
        state_values[n] = np.nanmean(pos_state_values)

        pos_state_values_1 = [V[model.get_number_from_node_tuple((n, p))] for p in possible_states]
        print(n, possible_states, pos_state_values_1)
        state_values_1[n] = np.nanmean(pos_state_values_1)

    print(state_values)
    plot_maze_stats(state_values, datatype="state_values",
                    save_file_name=os.path.join(save_file_path, f'state_values.png'),
                    display=False,
                    figtitle=f'state values \n alpha, beta, gamma, lambda = {title_params}')
    plt.clf()
    plt.close()
    print(state_values_1)
    plot_maze_stats(state_values_1, datatype="state_values",
                    save_file_name=os.path.join(save_file_path, f'state_values_1.png'),
                    display=False,
                    figtitle=f'state values \n alpha, beta, gamma, lambda = {title_params}')
    plt.clf()
    plt.close()
    return


def plot_exploration_efficiency(new_end_nodes_found, save_file_path, title_params):
    """
    new_end_nodes_found: dict() of how many steps -> how many distinct end nodes
    """
    plt.plot(new_end_nodes_found.keys(), new_end_nodes_found.values(), 'o-')
    plt.xscale('log', base=10)

    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("end nodes visited")
    plt.ylabel("new end nodes found")
    plt.savefig(os.path.join(save_file_path, f'exp_efficiency.png'))

    plt.show()
    plt.clf()
    plt.close()
    return


def compare_mouse_and_simulated(real_episodes, simulated_episodes):

    simulated_features = em.get_feature_vectors(simulated_episodes)
    print("simulated_features", simulated_features.shape)
    print(simulated_features)

    mouse_features = em.get_feature_vectors(real_episodes)
    print("mouse_features", mouse_features.shape)
    print(mouse_features)

    tsne = TSNE()
    tsne_results = tsne.fit_transform(simulated_features)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label='TDlambda')

    tsne_1 = TSNE()
    tsne_results_1 = tsne_1.fit_transform(mouse_features)
    plt.scatter(tsne_results_1[:, 0], tsne_results_1[:, 1], label='Real mouse')
    plt.xlabel('t-sne1')
    plt.ylabel('t-sne2')

    plt.legend()
    plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot()
    # f = 1
    ax.scatter(
        simulated_features[:, 0],
               simulated_features[:, 1],
        #         np.arange(len(simulated_features[:, f])),
               simulated_features[:, 2],
               color='r', label='TDlambda')
    ax.scatter(
        mouse_features[:, 0],
               mouse_features[:, 1],
        # np.arange(len(mouse_features[:, f])),
               mouse_features[:, 2],
               color='y', label='Mouse')
    # ax.set_xlabel('episodes')
    ax.set_xlabel('mean rotational_velocity')
    ax.set_ylabel('mean diffusivity')
    ax.set_zlabel('tortuosity')
    plt.legend()
    plt.show()
    return


def analyse_avg(model, V_all, reward_lengths_all, new_end_nodes_found_all, simulated_episodes, save_file_path, params):
    print(">>> Onto average analysis")
    median = np.nanmedian(reward_lengths_all, axis=0)
    # for arr in reward_lengths_all:
    #     plt.plot(arr[arr > 0], 'c.')
    # plot_reward_path_lengths(median[median > 0], save_file_path, params, dots=False)
    plt.clf()
    plt.close()

    new_end_nodes_found_avg = dict()
    for s, l in new_end_nodes_found_all.items():
        new_end_nodes_found_avg[s] = np.nanmean(l)

    plot_exploration_efficiency(new_end_nodes_found_avg, save_file_path, params)

    real_episodes = model.extract_trajectory_data()
    print(len(real_episodes))
    real_episodes = reduce(lambda x, y: x+y, real_episodes)
    # print(len(real_episodes))
    print(len(simulated_episodes))
    simulated_episodes = reduce(lambda x, y: x+y, simulated_episodes)
    # print(len(simulated_episodes)
    # print(simulated_episodes)
    # for r, s in zip([real_episodes], [simulated_episodes]):
    #     print("mouse i")
    #     compare_mouse_and_simulated([e[:model.X] for e in r], [e[:model.X] for e in s if e[-1] == RewardNode])
        # break

    # V_avg = np.nanmedian(V_all, axis=0)
    # analyse_state_values(model, V_avg, save_file_path, params)
    # analyse_state_values(model, V_avg, save_file_path, params)
    # from MM_Traj_Utils import PlotMazeFunction
    # from MM_Maze_Utils import  NewMaze
    # ma = NewMaze()
    # plot_maze_stats([e[:model.X] for e in simulated_episodes if e[-1] == RewardNode], 'states', plt.cm.get_cmap())
    return


def run(params_all):
    # Load the parameters fitted by stan for each mouse
    # with open('/Volumes/ssrde-home/run2/TDlambdaXsteps_best_sub_fits.p', 'rb') as f:
    #     params_all = pickle.load(f)

    MAX_LENGTH = 500000
    N_BOUTS_TO_GENERATE = 1

    # Import the model class you are interested in
    model = TDLambdaXStepsPrevNodeRewardReceived()
    simulation_results = model.simulate_multiple(params_all,
                                                 n=len(params_all),
                                                 MAX_LENGTH=MAX_LENGTH/10,
                                                 N_BOUTS_TO_GENERATE=N_BOUTS_TO_GENERATE)
    # analyse results
    for mouse, params in params_all.items():
        print("params:", params)
        save_file_path = f'/Users/usingla/mouse-maze/figs/' \
                         f'TDLambdaXStepsPrevNodeRewardReceived/' \
                         f'MAX_LENGTH={MAX_LENGTH}/' \
                         f'{params.__str__()}_rand{random.randint(1, 10000)}/'
        print(save_file_path)
        Path(save_file_path).mkdir(parents=True, exist_ok=True)
        success, stats = simulation_results[mouse]
        episodes_mouse = stats["episodes"]
        LL = stats["LL"]
        V = stats["V"]

        if success:
            print(len(episodes_mouse))
            with open(os.path.join(save_file_path,
                                   f'episodes_{mouse}_{params.__str__()}_LL={LL}.pkl'),
                      'wb') as f:
                pickle.dump(stats, f)
            # print(episodes_mouse)
            analyse_state_values(model, V, save_file_path, params)
            analyse_episodes(model, episodes_mouse, save_file_path, params)
        print(">>> Done with params!", params)

    return


def load_multiple(save_file_path):
    model = TDLambdaXStepsPrevNodeRewardReceived()
    import glob
    episode_files_list = glob.glob(save_file_path + '/*_rand*/episodes*')

    max_rewards = 200
    reward_lengths_all_matrix = np.ones((len(episode_files_list), max_rewards)) * -1
    V_all = np.ones((len(episode_files_list), model.S+1))
    new_end_nodes_found_all = defaultdict(list)
    data = episode_files_list
    all_episodes_agents = []
    for each in data:
        print("Loading", each)
        match = re.search(r'episodes_(\d+)_(\[.*])_LL', each)
        print(match.groups())
        agent_id = int(match.group(1))
        params = json.loads(match.group(2))
        print("agent_id", agent_id, "load params", params)
        with open(os.path.join(each), 'rb') as f:
            stats = pickle.load(f)
        episodes = stats["episodes"]
        LL = stats["LL"]
        V = stats["V"]
        print(len(episodes))
        visit_home_node, visit_reward_node, time_reward_node = get_reward_times(episodes)
        print(">>> Number of rewards", len(time_reward_node))
        reward_lengths_all_matrix[agent_id, :] = np.pad(
            time_reward_node, (0, max_rewards-len(time_reward_node)),
            'constant', constant_values=-1)
        V_all[agent_id, :] = V

        for s, v in em.exploration_efficiency(episodes).items():
            new_end_nodes_found_all[s].append(v)
        all_episodes_agents.append(episodes)
    analyse_avg(model, V_all, reward_lengths_all_matrix, new_end_nodes_found_all, all_episodes_agents, save_file_path, [])
    return


if __name__ == '__main__':
    # param_sets = []
    # for alpha in [0.15, 0.75]:
    #     for beta in [10, 50]:
    #         for lamda in [0.2, 0.4]:
    #             param_sets.append([alpha, beta, 0.89, lamda])
    # param_sets = dict([(i, [0.1, 3, 0.89, 0.7]) for i in range(1)])
    # param_sets = dict([(0, [0.1, 3, 0.89, 0.7]), (1, [0.3, 3, 0.89, 0.3])])
    # run(param_sets)

    # load episodes file if you want to analyse prev run data
    save_file_path = "/Users/usingla/mouse-maze/figs/TDLambdaXStepsPrevNodeRewardReceived/MAX_LENGTH=310000/2"
    load_multiple(save_file_path)
