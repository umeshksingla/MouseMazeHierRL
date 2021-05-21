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
from multiprocessing import Pool


import sys
module_path = '../src'
if module_path not in sys.path:
    sys.path.append(module_path)

from TDLambdaXStepsPrevNode_model import TDLambdaXStepsPrevNodeRewardReceived
from model_plot_utils import plot_trajectory
from parameters import HomeNode, RewardNode


def analyse(episodes_mouse, save_file_path, params):
    print("#Episodes", len(episodes_mouse))
    title_params = [round(p, 3) for p in params]
    visit_home_node = []
    visit_reward_node = []
    time_reward_node = []

    for i, traj in enumerate(episodes_mouse):
        print(i, ":", traj[:5], '...', traj[-5:])
        if traj.count(HomeNode):
            visit_home_node.append(i)
        if traj.count(RewardNode):
            visit_reward_node.append(i)
            time_reward_node.append(len(traj))
        else:
            time_reward_node.append(0)

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
    # print("time_reward_node", time_reward_node)
    plt.bar(range(len(time_reward_node)), time_reward_node, label='Steps to reward')
    plt.legend()
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("reward")
    plt.ylabel("number of steps")
    plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_bars.png'))
    # plt.show()
    plt.clf()
    plt.close()
    print("bout_lengths")
    plt.bar(range(len(episodes_mouse)), [len(e) for e in episodes_mouse], label='Bout length; from home to home')
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("time")
    plt.ylabel("number of steps")
    plt.legend()
    plt.savefig(os.path.join(save_file_path, f'bout_lengths_bars.png'))
    # plt.show()
    plt.clf()
    plt.close()
    actual_reward_times = list(filter(lambda x: x != 0, time_reward_node))
    plt.plot(range(len(actual_reward_times)), actual_reward_times, 'b.', label='Steps to reward')
    plt.title(f'alpha, beta, gamma, lambda = {title_params}')
    plt.xlabel("number of rewards")
    plt.ylabel("number of steps")
    plt.legend()
    plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_dots.png'))
    # plt.show()

    plt.clf()
    plt.close()

    # try fitting an exp decay
    # x = np.arange(len(actual_reward_times))
    # print(x)
    # y = np.max(actual_reward_times)*np.exp(-x/2) + np.min(actual_reward_times)
    # plt.plot(x, y)
    # plt.show()

    for i, traj in enumerate(episodes_mouse):
        print(i, ":", traj[:5], '...', traj[-5:])
        if len(episodes_mouse) >= 100 and i >=10 and i%5 != 0:
            # save every 5th graph when lots of episodes
            continue
        plot_trajectory([traj], 'all',
                        save_file_name=os.path.join(save_file_path, f'traj_{i}.png'),
                        display=False, figtitle=f'Traj {i} \n alpha, beta, gamma, lambda = {title_params}')
        plt.clf()
        plt.close()

    return

    # plt.plot(episodes_all_mice[mouse].keys(), [len(path) for e, path in episodes_all_mice[mouse].items()], 'b*')
    # plt.ylabel("Reward Path length")
    # plt.xlabel("time")
    # # plt.savefig(os.path.join(save_file_path, f'pathlength_vs_time_{mouse+1}.png'))
    # # plt.show()
    # plt.clf()
    # plt.close()
    #
    # bins = MAX_LENGTH
    # plt.hist([len(path) for e, path in episodes_all_mice[mouse].items()], bins, density=True)
    # plt.ylabel("Fraction")
    # plt.xlabel("Path length")
    # plt.title("Valid episodes: path length")
    # # plt.savefig(os.path.join(save_file_path, f'valid_epi_pathlength_dist_{mouse + 1}.png'))
    # # plt.show()
    # plt.clf()
    # plt.close()
    #
    # plt.hist([len(path) for e, path in invalid_episodes_all_mice[mouse].items()], bins, density=True)
    # plt.ylabel("Fraction")
    # plt.xlabel("Path length")
    # plt.title("Invalid episodes: path length")
    # # plt.savefig(os.path.join(save_file_path, f'invalid_epi_pathlength_dist_{mouse + 1}.png'))
    # # plt.show()
    # plt.clf()
    # plt.close()
    #
    # bins = MAX_LENGTH//5
    # plt.hist([len(path) for e, path in episodes_all_mice[mouse].items()], bins, density=True)
    # plt.ylabel("Fraction")
    # plt.xlabel("Path length")
    # plt.title("Valid episodes: path length")
    # # plt.savefig(os.path.join(save_file_path, f'valid_epi_pathlength_dist_zoomout_{mouse + 1}.png'))
    # # plt.show()
    # plt.clf()
    # plt.close()
    #
    # plt.hist([len(path) for e, path in invalid_episodes_all_mice[mouse].items()], bins, density=True)
    # plt.ylabel("Fraction")
    # plt.xlabel("Path length")
    # plt.title("Invalid episodes: path length")
    # # plt.savefig(os.path.join(save_file_path, f'invalid_epi_pathlength_dist_zoomout_{mouse + 1}.png'))
    # # plt.show()
    # plt.clf()
    # plt.close()
    #
    # print(stats)
    # total = stats[mouse]['count_total']
    #
    # initial_state_valid_counts = defaultdict(int)
    # for i, epi in episodes_all_mice[mouse].items():
    #     if epi:
    #         initial_state_valid_counts[epi[0]] += 1
    # initial_state_valid_counts = dict([(s, round(c/total * 100)) for s, c in initial_state_valid_counts.items()])
    # print("Led to valid episodes:", initial_state_valid_counts)
    # plt.plot(initial_state_valid_counts.keys(),
    #          initial_state_valid_counts.values(), 'b.', label='valid')
    #
    # initial_state_invalid_counts = defaultdict(int)
    # for i, epi in invalid_episodes_all_mice[mouse].items():
    #     if epi:
    #         initial_state_invalid_counts[epi[0]] += 1
    # initial_state_invalid_counts = dict([(s, round(c/total * 100)) for s, c in initial_state_invalid_counts.items()])
    # print("Led to invalid episodes:", initial_state_invalid_counts)
    #
    # plt.plot(initial_state_invalid_counts.keys(),
    #          initial_state_invalid_counts.values(), 'r.', label='invalid')
    # plt.title("Percent of valid or invalid episodes for an initial state")
    # plt.xlabel("initial state")
    # # plt.savefig(os.path.join(save_file_path, f'initial_state_valid_percent_{mouse + 1}.png'))
    # plt.legend()
    # # plt.show()
    # plt.clf()
    # plt.close()
    #
    # print("========================================")


def main(params):
    info('function main')
    # Load the parameters fitted by stan for each mouse
    # with open('/Volumes/ssrde-home/run2/TDlambdaXsteps_best_sub_fits.p', 'rb') as f:
    #     sub_fits = pickle.load(f)

    MAX_LENGTH = 10000
    N_BOUTS_TO_GENERATE = 1

    # Import the model class you are interested in
    model = TDLambdaXStepsPrevNodeRewardReceived()

    mouse = 0
    # params = sub_fits[mouse]
    print("params:", params)
    save_file_path = f'/Users/usingla/mouse-maze/figs/TDLambdaXStepsPrevNodeRewardReceived/MAX_LENGTH={MAX_LENGTH}/{params.__str__()}_rand{random.randint(1, 10000)}/'
    print(save_file_path)
    Path(save_file_path).mkdir(parents=True, exist_ok=True)
    episodes_all_mice, invalid_episodes_all_mice, loglikelihoods, success, stats = model.simulate({mouse: params}, MAX_LENGTH, N_BOUTS_TO_GENERATE)
    total_valid = len(episodes_all_mice[mouse])
    total_invalid = len(invalid_episodes_all_mice[mouse])
    print("Total valid", total_valid, "Total invalid", total_invalid)

    episodes_mouse = episodes_all_mice[mouse]
    LL = loglikelihoods[mouse]
    if success:
        print(len(episodes_mouse))
        with open(os.path.join(save_file_path,
                               f'episodes_{mouse + 1}_{params.__str__()}_LL={LL}.json'),
                  'w') as f:
            json.dump(episodes_mouse, f)
        print(episodes_mouse)
        analyse(episodes_mouse, save_file_path, params)
        print(">>> Done with param_set!", params)
    return f"Finished {params}"


def info(title):
    print(title)
    print('>>> module name:', __name__, 'parent process id:', os.getppid(), 'process id:', os.getpid())


def load(save_file_dir, epi_dir, epi_file_name, params=None):
    if not params:
        params = json.loads(epi_dir.split('_', 1)[0])
    print("load params", params)
    with open(os.path.join(save_file_dir, epi_dir, epi_file_name)) as f:
        episodes_mouse = json.load(f)
    print(len(episodes_mouse))
    analyse(episodes_mouse, os.path.join(save_file_dir, epi_dir), params)
    return


if __name__ == '__main__':
    # param_sets = []
    # for alpha in [0.15, 0.75]:
    #     for beta in [10, 50]:
    #         for lamda in [0.2, 0.4]:
    #             param_sets.append([alpha, beta, 0.89, lamda])
    # param_sets = [[0.75, 50, 0.89, 0.2]]
    # print(param_sets)
    # print(len(param_sets))
    # with Pool(4) as p:
    #     print(p.map(main, param_sets))

    # load episodes file if you want to analyse prev run data
    save_file_dir = "/Users/usingla/mouse-maze/figs/TDLambdaXStepsPrevNodeRewardReceived/MAX_LENGTH=20000"
    epi_dir = "[0.3, 3, 0.89, 0.3]_rand8227"
    epi_file_name = "episodes_1_[0.3, 3, 0.89, 0.3]_LL=-16122.135425940007.json"
    load(save_file_dir, epi_dir, epi_file_name)
    pass
