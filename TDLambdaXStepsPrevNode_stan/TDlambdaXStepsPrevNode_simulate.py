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

# import sys
# module_path = '../src'
# if module_path not in sys.path:
#     sys.path.append(module_path)

from TDLambdaXStepsPrevNode_model import TDLambdaXStepsPrevNodeRewardReceived
from model_plot_utils import plot_trajectory
from parameters import HomeNode, RewardNode


def analyse(episodes_mouse, save_file_path, params):
    print("#Episodes", len(episodes_mouse))
    visit_home_node = []
    visit_reward_node = []
    time_reward_node = []

    for i, traj in enumerate(episodes_mouse):
        print(traj[:5], '...', traj[-5:])
        plot_trajectory([traj], 'all',
                        save_file_name=os.path.join(save_file_path, f'traj_{i}.png'),
                        display=False, figtitle=f'Traj {i} \n alpha beta gamma lambda:{[round(p, 3) for p in params]}')
        plt.clf()
        plt.close()

        if traj.count(HomeNode):
            visit_home_node.append(i)
        if traj.count(RewardNode):
            visit_reward_node.append(i)
            time_reward_node.append(len(traj))
        else:
            time_reward_node.append(0)

    print("visit_home_node", visit_home_node)
    plt.plot(visit_home_node, [1]*len(visit_home_node), 'y.', label='home')
    print("visit_reward_node", visit_reward_node)
    plt.plot(visit_reward_node, [1]*len(visit_reward_node), 'b.', label='reward')
    plt.legend()
    plt.savefig(os.path.join(save_file_path, f'home_and_reward_visits.png'))
    plt.show()
    print("time_reward_node", time_reward_node)
    plt.bar(range(len(time_reward_node)), time_reward_node, label='Steps to reward')
    plt.legend()
    plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_bars.png'))
    plt.show()

    actual_reward_times = list(filter(lambda x: x != 0, time_reward_node))
    plt.plot(range(len(actual_reward_times)), actual_reward_times, 'b.', label='Steps to reward')
    plt.legend()
    plt.savefig(os.path.join(save_file_path, f'reward_path_lengths_dots.png'))
    plt.show()
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


def main():
    # Load the parameters fitted by stan for each mouse
    # with open('/Volumes/ssrde-home/run2/TDlambdaXsteps_best_sub_fits.p', 'rb') as f:
    #     sub_fits = pickle.load(f)

    MAX_LENGTH = 20000
    N_BOUTS_TO_GENERATE = 1

    # Import the model class you are interested in
    model = TDLambdaXStepsPrevNodeRewardReceived()

    for mouse in range(1):
        # params = sub_fits[mouse]
        params = [0.75, 2, 0.8852032811442605, 0.4166301008224827]
        print(params)
        save_file_path = f'/Users/usingla/mouse-maze/figs/TDLambdaXStepsPrevNodeRewardReceived/MAX_LENGTH={MAX_LENGTH}/{params.__str__()}_rand{random.randint(1, 10000)}/'
        Path(save_file_path).mkdir(parents=True, exist_ok=True)
        episodes_all_mice, invalid_episodes_all_mice, success, stats = model.simulate({mouse: params}, MAX_LENGTH, N_BOUTS_TO_GENERATE)
        total_valid = len(episodes_all_mice[mouse])
        total_invalid = len(invalid_episodes_all_mice[mouse])
        print("Total valid", total_valid, "Total invalid", total_invalid)
        #
        episodes_mouse = episodes_all_mice[mouse]
        # with open(os.path.join(save_file_path, f'episodes_{mouse + 1}_{params.__str__()}.json')) as f:
        #     episodes_mouse = json.load(f)
        if True:
            with open(os.path.join(save_file_path,
                                   f'episodes_{mouse + 1}_{params.__str__()}.json'),
                      'w') as f:
                json.dump(episodes_mouse, f)
            # print(episodes_mouse)
            analyse(episodes_mouse, save_file_path, params)
        return


if __name__ == '__main__':
    main()
