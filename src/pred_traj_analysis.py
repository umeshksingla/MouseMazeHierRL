"""
Post-training analysis on predicted trajectories

Really intended as a custom script file, and not part of the main source code.

"""

import pickle
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived
from model_plot_utils import plot_trajectory
from pathlib import Path
import os
import matplotlib.pyplot as plt
from collections import defaultdict

model = TDLambdaXStepsRewardReceived()
with open('/Volumes/ssrde-home/run2/TDlambdaXsteps_best_sub_fits.p', 'rb') as f:
    sub_fits = pickle.load(f)

area = 'endnodes-1'
MAX_LENGTH = 50
save_file_path = f'/Users/usingla/mouse-maze/figs/lamda_as_param/MAX_LENGTH={MAX_LENGTH}/{area}/'
Path(save_file_path).mkdir(parents=True, exist_ok=True)

for mouse in range(10):
    print(sub_fits[mouse])
    episodes_all_mice, invalid_episodes_all_mice, success, stats = model.simulate({mouse: sub_fits[mouse]}, MAX_LENGTH)
    total_valid = len(episodes_all_mice[mouse])
    total_invalid = len(invalid_episodes_all_mice[mouse])
    print("Total valid", total_valid, "Total invalid", total_invalid)
    if success:
        plot_trajectory(episodes_all_mice[mouse], 'all',
                        save_file_name=os.path.join(save_file_path, f'all_valid_traj_{mouse+1}.png'),
                        display=False)
        plt.clf()
        plt.close()

        plot_trajectory(invalid_episodes_all_mice[mouse], 'all',
                        save_file_name=os.path.join(save_file_path,
                                                    f'all_invalid_traj_{mouse + 1}.png'),
                        display=False)
        plt.clf()
        plt.close()

    plt.plot(episodes_all_mice[mouse].keys(), [len(path) for e, path in episodes_all_mice[mouse].items()], 'b*')
    plt.ylabel("Reward Path length")
    plt.xlabel("time")
    plt.savefig(os.path.join(save_file_path, f'pathlength_vs_time_{mouse+1}.png'))
    # plt.show()
    plt.clf()
    plt.close()

    bins = MAX_LENGTH
    plt.hist([len(path) for e, path in episodes_all_mice[mouse].items()], bins, density=True)
    plt.ylabel("Fraction")
    plt.xlabel("Path length")
    plt.title("Valid episodes: path length")
    plt.savefig(
        os.path.join(save_file_path, f'valid_epi_pathlength_dist_{mouse + 1}.png'))
    # plt.show()
    plt.clf()
    plt.close()

    plt.hist([len(path) for e, path in invalid_episodes_all_mice[mouse].items()], bins, density=True)
    plt.ylabel("Fraction")
    plt.xlabel("Path length")
    plt.title("Invalid episodes: path length")
    plt.savefig(
        os.path.join(save_file_path, f'invalid_epi_pathlength_dist_{mouse + 1}.png'))
    # plt.show()
    plt.clf()
    plt.close()

    bins = MAX_LENGTH//5
    plt.hist([len(path) for e, path in episodes_all_mice[mouse].items()], bins, density=True)
    plt.ylabel("Fraction")
    plt.xlabel("Path length")
    plt.title("Valid episodes: path length")
    plt.savefig(
        os.path.join(save_file_path, f'valid_epi_pathlength_dist_zoomout_{mouse + 1}.png'))
    # plt.show()
    plt.clf()
    plt.close()

    plt.hist([len(path) for e, path in invalid_episodes_all_mice[mouse].items()], bins, density=True)
    plt.ylabel("Fraction")
    plt.xlabel("Path length")
    plt.title("Invalid episodes: path length")
    plt.savefig(
        os.path.join(save_file_path, f'invalid_epi_pathlength_dist_zoomout_{mouse + 1}.png'))
    # plt.show()
    plt.clf()
    plt.close()

    print(stats)
    total = stats[mouse]['count_total']

    initial_state_valid_counts = defaultdict(int)
    for i, epi in episodes_all_mice[mouse].items():
        if epi:
            initial_state_valid_counts[epi[0]] += 1
    initial_state_valid_counts = dict([(s, round(c/total * 100)) for s, c in initial_state_valid_counts.items()])
    print("Led to valid episodes:", initial_state_valid_counts)
    plt.plot(initial_state_valid_counts.keys(),
             initial_state_valid_counts.values(), 'b.', label='valid')

    initial_state_invalid_counts = defaultdict(int)
    for i, epi in invalid_episodes_all_mice[mouse].items():
        if epi:
            initial_state_invalid_counts[epi[0]] += 1
    initial_state_invalid_counts = dict([(s, round(c/total * 100)) for s, c in initial_state_invalid_counts.items()])
    print("Led to invalid episodes:", initial_state_invalid_counts)

    plt.plot(initial_state_invalid_counts.keys(),
             initial_state_invalid_counts.values(), 'r.', label='invalid')
    plt.title("Percent of valid or invalid episodes for an initial state")
    plt.xlabel("initial state")
    plt.savefig(
        os.path.join(save_file_path, f'initial_state_valid_percent_{mouse + 1}.png'))
    plt.legend()
    # plt.show()
    plt.clf()
    plt.close()

    print("========================================")

'''
# generated

10 quad1-1
10 quad1-2
10 zero-1
10 zero-2
10 endnodes-1
10 no28_57_115_116-1
10 no28_57_115_116-2

25 'quad1-1'
25 'quad1-2'
25 zero-1
25 zero-2
25 no28_57_115_116-1
25 no28_57_115_116-2

50 'quad1-1'
50 'quad1-2'
50 zero-1
50 zero-2
50 endnodes-1

100 'quad1-1'
100 'quad1-2'
100 zero-1
100 zero-2
100 no28_57_115_116-1
100 no28_57_115_116-2
100 endnodes-1

200 'quad1-1'
200 'quad1-2'
200 'zero-1'
200 no28_57_115_116-1
200 no28_57_115_116-2
200 endnodes-1
'''
