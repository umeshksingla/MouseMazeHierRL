import os
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import ast
import glob
import re

from MM_Maze_Utils import NewMaze
from MM_Traj_Utils import SplitModeClips
from decision_bias_analysis_tools import ComputeFourBiasClips2
from plot_utils import plot_trajs, plot_episode_lengths, \
    plot_exploration_efficiency, plot_maze_stats, plot_visit_freq, plot_decision_biases
import evaluation_metrics as em
from utils import convert_episodes_to_traj_class


def analyse_episodes(stats, save_file_path, params):
    """todo
    episodes:

    """
    episodes = stats["episodes_positions"]
    # plot_reward_path_lengths(episodes, params, save_file_path)
    plot_episode_lengths(episodes, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(episodes, re=False, title=params, save_file_path=save_file_path)
    plot_visit_freq(stats["visit_frequency"], title=params, save_file_path=save_file_path)
    plot_maze_stats(stats["visit_frequency"], interpolate_cell_values=True, colormap_name='Blues',
                    colorbar_label="visit freq",
                    save_file_name=os.path.join(save_file_path, f'visit_frequency_maze.png'),
                    display=False,
                    figtitle=f'state visit freq\n{params}')

    plot_trajs(episodes, save_file_path, params)
    return


def analyse_state_values(model, V, save_file_path, params):
    """todo
    model:
    V:
    """
    state_values = model.get_maze_state_values(V)
    print("state_values", state_values)
    plot_maze_stats(state_values, interpolate_cell_values=True,
                    save_file_name=os.path.join(save_file_path, f'state_values.png'),
                    display=False,
                    figtitle=f'state values\n{params}')
    return


def load(save_file_path):
    """
    Takes episodes pickles from a local directory and plots exp efficiency for
    all of them on one graph. To be extended to other metrics.
    """

    episode_files_list = glob.glob(save_file_path + '/*_rand*/episodes*')
    episode_files_list = sorted(episode_files_list)
    c = 0

    plt.figure(figsize=(20, 10))
    plt.xscale('log', base=10)
    colormap = plt.cm.gist_ncar

    for i, each in enumerate(episode_files_list):
        print("Loading", each)
        match = re.search(r'episodes_(\d+)_(\{.*})_LL', each)
        agent_id = int(match.group(1))
        params = ast.literal_eval(match.group(2))
        print(agent_id, " == ", params)

        label = f'td-opt-{params["lamda"]}-lamda-{params["gamma"]}-gamma'
        print("agentId", agent_id, ":", label)

        with open(os.path.join(each), 'rb') as f:
            stats = pickle.load(f)
        episodes = stats["episodes"]
        new_end_nodes_found = em.exploration_efficiency(episodes, re=False)
        plt.plot(new_end_nodes_found.keys(), new_end_nodes_found.values(),
                 color=colormap(0.2 + (0.03) * c), linestyle='-', marker="o",
                 label=label)
        c += 1
        print()

    # DFS
    new_end_nodes_found_dfs = em.get_dfs_ee()
    plt.plot(new_end_nodes_found_dfs.keys(), new_end_nodes_found_dfs.values(), 'black', label='DFS')

    # one unrewarded animal
    new_end_nodes_found_unrew = em.get_unrewarded_ee()
    plt.plot(new_end_nodes_found_unrew.keys(),
             new_end_nodes_found_unrew.values(), color=colormap(0),
             linestyle='-.', label='Unrewarded: B5')

    # one rewarded animal
    new_end_nodes_found_rew = em.get_rewarded_ee()
    plt.plot(new_end_nodes_found_rew.keys(), new_end_nodes_found_rew.values(),
             'ro-', label='Rewarded: B1')
    plt.title("exploration efficiency as defined in orig paper")
    plt.xlabel("end nodes visited")
    plt.ylabel("new end nodes found")
    plt.legend()

    # sort legend labels with some logic if you need to
    # handles, labels = plt.gca().get_legend_handles_labels()
    # print(labels)
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0].split('-')[2]) if len(t[0].split('-')) > 2 else 1))
    # plt.legend(handles, labels)
    plt.show()
    return


def run(model, params_all, base_path, MAX_LENGTH, N_BOUTS_TO_GENERATE):

    # N_SIMULATIONS = 5
    # tfs = list()
    # for _ in range(N_SIMULATIONS):
    # TODO: when this is working, just ident all the code until the line that starts with tmp_tf
    #  this code with two loops is bad, but it should work. For long term use, please be kind to our future selves and do
    #  something decent that won't take much time to code.

    simulation_results = model.simulate_multiple(params_all, MAX_LENGTH=MAX_LENGTH, N_BOUTS_TO_GENERATE=N_BOUTS_TO_GENERATE)
    # analyse results
    for agent_id, params in params_all.items():
        print("params:", params)
        success, stats = simulation_results[agent_id]
        episodes = stats["episodes_positions"]
        LL = stats["LL"]
        V = stats["V"]
        if success:
            print("#Episodes: ", len(episodes))
            save_file_path = f'{base_path}/' \
                             f'{model.__class__.__name__}/' \
                             f'MAX_LENGTH={MAX_LENGTH}/' \
                             f'{params.__str__()}_rand{np.random.randint(1, 10000)}/'
            Path(save_file_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(save_file_path, f'episodes_{agent_id}_{params.__str__()}_LL={LL}.pkl'), 'wb') as f:
                pickle.dump(stats, f)
            # print("episodes", episodes)
            analyse_state_values(model, V, save_file_path, params)
            analyse_episodes(stats, save_file_path, params)
            print(">>> Done with params!", params, "- Check results at:", save_file_path)

    #             tmp_tf = convert_episodes_to_traj_class(stats["episodes"], stats["episodes_states"])
    #             tfs.append(tmp_tf)
    #
    # plot_decision_biases(tfs)

    return


if __name__ == '__main__':
    # np.random.seed(0)
    MAX_LENGTH = 4000
    N_BOUTS_TO_GENERATE = 1

    # base path to save figs or other results in
    base_path = '/Users/usingla/mouse-maze/figs'

    # 1. Load prev simulation(s) results saved locally
    # load(base_path + f'/TDLambdaOptimisticInitialization/MAX_LENGTH={MAX_LENGTH}/')
    # quit()

    # OR, 2. Run a new simulation

    # Import the model class you are interested in
    # from TD_UCB_model import TD_UCBpolicy
    # from Dyna_Qplus import DynaQPlus
    # model = DynaQPlus()
    from TDLambdaOptimisticInitialization import TDLambdaOptimisticInitialization
    model = TDLambdaOptimisticInitialization()
    param_sets = {

        # param_sets to try TDLambda+OptimisticInitialization
        # 1: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.0, "epsilon": 0.0},
        2: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "epsilon": 0.1},
        # 3: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.1, "epsilon": 0.0},
        # 4: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.5, "epsilon": 0.0},

        # param_sets to try dynaQ+
        # 12: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
        #      "epsilon": 0.0, "n_plan": 50000, "back_action": True},
        # 13: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
        #      "epsilon": 0.1, "n_plan": 50000, "back_action": True},
        # 14: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
        #      "epsilon": 0.5, "n_plan": 50000, "back_action": True},
        # 15: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
        #      "epsilon": 0.8, "n_plan": 50000, "back_action": True},

    }
    run(model, param_sets, base_path, MAX_LENGTH, N_BOUTS_TO_GENERATE)
