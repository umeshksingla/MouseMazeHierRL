import os
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import ast
import glob
import re

from plot_utils import plot_trajs, plot_episode_lengths, \
    plot_exploration_efficiency, plot_maze_stats, plot_visit_freq, \
    plot_decision_biases, plot_markov_fit_pooling, plot_markov_fit_non_pooling,\
    plot_trajectory_features, plot_reward_path_lengths, plot_outside_inside_ratio, \
    plot_percent_turns, plot_opposite_node_preference, plot_first_endnode_labels, \
    plot_end_node_revisits_level_halves, plot_end_node_revisits_level_all_time, \
    plot_unique_node_revisits_level_halves, plot_node_revisits_level_halves
import evaluation_metrics as em
from utils import convert_episodes_to_traj_class


def analyse_state_values(model, V, save_file_path, params):
    """todo
    model:
    V:
    """
    state_values = model.get_maze_state_values(V)
    # print("state_values", state_values)
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

    plt.figure(figsize=(20, 10))
    plt.xscale('log', base=10)
    colormap = plt.cm.gist_ncar

    for i, each in enumerate(episode_files_list):
        print("Loading", each)
        match = re.search(r'episodes_(\d+)_(\{.*})_LL', each)
        agent_id = int(match.group(1))
        params = ast.literal_eval(match.group(2))
        print(agent_id, " == ", params)

        label = f'myopic-{params["initial_beta"]}-beta-{params["initial_lambda"]}-lambda-{params["initial_alpha"]}-alpha'
        print("agentId", agent_id, ":", label)

        with open(os.path.join(each), 'rb') as f:
            stats = pickle.load(f)

    # sort legend labels with some logic if you need to
    # handles, labels = plt.gca().get_legend_handles_labels()
    # print(labels)
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0].split('-')[2]) if len(t[0].split('-')) > 2 else 1))
    # plt.legend(handles, labels)
    plt.show()
    return


def analyse_episodes(stats, save_file_path, params):
    episodes = stats["episodes_positions"]
    parameters = {
        'axes.labelsize': 12,
        'axes.titlesize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    }
    plt.rcParams.update(parameters)
    tf = convert_episodes_to_traj_class(episodes, stats["episodes_states"])
    plot_node_revisits_level_halves(tf, [6], title=params, save_file_path=save_file_path + 'n_revisits', display=False)
    plot_end_node_revisits_level_halves(tf, [6], title=params, save_file_path=save_file_path + 'en_revisits', display=False)
    plot_unique_node_revisits_level_halves(tf, [6], title=params, save_file_path=save_file_path + 'un_revisits', display=False)
    plot_first_endnode_labels(tf, title=params, save_file_path=save_file_path, display=False)
    plot_opposite_node_preference(tf, title=params, save_file_path=save_file_path, display=False)
    plot_percent_turns(tf, title=params, save_file_path=save_file_path, display=False)
    # plot_reward_path_lengths(episodes, params, save_file_path)
    plot_episode_lengths(episodes, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=False, le=6, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=False, le=5, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=False, le=4, title=params, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=False, le=3, title=params, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=False, le=2, title=params, save_file_path=save_file_path)
    plot_visit_freq(stats["normalized_visit_frequency"], by_level=False, title=params, save_file_path=save_file_path)
    plot_visit_freq(stats["normalized_visit_frequency_by_level"], by_level=True, title=params, save_file_path=save_file_path)

    # plot_markov_fit_non_pooling(episodes, re=False, title=params, save_file_path=save_file_path, display=False)
    plot_outside_inside_ratio(tf, re=False, title=params, save_file_path=save_file_path)
    plot_decision_biases(tf, re=False, title=params, save_file_path=save_file_path, display=False)
    # plot_trajectory_features(episodes, title=params, save_file_path=save_file_path, display=False)
    # plot_maze_stats(stats["visit_frequency"], interpolate_cell_values=True, colormap_name='Blues',
    #                 colorbar_label="visit freq",
    #                 save_file_name=os.path.join(save_file_path, f'visit_frequency_maze.png'),
    #                 display=False,
    #                 figtitle=f'state visit freq\n{params}')
    plot_markov_fit_pooling(episodes, re=False, title=params, save_file_path=save_file_path, display=False)
    plot_trajs(episodes, title=params, save_file_path=save_file_path)
    return


def run(model, params_all, base_path, VARIATION):
    MAX_LENGTH = int(VARIATION.split('_', 1)[0])
    var = VARIATION.split('_', 1)[1:]
    simulation_results = model.simulate_multiple(params_all, MAX_LENGTH)

    # analyse results
    for agent_id, params in params_all.items():
        print("params:", params)
        success, stats = simulation_results[agent_id]
        episodes = stats["episodes_positions"]
        LL = stats["LL"]
        model_name = model.__class__.__name__
        params['model'] = model_name
        if success:
            run_id = np.random.randint(1, 100000)
            print("#Episodes: ", len(episodes))
            save_file_path = f'{base_path}/' \
                             f'{model_name}/' \
                             f'MAX_LENGTH={MAX_LENGTH}/' \
                             f'{params.__str__()}_rand{run_id}_{var}/'
            Path(save_file_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(save_file_path, f'episodes_{agent_id}_{params.__str__()}_LL={LL}.pkl'), 'wb') as f:
                pickle.dump(stats, f)
            # analyse_state_values(model, V, save_file_path, params)
            analyse_episodes(stats, save_file_path, params)
            print(">>> Done with params!", params, "\nCheck results at:", save_file_path, "\nrun_id", run_id)
    return


if __name__ == '__main__':
    # np.random.seed(0)
    # should always start with an int denoting max number of steps in the total trajectory
    VARIATION = '20004'

    # base path to save figs or other results in
    base_path = '/Users/usingla/mouse-maze/figs'

    # 1. Load prev simulation(s) results saved locally
    # load(base_path + f'/BayesianQL/MAX_LENGTH={MAX_LENGTH}/')
    # quit()

    # OR, 2. Run a new simulation
    # Import the model class you are interested in

    # from Dyna_Qplus import DynaQPlus
    # model = DynaQPlus()
    # param_sets = {
    #     11: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
    #          "epsilon": 0.0, "n_plan": 100000, "bonus_in_planning": True},
    #     12: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
    #          "epsilon": 0.0, "n_plan": 100000, "bonus_in_planning": True},
    #     13: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
    #          "epsilon": 0.0, "n_plan": 100000, "bonus_in_planning": True},
    #     14: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
    #          "epsilon": 0.0, "n_plan": 100000, "bonus_in_planning": True},
    #     15: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
    #          "epsilon": 0.0, "n_plan": 100000, "bonus_in_planning": True},
    # }

    # from TDLambdaOptimisticInitialization import TDLambdaOptimisticInitialization
    # model = TDLambdaOptimisticInitialization()
    # param_sets = {
    #     # 1: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.1, "epsilon": 0.0},
    #     # 2: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.5, "epsilon": 0.0},
    #     # 3: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "epsilon": 0.0},
    #     # 4: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.8, "epsilon": 0.0},
    #     5: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.99, "epsilon": 0.0},
    # }

    # from EpsilonGreedy_model import EpsilonGreedy
    # model = EpsilonGreedy()
    # param_sets = {
    #     1: {"epsilon": 1.0},
    # }

    # from BayesianQL import BayesianQL
    # model = BayesianQL()
    # param_sets = {
    #     1: {"gamma": 0.99, "action_selection_method": 'q_sampling',
    #          "initial_alpha": 1.5, "initial_lambda": 3, "initial_beta": 0.75},
    #     2: {"gamma": 0.99, "action_selection_method": 'q_sampling',
    #          "initial_alpha": 1.5, "initial_lambda": 3, "initial_beta": 0.75},
    #     3: {"gamma": 0.99, "action_selection_method": 'q_sampling',
    #          "initial_alpha": 1.5, "initial_lambda": 3, "initial_beta": 0.75},
    #     4: {"gamma": 0.99, "action_selection_method": 'q_sampling',
    #          "initial_alpha": 1.5, "initial_lambda": 3, "initial_beta": 0.75},
    #     5: {"gamma": 0.99, "action_selection_method": 'q_sampling',
    #          "initial_alpha": 1.5, "initial_lambda": 3, "initial_beta": 0.75},
    # }

    # from EpsilonDirectionGreedy_model import EpsilonDirectionGreedy
    # model = EpsilonDirectionGreedy()
    # param_sets = {
    #     12: {"epsilon": 0.9, "is_strict": True, "version": 3, "remember_corners": False, "x": 2},
    # }

    # from CustomDirection_model import CustomDirection
    # model = CustomDirection()
    # param_sets = {
    #     1: {"is_strict": True, "version": 2, "z_type": "zipf"},
    #     2: {"is_strict": False, "version": 2, "z_type": "zipf"},
    # }

    # from IDDFS_model import IDDFS
    # model = IDDFS()
    # param_sets = {
    #     1: {},
    # }

    # from random_backprob_model import RandomLessBackProb
    # model = RandomLessBackProb()
    # param_sets = {
    #     1: {"back_prob": 0.3},
    # }

    # from EpsilonZGreedy_model import EpsilonZGreedy
    # model = EpsilonZGreedy()
    # param_sets = {
    #     # 4: {"epsilon": 0.4, "enable_alternate_action": True, "enable_LoS": False},
    #     5: {"epsilon": 0.3, "enable_alternate_action": True, "enable_LoS": False},
    # }

    from v1_model import BiasedModelV1
    model = BiasedModelV1()
    param_sets = {
        # 1: {'back_prob': 0.4},
        # 2: {'back_prob': 0.2},
        # 3: {'back_prob': 0.3},
        2: {'back_prob': 0.2, 'node_preferred_prob': 0.7},
        3: {'back_prob': 0.2, 'node_preferred_prob': 0.55},
        4: {'back_prob': 0.2, 'node_preferred_prob': 0.8},
        5: {'back_prob': 0.2, 'node_preferred_prob': 0.6},
        6: {'back_prob': 0.2, 'node_preferred_prob': 0.75},
    }

    run(model, param_sets, base_path, VARIATION)
