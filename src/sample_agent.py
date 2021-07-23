import os
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import ast
import glob
import re

from plot_utils import plot_trajs, plot_episode_lengths, \
    plot_exploration_efficiency, plot_reward_path_lengths, plot_maze_stats, plot_visit_freq
from utils import calculate_visit_frequency
import evaluation_metrics as em


def analyse_episodes(stats, save_file_path, params):
    """todo
    episodes:

    """
    episodes = stats["episodes"]
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

    plt.figure(figsize=(20, 10))
    plt.xscale('log', base=10)
    colormap = plt.cm.gist_ncar

    episode_files_list = glob.glob(save_file_path + '/*_rand*/episodes*')
    for i, each in enumerate(episode_files_list):
        print("Loading", each)
        match = re.search(r'episodes_(\d+)_(\{.*})_LL', each)
        print(match.group(1), " == ", match.group(2))
        agent_id = int(match.group(1))
        params = ast.literal_eval(match.group(2))
        eps, n_plan, k, lamda = params["epsilon"], params["n_plan"], params["k"], params["lamda"]

        if eps != 0.5:
            continue

        with open(os.path.join(each), 'rb') as f:
            stats = pickle.load(f)

        print("agentId", agent_id, ":", f'{k}-dyna-{n_plan}-plan-{eps}-eps-{lamda}-lamda')
        episodes = stats["episodes"]
        new_end_nodes_found = em.exploration_efficiency(episodes, re=False)
        plt.plot(new_end_nodes_found.keys(), new_end_nodes_found.values(),
                 color=colormap(0.1 + 0.03 * i), linestyle='-', marker="o",
                 label=f'{k}-dyna-{n_plan}-plan-{eps}-eps-{lamda}-lamda')
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
    plt.show()
    return


def run(model, params_all, base_path):

    MAX_LENGTH = 20001
    N_BOUTS_TO_GENERATE = 1

    simulation_results = model.simulate_multiple(params_all,
                                                 MAX_LENGTH=MAX_LENGTH,
                                                 N_BOUTS_TO_GENERATE=N_BOUTS_TO_GENERATE)
    # analyse results
    for agent_id, params in params_all.items():
        print("params:", params)
        success, stats = simulation_results[agent_id]
        episodes = stats["episodes"]
        LL = stats["LL"]
        V = stats["V"]
        if success:
            print("#Episodes: ", len(episodes))
            save_file_path = f'{base_path}/{model.__class__.__name__}/MAX_LENGTH={MAX_LENGTH}/' \
                             f'{params.__str__()}_rand{np.random.randint(1, 10000)}/'
            Path(save_file_path).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(save_file_path, f'episodes_{agent_id}_{params.__str__()}_LL={LL}.pkl'), 'wb') as f:
                pickle.dump(stats, f)
            # print("episodes", episodes)
            analyse_state_values(model, V, save_file_path, params)
            analyse_episodes(stats, save_file_path, params)
        print(">>> Done with params!", params)

    return


if __name__ == '__main__':
    np.random.seed(0)

    # base path to save figs or other results in
    base_path = '/Users/usingla/mouse-maze/figs'

    load(base_path + '/DynaQPlus/MAX_LENGTH=20001/')
    quit()

    # Import the model class you are interested in
    # from TDLambdaXStepsPrevNode_model import TDLambdaXStepsPrevNodeRewardReceived
    # from Epsilon3Greedy_model import Epsilon3Greedy
    # from Epsilon2Greedy_model import Epsilon2Greedy
    # from TD_UCB_model import TD_UCBpolicy
    from Dyna_Qplus import DynaQPlus
    model = DynaQPlus()
    param_sets = {
        0: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.0, "n_plan": 0, "back_action": True},
        1: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.1, "n_plan": 0, "back_action": True},
        2: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.5, "n_plan": 0, "back_action": True},
        3: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.8, "n_plan": 0, "back_action": True},

        4: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.0, "n_plan": 50, "back_action": True},
        5: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.1, "n_plan": 50, "back_action": True},
        6: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.5, "n_plan": 50, "back_action": True},
        7: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.8, "n_plan": 50, "back_action": True},

        8: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.0, "n_plan": 5000, "back_action": True},
        9: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.1, "n_plan": 5000, "back_action": True},
        10: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.5, "n_plan": 5000, "back_action": True},
        11: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.8, "n_plan": 5000, "back_action": True},

        12: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.0, "n_plan": 50000, "back_action": True},
        13: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.1, "n_plan": 50000, "back_action": True},
        14: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.5, "n_plan": 50000, "back_action": True},
        15: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.8, "n_plan": 50000, "back_action": True},

        16: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.0, "n_plan": 100000, "back_action": True},
        17: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.1, "n_plan": 100000, "back_action": True},
        18: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.5, "n_plan": 100000, "back_action": True},
        19: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.8, "n_plan": 100000, "back_action": True},

        20: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.0, "n_plan": 1000000, "back_action": True},
        21: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.1, "n_plan": 1000000, "back_action": True},
        22: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
            "epsilon": 0.5, "n_plan": 1000000, "back_action": True},
        23: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "k": 0.001,
             "epsilon": 0.8, "n_plan": 1000000, "back_action": True},
        # 0: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "c": 0.8, 'V': 'zero'},
        # 1: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "c": 0.8, 'V': 'zero'},
        #
        # 0: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "c": 0.1, 'V': 'zero'},
        # 3: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "c": 0.1, 'V': 'zero'},
        #
        # 4: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "c": 0.5, 'V': 'one'},
        # 5: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "c": 0.5, 'V': 'zero'},
        # 0: {"alpha": 0.1, "beta": 3, "gamma": 0.89, "lamda": 0.5},
    }
    run(model, param_sets, base_path)
