import os
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import ast
import glob
import re

from plot_utils import plot_trajs, plot_episode_lengths, \
    plot_exploration_efficiency, plot_maze_stats, plot_visit_freq_by_level, \
    plot_decision_biases, plot_markov_fit_pooling, plot_markov_fit_non_pooling,\
    plot_trajectory_features, plot_reward_runs, plot_outside_inside_ratio, \
    plot_percent_turns, plot_opposite_node_preference, plot_first_endnode_labels, \
    plot_end_node_revisits_level_halves, plot_end_node_revisits_level_all_time, \
    plot_unique_node_revisits_level_halves, plot_node_revisits_level_halves
import evaluation_metrics as em
from utils import convert_episodes_to_traj_class, split_trajectories_at_first_reward, split_trajectories_k_parts
from MM_Traj_Utils import LoadTrajFromPath
from parameters import RewNames, UnrewNamesSub, OUTDATA_PATH


def analyse_state_values(stats, save_file_path, params):
    state_values = stats["V"]
    # print("state_values", state_values)
    plot_maze_stats(state_values, interpolate_cell_values=True,
                    save_file_name=os.path.join(save_file_path, f'state_values.png'),
                    display=False,
                    figtitle=f'state values\n{params}')
    return


def load(runs, save_file_path):
    """
    Takes episodes pickles from a local directory and plots exp efficiency for
    all of them on one graph. To be extended to other metrics.
    """

    tfs_labels = []
    label_runids = []
    loaded_model_params = []
    c = 0
    for model_name, model_runs in runs:
        for run_id in model_runs:
            path = save_file_path + f'{model_name}/*/*_rand{run_id}_*/episodes*.pkl'
            print(path)
            files_list = glob.glob(path)
            print(files_list)
            assert len(files_list) == 1

            path = files_list[0]
            print(path)
            match = re.search(r'episodes_(\d+)_(\{.*})_(\[.*])_LL', path)
            agent_id = int(match.group(1))
            params = ast.literal_eval(match.group(2))
            var = ast.literal_eval(match.group(3))
            print(agent_id, " == ", params, var)

            label = params.get('model', model_name)
            label_runids.append(label)
            label_runids.append(str(run_id))

            with open(os.path.join(files_list[0]), 'rb') as f:
                stats = pickle.load(f)

            episodes = stats["episodes_positions"]
            tf = convert_episodes_to_traj_class(episodes, stats["episodes_states"])
            # tfs_labels.append((tf, params.get('model', model_name) + (var[0] if var else '')))
            tfs_labels.append((tf,
                               f'{params["model"]}' # + f' ε={params["epsilon"]} µ={params["mu"]}' if 'ezg' in params["model"] else 'BiasedWalk4'
                               # f'ez-mu={params["mu"]}'
                               # 'v2-' + str(params.get('staySQ', '')) # + params.get('memory_l5', '')  # + '-' + (var[0] if var else '')
                               ))
            loaded_model_params.append((c, run_id, params))
            c += 1
            print()

    save_path = save_file_path + f'./combined/{"-".join(label_runids)}/'
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + 'loaded_model_params.txt', 'w') as f:     # save a copy of params of models being plotted together
        for p in loaded_model_params: f.write(str(p) + '\n\n')

    title = ''
    rew = params['rew']
    plot_opposite_node_preference(tfs_labels, title=title, save_file_path=save_path, display=False)
    plot_exploration_efficiency(tfs_labels, re=rew, le=6, title=title, save_file_path=save_path)
    plot_exploration_efficiency(tfs_labels, re=rew, le=5, title=title, save_file_path=save_path)
    plot_exploration_efficiency(tfs_labels, re=rew, le=4, title=title, save_file_path=save_path)
    plot_exploration_efficiency(tfs_labels, re=rew, le=3, title=title, save_file_path=save_path)
    plot_decision_biases(tfs_labels, re=rew, title=title, save_file_path=save_path)
    plot_node_revisits_level_halves(tfs_labels, re=rew, title=title, level_to_plot=6, save_file_path=save_path)
    plot_visit_freq_by_level(tfs_labels, re=rew, title=title, save_file_path=save_path)
    plot_first_endnode_labels(tfs_labels, re=rew, title=title, save_file_path=save_path)
    plot_outside_inside_ratio(tfs_labels, re=rew, title=title, save_file_path=save_path)
    plot_percent_turns(tfs_labels, title=title, save_file_path=save_path)
    plot_episode_lengths(tfs_labels, title=title, save_file_path=save_path)

    for i, (tf, label) in enumerate(tfs_labels):
        # if 'BiasedWalk4' in label:
        #     continue
        plot_markov_fit_pooling(tf, i, re=rew, title=title, save_file_path=save_path)

    return


def analyse_episodes(stats, save_file_path, params):
    episodes = stats["episodes_positions"]
    rew = False
    parameters = {
        'axes.labelsize': 12,
        'axes.titlesize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    }
    plt.rcParams.update(parameters)
    traj = convert_episodes_to_traj_class(episodes, stats["episodes_states"])
    tf = [(traj, params['model'])]

    additional_title = str(params)

    plot_exploration_efficiency(tf, re=rew, le=6, title=additional_title, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=rew, le=5, title=additional_title, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=rew, le=4, title=additional_title, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=3, title=additional_title, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=rew, le=6, half='separate', title=additional_title, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=rew, le=6, half=1, title=additional_title, save_file_path=save_file_path)
    # plot_exploration_efficiency(tf, re=rew, le=6, half=2, title=additional_title, save_file_path=save_file_path)
    # plot_visit_freq_by_level(tf, re=rew, title=additional_title, save_file_path=save_file_path)
    plot_decision_biases(tf, re=rew, title=additional_title, save_file_path=save_file_path, display=False)
    # plot_node_revisits_level_halves(tf, 6,  re=rew, title=additional_title, save_file_path=save_file_path, display=False)
    plot_first_endnode_labels(tf, re=rew, title=additional_title, save_file_path=save_file_path, display=False)
    # plot_opposite_node_preference(tf, re=rew, title=additional_title, save_file_path=save_file_path, display=False)
    plot_percent_turns(tf, re=rew, title=additional_title, save_file_path=save_file_path, display=False)
    # plot_episode_lengths(tf, re=rew, title=additional_title, save_file_path=save_file_path)

    plot_outside_inside_ratio(tf, re=rew, title=additional_title, save_file_path=save_file_path)
    # plot_markov_fit_pooling(traj, params['model'], re=rew, title=additional_title, save_file_path=save_file_path, display=False)
    plot_trajs(episodes, title=params, save_file_path=save_file_path)

    return


def run(model, params_all, base_path, VARIATION, analyze=True):

    MAX_LENGTH = int(VARIATION.split('_', 1)[0])
    var = VARIATION.split('_', 1)[1:]
    simulation_results = model.simulate_multiple(params_all, MAX_LENGTH)

    run_ids = []
    for agent_id, params in enumerate(params_all):
        print("params:", params)
        _, stats = simulation_results[agent_id]
        episodes = stats["episodes_positions"]
        assert stats["agentId"] == agent_id
        if 'model' not in params:
            params['model'] = model.__class__.__name__
        # params['version'] = getattr(model, 'version', '')
        run_id = np.random.randint(1, 1000000)
        print("#Episodes: ", len(episodes))
        save_file_path = f'{base_path}/' \
                         f'{ params["model"]}/' \
                         f'MAX_LENGTH={MAX_LENGTH}/' \
                         f'{params.__str__()}_rand{run_id}_{var}/'
        Path(save_file_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_file_path,
                               f'episodes_{agent_id}_{params.__str__()}_{var}.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        if analyze:
            analyse_episodes(stats, save_file_path, params)
        run_ids.append(run_id)
        print(">>> Done with params!", params, "\nCheck results at:", save_file_path, "\nrun_id", run_id)
    return run_ids


if __name__ == '__main__':

    # ez-greedy model (with alternative options, the main model)
    from TeAltOptions_model import TeAltOptions

    param_sets = [{'mu': 2, 'model': f"TeAltOptions2"}]
    print(param_sets)
    base_path = '/Users/us3519/mouse-maze/figs/may28/'
    run_ids = run(TeAltOptions(), param_sets, base_path, '50000', analyze=False)
    print(run_ids)

    # # OR, Biased Walk model
    # from BiasedWalk4 import BiasedWalk4
    # param_sets = [{'rew': False}]
    # runids = run(BiasedWalk4(), param_sets, '/Users/us3519/mouse-maze/figs/may28', '20000', analyze=True)
    # print(runids)

