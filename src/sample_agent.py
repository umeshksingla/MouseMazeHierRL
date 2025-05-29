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
        if 'BiasedWalk4' in label:
            continue
        plot_markov_fit_pooling(tf, i, re=rew, title=title, save_file_path=save_path)

    return


def analyse_episodes(stats, save_file_path, params):
    episodes = stats["episodes_positions"]
    rew = params['rew']
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
    # plot_reward_runs(tf, title=params, save_file_path=save_file_path)
    # plot_reward_path_lengths(tf, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=6, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=5, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=4, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=3, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=6, half='separate', title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=6, half=1, title=params, save_file_path=save_file_path)
    plot_exploration_efficiency(tf, re=rew, le=6, half=2, title=params, save_file_path=save_file_path)
    plot_visit_freq_by_level(tf, re=rew, title=params, save_file_path=save_file_path)
    plot_decision_biases(tf, re=rew, title=params, save_file_path=save_file_path, display=False)
    plot_node_revisits_level_halves(tf, 6,  re=rew, title=params, save_file_path=save_file_path, display=False)
    plot_first_endnode_labels(tf, re=rew, title=params, save_file_path=save_file_path, display=False)
    plot_opposite_node_preference(tf, re=rew, title=params, save_file_path=save_file_path, display=False)
    plot_percent_turns(tf, re=rew, title=params, save_file_path=save_file_path, display=False)
    plot_episode_lengths(tf, re=rew, title=params, save_file_path=save_file_path)

    plot_outside_inside_ratio(tf, re=rew, title=params, save_file_path=save_file_path)
    plot_markov_fit_pooling(traj, params['model'], re=rew, title=params, save_file_path=save_file_path, display=False)
    # plot_trajs(episodes, title=params, save_file_path=save_file_path)

    # plot_end_node_revisits_level_halves(tf, [6], title=params, save_file_path=save_file_path + 'en_revisits', display=False)
    # plot_unique_node_revisits_level_halves(tf, [6], title=params, save_file_path=save_file_path + 'un_revisits', display=False)
    # plot_exploration_efficiency(tf, re=False, le=2, title=params, save_file_path=save_file_path)
    # plot_trajectory_features(episodes, title=params, save_file_path=save_file_path, display=False)
    # plot_opposite_node_preference(tf, title=params, save_file_path=save_file_path, display=False)
    # plot_markov_fit_non_pooling(episodes, re=False, title=params, save_file_path=save_file_path, display=False)
    # plot_maze_stats(stats["visit_frequency"], interpolate_cell_values=True, colormap_name='Blues',
    #                 colorbar_label="visit freq",
    #                 save_file_name=os.path.join(save_file_path, f'visit_frequency_maze.png'),
    #                 display=False,
    #                 figtitle=f'state visit freq\n{params}')

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
                               f'episodes_{agent_id}_{params.__str__()}_{var}_LL={stats["LL"]}.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        if analyze:
            analyse_episodes(stats, save_file_path, params)
        run_ids.append(run_id)
        print(">>> Done with params!", params, "\nCheck results at:", save_file_path, "\nrun_id", run_id)
    return run_ids


if __name__ == '__main__':
    # np.random.seed(0)
    # should always start with an int denoting max number of steps in the total trajectory
    # VARIATION = '20005_moreprobopp'

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

    # from v1_model import BiasedModelV1
    # model = BiasedModelV1()
    # param_sets = {
    #     2: {'back_prob': 0.2, 'node_preferred_prob': 0.75, 'model': 'V2'},
    #     # 3: {'back_prob': 0.2, 'node_preferred_prob': 0.55},
    #     # 4: {'back_prob': 0.2, 'node_preferred_prob': 0.8},
    #     # 5: {'back_prob': 0.2, 'node_preferred_prob': 0.6},
    #     # 6: {'back_prob': 0.2, 'node_preferred_prob': 0.75},
    # }
    # run(model, param_sets, base_path, VARIATION)

    # base path to save figs or other results in
    base_path = '/Users/usingla/mouse-maze/figs/'
    # load([
    #     # ('BiasedModelV1', 41724),
    #     ('BiasedModelV1', 77428),    # v2
    #     # ('BiasedModelV1', '71316'),
    #     ('EpsilonZGreedy', 22693),
    #     ('BiasedWalkMM', 43600),
    #     # ('BiasedModelV1', 84397)  # v1
    # ], base_path)

    load([
        # # ez original
        # ('EpsilonZGreedy', 22693),

        # check
        # ('EpsilonZGreedy', 85434),

        # strong 1
        # ('EpsilonZGreedy', 349508),
        # ('EpsilonZGreedy', 919586),


        # ('EpsilonZGreedy', 900664),
        #
        # # strong 01
        # ('EpsilonZGreedy', 572794),
        # ('EpsilonZGreedy', 622893),

        # # strong 1 d0
        # ('EpsilonZGreedy', 528134),

        # weak r12 d0
        # ('EpsilonZGreedy', 593958),

        # # weak 2 d0
        # ('EpsilonZGreedy', 748594),

        # biased model
        # ('BiasedModel', 77428),

        # coeff model
        # ('CoeffModel', 905105),

        # e = 0.1
        # ('AA', 72432),
        # ('AA', 64435),
        # ('AA', 44679),
        # ('AA', 147363),

        # e = 0.3
        # ('AA', 43235),
        # ('AA', 8147),
        # ('AA', 181887),
        # ('AA', 348276),

        # # e = 1.0
        # ('AA', 55403),
        # ('AA', 831945),
        # ('AA', 76199),
        # ('AA', 412521),

        # e = 0.1, 0.3, 1.0 random in central l0-2 or l0-3
        # ('AA', 44679),
        # ('AA', 181887),
        # ('AA', 76199),

        # mu = 2: strong, weak, absent - strong and absent good eff, left bias good in absent, P similar, fraction good in weak (BRING DOWN L6 FRACTION AND UP L4)
        # ('EpsilonZGreedy', 87716),
        # ('EpsilonZGreedy', 829236),
        # ('EpsilonZGreedy', 608501),

        # mu = 1.9: strong, weak, absent - strong and absent good eff, left bias good in most, P similar, fraction good in weak (BRING DOWN L6 FRACTION AND UP L4)
        # ('EpsilonZGreedy', 914265),
        # ('EpsilonZGreedy', 853630),
        # ('EpsilonZGreedy', 252508),

        # mu = 1.8: strong, weak, absent - strong and absent okay eff, left bias good in most, P similar, fraction good in weak (BRING DOWN L6 FRACTION AND UP L4)
        # ('EpsilonZGreedy', 312597),
        # ('EpsilonZGreedy', 767061),
        # ('EpsilonZGreedy', 940515),

        # mu = 1.7: strong, weak, absent - strong and absent okay eff, left bias good in absent/weak, P similar, fraction good in weak (BRING DOWN L6 FRACTION AND UP L4)
        # ('EpsilonZGreedy', 467819),
        # ('EpsilonZGreedy', 572280),
        # ('EpsilonZGreedy', 86888),

        # ('V2', [933613, 577107, 599346, 889774, 359900])    # staySq 0.7, stayQ varies

        # ('V2', [446726, 116325, 948619, 289226, 253401])    # staySq 0.65, stayQ varies

        # ('V2', [263475, 770165, 398401, 962502, 595252, 860300])  # staySQ varies, stayQ 0.85

        # ('V2', [263475, 962502, 846765])  # staySQ varies, stayQ 0.85

        # ('V2', [263475, 962502, 846765])  # staySQ varies, stayQ 0.85

        # ('V2', [962502, 599346]),
        # ('EpsilonZGreedy', [22693, 829236])
        # ('V3', [402647, 613683, 290052])  # no diff between opp straight biases at subq and q level
        # ('V3', [716722, 801808])    # all opp straight bent etc equal

        # ('V3', [164631]),   # 887275
        # ('BiasedWalk4', [384434]),
        # ('EpsilonZGreedy', [22693])

        # ('V2', [799411, 570294, 838355, 325609, 228337])    # multiple simulations for V2 with same params
        # ('V3', [191820, 493692, 360591, 383989, 58814])  # multiple simulations for V3 with same params
        # ('BiasedWalk4', [190172, 483041, 655539, 529415, 397616])  # multiple simulations for BiasedWalk4 with same params

        # ('EpsilonZGreedy', [655851, 972319, 898791, 464215, 216742])    # multiple simulations for EpsilonZGreedy with same params

        # ('EpsilonZGreedy', [475022, 325380, 57433, 42260, 406924, 459779])    # multiple simulations for EpsilonZGreedy with diff mu but e=0.3
        # ('EpsilonZGreedy', [601047, 292311, 866783, 716345, 426014, 23484])     # multiple simulations for EpsilonZGreedy with diff mu but e=0.2
        # ('EpsilonZGreedy', [675911, 981948, 162233, 284774, 640510, 700054])  # multiple simulations for EpsilonZGreedy with diff mu but e=0.4
        # ('EpsilonZGreedy', [783121, 408644, 745470, 292030, 259926, 336503])     # multiple simulations for EpsilonZGreedy with diff mu but e=0.4 and duration not 0 in random
        # ('EpsilonZGreedy', [603896, 1901, 530173, 46841, 349521, 308086])     # multiple simulations for EpsilonZGreedy with diff mu but e=0.3 and duration not 0 in random
        # ('EpsilonZGreedy', [410642, 511540, 701584, 778068, 825601, 712320])    # multiple simulations for EpsilonZGreedy with diff mu but e=0.3 and duration not 0 in random, +1 in zipf

        # ('EpsilonZGreedy', [274045]),
        # ('V2', [799411]),
        # ('V3', [360591]),
        # ('BiasedWalk4', [190172]),

        # ('EpsilonZGreedy', [274045, 351191, 310880, 8908])    # multiple simulations for EpsilonZGreedy e=0.3, with duration 0 in random
        # ('EpsilonZGreedy', [513435, 60530, 989176, 96801])    # multiple simulations for EpsilonZGreedy e=0.35, with duration 0 in random
        # ('EpsilonZGreedy', [714363, 19392, 34583, 458679])    # multiple simulations for EpsilonZGreedy e=0.25, with duration 0 in random
        # ('EpsilonZGreedy', [53487, 859163, 917598, 992250])   # multiple simulations for EpsilonZGreedy e=0.4, with duration 0 in random
        # ('EpsilonZGreedy', [187077, 697240, 879930, 155774])    # multiple simulations for EpsilonZGreedy e=1, with duration 0 in random

        # ('EpsilonZGreedy', [506691])    # multiple simulations for EpsilonZGreedy e=1, different mu, with duration 0 in random
        # ('EpsilonZGreedy', [534614, 355055, 516984, 671772, 748921])
            # multiple simulations for EpsilonZGreedy e=1, different mu, strong memory, with duration 0 in random
        # ('EpsilonZGreedy', [806473, 848916, 54208, 9402, 667047])
        # multiple simulations for EpsilonZGreedy e=1, different mu, weak memory, with duration 0 in random
        # ('EpsilonZGreedy', [919975, 556059, 667704, 86919, 231990])
        # multiple simulations for EpsilonZGreedy e=1, different mu, weak memory, with duration 0 in random

        # ('Custom', [382029]),
        # ('Levy', [184722]),
        # # ('Optimal', [58410]),
        # ('BiasedWalk4', [50173])

        # ('EZCustom', [684225, 692398, 812279, 866072, 690742]),
        # ('EZCustom', [778597, 772763, 68980]),

        ('LW', [322469]),
        ('BiasedWalk4', [927154]),

    ], base_path)
