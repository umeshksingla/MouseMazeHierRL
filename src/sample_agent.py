import os
import numpy as np
from pathlib import Path
import pickle

from plot_utils import plot_trajs, plot_episode_lengths, \
    plot_exploration_efficiency, plot_reward_path_lengths, plot_maze_stats


def analyse_episodes(episodes, save_file_path, params):
    """todo
    episodes:

    """
    plot_reward_path_lengths(episodes, save_file_path, params)
    plot_episode_lengths(episodes, save_file_path, params)
    plot_exploration_efficiency(episodes, save_file_path, params)
    # plot_trajs(episodes, save_file_path, params)
    return


def analyse_state_values(model, V, save_file_path, params):
    """todo
    model:
    V:
    """
    state_values = model.get_maze_state_values(V)
    print("state_values", state_values)
    plot_maze_stats(state_values, datatype="state_values",
                    save_file_name=os.path.join(save_file_path, f'state_values.png'),
                    display=False,
                    figtitle=f'state values \n params = {params}')
    return


def run(model, params_all, base_path):

    MAX_LENGTH = 3000
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
            print("episodes", episodes)
            analyse_state_values(model, V, save_file_path, params)
            analyse_episodes(episodes, save_file_path, params)
        print(">>> Done with params!", params)

    return


if __name__ == '__main__':

    # base path to save figs or other results in
    base_path = '/Users/usingla/mouse-maze/figs'

    # Import the model class you are interested in
    # from TDLambdaXStepsPrevNode_model import TDLambdaXStepsPrevNodeRewardReceived
    from Epsilon3Greedy_model import Epsilon3Greedy
    # from Epsilon2Greedy_model import Epsilon2Greedy
    model = Epsilon3Greedy()
    param_sets = {
        0: {"alpha": 0.1, "gamma": 0.9, "lamda": 0.7, "epsilon": 0.5},
        # 0: {"alpha": 0.1, "beta": 3, "gamma": 0.89, "lamda": 0.5},
    }
    run(model, param_sets, base_path)
