import pickle
from TDLambdaXSteps_model import TDLambdaXStepsRewardReceived
from model_plot_utils import plot_trajectory
from pathlib import Path
import os
import matplotlib.pyplot as plt

model = TDLambdaXStepsRewardReceived()
with open('/Volumes/ssrde-home/run2/TDlambdaXsteps_best_sub_fits.p', 'rb') as f:
    sub_fits = pickle.load(f)

area = 'quad123_100'
MAX_LENGTH = 50
save_file_path = f'../TDlambdaXsteps_stan/figs/lamda_as_param/MAX_LENGTH={MAX_LENGTH}/{area}/'
Path(save_file_path).mkdir(parents=True, exist_ok=True)

for mouse in range(10):
    print(sub_fits[mouse])
    episodes_all_mice, success, stats = model.simulate({mouse: sub_fits[mouse]}, MAX_LENGTH)
    if success:
        plot_trajectory(episodes_all_mice[mouse], 'all',
                        save_file_name=os.path.join(save_file_path, f'Figure_{mouse+1}.png'),
                        display=False)
        print(stats)
        # total = stats[mouse]['count_total']
        # print(stats[mouse]['invalid_initial_state_counts'].values())
        # print(sum(stats[mouse]['invalid_initial_state_counts'].values())/total)
        # for state, count in stats[mouse]['invalid_initial_state_counts'].items():
        #     # print(state, round(count/total, 3))
        #     stats[mouse]['invalid_initial_state_counts'][state] = round(count/total, 3)
        # plt.plot(stats[mouse]['invalid_initial_state_counts'].keys(), stats[mouse]['invalid_initial_state_counts'].values(), 'r*')
        # plt.show()
    print("========================================")

