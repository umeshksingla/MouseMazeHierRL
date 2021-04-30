== STAN Modelling ==

~ PART I: Parameter Recovery ~

1. Adjust parameters in "submit_sim" to simulate new trajectories in stan/pre_reward_traj/
2. cd to /sphere/mattar-lab/stan and submit batch job to cluster with "sbatch submit_sim sim.py"
3. After trajectories are simulated, adjust parameters in "submit_stan_array" for fitting to the newly simulated trajectories
4. cd to /sphere/mattar-lab/stan and submit batch job (parameter fitting) to cluster with "sbatch submit_stan_array run_td.py"

~ PART II: Model fitting to actual data ~

1. In "submit_stan", set TRAJ_TYPE to real
2. cd to /sphere/mattar-lab/stan and submit batch job to cluster with "sbatch submit_stan run_td.py"