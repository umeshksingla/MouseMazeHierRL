== STAN Modelling ==

~ PART I: Parameter Recovery ~

1. Adjust parameters in "submit_genPred_paramRecovery" to generate new predicted trajectories in stan/traj_data/pred_traj/
2. cd to /sphere/mattar-lab/stan and submit batch job to cluster with "sbatch submit_genPred_paramRecovery sim.py"
3. After trajectories are simulated, adjust parameters in "submit_stan_fit" for fitting to the newly predicted trajectories
4. cd to /sphere/mattar-lab/stan and submit batch job (parameter fitting) to cluster with "sbatch submit_stan_fit run_td.py"

~ PART II: Model fitting to actual data ~

1. In "submit_stan_fit", set TRAJ_TYPE to real and edit other parameters for fitting to real data.
2. cd to /sphere/mattar-lab/stan and submit batch job to cluster with "sbatch submit_stan_fit run_td.py"