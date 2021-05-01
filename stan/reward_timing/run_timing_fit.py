import pystan
import numpy as np
import pickle
import matplotlib.pyplot as plt

main_dir = 'Z:/stan/reward_timing/'
stan_file = main_dir + 'timing_fit.stan'
samples = np.load(main_dir + 'cdf_samples_e5_s10.npy')
mu = 90  # units in seconds
N = int(1e6)
#samples = np.random.normal(mu,sigma,N) + np.random.exponential(scale, N)  # scale in stan = 1 / scale in numpy
#plt.figure()
#plt.hist(samples)
model_data={
    'NUMsamples':len(samples),
    'samples':samples,
    'mu': mu
}
sm = pystan.StanModel(file=stan_file)
fit = sm.sampling(data=model_data, iter=2000, chains=4, warmup=250, control={'max_treedepth':20, 'adapt_delta':0.95}, n_jobs=1)

# Save fit
with open(main_dir + 'fit.pkl', 'wb') as f:
    pickle.dump({'model': sm, 'fit': fit}, f, protocol=-1)
fit_data = open(main_dir + 'fit_data.txt', 'w')
print(fit, file=fit_data)
fit_data.close()