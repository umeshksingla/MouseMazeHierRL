data{
    int NUMsamples;
    real samples[NUMsamples];
    real mu;
}
parameters{
    //real <lower=0.001>exp_scale;
    real <lower=0.001>sigma;
}
model{
    sigma ~ normal(0,1);
    //exp_scale ~ normal(0,100);

    for (s in 1:NUMsamples){
        real w = 0.5;
        target += log(w*exp(normal_lpdf(samples[s] | mu, sigma)) + (1-w)*exp(exponential_lpdf(samples[s] | 5)));
    }
}
generated quantities{
    real loglk;
    real rand_loglk;

    // Initializing log likelihood
    loglk = 0;
    rand_loglk = 0;

    for (s in 1:NUMsamples){
        real w = 0.5;
        loglk += log(w * exp(normal_lpdf(samples[s] | mu, sigma)) + (1-w) * exp(exponential_lpdf(samples[s] | 5)));
        rand_loglk += log(0.5);
    }
}