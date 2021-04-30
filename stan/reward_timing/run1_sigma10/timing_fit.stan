data{
    int NUMsamples;
    real samples[NUMsamples];
    real mu;
}
parameters{
    //real exp_scale;
    real sigma;
}
model{
    sigma ~ normal(0,1);

    for (s in 1:NUMsamples){
        samples[s] ~ normal(mu, sigma);
    }
}