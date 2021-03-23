data{
    int N;                        // number of rewarded mice
    int B;                        // maximum number of bouts until the first reward was sampled
    int BL;                       // maximum bout length until the first reward was sampled
    int S;                        // number of states/nodes in the maze
    int A;                        // number of actions available at each state
    int RT;                       // reward magnitude at the reward port (node 116)
    real Q0[N,S+1,A];             // initial state-action values for each mouse
    int nodemap[S+1,A];          // node labels corresponding to state-action values in the Q lookup table
    int TrajS[N,B,BL];            // trajectories of all mice up until the first reward was sampled
    int TrajA[N,B,BL];            // corresponding actions taken at each point in TrajS to transition to the next state
                                  // TrajA actions: 1,2,3 correspond to column positions on nodemap for each row position (states)
    int UB[3];                    // upper bounds for each free parameter
}
transformed data {
    real alpha_UB;
    real beta_UB;
    real gamma_UB;

    // Storing parameter upper bounds
    alpha_UB = UB[1];
    beta_UB = UB[2];
    gamma_UB = UB[3];
}
parameters{
    // population parameters
    real alpha_mu;
    real <lower=0.001> alpha_sd;
    //real beta_mu;
    //real <lower = 0.001> beta_sd;
    real <lower=1> beta_u;
    real <lower=1> beta_l;
    real gamma_mu;
    real <lower = 0.001> gamma_sd;

    // agent parameters
    real alpha_sub[N];
    real <lower = 0.001, upper=0.999> beta_sub[N];
    real gamma_sub[N];
}
model{
    // Declare model parameters
    real Q[N,S+1,A];
    int s;
    int a;
    int R;
    int sprime;
    int aprime;
    vector[3] sprime_values_beta;

    // sampling population level parameters
    alpha_mu ~ normal(0, 1);
    alpha_sd ~ normal(0, 1);
    //beta_mu ~ normal(0, 1);
    //beta_sd ~ normal(0, 1);
    beta_u ~ normal(1, 100);
    beta_l ~ normal(1, 100);
    gamma_mu ~ normal(0, 1);
    gamma_sd ~ normal(0, 1);

    // Initialize state-action values for a mouse
    Q = Q0;

    for (n in 1:N){
        real alpha;
        real beta;
        real gamma;

        // sampling agent parameters
        alpha_sub[n] ~ normal(alpha_mu, alpha_sd);
        alpha = Phi_approx(alpha_sub[n]);
        //beta_sub[n] ~ normal(beta_mu, beta_sd);
        //beta = beta_UB * Phi_approx(beta_sub[n]);
        beta_sub[n] ~ beta(beta_u, beta_l);
        beta = beta_UB * beta_sub[n];
        gamma_sub[n] ~ normal(gamma_mu, gamma_sd);
        gamma = gamma_UB * Phi_approx(gamma_sub[n]);

        // Initialize starting state, s0 to node 0
        s = 0;

        for (b in 1:B){
            // Begin episode if the mouse enters the maze during this bout
            // i.e. when the first node position is 0 and second node position is not 127
            if (TrajS[n,b,1] == 0 && TrajS[n,b,2] != 127){

                // Begin stepping through the episode
                // Get a vector of 3 possible state-action values for the current starting state, s = 0
                for (i in 1:A){
                    sprime_values_beta[i] = Q[n, nodemap[1,i]+1, i] * beta;
                }

                // Update the likelihood of choosing action, a from state s = 0
                TrajA[n,b,1] ~ categorical_logit(sprime_values_beta);

                // Select action to take from s
                a = TrajA[n,b,1];

                for (step in 2:BL){

                    // Transition to the next state
                    sprime = TrajS[n,b,step];

                    // Checking if the trajectory is still valid
                    if (sprime != -1){
                        // Sample a reward
                        if (sprime == 116){
                            R = RT;
                        }
                        else{
                            R = 0;
                        }

                        // At terminal states, Q[n, sprime+1, aprime] = 0
                        if (sprime == 116){
                            aprime = 1;
                        }
                        else if (sprime == 127){
                            aprime = 2;
                        }
                        else if (nodemap[sprime+1,2] == -1){
                            // Select action to take from sprime
                            aprime = TrajA[n,b,step];
                        }
                        else {
                            // Select an action
                            // Get a vector of 3 possible state-action values for the state, sprime
                            for (i in 1:A){
                                sprime_values_beta[i] = Q[n,nodemap[sprime+1,i]+1,i] * beta;
                            }

                            // Update the likelihood of choosing action, aprime from state sprime
                            TrajA[n,b,step] ~ categorical_logit(sprime_values_beta);

                            // Select action to take from sprime
                            aprime = TrajA[n,b,step];
                        }

                        // Update previous state-action value
                        Q[n,s+1,a] += alpha * (R + gamma * Q[n,sprime+1,aprime] - Q[n,s+1,a]);

                        // Update new state and action to old state and action
                        s = sprime;
                        a = aprime;

                        // Check if current state is a terminal state
                        if (s == 127 || s == 116){
                            break;
                        }
                    }
                    else{
                        // End the current bout/episode and move on to the next one
                        break;
                    }
                }
            }
        }
    }
}
generated quantities{
    real alpha_mu_phi;
    //real beta_mu_phi;
    real beta_hyper_mean;
    real gamma_mu_phi;
    real alpha_sub_phi[N];
    real beta_sub_phi[N];
    real gamma_sub_phi[N];
    real log_LL[N];

    // Preparing fitted parameters for output to file and model summary
    alpha_mu_phi = Phi_approx(alpha_mu);
    alpha_sub_phi = Phi_approx(alpha_sub);
    //beta_mu_phi = beta_UB * Phi_approx(beta_mu);
    //beta_sub_phi = Phi_approx(beta_sub);
    beta_hyper_mean = (beta_u * beta_UB) / (beta_u + beta_l);
    beta_sub_phi = beta_sub;
    gamma_mu_phi = gamma_UB * Phi_approx(gamma_mu);
    gamma_sub_phi = Phi_approx(gamma_sub);

    for (n in 1:N){
        beta_sub_phi[n] = beta_UB * beta_sub_phi[n];
        gamma_sub_phi[n] = gamma_UB * gamma_sub_phi[n];
    }

    // Calculating likelihood
    for (n in 1:N){

        // Initialize model parameters
        int s;
        int a;
        int R;
        int sprime;
        int aprime;
        vector[3] sprime_values_beta;
        real Q[S+1,A];

        // Initialize starting state, s0 to node 0 and state-action values
        s = 0;
        Q = Q0[n,:,:];

        // Initialize likelihood for each subject
        log_LL[n] = 0;

        for (b in 1:B){
            // Begin episode if the mouse enters the maze during this bout i.e. when the first node position is 0 and not 127

            if (TrajS[n,b,1] == 0 && TrajS[n,b,2] != 127){
                // Begin stepping through the episode

                // Get a vector of 3 possible state-action values for the current starting state, s = 0
                for (i in 1:A){
                    sprime_values_beta[i] = Q[nodemap[1,i]+1,i] * beta_sub_phi[n];
                }

                // Update the likelihood of choosing action, a from state s = 0
                log_LL[n] += categorical_logit_lpmf( TrajA[n,b,1] | sprime_values_beta );

                // Select action to take from s
                a = TrajA[n,b,1];

                for (step in 2:BL){

                    // Transition to the next state
                    sprime = TrajS[n,b,step];

                    // Checking if the trajectory is still valid
                    if (sprime != -1){
                        // Sample a reward
                        if (sprime == 116){
                            R = RT;
                        }
                        else{
                            R = 0;
                        }

                        // At terminal states, Q[sprime+1, aprime] = 0
                        if (sprime == 116){
                            aprime = 1;
                        }
                        else if (sprime == 127){
                            aprime = 2;
                        }
                        else if (nodemap[sprime+1,2] == -1){
                            // Select action to take from sprime
                            aprime = TrajA[n,b,step];
                        }
                        else {
                            // Select an action
                            // Get a vector of 3 possible state-action values for the state, sprime
                            for (i in 1:A){
                                sprime_values_beta[i] = Q[nodemap[sprime+1,i]+1,i] * beta_sub_phi[n];
                            }

                            // Update the likelihood of choosing action, aprime from state sprime
                            log_LL[n] += categorical_logit_lpmf( TrajA[n,b,step] | sprime_values_beta );

                            // Select action to take from sprime
                            aprime = TrajA[n,b,step];
                        }

                        // Update previous state-action value
                        Q[s+1,a] += alpha_sub_phi[n] * (R + gamma_sub_phi[n] * Q[sprime+1,aprime] - Q[s+1,a]);

                        // Update new state and action to old state and action
                        s = sprime;
                        a = aprime;

                        // Check if current state is a terminal state
                        if (s == 127 || s == 116){
                            break;
                        }
                    }
                    else{
                        // End the current bout/episode and move on to the next one
                        break;
                    }
                }
            }
        }
    }
}