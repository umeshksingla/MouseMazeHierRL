data{
    int N;                        // number of rewarded mice
    int B;                        // maximum number of bouts until the first reward was sampled
    int BL;                       // maximum bout length until the first reward was sampled
    int S;                        // number of states/nodes in the maze
    int A;                        // number of actions available at each state
    int RT;                       // reward magnitude at the reward port (node 116)
    real V0[N,S+1];               // initial state values for each mouse
    int nodemap[S+1,A];           // node labels corresponding to state-action values in the Q lookup table
    int TrajS[N,B,BL];            // trajectories of all mice up until the first reward was sampled
    int TrajA[N,B,BL];            // corresponding actions taken at each point in TrajS to transition to the next state
                                  // TrajA actions: 1,2,3 correspond to column positions on nodemap for each row position (states)
    int nonRew_RVisits[N,B];      // Number of visits to the reward node for each bout
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
    real beta_mu;
    real <lower = 0.001> beta_sd;
    //real <lower=1> beta_u;
    //real <lower=1> beta_l;
    real gamma_mu;
    real <lower = 0.001> gamma_sd;

    // agent parameters
    real alpha_sub[N];
    real <lower = 0.001, upper=0.999> beta_sub[N];
    real gamma_sub[N];
}
model{
    // Declare model parameters
    real V[N,S+1];
    int s;
    int a;
    int R;
    int sprime;
    vector[3] sprime_values_beta;

    // sampling population level parameters
    alpha_mu ~ normal(0, 1);
    alpha_sd ~ normal(0, 1);
    beta_mu ~ normal(0, 1);
    beta_sd ~ normal(0, 1);
    //beta_u ~ normal(1, 100);
    //beta_l ~ normal(1, 100);
    gamma_mu ~ normal(0, 1);
    gamma_sd ~ normal(0, 1);

    // Initialize state values for a mouse
    V = V0;

    for (n in 1:N){
        real alpha;
        real beta;
        real gamma;
        real RCount;

        // sampling agent parameters
        alpha_sub[n] ~ normal(alpha_mu, alpha_sd);
        alpha = Phi_approx(alpha_sub[n]);
        beta_sub[n] ~ normal(beta_mu, beta_sd);
        beta = beta_UB * Phi_approx(beta_sub[n]);
        //beta_sub[n] ~ beta(beta_u, beta_l);
        //beta = beta_UB * beta_sub[n];
        gamma_sub[n] ~ normal(gamma_mu, gamma_sd);
        gamma = gamma_UB * Phi_approx(gamma_sub[n]);

        // Initialize starting state, s0 to node 0
        s = 0;

        for (b in 1:B){
            // Begin episode if the mouse enters the maze during this bout
            // i.e. when the first node position is 0 and second node position is not 127
            RCount = 0;

            if (TrajS[n,b,1] == 0 && TrajS[n,b,2] != 127){

                // Begin stepping through the episode
                for (step in 2:BL){

                    // Transition to the next state
                    sprime = TrajS[n,b,step];

                    // Checking if the trajectory is still valid
                    if (sprime != -1){
                        // Sample a reward
                        if (sprime == 116){
                            RCount += 1;
                            if (RCount >= nonRew_RVisits[n,b]){
                                R = RT;
                            }
                            else{
                                R = 0;
                            }
                        }
                        else{
                            R = 0;
                        }

                        // At terminal states, V[n, sprime+1] = 0
                        if (sprime!=116 && sprime!=127 && nodemap[sprime+1,2]!=-1){
                            // Select an action
                            // Get a vector of 3 possible state-action values for the state, sprime
                            for (i in 1:A){
                                sprime_values_beta[i] = V[n,nodemap[sprime+1,i]+1] * beta;
                            }

                            // Update the likelihood of transitioning to state sprime from s
                            TrajA[n,b,step] ~ categorical_logit(sprime_values_beta);
                        }

                        // Update previous state-action value
                        V[n,s+1] += alpha * (R + gamma * V[n,sprime+1] - V[n,s+1]);

                        // Update new state and action to old state and action
                        s = sprime;

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
    real beta_mu_phi;
    //real beta_hyper_mean;
    real gamma_mu_phi;
    real alpha_sub_phi[N];
    real beta_sub_phi[N];
    real gamma_sub_phi[N];
    real log_LL[N];

    // Preparing fitted parameters for output to file and model summary
    alpha_mu_phi = Phi_approx(alpha_mu);
    alpha_sub_phi = Phi_approx(alpha_sub);
    beta_mu_phi = beta_UB * Phi_approx(beta_mu);
    beta_sub_phi = Phi_approx(beta_sub);
    //beta_hyper_mean = (beta_u * beta_UB) / (beta_u + beta_l);
    //beta_sub_phi = beta_sub;
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
        vector[3] sprime_values_beta;
        real V[S+1];
        int RCount;

        // Initialize starting state, s0 to node 0 and state-action values
        s = 0;
        V = V0[n,:];

        // Initialize likelihood for each subject
        log_LL[n] = 0;

        for (b in 1:B){
            // Begin episode if the mouse enters the maze during this bout i.e. when the first node position is 0 and not 127
            RCount = 0;

            if (TrajS[n,b,1] == 0 && TrajS[n,b,2] != 127){

                // Begin stepping through the episode
                for (step in 2:BL){

                    // Transition to the next state
                    sprime = TrajS[n,b,step];

                    // Checking if the trajectory is still valid
                    if (sprime != -1){
                        // Sample a reward
                        if (sprime == 116){
                            RCount += 1;
                            if (RCount >= nonRew_RVisits[n,b]){
                                R = RT;
                            }
                            else{
                                R = 0;
                            }
                        }
                        else{
                            R = 0;
                        }

                        // At terminal states, Q[sprime+1, aprime] = 0
                        if (sprime!=116 && sprime!=127 && nodemap[sprime+1,2]!=-1){
                            // Select an action
                            // Get a vector of 3 possible state-action values for the state, sprime
                            for (i in 1:A){
                                sprime_values_beta[i] = V[nodemap[sprime+1,i]+1] * beta_sub_phi[n];
                            }

                            // Update the likelihood of choosing action, aprime from state sprime
                            log_LL[n] += categorical_logit_lpmf( TrajA[n,b,step] | sprime_values_beta );
                        }

                        // Update previous state-action value
                        V[s+1] += alpha_sub_phi[n] * (R + gamma_sub_phi[n] * V[sprime+1] - V[s+1]);

                        // Update new state and action to old state and action
                        s = sprime;

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