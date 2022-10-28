data{
    int N;                        // number of rewarded mice
    int B;                        // maximum number of bouts throughout the experiment
    int BL;                       // maximum bout length until the first reward was sampled
    int S;                        // number of states/nodes in the maze
    int A;                        // number of actions available at each state
    int RewardNodeMag;            // reward magnitude at the reward port (node 116)
    matrix[N,S] V0;               // initial state values for each mouse
    row_vector[S] e0;             // initial values of zero for the eligibility trace
    int nodemap[S,A];           // node labels corresponding to state-action values in the Q lookup table
    int InvalidState;             // placeholder for invalid states
    int HomeNode;                 // node at maze entrance
    int WaterPortState;           // state with liquid reward
    int TrajS[N,B,BL];            // trajectories of all mice up until the first reward was sampled
    int TrajA[N,B,BL];            // corresponding actions taken at each point in TrajS to transition to the next state
                                  // TrajA actions: 1,2,3 correspond to column positions on nodemap for each row position (states)
    int NUM_PARAMS;               // number of free parameters
    real UB[NUM_PARAMS];          // upper bounds for each free parameter
}
transformed data {
    real alpha_UB;
    real beta_UB;
    real gamma_UB;
    real lamda_UB;

    // Storing parameter upper bounds
    alpha_UB = UB[1];
    beta_UB = UB[2];
    gamma_UB = UB[3];
    lamda_UB = UB[4];
}
parameters{
    // population parameters
    real alpha_mu;
    real <lower=0.001> alpha_sd;
    real beta_mu;
    real <lower = 0.001> beta_sd;
    real gamma_mu;
    real <lower = 0.001> gamma_sd;
    real lamda_mu;
    real <lower = 0.001> lamda_sd;

    // agent parameters
    real alpha_sub[N];
    real beta_sub[N];
    real gamma_sub[N];
    real lamda_sub[N];
}
model{
    // Declare model parameters
    row_vector[S] V;
    row_vector[S] e;
    int s;
    int a;
    int R;
    int sprime;
    int nextS[A];
    vector[A] nextV;
    int sIsEndNode;
    real td_error;
    vector[A] sprime_values_beta;

    // sampling population level parameters
    alpha_mu ~ normal(0, 1);
    alpha_sd ~ normal(0, 1);
    beta_mu ~ normal(0, 1);
    beta_sd ~ normal(0, 1);
    gamma_mu ~ normal(0, 1);
    gamma_sd ~ normal(0, 1);
    lamda_mu ~ normal(0, 1);
    lamda_sd ~ normal(0, 1);

    for (n in 1:N){
        real alpha;
        real beta;
        real gamma;
        real lamda;

        // sampling agent parameters
        alpha_sub[n] ~ normal(alpha_mu, alpha_sd);
        alpha = Phi_approx(alpha_sub[n]);
        beta_sub[n] ~ normal(beta_mu, beta_sd);
        beta = beta_UB * Phi_approx(beta_sub[n]);
        gamma_sub[n] ~ normal(gamma_mu, gamma_sd);
        gamma = Phi_approx(gamma_sub[n]);
        lamda_sub[n] ~ normal(lamda_mu, lamda_sd);
        lamda = Phi_approx(lamda_sub[n]);

        // Initialize state values for all mice
        V = V0[n];

        // Initialize eligibility trace for a mouse
        e = e0;

        for (b in 1:B){
            // Checking if the current episode is valid by looking at the first step of the episode
            if (TrajS[n,b,1] != InvalidState){

                // Begin stepping through the valid episode
                for (step in 2:BL){
                    s = TrajS[n,b,step-1];     // current state
                    sprime = TrajS[n,b,step];  // future state

                    // Checking if the trajectory is still valid
                    if (sprime == InvalidState){
                        // End the current bout/episode and move on to the next one
                        break;
                    }

                    // Select an action
                    // Get a vector of 3 possible state-action values for the state, sprime
                    sIsEndNode = 0;
                    for (i in 1:A){
                        if (nodemap[s+1,i] == InvalidState){
                            // Invalid states as potential options only occur when the current state, s is an end node
                            sIsEndNode = 1;
                            break;  // at an end node there is only one sprime state to transition with
                                    // a probability of 1. i.e. log(prob) = 0
                        }
                        nextS[i] = nodemap[s+1,i];
                        nextV[i] = V[nextS[i]+1];
                    }

                    if (sIsEndNode == 0){
                        sprime_values_beta = beta * nextV;

                        // Update the likelihood of transitioning to state, sprime from state, s
                        TrajA[n,b,step-1] ~ categorical_logit(sprime_values_beta);
                    }

                    // Sample a reward
                    if (sprime == WaterPortState){
                        R = RewardNodeMag;
                    }
                    else{
                        R = 0;
                    }

                    // Calculate error signal for current state
                    td_error = R + gamma * V[sprime+1] - V[s+1];

                    // Update the eligibility trace for the current state, s to 1
                    e[s+1] = 1;

                    // Propagate value to all other states
                    V += alpha * td_error * e;
                    e = gamma * lamda * e;
                }
            }
        }
    }
}
generated quantities{
    real alpha_mu_phi;
    real alpha_sub_phi[N];
    real beta_mu_phi;
    real beta_sub_phi[N];
    real gamma_mu_phi;
    real gamma_sub_phi[N];
    real lamda_mu_phi;
    real lamda_sub_phi[N];
    real log_LL[N];

    // Preparing fitted parameters for output to file and model summary
    alpha_mu_phi = Phi_approx(alpha_mu);
    alpha_sub_phi = Phi_approx(alpha_sub);
    beta_mu_phi = beta_UB * Phi_approx(beta_mu);
    beta_sub_phi = Phi_approx(beta_sub);
    gamma_mu_phi = Phi_approx(gamma_mu);
    gamma_sub_phi = Phi_approx(gamma_sub);
    lamda_mu_phi = Phi_approx(lamda_mu);
    lamda_sub_phi = Phi_approx(lamda_sub);

    for (n in 1:N){
        beta_sub_phi[n] = beta_UB * beta_sub_phi[n];
    }

    // Calculating likelihood
    for (n in 1:N){

        // Initialize model parameters
        int s;
        int a;
        int R;
        int sprime;
        int nextS[A];
        vector[A] nextV;
        vector[3] sprime_values_beta;
        int sIsEndNode;
        real td_error;
        row_vector[S] e;
        row_vector[S] V;

        // Initialize state-values
        V = V0[n];

        // Initialize eligibility trace for a mouse
        e = e0;

        // Initialize likelihood for each subject
        log_LL[n] = 0;

        for (b in 1:B){
            // Checking if the current episode is valid by looking at the first step of the episode
            if (TrajS[n,b,1] != InvalidState){

                // Begin stepping through the valid episode
                for (step in 2:BL){

                    s = TrajS[n,b,step-1];     // current state
                    sprime = TrajS[n,b,step];  // future state

                    // Checking if the trajectory is still valid
                    if (sprime == InvalidState){
                        // End the current bout/episode and move on to the next one
                        break;
                    }

                    // Select an action
                    // Get a vector of 3 possible state-action values for the transition from s -> sprime
                    sIsEndNode = 0;
                    for (i in 1:A){
                        if (nodemap[s+1,i] == InvalidState){
                            // Invalid states as potential options only occur when the current state, s is an end node
                            sIsEndNode = 1;
                            break;  // at an end node there is only one sprime state to transition with
                                    // a probability of 1. i.e. log(prob) = 0
                        }
                        nextS[i] = nodemap[s+1,i];
                        nextV[i] = V[nextS[i]+1];
                    }

                    if (sIsEndNode == 0){
                        sprime_values_beta = beta_sub_phi[n] * nextV;

                        // Update the likelihood of transitioning to state, sprime from state, s
                        log_LL[n] += categorical_logit_lpmf( TrajA[n,b,step-1] | sprime_values_beta );
                    }

                    // Sample a reward
                    if (sprime == WaterPortState){
                        R = RewardNodeMag;
                    }
                    else{
                        R = 0;
                    }

                    // Calculate error signal for current state
                    td_error = R + gamma_sub_phi[n] * V[sprime+1] - V[s+1];

                    // Update the eligibility trace for the current state, s to 1
                    e[s+1] = 1;

                    // Propagate value to all other states
                    V += alpha_sub_phi[n] * td_error * e;
                    e = gamma_sub_phi[n] * lamda_sub_phi[n] * e;
                }
            }
        }
    }
}