data{
    int N;                        // number of rewarded mice
    int B;                        // maximum number of bouts until the first reward was sampled
    int BL;                       // maximum bout length until the first reward was sampled
    int S;                        // number of states/nodes in the maze
    int A;                        // number of actions available at each state
    int RewardNodeMag;            // reward magnitude at the reward port (node 116)
    real V0[N,S];               // initial state values for each mouse
    int nodemap[S,A];           // node labels corresponding to state-action values in the Q lookup table
    int InvalidState;             // placeholder for invalid states
    int HomeNode;                 // node at maze entrance
    int StartNode;                // node at which all episodes begin
    int RewardNode;               // node where liquid reward is located
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
    real gamma_mu;
    real <lower = 0.001> gamma_sd;

    // agent parameters
    real alpha_sub[N];
    real beta_sub[N];
    real gamma_sub[N];
}
model{
    // Declare model parameters
    real V[S];
    int s;
    int a;
    int R;
    int sprime;
    int nextStateOption;
    int sIsEndNode;
    vector[3] sprime_values_beta;

    // sampling population level parameters
    alpha_mu ~ normal(0, 1);
    alpha_sd ~ normal(0, 1);
    beta_mu ~ normal(0, 1);
    beta_sd ~ normal(0, 1);
    gamma_mu ~ normal(0, 1);
    gamma_sd ~ normal(0, 1);

    for (n in 1:N){
        real alpha;
        real beta;
        real gamma;

        // sampling agent parameters
        alpha_sub[n] ~ normal(alpha_mu, alpha_sd);
        alpha = Phi_approx(alpha_sub[n]);
        beta_sub[n] ~ normal(beta_mu, beta_sd);
        beta = beta_UB * Phi_approx(beta_sub[n]);
        gamma_sub[n] ~ normal(gamma_mu, gamma_sd);
        gamma = gamma_UB * Phi_approx(gamma_sub[n]);

        // Initialize state values for a mouse
        V = V0[n,:];

        // Initialize starting state, s0 to node 0
        s = 0;

        for (b in 1:B){
            // Begin an episode
            for (step in 2:BL){
                // Begin stepping through the episode

                // Transition to the next state
                sprime = TrajS[n,b,step];

                // Checking if the trajectory is still valid
                if (sprime != InvalidState){
                    // Sample a reward
                    if (sprime == RewardNode){
                        R = RewardNodeMag;
                    }
                    else{
                        R = 0;
                    }

                    // Select an action
                    // Get a vector of 3 possible state-action values for the state, sprime
                    sIsEndNode = 0;
                    for (i in 1:A){
                        nextStateOption = nodemap[s+1,i];
                        if (nextStateOption == InvalidState){
                            // Invalid states as potential options only occur when the current state, s is an end node
                            sIsEndNode = 1;
                            break;  // at an end node there is only one sprime state to transition with
                                    // a probability of 1. i.e. log(prob) = 0
                        }
                        else{
                            sprime_values_beta[i] = V[nextStateOption + 1] * beta;
                        }
                    }

                    // Update the likelihood of transitioning to state, sprime from state, s
                    if (sIsEndNode == 0){
                        TrajA[n,b,step-1] ~ categorical_logit(sprime_values_beta);
                    }

                    // Update previous state-action value
                    V[s+1] += alpha * (R + gamma * V[sprime+1] - V[s+1]);

                    // Update new state and action to old state and action
                    s = sprime;

                    // Check if current state is a terminal state
                    if (s == HomeNode || s == RewardNode){
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
generated quantities{
    real alpha_mu_phi;
    real beta_mu_phi;
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
        int nextStateOption;
        vector[3] sprime_values_beta;
        real V[S];

        // Initialize starting state, s0 to node 0 and state-action values
        s = 0;
        V = V0[n,:];

        // Initialize likelihood for each subject
        log_LL[n] = 0;

        for (b in 1:B){
            // Begin episode
            for (step in 2:BL){
                // Begin stepping through the episode

                // Transition to the next state
                sprime = TrajS[n,b,step];

                // Checking if the trajectory is still valid
                if (sprime != InvalidState){
                    // Sample a reward
                    if (sprime == RewardNode){
                        R = RewardNodeMag;
                    }
                    else{
                        R = 0;
                    }

                    // Select an action
                    // Get a vector of 3 possible state-action values for the transition from s -> sprime
                    sIsEndNode = 0;
                    for (i in 1:A){
                        nextStateOption = nodemap[s+1,i];
                        if (nextStateOption == InvalidState){
                            // Invalid states as potential options only occur when the current state, s is an end node
                            sIsEndNode = 1;
                            break;  // at an end node there is only one sprime state to transition with
                                    // a probability of 1. i.e. log(prob) = 0
                        }
                        else{
                            sprime_values_beta[i] = V[nextStateOption + 1] * beta_sub_phi[n];
                        }
                    }

                    // Update the likelihood of choosing action, aprime from state sprime
                    if (sIsEndNode == 0){
                        log_LL[n] += categorical_logit_lpmf( TrajA[n,b,step-1] | sprime_values_beta );
                    }

                    // Update previous state-action value
                    V[s+1] += alpha_sub_phi[n] * (R + gamma_sub_phi[n] * V[sprime+1] - V[s+1]);

                    // Update new state and action to old state and action
                    s = sprime;

                    // Check if current state is a terminal state
                    if (s == HomeNode || s == RewardNode){
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