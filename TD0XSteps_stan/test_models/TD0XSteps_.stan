data{
    int N;                        // number of rewarded mice
    int B;                        // maximum number of bouts
    int BL;                       // maximum bout length
    int S;                        // number of states
    int A;                        // number of actions available at each state
    int RewardNodeMag;            // reward magnitude at the reward port
    real V0[N,S];                 // initial state values for each mouse
    int nodemap[S,A];             // node labels corresponding to state-action values in the Q lookup table
    int InvalidState;             // placeholder for invalid states
    int HomeNode;                 // node at maze entrance
    int StartNode;                // node at which all episodes begin
    int RewardNode;               // node where reward is located
    int TrajS[N,B,BL];            // trajectories of all mice
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
    int s_current;
    int a;
    int R;
    int s_next;
    int s_next_i;
    int IsCurrentStateEndNode;
    real td_error;
    vector[3] s_next_values_beta;

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
        real lamda;

        // sampling agent parameters
        alpha_sub[n] ~ normal(alpha_mu, alpha_sd);
        alpha = Phi_approx(alpha_sub[n]);
        beta_sub[n] ~ normal(beta_mu, beta_sd);
        beta = beta_UB * Phi_approx(beta_sub[n]);
        gamma_sub[n] ~ normal(gamma_mu, gamma_sd);
        gamma = gamma_UB * Phi_approx(gamma_sub[n]);

        // Initialize state values for this mice
        V = V0[n,:];

        // Loop through each bout
        for (b in 1:B){

            if (TrajS[n,b,1] == InvalidState){
            // when first node of a bout is invalid,
            // no need to check the rest of the bouts
            // they will just be -1s required to fill the matrix
                break;
            }

            // Begin stepping through the bout
            for (step in 2:BL){

                s_current = TrajS[n,b,step-1];  // current state
                s_next = TrajS[n,b,step];   // actual next state

                // Checking if the trajectory is still valid
                if (s_next == InvalidState)
                    break;

                // Sample a reward
                R = (s_next == RewardNode) ? RewardNodeMag : 0;

                // Possible next states by taking an action in state s_current
                IsCurrentStateEndNode = 0;
                for (i in 1:A){
                    s_next_i = nodemap[s_current+1,i];   // s_current+1 because of indexing difference in py and stan
                    if (s_next_i != InvalidState) {
                        IsCurrentStateEndNode = 1;
                        break;
                    }
                    s_next_values_beta[i] = V[s_next_i+1] * beta;
                    print("s_next_i ", s_next_i, "Vs_next_i", V[s_next_i+1], "s_next_beta_i ", s_next_values_beta[i], " s_current ", s_current, " beta ", beta);
                }

                print("probs: ", s_next_values_beta, " current state: ", s_current);

                // Update the likelihood of transitioning from state s to state s_next
                print("log density before =", target(), " n ", n, " b ", b, " step-1 ", step-1);
                if (!IsCurrentStateEndNode)
                    TrajA[n,b,step-1] ~ categorical_logit(s_next_values_beta);
                print("log density after =", target(), " n ", n, " b ", b, " step-1 ", step-1);

                // Calculate error signal for current state
                td_error = R + gamma * V[s_next+1] - V[s_current+1];

                // Propagate value to all other states
                for (j in 1:S){
                    V[s_current+1] += alpha * td_error;
                }

                // Check if current state is a terminal state
                // if (s_current == HomeNode || s_current == RewardNode)
                //    break;
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
    }

    // Calculating likelihood
    for (n in 1:N){

        // Initialize model parameters
        int s_current;
        int a_true;
        int R;
        int s_next;
        int s_next_i;
        int IsCurrentStateEndNode;
        real td_error;
        vector[3] s_next_values_beta;
        real V[S];

        // Initialize state values for this mice
        V = V0[n,:];

        // Initialize likelihood for each subject
        log_LL[n] = 0;

        // Loop through each bout
        for (b in 1:B){

            if (TrajS[n,b,1] == InvalidState){
            // when first node of a bout is invalid,
            // no need to check the rest of the bouts
            // they will just be -1s required to fill the matrix
                break;
            }

            // Begin stepping through the bout
            for (step in 2:BL){

                s_current = TrajS[n,b,step-1];  // current state
                a_true = TrajA[n,b,step-1];      // true action taken in current state
                s_next = TrajS[n,b,step];       // true next state

                // Checking if the trajectory is still valid
                if (s_next == InvalidState)
                    break;

                // Sample a reward
                R = (s_next == RewardNode) ? RewardNodeMag : 0;

                // Possible next states by taking an action i in state s_current
                IsCurrentStateEndNode = 0;
                for (i in 1:A){
                    s_next_i = nodemap[s_current+1,i];
                    if (s_next_i != InvalidState) {
                        IsCurrentStateEndNode = 1;
                        break;
                    }
                    s_next_values_beta[i] = V[s_next_i+1] * beta_sub_phi[n];;
                }

                // Update the likelihood of choosing action a_true in state s_current
                print("log_LL before =", log_LL[n], " n ",n," b ",b," step-1 ",step-1," a_true ",a_true," s_next_values_beta ",s_next_values_beta);
                if (!IsCurrentStateEndNode)
                    log_LL[n] += categorical_logit_lpmf( a_true | s_next_values_beta );
                print("log_LL after =", log_LL[n], " n ",n," b ",b," step-1 ",step-1," a_true ",a_true," s_next_values_beta ",s_next_values_beta);

                // Calculate error signal for current state
                td_error = R + gamma_sub_phi[n] * V[s_next+1] - V[s_current+1];

                // Propagate value to all other states
                for (j in 1:S){
                    V[s_current+1] += alpha_sub_phi[n] * td_error;
                }

                // Check if current state is a terminal state
                // if (s_current == HomeNode || s_current == RewardNode)
                //    break;
            }
        }
    }
}