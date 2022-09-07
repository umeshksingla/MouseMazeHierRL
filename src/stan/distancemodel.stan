data {
    int K;
    int B;
    int N_CELLS;
    int N_NODES;
    int N_ACTIONS;
    int MAX_BL;
    int TrajS[B, MAX_BL];            // trajectories of a mice
    int TrajL[B];
    int TrajA[B, MAX_BL-1];
    int INVALID_STATE;

    int NODE_CELL_MAPPING[N_NODES];
    int ACTION_NODE_MATRIX[N_NODES,N_NODES,N_ACTIONS];
    row_vector[2] CELL_XY[N_CELLS];
}

parameters {
  vector<lower=-1, upper=1>[K] c;
}

//transformed parameters {
//  vector[K] c = append_row(c_raw, 1-sum(c_raw));
//}

model {
    vector[N_ACTIONS] V;
    int prev_n;
    int n;
    int next_n;
    int actual_next_n;
    int actual_a;

    row_vector[2] choice_cell_coors;
    row_vector[2] p_cell_coors;
    int next_possible_action_nodes[N_ACTIONS];

    vector[K] d;

    c ~ normal(0, 1);

    for (b in 1:B) {
        for (step in K:TrajL[b]-1) {
            prev_n = TrajS[b, step-1];
            n = TrajS[b, step];
            actual_next_n = TrajS[b, step+1];
            actual_a = TrajA[b, step]+1;
            next_possible_action_nodes = ACTION_NODE_MATRIX[prev_n+1][n+1];
            //print(">>> ns: ", prev_n, " ", n, " ", next_possible_action_nodes, " ", actual_next_n, " ", actual_a);
            for (a in 1:size(next_possible_action_nodes)) {
                next_n = next_possible_action_nodes[a];
                if (next_n == INVALID_STATE) {
                    V[a] = -1e10;
                    continue;
                }
                choice_cell_coors = CELL_XY[NODE_CELL_MAPPING[next_n+1]+1];
                // print("coors: ", next_n, " ", choice_cell_coors)
                for (i in step-K+1:step) {
                    // print("i ",i, TrajS[b, i]);
                    p_cell_coors = CELL_XY[NODE_CELL_MAPPING[TrajS[b, i]+1]+1];
                    d[step-i+1] = distance(choice_cell_coors, p_cell_coors);
                }
                //print("c ", c, " d ", d);
                V[a] = dot_product(c, d);
                //print("a:", a, " ", next_n, " ", V[a]);
            }
            //print("actual a: ", TrajA[b, step]+1, " values: ", V);
            //print(target());
            actual_a ~ categorical_logit(V);
            //print(target());
        }
    }
}
