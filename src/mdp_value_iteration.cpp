#include "mdp_solution/mdp_value_iteration.hpp"

namespace mdp_solution {
Eigen::MatrixXd MDPValueIteration::value_iteration(Eigen::MatrixXd map) {
    std::cout << "Start value iteration" << std::endl;
    Eigen::MatrixXd init_map = map;
    while (1) {
        Eigen::MatrixXd next_map = init_map;
        double error = 0;
        for (int row = 0; row < map.rows(); row++) {
            for (int col = 0; col < map.cols(); col++) {
                if ((row == 0 && col == 2) || (row == 1 && col == 1)) {
                    continue;
                }
                double max_reward = -1 * INFINITY;
                for (int i = 0; i < action_.size(); i++) {
                    double reward = get_reward(map, row, col, i);
                    if (max_reward < reward) {
                        max_reward = reward;
                    }   // for each state, choose the highest value action
                }
                next_map(row, col) = max_reward;
                Eigen::MatrixXd error_map = map - next_map;
                error = error_map.array().abs().maxCoeff();
                // choose the maximum value of the error matrix to ensure that
                // all errors are less than the threshold
            }
        }
        map = next_map;
        print_map(map);
        if (error < threshold_) {
            break;
        }
    }

    return map;
}
Eigen::MatrixXi MDPValueIteration::get_optimal_policy(Eigen::MatrixXd map) {
    Eigen::MatrixXi policy = Eigen::MatrixXi::Zero(3, 3);
    for (int row = 0; row < map.rows(); row++) {
        for (int col = 0; col < map.cols(); col++) {
            if (row == 0 && col == 2) {
                continue;
            }
            int max_action = 0;
            double max_reward = -1 * INFINITY;
            for (int i = 0; i < action_.size(); i++) {
                double reward = get_reward(map, row, col, i);
                if (max_reward < reward) {
                    max_reward = reward;
                    max_action = i;
                } 
            } // record the highest value action as the optimal policy
            policy(row, col) = max_action;
        }
    }

    return policy;
}
}