#include "mdp_solution/mdp_policy_iteration.hpp"

namespace mdp_solution {
Eigen::MatrixXd MDPPolicyIteration::policy_evaluation(Eigen::MatrixXi policy, Eigen::MatrixXd map) {
    Eigen::MatrixXd init_map = map;
    while (1) {
        Eigen::MatrixXd next_map = init_map;
        double error = 0;
        for (int row = 0; row < map.rows(); row++) {
            for (int col = 0; col < map.cols(); col++) {
                if (row == 0 && col == 2) {
                    continue;
                }
                next_map(row, col) = get_reward(map, row, col, policy(row, col));
                Eigen::MatrixXd error_map = map - next_map;
                error = error_map.array().abs().maxCoeff();
                // choose the maximum value of the error matrix to ensure that
                // all errors are less than the threshold
            }
        }
        map = next_map;
        if (error < threshold_) {
            break;
        }
    }

    return map;
}
Eigen::MatrixXi MDPPolicyIteration::policy_iteration(Eigen::MatrixXi policy, Eigen::MatrixXd map) {
    std::cout << "Start policy iteration" << std::endl;
    while (1) {
        map = policy_evaluation(policy, map);
        bool update_flag = false;
        for (int row = 0; row < map.rows(); row++) {
            for (int col = 0; col < map.cols(); col++) {
                if ((row == 0 && col == 2) || (row == 1 && col == 1))  {
                    continue;
                }
                double max_action = 0;
                double max_reward = -1 * INFINITY;
                for (int i = 0; i < action_.size(); i++) {
                    double reward = get_reward(map, row, col, i);
                    if (max_reward < reward) {
                        max_action = i;
                        max_reward = reward;
                    }   // for each state, choose the highest value action
                }
                if (max_reward > get_reward(map, row, col, policy(row, col))) {
                    policy(row, col) = max_action; 
                    update_flag = true;
                }   // record the highest value action as the new policy
            }
        }
        if (!update_flag) {
            break;
        }
        print_map(policy);
    }

    return policy;
}
}