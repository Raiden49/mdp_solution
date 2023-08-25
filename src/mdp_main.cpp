#include <memory>

#include "mdp_solution/mdp_value_iteration.hpp"
#include "mdp_solution/mdp_policy_iteration.hpp"

int main() {
    double gamma = 0.9;
    double threshold = 0.0001;
    double common_reward = 0;
    
    // value iteration
    std::cout << "************************" << std::endl;
    auto mdp_value = std::make_shared<mdp_solution::MDPValueIteration>(gamma, threshold, common_reward);
    std::cout << "Init map:" << std::endl;
    mdp_value->print_map(mdp_value->map_);
    Eigen::MatrixXd map = mdp_value->value_iteration(mdp_value->map_);
    Eigen::MatrixXi optimal_policy = mdp_value->get_optimal_policy(map);
    std::cout << "The optimal policy:" << std::endl;
    mdp_value->print_map(optimal_policy);

    // policy iteration
    std::cout << "************************" << std::endl;
    auto mdp_policy = std::make_shared<mdp_solution::MDPPolicyIteration>(gamma, threshold, common_reward);
    std::cout << "Init map:" << std::endl;
    mdp_policy->print_map(mdp_policy->map_);
    Eigen::MatrixXi policy = Eigen::MatrixXi::Zero(3, 3);
    policy = mdp_policy->policy_iteration(policy, mdp_policy->map_);
    std::cout << "The optimal policy:" << std::endl;
    mdp_policy->print_map(policy);

    return 0;
}