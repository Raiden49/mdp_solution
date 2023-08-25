#ifndef MDP_POLICY_ITERATION_HPP_
#define MDP_POLICY_ITERATION_HPP_

#include "mdp_solution/mdp_interface.hpp"

namespace mdp_solution {
class MDPPolicyIteration : public MDPInterface {
    public:
        /**
         * @brief Construct a new MDPPolicyIteration object
         * 
         * @param gamma Discount factor
         * @param threshold Threshold for the end of the algorithm
         * @param common_reward The value of non-traps and target states
         */
        MDPPolicyIteration(double gamma, double threshold, double common_reward) 
            : MDPInterface(gamma, threshold,  common_reward) {}
        /**
         * @brief Policy evaluation function 
         * 
         * @param policy Iterated policy matrix
         * @param map Map with information about each state
         * @return Eigen::MatrixXd. Iterated map information
         */
        Eigen::MatrixXd policy_evaluation(Eigen::MatrixXi policy, Eigen::MatrixXd map);
        /**
         * @brief Get the optimal policy function
         * 
         * @param policy Iterated policy matrix
         * @param map Map with information about each state
         * @return Eigen::MatrixXi. The optimal policy
         */
        Eigen::MatrixXi policy_iteration(Eigen::MatrixXi policy, Eigen::MatrixXd map);
};
}

#endif // MDP_POLICY_ITERATION_HPP_