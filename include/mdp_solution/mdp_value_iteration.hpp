#ifndef MDP_VALUE_ITERATION_HPP_
#define MAP_VALUE_ITERATION_HPP_

#include "mdp_interface.hpp"

namespace mdp_solution {
class MDPValueIteration : public MDPInterface {
    public:
        /**
         * @brief Construct a new MDPValueIteration object
         * 
         * @param gamma Discount factor
         * @param threshold Threshold for the end of the algorithm
         * @param common_reward The value of non-traps and target states
         */
        MDPValueIteration(double gamma, double threshold, double common_reward) 
            : MDPInterface(gamma, threshold,  common_reward) {}
        /**
         * @brief value iteration function 
         * 
         * @param map Map with information about each state
         * @return Eigen::MatrixXd. Iterated map information
         */
        Eigen::MatrixXd value_iteration(Eigen::MatrixXd map);
        /**
         * @brief Get the optimal policy function
         * 
         * @param map Map with information about each state
         * @return Eigen::MatrixXi. The optimal policy
         */
        Eigen::MatrixXi get_optimal_policy(Eigen::MatrixXd map);
};
}

#endif // MAP_VALUE_ITERATION_HPP_