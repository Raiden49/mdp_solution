#ifndef MDP_INTERFACE_HPP_
#define MDP_INTERFACE_HPP_

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace mdp_solution {
class MDPInterface {
    public:
        /**
         * @brief Construct a new MDPInterface object
         * 
         * @param gamma Discount factor
         * @param thread Threshold for the end of the algorithm
         * @param common_reward The value of non-traps and target states
         */
        MDPInterface(double gamma, double threshold, double common_reward): 
            gamma_(gamma), threshold_(threshold), common_reward_(common_reward) {
            map_(1, 1) = trap_reward_;   // trap
            map_(0, 2) = 10;    // target
            action_.push_back(std::make_pair(-1, 0));   // up
            action_.push_back(std::make_pair(1, 0));    // down
            action_.push_back(std::make_pair(0, -1));   // left
            action_.push_back(std::make_pair(0, 1));    // right
        }
        /**
         * @brief Output environmental information
         * 
         * @param map Map with information about each state
         * @param print_policy Determining whether to print the policy
         */
        void print_map(Eigen::MatrixXd map, bool print_policy = false);
        void print_map(Eigen::MatrixXi map, bool print_policy = true);
        
        /**
         * @brief Get the map info object
         * 
         * @param map Map with information about each state
         * @param row A row of the map matrix
         * @param col A col of the map matrix
         * @param action Action taken
         * @return int. Status information after executing an action
         */
        int get_map_info(Eigen::MatrixXd map, int row, int col, int action);

        /**
         * @brief Get the reward object
         * 
         * @param map Map with information about each state
         * @param row A row of the map matrix
         * @param col A col of the map matrix
         * @param action Action taken
         * @return double. Get the reward value after taking an action
         */
        double get_reward(Eigen::MatrixXd map, int row, int col, int action);
    public:
        Eigen::MatrixXd map_ = Eigen::MatrixXd::Zero(3, 3);
        std::vector<std::pair<int, int>> action_;
        std::string action_info[4] = {"up    ", "down  ", "left  ", "right "};
        double gamma_;
        double threshold_;
        double common_reward_;
        double trap_reward_ = -10;
};
}

#endif // MDP_INTERFACE_HPP_