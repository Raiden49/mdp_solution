#include "mdp_solution/mdp_interface.hpp"

namespace mdp_solution {
void MDPInterface::print_map(Eigen::MatrixXd map, bool print_policy) {
    std::string str = "";
    std::string temp = "";
    for (int row = 0; row < map.rows(); row++) {
        str += "|";
        for (int col = 0; col < map.cols(); col++) {
            temp = std::to_string(map(row, col));
            str += " " + temp + " |";
        }
        str += "\n";
    }
    std::cout << str << std::endl;
}
void MDPInterface::print_map(Eigen::MatrixXi map, bool print_policy) {
    std::string str = "";
    std::string temp = "";
    for (int row = 0; row < map.rows(); row++) {
        str += "|";
        for (int col = 0; col < map.cols(); col++) {
            if (row == 0 && col == 2) {
                temp = "target";
            }
            else {
                temp = action_info[map(row, col)];
            }
            str += " " + temp + " |";
        }
        str += "\n";
    }
    std::cout << str << std::endl;
}
int MDPInterface::get_map_info(Eigen::MatrixXd map, int row, int col, int action) {
    int drow = action_[action].first;
    int dcol = action_[action].second;
    row = row + drow;
    col = col + dcol;
    if (row < 0 || col < 0 || row >= map.rows() || col >= map.cols()) {
        return map(row - drow, col - dcol);
    }
    else {
        return map(row, col);
    }
}
double MDPInterface::get_reward(Eigen::MatrixXd map, int row, int col, int action) {
    double reward = common_reward_;
    reward += gamma_ * get_map_info(map, row, col, action); 
    // v = reward + gamma * P_pi * v
    // reward = common_reward_
    // P_pi = 1(matrix)
    
    return reward;
}
}