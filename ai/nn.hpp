#ifndef __NN_HPP__
#define __NN_HPP__
#include "game.hpp"
#include "common.hpp"
#include <vector>
namespace nn {
constexpr inline int FEAT_SIZE = 2;
typedef std::vector<std::vector<int>> Feature;
typedef double NNScore;

inline NNScore to_nnscore(const float sc) {
    auto score = static_cast<int>(sc * 10000);
    if (score >= 10000) {
        score = 9999;
    } else if (score <= -10000) {
        score = -9999;
    }
    return static_cast<NNScore>(static_cast<double>(score) / 10000.0);
}

Feature feature(const game::Position &pos) {
    Feature feat(FEAT_SIZE, std::vector<int>(SQUARE_SIZE, 0));
    // int reach_point[SQUARE_SIZE] = {};
    // pos.reach_sq(reach_point);
    // int dangerous_point[SQUARE_SIZE] = {};
    // pos.dangerous_sq(dangerous_point);
    // auto reach_sq_num = pos.piece_count(reach_point);
    // auto dangerous_sq_num = pos.piece_count(dangerous_point);
    REP_POS(i) {
        feat[0][i] = pos.self(i);
        feat[1][i] = pos.enemy(i);
        // feat[2][i] = reach_point[i];
        // feat[3][i] = dangerous_point[i];
        // feat[4][i] = (reach_sq_num == 0) ? 1 : 0;
        // feat[5][i] = (reach_sq_num == 1) ? 1 : 0;
        // feat[6][i] = (reach_sq_num == 2) ? 1 : 0;
        // feat[7][i] = (dangerous_sq_num == 0) ? 1 : 0;
        // feat[8][i] = (dangerous_sq_num == 1) ? 1 : 0;
        // feat[9][i] = (dangerous_sq_num == 2) ? 1 : 0;
    }
    return feat;
}
void test_nn() {
}

}
#endif