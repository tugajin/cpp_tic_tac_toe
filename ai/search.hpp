#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include <climits>
#include "game.hpp"
#include "common.hpp"
#include "movelegal.hpp"

namespace search {

constexpr int SEARCH_MATE = 10000;
constexpr int SEARCH_MAX  = 20000;
constexpr int SEARCH_MIN  = -SEARCH_MAX;

int search(game::Position &pos, int alpha, int beta, int depth);

Move search_root(game::Position &pos, int depth, int &best_sc) {
    auto best_move = MOVE_NONE;
    auto best_score = -SEARCH_MATE;
    auto alpha = SEARCH_MIN;
    auto beta = SEARCH_MAX;
    movelist::MoveList ml;
    gen::legal_moves(pos,ml);

    for (const auto m : ml) {
        auto next_pos = pos.next(m);
        const auto score = -search(next_pos,-beta, -alpha, depth-1);
        if (score > best_score) {
            best_score = score;
            best_move = m;
            if (score > alpha) {
                alpha = score;
            }
        }
    }
    best_sc = best_score;
    return best_move;
}

int search(game::Position &pos, int alpha, int beta, int depth) {
    ASSERT2(pos.is_ok(),{
        Tee<<pos<<std::endl;
    });
    ASSERT(alpha < beta);
    if (pos.is_draw()) {
        return 0;
    }
    if (pos.is_lose()) {
        return -SEARCH_MATE + 10;
    }
    if (pos.is_win()) {
        return SEARCH_MATE - 10;
    }
    if (depth < 0) {
        return 0;
    }
    movelist::MoveList ml;
    gen::legal_moves(pos, ml);
    auto best_score = SEARCH_MIN;
    for (const auto m : ml) {
        auto next_pos = pos.next(m);
        const auto score = -search(next_pos, -beta, -alpha, depth-1);
        if (score > best_score) {
            best_score = score;
            alpha = score;
            if (score >= beta) {
                return best_score;
            }
        }
    }
    if (best_score == SEARCH_MIN) {
        return -SEARCH_MATE + 7;
    }
    return best_score;
}
void test_search() {
}

}
#endif