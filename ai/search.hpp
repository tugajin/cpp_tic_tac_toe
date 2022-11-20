#include "game.hpp"
#include "common.hpp"

int search(Position pos, int alpha, int beta) {
    ASSERT2(pos.is_ok(),{
        Tee<<pos<<std::endl;
    });
    if (pos.is_draw()) {
        return 0;
    }
    if (pos.is_lose()) {
        return -1;
    }
    MoveList ml;
    pos.legal_actions(ml);
    auto best_score = -1;
    for (const auto m : ml) {
        auto next_pos = pos.next(m);
        const auto score = -search(next_pos, -beta, -alpha);
        if (score > best_score) {
            best_score = score;
            alpha = score;
            if (score >= beta) {
                return score;
            }
        }
    }
    return best_score;
}
void test_search() {
    Position pos;
    Tee<<search(pos,-3,3)<<std::endl;
}