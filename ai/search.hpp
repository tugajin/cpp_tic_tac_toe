#include <unordered_map>
#include "game.hpp"
#include "common.hpp"

int search(Position &pos, int alpha, int beta) {
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
    pos.legal_moves(ml);
    auto best_score = -1;
    for (const auto m : ml) {
        ASSERT2(move_is_ok(m),{
            Tee<<pos<<std::endl;
            Tee<<m<<std::endl;
        });
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
uint32 perft(Position &pos, std::unordered_map<uint32, int> &hash) {
    ASSERT2(pos.is_ok(),{
        Tee<<pos<<std::endl;
    });
    if (hash.count(pos.hash_key()) == 0) {
        hash[pos.hash_key()] = 1;
    }
    if (pos.is_draw()) {
        return 1;
    }
    if (pos.is_lose()) {
        return 1;
    }
    MoveList ml;
    pos.legal_moves(ml);
    auto node_num = 0u;
    for (const auto m : ml) {
        ASSERT2(move_is_ok(m),{
            Tee<<pos<<std::endl;
            Tee<<m<<std::endl;
        });
        auto next_pos = pos.next(m);
        node_num += perft(next_pos, hash);
    }
    return node_num;
}
void test_search() {
    std::unordered_map<uint32,int> hash;
    Position pos;
    Tee<<perft(pos,hash)<<std::endl;
    Tee<<hash.size()<<std::endl;
}