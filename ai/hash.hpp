#ifndef __HASH_HPP__
#define __HASH_HPP__
#include "game.hpp"

namespace hash {

inline constexpr Key START_HASH_KEY = 0UL;

game::Position from_hash(const Key key) {
    return game::Position(key);
}
Key hash_key(const game::Position &pos) {
    Key k = Key(0);
    REP_POS(i) {
        k <<= 2;
        const auto sp = (pos.turn() == BLACK) ? pos.self(i) : pos.enemy(i);
        const auto ep = (pos.turn() == BLACK) ? pos.enemy(i) : pos.self(i);
        if (sp == 1) {
            k |= 1;
        } else if (ep == 1) {
            k |= 2;
        } else {
            k |= 0;
        }
    }
    k <<= 1;
    if (pos.turn() == WHITE) {
        k |= 1;
    }
    return k;
}

game::Position hirate() {
    return from_hash(START_HASH_KEY);
}

void test_hash() {
}
}
#endif