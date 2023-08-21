#ifndef __MOVELEGAL_HPP__
#define __MOVELEGAL_HPP__

#include "game.hpp"
#include "movelist.hpp"

namespace gen {
void legal_moves(const game::Position &pos, movelist::MoveList &ml) {
    REP_POS(index) {
        if (pos.self(index) == 0 && pos.enemy(index) == 0) {
            ml.add(Move(index));
        }
    }
}
void test_gen() {
#if DEBUG
    REP(i,10000000) {
        auto pos = game::Position();
        while (1) {
            Tee<<pos<<std::endl;
            auto mirror = pos.mirror();
            auto mirror2 = mirror.mirror();
            ASSERT(pos.history() == mirror2.history());

            auto rotate90 = pos.rotate();
            auto rotate180 = rotate90.rotate();
            auto rotate270 = rotate180.rotate();
            auto rotate360 = rotate270.rotate();
            ASSERT(pos.history() == rotate360.history());

            if (pos.is_done()) {
                break;
            }
            movelist::MoveList ml;
            legal_moves(pos,ml);
            const auto action = ml[my_rand(ml.len())];
            pos = pos.next(action);
        }
    }
#endif
}
}

#endif