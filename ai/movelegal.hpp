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
}

#endif