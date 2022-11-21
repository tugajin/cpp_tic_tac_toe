#ifndef __GAME_HPP__
#define __GAME_HPP__

#include "common.hpp"
#include "util.hpp"
#include "movelist.hpp"


class Position {
public:
    Position() {
        FOREACH_POS(i){
            this->self_pieces[i] = this->enemy_pieces[i] = 0;
        }
        this->pos_turn = BLACK;
    }
    Position(const int self_pieces[], const int enemy_pieces[], const Color turn) {
        FOREACH_POS(i) {
            this->self_pieces[i] = self_pieces[i];
            this->enemy_pieces[i] = enemy_pieces[i];
        }
        this->pos_turn = turn;
    }
    int piece_count(const int pieces[]) const {
        auto count = 0;
        FOREACH_POS(i) {
            if (pieces[i] == 1) {
                count++;
            }
        }
        return count;
    }
    int all_piece_count() const {
        return this->piece_count(this->self_pieces) + this->piece_count(this->enemy_pieces);
    }
    bool is_lose() const {
        auto is_comp = [&](const int x, const int y, const int inc_x, const int inc_y) {
            for (int sq_x = x, sq_y = y; sq_is_ok(sq_x) && sq_is_ok(sq_y); sq_x += inc_x, sq_y += inc_y) {
                if (this->enemy(sq_x *3 + sq_y) == 0) {
                    return false;
                }
            }
            return true;
        };
        if (is_comp(0,0,1,1)) { return true; }
        if (is_comp(0,2,1,-1)) { return true; }
        for (auto i = 0; i < 3; i++) {
            if (is_comp(i,0,0,1)) { return true; }
            if (is_comp(0,i,1,0)) { return true; }
        }
        return false;
    }
    void dangerous_sq(int square[]) const {
        auto find_dangerous_sq = [&](const int x, const int y, const int inc_x, const int inc_y) {
            auto self_num = 0;
            auto dangerous_num = 0;
            int dangerous_point[8] = {};
            for (int sq_x = x, sq_y = y; sq_is_ok(sq_x) && sq_is_ok(sq_y); sq_x += inc_x, sq_y += inc_y) {
                const auto sq = sq_x *3 + sq_y;
                if (this->self(sq) == 1) {
                    self_num++;
                }
                if (this->self(sq) == 0 && this->enemy(sq) == 0) {
                    dangerous_point[dangerous_num++] = sq;
                }
            }
            if (self_num == 2) {
                for (auto i = 0; i < dangerous_num; i++) {
                    square[dangerous_point[i]] = 1;
                }
            }
        };
        FOREACH_POS(i) { square[i] = 0; }
        find_dangerous_sq(0,0,1,1);
        find_dangerous_sq(0,2,1,-1);
        for (auto i = 0; i < 3; i++) {
            find_dangerous_sq(i,0,0,1);
            find_dangerous_sq(0,i,1,0);
        }
    }
    int has_win() const {
        int sq[POS_SIZE] = {};
        this->dangerous_sq(sq);
        FOREACH_POS(i) {
            if (sq[i] == 1) {
                return i;
            }
        }
        return -1;
    }
    bool is_draw() const {
        return this->all_piece_count() == POS_SIZE;
    }
    bool is_done() const {
        return this->is_lose() || this->is_draw();
    }
    Position next(const Move action) const {
        auto p = Position(this->enemy_pieces, this->self_pieces, change_turn(this->pos_turn));
        p.enemy_pieces[action] = 1;
        return p;
    }
    void legal_moves(MoveList &ml) const {
        FOREACH_POS(pos) {
            if (this->self_pieces[pos] == 0 && this->enemy_pieces[pos] == 0) {
                ml.add(Move(pos));
            }
        }
    }
    Color turn() const {
        return this->pos_turn;
    }
    int self(const int sq) const {
        return this->self_pieces[sq];
    }
    int enemy(const int sq) const {
        return this->enemy_pieces[sq];
    }
    bool is_ok() const {
        if (this->pos_turn != BLACK && this->pos_turn != WHITE) {
            return false;
        }
        FOREACH_POS(i) {
            if (this->self(i) == 1 && this->enemy(i) == 1) {
                return false;
            }
        }
        const auto self_num = this->piece_count(this->self_pieces);
        const auto enemy_num = this->piece_count(this->enemy_pieces);
        if (this->turn() == BLACK) {
            if (self_num != enemy_num) {
                return false;
            }
        } else {
            if (self_num + 1 != enemy_num) {
                return false;
            }
        }
        return true;
    }
    uint32 hash_key() const {
        auto hash_key_body = [&](const int sp[], const int ep[]) {
            uint32 k = 0;
            FOREACH_POS(i) {
                k *= 10;
                if (sp[i] == 1) {
                    k += 2;
                } else if (ep[i] == 1) {
                    k += 3;
                } else {
                    k += 1;
                }
            }
            return k;
        };
        if (this->turn() == BLACK) {
            return hash_key_body(this->self_pieces, this->enemy_pieces);
        } else {
            return hash_key_body(this->enemy_pieces, this->self_pieces);
        }
    }
	friend std::ostream& operator<<(std::ostream& os, const Position& pos) {
        os << (pos.hash_key())<<std::endl;
        if (pos.turn() == BLACK) {
            os << "BLACK" << std::endl;
        } else {
            os << "WHITE" << std::endl;
        }
        for (auto x = 0; x < 3; x ++) {
            for (auto y = 0; y < 3; y++) {
                const auto sq = x * 3 + y;
                if (pos.turn() == BLACK) {
                    if (pos.self_pieces[sq] == 1) {
                        os << "o";
                    } else if (pos.enemy_pieces[sq] == 1) {
                        os << "x";
                    } else {
                        os << "-";
                    }
                } else {
                    if (pos.self_pieces[sq] == 1) {
                        os << "x";
                    } else if (pos.enemy_pieces[sq] == 1) {
                        os << "o";
                    } else {
                        os << "-";
                    }
                }
            }
            os << std::endl;
        }
		return os;
	}
private:
    int self_pieces[POS_SIZE];
    int enemy_pieces[POS_SIZE];
    Color pos_turn;
};
void test_pos() {
    
    Position pos;
    MoveList ml;
    pos.legal_moves(ml);
    ASSERT2(ml.len() == 9, {
        Tee<<pos<<std::endl;
        Tee<<"ml_len:"<<ml.len()<<std::endl;
    });
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());
    ASSERT(pos.is_ok());

    pos = pos.next(ml[0]);
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 8);
    ASSERT(pos.turn() == WHITE);
    ASSERT(pos.is_ok());
    ASSERT2(pos.enemy(0) == 1,{
        Tee<<pos<<std::endl;
        Tee<<pos.self(0)<<std::endl;
        Tee<<pos.enemy(0)<<std::endl;
    });
    
    pos = pos.next(Move(3));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 7);
    ASSERT(pos.is_ok());
    ASSERT2(pos.enemy(3) == 1,{
        Tee<<pos<<std::endl;
        Tee<<pos.self(0)<<std::endl;
        Tee<<pos.enemy(0)<<std::endl;
    });
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());

    pos = pos.next(Move(1));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 6);
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());
    ASSERT(pos.is_ok());

    pos = pos.next(Move(4));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 5);
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());
    ASSERT(pos.is_ok());
    ASSERT(pos.has_win() >= 0);

    pos = pos.next(Move(2));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 4);
    ASSERT2(pos.is_done(),{
        Tee<<pos<<std::endl;
    });
    ASSERT(pos.is_lose());
    ASSERT(pos.is_ok());
}
#endif