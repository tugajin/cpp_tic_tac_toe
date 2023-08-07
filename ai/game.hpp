#ifndef __GAME_HPP__
#define __GAME_HPP__

#include <array>
#include <bitset>
#include <functional>
#include "common.hpp"
#include "util.hpp"
#include "movelist.hpp"
namespace game {
class Position;
}
namespace hash {
Key hash_key(const game::Position &pos);
}
namespace game {
class Position {
public:
    Position() {
        REP_POS(i){
            this->self_pieces[i] = this->enemy_pieces[i] = 0;
        }
        this->pos_turn = BLACK;
    }
    Position(const int self_pieces[], const int enemy_pieces[], const Color turn) {
        REP_POS(i) {
            this->self_pieces[i] = self_pieces[i];
            this->enemy_pieces[i] = enemy_pieces[i];
        }
        this->pos_turn = turn;
    }
    Position(const uint32 h) {
        auto hash = h;
        if (hash & 1) {
            this->pos_turn = WHITE;
        } else {
            this->pos_turn = BLACK;
        }
        hash >>= 1;
        int self_pieces[SQUARE_SIZE] = {};
        int enemy_pieces[SQUARE_SIZE] = {};
        REP_POS(i) {
            auto piece = hash & 3;
            const auto sq = SQUARE_SIZE - i - 1;
            if (piece == 0) {
            } else if (piece == 1) {
                self_pieces[sq] = 1;
            } else if (piece == 2) {
                enemy_pieces[sq] = 1;
            }
            hash >>= 2;
        }
        REP_POS(i) {
            if (this->turn() == BLACK) {
                this->self_pieces[i] = self_pieces[i];
                this->enemy_pieces[i] = enemy_pieces[i];
            } else {
                this->self_pieces[i] = enemy_pieces[i];
                this->enemy_pieces[i] = self_pieces[i];
            }
        }
    }
    int piece_count(const int pieces[]) const {
        auto count = 0;
        REP_POS(i) {
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
        REP(i,3) {
            if (is_comp(i,0,0,1)) { return true; }
            if (is_comp(0,i,1,0)) { return true; }
        }
        return false;
    }
    void find_special_sp(int square[], std::function<void(const int x, const int y, const int inc_x, const int inc_y)> func) const {
        REP_POS(i) { square[i] = 0; }
        func(0,0,1,1);
        func(0,2,1,-1);
        REP(i,3) {
            func(i,0,0,1);
            func(0,i,1,0);
        }
    }
    void reach_sq(int square[]) const {
        auto func = [&](const int x, const int y, const int inc_x, const int inc_y) {
            auto self_num = 0;
            auto reach_num = 0;
            int reach_point[8] = {};
            for (int sq_x = x, sq_y = y; sq_is_ok(sq_x) && sq_is_ok(sq_y); sq_x += inc_x, sq_y += inc_y) {
                const auto sq = sq_x *3 + sq_y;
                if (this->self(sq) == 1) {
                    self_num++;
                }
                if (this->self(sq) == 0 && this->enemy(sq) == 0) {
                    reach_point[reach_num++] = sq;
                }
            }
            if (self_num == 2) {
                REP(i, reach_num) {
                    square[reach_point[i]] = 1;
                }
            }
        };
        this->find_special_sp(square,func);
    }
    void dangerous_sq(int square[]) const {
        auto func = [&](const int x, const int y, const int inc_x, const int inc_y) {
            auto enemy_num = 0;
            auto dangerous_num = 0;
            int dangerous_point[8] = {};
            for (int sq_x = x, sq_y = y; sq_is_ok(sq_x) && sq_is_ok(sq_y); sq_x += inc_x, sq_y += inc_y) {
                const auto sq = sq_x *3 + sq_y;
                if (this->enemy(sq) == 1) {
                    enemy_num++;
                }
                if (this->self(sq) == 0 && this->enemy(sq) == 0) {
                    dangerous_point[dangerous_num++] = sq;
                }
            }
            if (enemy_num == 2) {
                REP(i, dangerous_num) {
                    square[dangerous_point[i]] = 1;
                }
            }
        };
        this->find_special_sp(square,func);
    }
    bool is_win() const {
        int sq[SQUARE_SIZE] = {};
        this->reach_sq(sq);
        REP_POS(i) {
            if (sq[i] == 1) {
                return true;
            }
        }
        return false;
    }
    bool is_draw() const {
        return this->all_piece_count() == SQUARE_SIZE;
    }
    bool is_done() const {
        return this->is_lose() || this->is_draw();
    }
    int ply() const {
        return this->all_piece_count();
    }
    Position next(const Move action) const {
        auto p = Position(this->enemy_pieces, this->self_pieces, change_turn(this->pos_turn));
        p.enemy_pieces[action] = 1;
        return p;
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
        REP_POS(i) {
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

    std::string str() const {
        std::string str = to_string(std::bitset<19>(this->history())) + "\n";
        str += to_string(this->history()) + "\n";
        str += color_str(this->turn()) + "\n";
        REP(x, 3) {
            REP(y, 3) {
                const auto sq = x * 3 + y;
                if (this->turn() == BLACK) {
                    if (this->self_pieces[sq] == 1) {
                        str += "o";
                    } else if (this->enemy_pieces[sq] == 1) {
                        str += "x";
                    } else {
                        str += "-";
                    }
                } else {
                    if (this->self_pieces[sq] == 1) {
                        str += "x";
                    } else if (this->enemy_pieces[sq] == 1) {
                        str += "o";
                    } else {
                        str += "-";
                    }
                }
            }
            str += "\n";
        }
        return str;
    }
    Key history() const {
        return hash::hash_key(*this);
    }
	friend std::ostream& operator<<(std::ostream& os, const Position& pos) {
        os << pos.str();
		return os;
	}

private:
    int self_pieces[SQUARE_SIZE];
    int enemy_pieces[SQUARE_SIZE];
    Color pos_turn;
};

void test_pos() {
}    
void test_nn() {
}
}
#endif