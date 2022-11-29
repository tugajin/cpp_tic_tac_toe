#ifndef __GAME_HPP__
#define __GAME_HPP__

#include <array>
#include <bitset>
#include <functional>
#include "common.hpp"
#include "util.hpp"
#include "movelist.hpp"

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
        int self_pieces[POS_SIZE] = {};
        int enemy_pieces[POS_SIZE] = {};
        REP_POS(i) {
            auto piece = hash & 3;
            const auto sq = POS_SIZE - i - 1;
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
    int has_win() const {
        int sq[POS_SIZE] = {};
        this->reach_sq(sq);
        REP_POS(i) {
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
    void legal_moves(movelist::MoveList &ml) const {
        REP_POS(pos) {
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
    uint32 hash_key() const {
        auto hash_key_body = [&](const int sp[], const int ep[]) {
            uint32 k = 0;
            REP_POS(i) {
                k <<= 2;
                if (sp[i] == 1) {
                    k |= 1;
                } else if (ep[i] == 1) {
                    k |= 2;
                } else {
                    k |= 0;
                }
            }
            k <<= 1;
            if (this->turn() == WHITE) {
                k |= 1;
            }
            return k;
        };
        if (this->turn() == BLACK) {
            return hash_key_body(this->self_pieces, this->enemy_pieces);
        } else {
            return hash_key_body(this->enemy_pieces, this->self_pieces);
        }
    }
    std::string str() const {
        std::string str = to_string(std::bitset<19>(this->hash_key())) + "\n";
        str += to_string(this->hash_key()) + "\n";
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
	friend std::ostream& operator<<(std::ostream& os, const Position& pos) {
        os << pos.str();
		return os;
	}
    Feature feature() const {
        Feature feat(FEAT_SIZE, std::vector<int>(POS_SIZE, 0));
        //channel file rank
        int reach_point[POS_SIZE] = {};
        this->reach_sq(reach_point);
        int dangerous_point[POS_SIZE] = {};
        this->dangerous_sq(dangerous_point);
        REP_POS(i) {
            feat[0][i] = this->self(i);
            feat[1][i] = this->enemy(i);
            feat[2][i] = reach_point[i];
            feat[3][i] = dangerous_point[i];
        }
        return feat;
    }
private:
    int self_pieces[POS_SIZE];
    int enemy_pieces[POS_SIZE];
    Color pos_turn;
};
Position from_hash(const uint32 h) {
    return Position(h);
}
void test_pos() {
    
    Position pos;
    movelist::MoveList ml;
    pos.legal_moves(ml);
    ASSERT2(ml.len() == 9, {
        Tee<<pos<<std::endl;
        Tee<<"ml_len:"<<ml.len()<<std::endl;
    });
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());
    ASSERT(pos.is_ok());
    Position pos2 = Position(pos.hash_key());
    ASSERT2(pos.hash_key() == pos2.hash_key(),{
        Tee<<pos<<std::endl;
        Tee<<pos2<<std::endl;
    });

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
    pos2 = Position(pos.hash_key());
    ASSERT2(pos.hash_key() == pos2.hash_key(),{
        Tee<<pos<<std::endl;
        Tee<<pos2<<std::endl;
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
    pos2 = Position(pos.hash_key());
    ASSERT2(pos.hash_key() == pos2.hash_key(),{
        Tee<<pos<<std::endl;
        Tee<<pos2<<std::endl;
    });

    pos = pos.next(Move(1));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 6);
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());
    ASSERT(pos.is_ok());
    pos2 = Position(pos.hash_key());
    ASSERT2(pos.hash_key() == pos2.hash_key(),{
        Tee<<pos<<std::endl;
        Tee<<pos2<<std::endl;
    });

    pos = pos.next(Move(4));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 5);
    ASSERT(!pos.is_done());
    ASSERT(!pos.is_lose());
    ASSERT(pos.is_ok());
    ASSERT(pos.has_win() >= 0);
    pos2 = Position(pos.hash_key());
    ASSERT2(pos.hash_key() == pos2.hash_key(),{
        Tee<<pos<<std::endl;
        Tee<<pos2<<std::endl;
    });

    pos = pos.next(Move(2));
    ml.init();
    pos.legal_moves(ml);
    ASSERT(ml.len() == 4);
    ASSERT2(pos.is_done(),{
        Tee<<pos<<std::endl;
    });
    ASSERT(pos.is_lose());
    ASSERT(pos.is_ok());
    pos2 = Position(pos.hash_key());
    ASSERT2(pos.hash_key() == pos2.hash_key(),{
        Tee<<pos<<std::endl;
        Tee<<pos2<<std::endl;
    });
}
void test_nn() {
    // Position pos;
    // std::array<Move,7> ml = {Move(0), Move(1), Move(4), Move(8), Move(6), Move(2), Move(3)};
    // for (auto m : ml) {
    //     Tee<<pos<<std::endl;
    //     auto f = pos.feature();
    //     auto f0 = torch::tensor(torch::ArrayRef<int>(f[0]));
    //     auto f1 = torch::tensor(torch::ArrayRef<int>(f[1]));
    //     auto f2 = torch::tensor(torch::ArrayRef<int>(f[2]));
    //     auto f3 = torch::tensor(torch::ArrayRef<int>(f[3]));
    //     auto f_all = torch::cat({f0, f1, f2, f3}).reshape({1,4,3,3});
    //     Tee<<f_all<<std::endl;
    //     Tee<<"--------------------------\n";
    //     pos = pos.next(m);
    // }
    // Position pos;
    // std::vector<at::Tensor> tensor_list;
    // auto f = pos.feature();
    // auto f0 = torch::tensor(torch::ArrayRef<int>(f[0]));
    // auto f1 = torch::tensor(torch::ArrayRef<int>(f[1]));
    // auto f2 = torch::tensor(torch::ArrayRef<int>(f[2]));
    // auto f3 = torch::tensor(torch::ArrayRef<int>(f[3]));
    // auto f_all = torch::cat({f0, f1, f2, f3}).reshape({FEAT_SIZE, 3, 3});
    // tensor_list.push_back(f_all);
    // tensor_list.push_back(f_all);
    // torch::Tensor tsr = torch::stack(tensor_list);
    // Tee<<tsr<<std::endl;
}
}
#endif