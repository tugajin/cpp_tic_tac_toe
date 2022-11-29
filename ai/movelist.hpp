#ifndef __MOVELIST_HPP__
#define __MOVELIST_HPP__

#include "common.hpp"
#include "util.hpp"
namespace movelist {
class MoveList {
private:
    static constexpr int MAX_LIST_SIZE = 9;
    Move moves[MAX_LIST_SIZE];
    Move *curr;
public:
    MoveList() {
        this->init();
    }
    void init() {
        this->curr = this->moves;
    }
    int len() const {
        return this->curr - this->moves;
    }
    void add(const Move m) {
        (*this->curr++) = m;
    }
    Move* begin() {
        return this->moves;
    }
    Move* end() {
        return this->curr;
    }
	Move operator [] (const int i) const {
		return *(this->moves+i);
	}
    std::string str() const {
        std::string s;
        REP(i, this->len()) {
            s += to_string(i) + ":" + to_string(static_cast<int>(this->moves[i])) +"\n";
        }
        return s;
    }
	friend std::ostream& operator<<(std::ostream& os,  const MoveList& ml) {
        os << ml.str();
		return os;
	}
};
void test_move_list() {
    MoveList ml;
    ASSERT(ml.len() == 0);
    REP(i, 9) {
        const auto m = Move(i);
        ml.add(m);
        ASSERT2(ml.len() == i+1,{
            Tee<<"ml_len:"<<ml.len()<<std::endl;
            Tee<<"index:"<<i+1<<std::endl;
        });
        ASSERT(ml[i] == m);
    }
    ml.init();
    ASSERT(ml.len() == 0);
}
}
#endif