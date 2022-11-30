#ifndef __COMMON_HPP__
#define __COMMON_HPP__

constexpr int POS_SIZE = 9;
constexpr int FEAT_SIZE = 10;

#define REP(i,e) for (auto (i) = 0; (i) < (e); ++(i))
#define REP_POS(i) for (auto (i) = 0; (i) < POS_SIZE; ++(i))

typedef std::vector<std::vector<int>> Feature;
typedef double NNScore;

enum Move : int {
    MOVE_NONE = -1
};
enum Color : int {
    BLACK = 1, WHITE = -1
};
std::string color_str(const Color c) {
    if (c == BLACK) {
        return "BLACK";
    } else {
        return "WHITE";
    }
}
std::ostream& operator<<(std::ostream& os, const Color c) {
    os << color_str(c);
    return os;
}
inline Color change_turn(const Color turn) {
    return static_cast<Color>(static_cast<int>(turn) * -1);
}
inline bool move_is_ok(const Move m) {
    const auto v = static_cast<int>(m);
    return (v >= 0 && v <= 8);
}
inline bool sq_is_ok(const int sq) {
    return (sq >= 0 && sq <= 2);
}

NNScore to_nnscore(const float sc) {
    auto score = static_cast<int>(sc * 10000);
    if (score >= 10000) {
        score = 9999;
    } else if (score <= -10000) {
        score = -9999;
    }
    return static_cast<NNScore>(static_cast<double>(score) / 10000.0);
}

#endif