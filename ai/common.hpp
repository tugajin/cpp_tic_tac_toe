#ifndef __COMMON_HPP__
#define __COMMON_HPP__

constexpr int POS_SIZE = 9;
#define FOREACH_POS(i) for (auto (i) = 0; (i) < POS_SIZE; (i)++)

typedef std::array<std::array<int,9>,4> Feature;
typedef double NNScore;

enum Move : int {
    MOVE_NONE = -1
};
enum Color : int {
    BLACK = 1, WHITE = -1
};

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
inline void init_feat(Feature &feat) {
    for (auto &f : feat) {
        for (auto &f2 : f) {
            f2 = 0;
        }
    }
}

#endif