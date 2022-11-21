#ifndef __COMMON_HPP__
#define __COMMON_HPP__

constexpr int POS_SIZE = 9;
#define FOREACH_POS(i) for (auto (i) = 0; (i) < POS_SIZE; (i)++)

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
    if (v < 0) {
        return false;
    }
    if (v > 9) {
        return false;
    }
    return true;
}
inline bool sq_is_ok(const int sq) {
    return (sq >= 0 && sq <= 2);
}
#endif