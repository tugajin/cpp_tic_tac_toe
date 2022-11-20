#ifndef __COMMON_HPP__
#define __COMMON_HPP__
enum Move : int {
    MOVE_NONE = -1
};
enum Color : int {
    BLACK = 1, WHITE = -1
};
inline Color change_turn(const Color turn) {
    return static_cast<Color>(static_cast<int>(turn) * -1);
}
#endif