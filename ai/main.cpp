#include <iostream>
#include <torch/torch.h>
#include "game.hpp"
#include "movelist.hpp"
#include "util.hpp"
#include "search.hpp"

TeeStream Tee;

int main(int argc, char *argv[]){
    test_move_list();
    test_pos();
    test_search();
    return 0;
}