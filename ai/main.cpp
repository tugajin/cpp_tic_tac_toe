#include "game.hpp"
#include "movelist.hpp"
#include "util.hpp"
#include "search.hpp"
#include "selfplay.hpp"
#include "ubfm.hpp"

TeeStream Tee;
namespace ubfm {
UBFMSearcherGlobal g_searcher_global;
}
namespace selfplay {
ReplayBuffer g_replay_buffer;
}

int main(int argc, char *argv[]){
    //movelist::test_move_list();
    //game::test_pos();
    //search::test_search();
    //game::test_nn();
    //selfplay::test_selfplay();
    ubfm::test_ubfm();
    //selfplay::execute_selfplay();
    return 0;
}