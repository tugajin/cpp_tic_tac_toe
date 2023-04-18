#ifndef __SELFPLAY_HPP__
#define __SELFPLAY_HPP__

#include <vector>
#include <utility>
#include <thread>
#include <fstream>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include "nlohmann/json.hpp"
#include "util.hpp"
#include "common.hpp"
#include "game.hpp"
#include "ubfm.hpp"

namespace selfplay {

using json = nlohmann::json;

class ReplayBuffer {
public:
    ReplayBuffer() {
        this->info = {};
    }
    void open() {
        this->info.clear();
        std::filesystem::create_directory("data");
        auto filename = "./data/" + timestamp() + "_" + to_string(my_rand(9999)) + ".json";
        this->ofs.open(filename);
    }
    void close() {
        this->ofs.close();
        this->info.clear();
    }
    void push_back(const uint32 hash, const NNScore sc) {
        info.push_back({{"p", hash},{"s", sc}});
    }
    void write_data() {
        this->ofs<<this->info.dump();
    }
private:
    json info;
    std::ofstream ofs;
};

extern ReplayBuffer g_replay_buffer;

void push_back(const uint32 hash, const NNScore score) {
    g_replay_buffer.push_back(hash, score);
}

Move execute_descent(game::Position &pos) {
    assert(ubfm::g_searcher_global.root_node.n == 0);
    ubfm::g_searcher_global.root_node.pos = pos;
    ubfm::g_searcher_global.run();
    ubfm::g_searcher_global.join();
    ubfm::g_searcher_global.choice_best_move_e_greedy();
    ubfm::g_searcher_global.add_replay_buffer(&ubfm::g_searcher_global.root_node);
    return ubfm::g_searcher_global.root_node.best_move;
}

void execute_selfplay() {
    
    ubfm::g_searcher_global.GPU_NUM = 1;
    ubfm::g_searcher_global.THREAD_NUM = 1;
    ubfm::g_searcher_global.IS_DESCENT = true;
    ubfm::g_searcher_global.TEMPERATURE = 0.2;
    ubfm::g_searcher_global.DESCENT_PO_NUM = 50;

    ubfm::g_searcher_global.init();

    REP(i, INT_MAX) {
        game::Position pos;
        g_replay_buffer.open();
        while(true) {
            ubfm::g_searcher_global.clear_tree();
            if (pos.is_done()) {
                g_replay_buffer.write_data();
                g_replay_buffer.close();
                break;
            }
            const auto best_move = execute_descent(pos);
            pos = pos.next(best_move);
        }
        if (i && i % 10 == 0) {
            Tee<<"\n";
            ubfm::g_searcher_global.load_model();
        }
        Tee<<".";
    }
}
void test_nn() {
}
void test_selfplay() {
    std::thread th_a(test_nn);
    th_a.join();
}

}
#endif