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

void execute_selfplay() {
    ubfm::g_searcher.allocate();
    ubfm::g_searcher.load_model();
    Tee<<"selfplay\n";
    REP(i, INT_MAX) {
        game::Position pos;
        g_replay_buffer.open();
        while(true) {
            if (pos.is_done()) {
                g_replay_buffer.write_data();
                g_replay_buffer.close();
                break;
            }
            ubfm::g_searcher.search<true>(pos, 50);
            auto best_move = ubfm::g_searcher.best_move();
            pos = pos.next(best_move);
        }
        if (i && i % 10 == 0) {
            Tee<<"\n";
            ubfm::g_searcher.load_model();
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