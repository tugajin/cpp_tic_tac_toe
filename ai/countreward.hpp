#ifndef __COUNT_REWARD_HPP__
#define __COUNT_REWARD_HPP__

#include "common.hpp"
#include "util.hpp"
#include "nlohmann/json.hpp"
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

namespace reward {

using json = nlohmann::json;

class CountReward {
private:
    json info;
    std::string path;
    int id;
public:
    CountReward(const int id) : id(id) {
        this->info = {};
        this->path = "count" + to_string(id) + ".json";
        this->load();
    }
    void update(const Key k) {
        if (!this->info.contains(to_string(k))) {
            this->info[to_string(k)] = 1;
        } else {
            int item = this->info[to_string(k)];
            this->info[to_string(k)] = item + 1;
        }
    }
    uint64 get(const Key k) const {
        if (this->info.contains(to_string(k))) {
            return this->info[to_string(k)];
        } else {
            return 0ul;
        }
    } 
    std::size_t size() const {
        return this->info.size();
    }
    void clean(std::size_t size) {
        ASSERT(size>0);
        if (this->info.size() < 1000000) {
            return;
        }
        std::vector<std::string> del_keys;
        for (auto &item : this->info.items()) {
            const auto k = item.key();
            const auto num = item.value();
            if (num < size) {
                del_keys.push_back(k);
            }
        }
        for (auto key : del_keys) {
            this->info.erase(key);
        }
        this->clean(size+1);
    }
    void load() {
        if (!is_exists_file(this->path)) {
            Tee << "not found count reward file\n";
            return;
        }
        std::ifstream f(this->path);
        this->info = json::parse(f);
    }
    void dump() {
        //Tee<<"cw:"<<this->info.size()<<std::endl;
        std::ofstream ofs(CountReward::path);
        ofs<<this->info.dump();
    }
};
void test_reward() {
}
}

#endif