#ifndef __SELFPLAY_HPP__
#define __SELFPLAY_HPP__

#include <vector>
#include <utility>
#include <thread>
#include <fstream>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "util.hpp"
#include "common.hpp"
#include "game.hpp"
#include "ubfm.hpp"
#include "nn.hpp"
#include "countreward.hpp"
#include "search.hpp"

#define DEBUG_OUT 0

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
    void push_back(const Key hash, const nn::NNScore sc) {
        info.push_back({{"p", hash},{"s", sc},{"r", -99}});
    }
    void overwrite_result(double result) {
        for(auto &item :this->info) {
            item["r"] = result;
            result *= -1;
        }
    }
    void write_data() {
        this->ofs<<this->info.dump();
    }
private:
    json info;
    std::ofstream ofs;
};

class DescentSearcherLocal ;

class SelfPlayWorker : public ubfm::UBFMSearcherGlobal {
public:
#if DEBUG_OUT
    static constexpr int NUM = 1;
#else
    static constexpr int NUM = 4;
#endif
    void run_descent();
    void join();
    void init();
private:
    std::vector<DescentSearcherLocal> worker;
};

class DescentSearcherLocal : public ubfm::UBFMSearcherLocal {
public:
    DescentSearcherLocal(const int id, const int gpu_id, SelfPlayWorker * selfplay) :
    ubfm::UBFMSearcherLocal(id, gpu_id, selfplay),cw(reward::CountReward(id)) {
    }
    void run_descent();
    void search_descent(const uint32 simulation_num);
    void selfplay();
    ReplayBuffer replay_buffer;
    reward::CountReward cw;
private:
    void evaluate_descent(ubfm::Node *node);
    Move execute_descent(game::Position &pos);
    bool interrupt_descent(const uint32 current_num, const uint32 simulation_num) const;
    void choice_best_move_count();
    int po_num = 100;
};

class SelfPlayInfo {
private:
    int num;
    int win_num;
    int lose_num;
    int draw_num;
    uint64 sum_ply;
    int num_limit;
    Lockable lock;
public:
    void init() {
        win_num = lose_num = draw_num = 0;
        num = num_limit = 0;
    }
    void inc(const int result, const int ply) {
        lock.lock();
        num++;
        sum_ply += ply;
        switch(result) {
            case 0:
                draw_num++;
                break;
            case 1:
                win_num++;
                break;
            case -1:
                lose_num++;
                break;
        }
        lock.unlock();
    }
    double win_rate() const {
        double rate = (double(win_num) + (double(draw_num)/2)) / (double(win_num) + double(lose_num) + double(draw_num));
        return rate;
    }
    int sum_num() const {
        return win_num + lose_num + draw_num;
    }
    double avg_ply() const {
        return double(sum_ply)/double(this->sum_num());
    }
    std::string str() {
        std::string ret = "------------selfplay info------------\n";
        const auto rate = this->win_rate();
        ret += "win:" + to_string(win_num) 
            + " lose:" + to_string(lose_num) 
            + " draw:" + to_string(draw_num) 
            + " sum:" + to_string(this->sum_num())
            + " rate:" + to_string(rate) 
            + " avg_ply:" + to_string(this->avg_ply()); 
        ret += "\n";
        return ret;
    }
    void set_limit(const int n) {
        num_limit = n;
    }
    bool is_end() const {
        return num >= num_limit;
    }
    void dump() const {
        std::ofstream ofs( "selfplay_result.csv", std::ios::out);
        ofs<<"win_num,lose_num,draw_num,percent,avg_ply,loss\n";
        ofs<<this->win_num<<","
           <<this->lose_num<<","
           <<this->draw_num<<","
           <<this->win_rate()<<","
           <<this->avg_ply()<<"\n";
    }
};

extern SelfPlayWorker g_selfplay_worker[SelfPlayWorker::NUM];
extern int g_thread_counter;
extern SelfPlayInfo g_selfplay_info;

void execute_selfplay(const int num) {
    g_thread_counter = 0;
    g_selfplay_info.init();
    g_selfplay_info.set_limit(num);
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].init();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].run_descent();
    }
    REP(i, SelfPlayWorker::NUM) {
        selfplay::g_selfplay_worker[i].join();
    }
    Tee<<g_selfplay_info.str();
}

void SelfPlayWorker::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    this->worker.emplace_back(g_thread_counter,0,this);
    this->clear_tree();
    g_thread_counter++;
}

void SelfPlayWorker::run_descent() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].run_descent();
}

void SelfPlayWorker::join() {
    ASSERT(this->worker.size() > 0);
    this->worker[0].join();
}

void DescentSearcherLocal::run_descent() {
    this->thread = new std::thread([this]() {
        this->selfplay();
    });
}
void DescentSearcherLocal::search_descent(const uint32 simulation_num) {
    for(auto i = 0u ;; ++i) {
#if DEBUG_OUT
        Tee<<"start simulation:" << i <<"/"<<simulation_num<<"\r";
#endif
        const auto interrupt = this->interrupt_descent(i, simulation_num);
        if (interrupt) {
            break;
        }
        this->evaluate_descent(this->root_node());
    }
}

void DescentSearcherLocal::evaluate_descent(ubfm::Node *node) {
    //ASSERT(!node->is_resolved());
    node->n++;
    // Timer timer;
    // timer.start();
    ASSERT2(node->pos.is_ok(),{
        Tee<<node->pos<<std::endl;
    })
    ASSERT(std::fabs(node->w) <= 1);
    // ASSERT2(node->is_ok(),{
    //     Tee<<node->str()<<std::endl;
    // });
    
    if (node->pos.is_draw()) {
        node->w = nn::NNScore(0.0);
        node->state = ubfm::NodeState::NodeDraw;
        return;
    } 
    if (node->pos.is_lose()) {
        node->w = ubfm::score_lose(node->ply);
        node->state = ubfm::NodeState::NodeLose;
        return;
    }
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
    } else {
        auto next_node = this->next_child<true>(node);
        ASSERT2(next_node != nullptr,{
            Tee<<node->str()<<std::endl;
        })
        this->evaluate_descent(next_node);
    }
    this->update_node(node);
}

bool DescentSearcherLocal::interrupt_descent(const uint32 current_num, const uint32 simulation_num) const {
    if (this->root_node()->is_resolved()) {
        return true;
    }
    // 最大基準の回数の2倍まではやってみる
    if (current_num >= simulation_num) {
        //Tee<<"over\n";
        return true;
    }
    auto mated_num = 0;
    auto mate_num = 0;
    auto root_node = this->root_node();
    REP(i, root_node->child_len) {
        const auto child = root_node->child(i);
        if (-child->w <= -0.9) {
            mated_num++;
        }
        if (-child->w >= 0.9) {
            mate_num++;
        }
    }
    // 最低2回くらいはやる
    const auto can_stop = current_num > static_cast<uint32>(2 * root_node->child_len);
    // 全部負けなら終わり
    if (can_stop && mated_num == root_node->child_len) {
        return true;
    }
    // 1手以外は全部負けなら終わり
    if (can_stop && mated_num + 1 == root_node->child_len) {
        return true;
    }
    //勝ちを見つけたら終わり
    if (can_stop && mate_num > 0) {
        return true;
    }
    if (current_num >= simulation_num) {
        nn::NNScore max_score = static_cast<nn::NNScore>(-1.0);
        auto max_num = -1;
        auto max_score_index = -1;
        auto max_num_index = -1;
        REP(i, root_node->child_len) {
            ASSERT(i>=0);
            ASSERT(i<root_node->child_len);
            const auto child = root_node->child(i);
            if (-child->w > max_score) {
                max_score = -child->w;
                max_score_index = i;
            }
            if (child->n > max_num) {
                max_num = child->n;
                max_num_index = i;
            }
        }
        // 一番選択した手と一番勝ちが高い手が不一致なら追加探索
        if (max_score_index != max_num_index) {
            //Tee<<"+ ";
            return false;
        }
        return true;
    }
    return false;
}

Move DescentSearcherLocal::execute_descent(game::Position &pos) {
    this->root_node()->pos = pos;
    this->search_descent(this->po_num);
    this->choice_best_move_count();
    const auto k = hash::hash_key(this->root_node()->pos);
    const auto w = this->root_node()->w;
    this->replay_buffer.push_back(k,w);
   return this->root_node()->best_move;
}

void DescentSearcherLocal::choice_best_move_count() {
    std::vector<double> scores;
    std::vector<uint64> num;
    REP(i, this->root_node()->child_len) {

        ASSERT(i>=0);
        ASSERT(i<this->root_node()->child_len);
        auto child = this->root_node()->child(i);
        const auto r = 1 + cw.get(child->pos.history());
        scores.push_back((1 / std::sqrt(r)));
        num.push_back(r);
    }
    REP(i, this->root_node()->child_len) {
        ASSERT(i>=0);
        ASSERT(i<this->root_node()->child_len);
        auto child = this->root_node()->child(i);
        const auto reward = scores[i] * 0.8;
        //const auto oracle = -oracle::g_oracle.result(child->pos);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node()->child_len) {
                    scores[i] = 0;
                }
                scores[i] = 1;
                break;
            } else if (child->is_draw()) {
                scores[i] = 0;
            } else if (child->is_win()) {
                scores[i] = -1.0 ;
            }
        } else {
            scores[i] = /*child->n + 1.0 */- child->w + reward;
        }
#if DEBUG_OUT
        Tee<<"n:"<<padding_str(to_string(child->n),3) 
            << " w:" << padding_str(to_string(-child->w),7) 
          //  << " oracle:" << padding_str(to_string(oracle),2) 
            << " org:" << padding_str(to_string(num[i]),3)<<" "
            <<move_str(child->parent_move)<<std::endl;
#endif
    }
    auto index = -1;
    auto iter = std::max_element(scores.begin(), scores.end());
    index = std::distance(scores.begin(), iter);

    ASSERT(index>=0);
    ASSERT(index<this->root_node()->child_len);
    auto child = this->root_node()->child(index);
    this->root_node()->best_move = child->parent_move;
    this->root_node()->w = -child->w;
}

nn::NNScore int_to_nn(const int sc) {
    if (sc >= (search::SEARCH_MATE - 100)) {
        return nn::NNScore(1);
    } else if (sc <= -(search::SEARCH_MATE - 100)) {
        return nn::NNScore(-1);
    }
    return std::tanh(static_cast<double>(sc)/1000.0);
}

void DescentSearcherLocal::selfplay() {
    Tee<<"start selfplay\n";
    for(auto i = 0; !g_selfplay_info.is_end() ; i++) {
        game::Position pos;
        pos = hash::hirate();
        this->replay_buffer.open();
        while(true) {
#if DEBUG_OUT
            Tee<<"自己対局("<<this->thread_id<<")："<<i<<":"<<pos.ply()<<std::endl;
            Tee<<pos<<std::endl;
#endif
            this->global->clear_tree();

            if (pos.is_lose() || pos.is_draw()) {
                auto result = 0.0;
                if (pos.is_draw()) {
                    result = 0.0;
                    g_selfplay_info.inc(0,pos.ply());
                }
                if (pos.is_lose()) {
                    if (pos.turn() == BLACK) {
                        result = -1.0;
                        g_selfplay_info.inc(-1,pos.ply());
                    } else {
                        result = 1.0;
                        g_selfplay_info.inc(1,pos.ply());
                    }
                }
                this->replay_buffer.overwrite_result(result);
                this->replay_buffer.write_data();
                this->replay_buffer.close();
                break;
            }
#if 1
            this->po_num = 50;
            
            auto best_move = execute_descent(pos);
           
            this->cw.update(pos.history());
#else
            auto sc = search::SEARCH_MIN;
            auto best_move = search::search_root(pos, 5, sc);
            const auto k = hash::hash_key(pos);
            const auto w = int_to_nn(sc);
            this->replay_buffer.push_back(k,w);
#endif
            pos = pos.next(best_move);
        }
        if (i % 10 == 0) {
            this->cw.dump();
        }
        if (this->thread_id == 0 && i % 10 == 0) {
            Tee<<g_selfplay_info.str();
        }
    }
    g_selfplay_info.dump();
}
}
#endif
