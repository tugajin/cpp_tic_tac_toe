#ifndef __CNS_HPP__
#define __CNS_HPP__

#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include "nlohmann/json.hpp"
#include "common.hpp"
#include "util.hpp"
#include "game.hpp"
#include "thread.hpp"
#include "nn.hpp"
#include "countreward.hpp"
#include "model.hpp"
#include "ubfm.hpp"

namespace cns {

enum ConspiracyNumber : uint32 {
    CONSPIRACY_INIT = 0u,
    CONSPIRACY_ONE = 1u,
    CONSPIRACY_MAX = 3000u,
};

enum NodeType : int {
    INIT_NODE = 0,
    MAX_NODE = 1,
    MIN_NODE = 2,
};

ConspiracyNumber operator+(ConspiracyNumber l, ConspiracyNumber r) {
  return static_cast<ConspiracyNumber>(static_cast<uint32>(l) + static_cast<uint32>(r));
}

void operator+=(ConspiracyNumber& l, ConspiracyNumber r) {
  l = l + r;
}

inline bool conspiracy_is_ok(const ConspiracyNumber c) {
    const uint32 v = static_cast<uint32>(c);
    return v <= 3000u;
}

inline ConspiracyNumber calc_leaf_pn(const nn::NNScore node_score, const nn::NNScore w, const bool is_terminal) {
    if (w <= node_score) {
        return CONSPIRACY_INIT;
    } else {
        if (!is_terminal) {
            return CONSPIRACY_ONE;
        } else {
            return CONSPIRACY_MAX;
        }
    }
}
inline ConspiracyNumber calc_leaf_dn(const nn::NNScore node_score, const nn::NNScore w, const bool is_terminal) {
    if (w >= node_score) {
        return CONSPIRACY_INIT;
    } else {
        if (!is_terminal) {
            return CONSPIRACY_ONE;
        } else {
            return CONSPIRACY_MAX;
        }
    }
}

class Node {
public:
    Node() : child_nodes(nullptr),
             parent_move(MOVE_NONE),
             best_move(MOVE_NONE),
             w(0.0),
             n(0u),
             pn(CONSPIRACY_INIT),
             dn(CONSPIRACY_INIT),
             node_type(INIT_NODE),
             child_len(-1),
             ply(-1){}
   Node( const Node & ) = delete ;
   Node & operator = ( const Node & ) = delete ;
   Node & operator = ( const Node && n) = delete;
    void init() {
        this->w = 0.0;
        this->n = 0u;
        this->pn = this->dn = CONSPIRACY_INIT;
        this->parent_move = this->best_move = MOVE_NONE;
        this->node_type = INIT_NODE;
        this->child_len = -1;
        this->child_nodes = nullptr;
        this->ply = 0;
    }
    bool is_terminal() const {
        return child_len == -1;
    }
    bool is_resolved() const {
        return (this->pn >= CONSPIRACY_MAX) || (this->dn >= CONSPIRACY_MAX);
    }
    std::string str(const bool is_root = true) const {
        const std::string padding = is_root ? "" :"        ";
        const std::string node_type_str = node_type == MAX_NODE ? "MAX_NODE" : "MIN_NODE";
        std::string str = "---------------------------\n";
        if (is_root) { str += pos.str(); }
        str += padding + "w:" + to_string(w) + "\n";
        str += padding + "n:" + to_string(n) + "\n";
        str += padding + "child_len:" + to_string(child_len) + "\n";
        str += padding + "ply:" + to_string(ply) + "\n";
        str += padding + "parent_move:" + move_str(parent_move) + "\n";
        str += padding + "best_move:" + move_str(best_move) + "\n";
        str += padding + "pn:" + to_string(pn) + " dn:" + to_string(dn) + "\n";
        str += padding + node_type_str + "\n";
        str += "---------------------------\n";
        if (is_root && !this->is_terminal()) {
            str += "child\n";
            REP(i, this->child_len) {
                str += "no:" + to_string(i) + "\n";
                ASSERT(i>=0);
                ASSERT(i<this->child_len);
                auto child = this->child(i);
                str += child->str(false);
            }
        }
        return str;
    }
    Node* child(const int index) const {
        ASSERT(index < child_len);
        ASSERT(index >= 0);
        return child_nodes[index].get();
    }
    bool is_ok() const {
        return true;
    }
    game::Position pos;
    std::unique_ptr<std::unique_ptr<Node>[]> child_nodes;
    Lockable lock_node;
    Move parent_move;
    Move best_move;
    nn::NNScore w;
    uint32 n;
    ConspiracyNumber pn;
    ConspiracyNumber dn;
    NodeType node_type;
    int child_len;
    int ply;
};
class CNSSearcherGlobal;

class CNSSearcherLocal {
public:
    CNSSearcherLocal(int id, int gpu_id, CNSSearcherGlobal * global) : 
                     global(global),
                     thread(nullptr),
                     thread_id(id),
                     gpu_id(gpu_id) {
    }
    void search(const uint32 simulation_num);
    void search_descent(const uint32 simulation_num);
    void selfplay() {};
    bool is_ok();
    void run();
    void join();
protected:
    void evaluate(Node *node);
    void predict(Node *node);
    void predict_root();
    void set_window(const nn::NNScore w);
    void select_mpn(Node *node);
    void expand();
    void update_node();
    bool interrupt(const uint32 current_num, const uint32 simulation_num) const;
    Node *root_node() const ;

    CNSSearcherGlobal *global;
    std::thread *thread;
    std::vector<Node*> pv;
    nn::NNScore w_max;
    nn::NNScore w_min;
    int thread_id;
    int gpu_id;
};

class CNSSearcherGlobal {
public:
    CNSSearcherGlobal() :
                       THREAD_NUM(1){}
    CNSSearcherGlobal(const int thread_num) : 
                       THREAD_NUM(thread_num){}
    Node root_node;
    void init();
    void clear_tree();
    void run();
    void join();
    void choice_best_move();

    int THREAD_NUM;
protected:
    std::vector<CNSSearcherLocal> worker;
};

extern CNSSearcherGlobal g_searcher_global;

Move think_cns(game::Position &pos) {
    g_searcher_global.init();
    g_searcher_global.root_node.pos = pos;
    g_searcher_global.run();
    g_searcher_global.join();
    //g_searcher_global.choice_best_move();
    return g_searcher_global.root_node.best_move;
}

Node *CNSSearcherLocal::root_node() const {
    return &this->global->root_node;
}

void CNSSearcherGlobal::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    REP(i, CNSSearcherGlobal::THREAD_NUM) {
        this->worker.emplace_back(i,0,this);
    }
    this->clear_tree();
}

void CNSSearcherGlobal::clear_tree() {
    this->root_node.init();
}

void CNSSearcherGlobal::run() {
    REP(i, CNSSearcherGlobal::THREAD_NUM) {
        this->worker[i].run();
    }
}

void CNSSearcherGlobal::join() {
    REP(i, CNSSearcherGlobal::THREAD_NUM) {
        this->worker[i].join();
    }
}

void CNSSearcherGlobal::choice_best_move() {
    std::vector<double> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(0.0);
    }
    REP(i, this->root_node.child_len) {
        ASSERT(i>=0);
        ASSERT(i<this->root_node.child_len);
        auto child = this->root_node.child(i);
        scores[i] = child->w;
    }
    auto index = -1;
    auto iter = std::max_element(scores.begin(), scores.end());
    index = std::distance(scores.begin(), iter);
    ASSERT(index>=0);
    ASSERT(index<this->root_node.child_len);
    auto child = this->root_node.child(index);
    this->root_node.best_move = child->parent_move;
}

void CNSSearcherLocal::run() {
	this->thread = new std::thread([this]() {
        this->search(int(2000 / this->global->THREAD_NUM));
    });
}
void CNSSearcherLocal::join() {
    this->thread->join();
    delete this->thread;
}

bool CNSSearcherLocal::interrupt(const uint32 current_num, const uint32 simulation_num) const {
    if (this->root_node()->is_resolved()) {
        return true;
    }
    if (current_num >= simulation_num) {
        return true;
    }
    return false;
}

void CNSSearcherLocal::search(const uint32 simulation_num) {
    
    const auto is_out = (this->thread_id == 0) && (this->gpu_id == 0);
    this->predict_root();
    for (auto i = 0u; !this->interrupt(i, simulation_num); ++i) {
        //Tee<<"start simulation:" << i <<"/"<<simulation_num<<"\r";
        this->set_window(this->root_node()->w);
        this->select_mpn(this->root_node());
        this->expand();
        this->update_node();
        // Tee<<"\n root info\n";
        // Tee<<this->w_min<<" < "<< this->w_max<<std::endl;
        // Tee<<this->root_node()->str()<<std::endl;
        // Tee<<"\n";
    }
    if (false) {
        Tee<<this->root_node()->str()<<std::endl;
    }
}

void CNSSearcherLocal::set_window(const nn::NNScore w) {
    this->w_max = w + 0.1;
    this->w_min = w - 0.1;
    this->pv.clear();
}

void CNSSearcherLocal::select_mpn(Node *node) {
    this->pv.push_back(node);
    node->n++;
    if (node->is_terminal()) {
        return;
    }
    Node* best_node = nullptr;
    auto best_n = CONSPIRACY_MAX;

    if (node->node_type == MAX_NODE) {
        auto best_score = nn::NNScore(-1);
        for (auto i = 0; i < node->child_len; i++) {
            auto child = node->child(i);
            if (child->pn < best_n) {
                best_node = child;
                best_n = child->pn;
                best_score = child->w;
            } else if (child->pn == best_n && child->w > best_score ) {
                best_node = child;
                best_score = child->w;
            }
        }
        auto best_score2 = nn::NNScore(-1);
        Node* best_node2 = nullptr;
        for (auto i = 0; i < node->child_len; i++) {
            auto child = node->child(i);
            if (child->w > best_score2) {
                best_node2 = child;
                best_score2 = child->w;
            } 
        }
    } else if (node->node_type == MIN_NODE) {
        auto best_score = nn::NNScore(1);
        for (auto i = 0; i < node->child_len; i++) {
            auto child = node->child(i);
            if (child->dn < best_n) {
                best_node = child;
                best_n = child->dn;
                best_score = child->w;
            } else if (child->dn == best_n && child->w < best_score) {
                best_node = child;
                best_score = child->w;
            }
        }
        auto best_score2 = nn::NNScore(1);
        Node* best_node2 = nullptr;
        for (auto i = 0; i < node->child_len; i++) {
            auto child = node->child(i);
            if (child->w < best_score2) {
                best_node2 = child;
                best_score2 = child->w;
            } 
        }
    } else {
        ASSERT(false);
    }
    ASSERT2(best_node != nullptr,{
        Tee<<"not found best_node\n";
        Tee<<node->str()<<std::endl;
    });
    this->select_mpn(best_node);
}

void CNSSearcherLocal::expand() {
    auto node = this->pv.back();
    auto moveList = movelist::MoveList();
    gen::legal_moves(node->pos, moveList);
    
    node->child_len = moveList.len();
    node->child_nodes = std::make_unique<std::unique_ptr<Node>[]>(node->child_len);
    const auto child_node_type = (node->node_type == MAX_NODE) ? MIN_NODE : MAX_NODE;
    REP(i, node->child_len) {
        auto next_pos = node->pos.next(moveList[i]);
        node->child_nodes[i] = std::make_unique<Node>();
        auto next_node = node->child_nodes[i].get();
        next_node->pos = next_pos;
        next_node->ply = node->ply+1;
        next_node->parent_move = moveList[i];
        next_node->node_type = child_node_type;
    }
    predict(node);
}
void CNSSearcherLocal::predict_root() {

    auto node = this->root_node();

    node->node_type = MAX_NODE;
    
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> outputs_list;
    auto &pos = node->pos;
    feat_list.push_back(nn::feature(pos));
    model::predict(this->gpu_id, feat_list, outputs_list);
    auto score = outputs_list[0];
    auto is_terminal = false;
    if (score >= nn::NNScore(1.0)) {
        score = nn::NNScore(0.8999);
    } else if (score <= nn::NNScore(-1.0)) {
        score = nn::NNScore(-0.8999);
    }
    if (pos.is_draw()) {
        score = nn::NNScore(0.0);
        is_terminal = true;
    } else if (pos.is_lose()) {
        score = ubfm::score_lose(1);
        is_terminal = true;
    } else if (pos.is_win()) {
        score = ubfm::score_win(1);
        is_terminal = true;
    }
    node->w = score;
    this->set_window(node->w);
    node->pn = calc_leaf_pn(score, this->w_max, is_terminal);
    node->dn = calc_leaf_dn(score, this->w_min, is_terminal);
}

void CNSSearcherLocal::predict(Node *node) {
    
    ASSERT2(node->child_len > 0,{
        Tee<<node->pos<<std::endl;
    });
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> outputs_list;
    REP(i, node->child_len) {
        ASSERT(i>=0);
        ASSERT(i<node->child_len);
        auto child = node->child(i);
        auto &pos = child->pos;
        feat_list.push_back(nn::feature(pos));
    }
    model::predict(this->gpu_id, feat_list, outputs_list);

    REP(i, node->child_len) {
        auto score = outputs_list[i];
        if (score >= nn::NNScore(1.0)) {
            score = nn::NNScore(0.8999);
        } else if (score <= nn::NNScore(-1.0)) {
            score = nn::NNScore(-0.8999);
        }
        ASSERT(i>=0);
        ASSERT(i<node->child_len);

        auto child = node->child(i);
        auto &pos = child->pos;
        
        auto is_terminal = false;
        if (pos.is_draw()) {
            score = nn::NNScore(0.0);
            is_terminal = true;
        } else if (pos.is_lose()) {
            score = ubfm::score_lose(child->ply);
            is_terminal = true;
        } else if (pos.is_win()) {
            score = ubfm::score_win(child->ply);
            is_terminal = true;
        }
        if (pos.turn() != this->root_node()->pos.turn()) {
            score = -score;
        }
        child->w = score;
        child->pn = calc_leaf_pn(score, this->w_max, is_terminal);
        child->dn = calc_leaf_dn(score, this->w_min, is_terminal);
    }
}

void CNSSearcherLocal::update_node() {
    while(!this->pv.empty()) {
        auto node = this->pv.back();
        if (node->node_type == MAX_NODE) {
            auto min_pn = CONSPIRACY_MAX;
            auto sum_dn = CONSPIRACY_INIT;
            auto max_score = nn::NNScore(-1);
            for(auto i = 0; i < node->child_len; i++) {
                auto child = node->child(i);
                if (min_pn > child->pn) {
                    min_pn = child->pn;
                } 
                if (max_score < child->w) {
                    max_score = child->w;
                }
                sum_dn += child->dn;
            }
            node->pn = min_pn;
            node->dn = std::min(sum_dn, CONSPIRACY_MAX);
            node->w = max_score;
        } else if (node->node_type == MIN_NODE) {
            auto min_dn = CONSPIRACY_MAX;
            auto sum_pn = CONSPIRACY_INIT;
            auto min_score = nn::NNScore(1);
            for(auto i = 0; i < node->child_len; i++) {
                auto child = node->child(i);
                if (min_dn > child->dn) {
                    min_dn = child->dn;
                }
                if (min_score > child->w) {
                    min_score = child->w;
                }
                sum_pn += child->pn;
            }
            node->pn = std::min(sum_pn, CONSPIRACY_MAX);
            node->dn = min_dn;
            node->w = min_score;
        } else {
            ASSERT(false);
        }
        this->pv.pop_back();
    }
    ASSERT(this->pv.empty());
}

using json = nlohmann::json;

void test_cns() {
    std::ifstream f("/home/tugajin/Documents/cpp_tic_tac_toe/oracle/all_pos.json");
    json info = json::parse(f);
    for (auto s : info) {
        auto key = s.get<uint64>();
        auto pos = hash::from_hash(static_cast<Key>(key));
        //Tee<<pos<<std::endl;
        cns::g_searcher_global.init();
        cns::think_cns(pos);
        ubfm::g_searcher_global.init();
        ubfm::think_ubfm(pos);
        Tee<<key<<","<<cns::g_searcher_global.root_node.n<<","<<ubfm::g_searcher_global.root_node.n<<std::endl;
    }
    // {
    //     auto pos = hash::from_hash(hash::START_HASH_KEY);
    //     pos = pos.next(Move(0));
    //     pos = pos.next(Move(1));
    //     Tee<<pos<<std::endl;
    //     cns::g_searcher_global.init();
    //     cns::think_cns(pos);
    // }
}

}
#endif
