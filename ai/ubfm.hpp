#ifndef __UBFM_HPP__
#define __UBFM_HPP__

#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include <torch/torch.h>
#include <torch/script.h>
#include "nlohmann/json.hpp"
#include "common.hpp"
#include "util.hpp"
#include "game.hpp"
#include "selfplay.hpp"
#include "thread.hpp"

namespace selfplay {
void push_back(const uint32 hash, const NNScore score);
}

namespace ubfm {

enum NodeState : int {
    NodeDraw = 0,
    NodeWin = 1,
    NodeLose = 2,
    NodeUnknown = -1
};

inline NNScore score_win(const int ply) {
    return NNScore(0.9999) - NNScore(static_cast<double>(ply) / 100.0);
}
inline NNScore score_lose(const int ply) {
    return -score_win(ply);
}

std::string node_state_str(const NodeState n) {
    switch (n) {
        case NodeWin:
            return "NODE_WIN";
        case NodeLose:
            return "NODE_LOSE";
        case NodeDraw:
            return "NODE_DRAW";
        case NodeUnknown:
            return "NODE_UNKNOWN";
        default:
            return "ERROR_NODE_STATE";
    }
}
std::ostream& operator<<(std::ostream& os, const NodeState n) {
    os << node_state_str(n);
    return os;
}
class Node {
public:
    Node() : w(0.0),
             n(0),
             parent_move(MOVE_NONE),
             best_move(MOVE_NONE),
             child_len(-1),
             child_nodes(nullptr),
             ply(-1),
             state(NodeUnknown) {}
   Node( const Node & ) = delete ;
   Node & operator = ( const Node & ) = delete ;
   Node & operator = ( const Node && n) = delete;
    void init() {
        this->w = 0.0;
        this->n = 0;
        this->parent_move = this->best_move = MOVE_NONE;
        this->child_len = -1;
        this->child_nodes = nullptr;
        this->ply = 0;
        this->state = NodeUnknown;
    }
    bool is_resolved() const {
        return state != NodeUnknown;
    }
    bool is_win() const {
        return state == NodeWin;
    }
    bool is_lose() const {
        return state == NodeLose;
    }
    bool is_draw() const {
        return state == NodeDraw;
    }
    bool is_terminal() const {
        return child_len == -1;
    }
        
    std::string str(const bool is_root = true) const {
        const std::string padding = is_root ? "" :"        ";
        std::string str = "---------------------------\n";
        if (is_root) { str += pos.str(); }
        str += padding + "w:" + to_string(w) + "\n";
        str += padding + "n:" + to_string(n) + "\n";
        str += padding + "child_len:" + to_string(child_len) + "\n";
        str += padding + "ply:" + to_string(ply) + "\n";
        str += padding + "parent_move:" + to_string(parent_move) + "\n";
        str += padding + "best_move:" + to_string(best_move) + "\n";
        str += padding + "state:" + to_string(state) + "\n";
        str += "---------------------------\n";
        if (is_root && !this->is_terminal()) {
            str += "child\n";
            REP(i, this->child_len) {
                str += "no:" + to_string(i) + "\n";
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
    game::Position pos;
    std::unique_ptr<std::unique_ptr<Node>[]> child_nodes;
    Lockable lock_node;
    Move parent_move;
    Move best_move;
    NodeState state;
    NNScore w;
    uint32 n;
    int child_len;
    int ply;

};
class UBFMSearcherGroup;

class UBFMSearcherLocal {
public:
    UBFMSearcherLocal(UBFMSearcherGroup *g, int id) : 
                     group(g),
                     thread(nullptr),
                     thread_id(id){
    }
    // UBFMSearcherLocal(UBFMSearcherLocal &&local) : thread(nullptr),
    //                                              thread_id(local.thread_id),
    //                                              group(local.group) {
    //     //this->root_node = std::move(local.root_node);
    // }
    // UBFMSearcherLocal& operator=(UBFMSearcherLocal &&local) {
    //    // this->root_node = std::move(local.root_node);
    //     return *this;
    // }
    template<bool is_descent = false>void search(const game::Position &pos, const int simulation_num);
    bool is_ok();
    void run();
    void join();
private:
    void evaluate(Node *node);
    void evaluate_descent(Node *node);
    void predict(Node *node);
    void expand(Node *node);
    Node *next_child(const Node *node) const;
    Node *next_child2(const Node *node) const;
    void update_node(Node *node);
    void add_node(const game::Position &pos,const Move parent_move, int ply);
    int thread_id;
    std::thread *thread;
    UBFMSearcherGroup *group;
};

class UBFMSearcherGroup {
public:
    UBFMSearcherGroup(const int id): 
                    gpu_id(id),
                    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, id){
        if (torch::cuda::is_available()) {
            Tee<<"GPU:"<<id<<"\n";
        } else {
            Tee<<"CPU:"<<id<<"\n";
        }
    }
    // UBFMSearcherGroup(UBFMSearcherGroup &&local) :
    //                 device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, 0) {
    //    // this->root_node = std::move(local.root_node);
    // }
    /*UBFMSearcherGroup& operator=(UBFMSearcherGroup &&p) {
        return *this;
    }*/
    void init();
    void load_model();
    void run();
    void join();
    torch::Device device;
    torch::jit::script::Module module;
    Lockable lock_gpu;
    int gpu_id;

private:
    std::vector<UBFMSearcherLocal> worker;
};

class UBFMSearcherGlobal {
public:
    std::unique_ptr<Node> nodes;
    Node root_node;
    void init();
    void load_model();
    void clear_tree();
    void run();
    void join();
    void choice_best_move();
    void choice_best_move_e_greedy();
    void add_replay_buffer(const Node * node) const ;

    int GPU_NUM = 1;
    int THREAD_NUM = 1;
    double TEMPERATURE = 0.0;
    uint32 DESCENT_PO_NUM = 2000;
    bool IS_DESCENT = false;
private:
    std::vector<UBFMSearcherGroup> worker;
};

extern UBFMSearcherGlobal g_searcher_global;

NNScore noise(double div = 1000) {
    auto sign = rand_double();
    auto value = rand_double() / div;
    return sign >= 0.5 ? value : -value;
}

Move think_ubfm(game::Position &pos) {
    assert(g_searcher_global.root_node.n == 0);
    g_searcher_global.root_node.pos = pos;
    g_searcher_global.run();
    g_searcher_global.join();
    g_searcher_global.choice_best_move();
    return g_searcher_global.root_node.best_move;
}

void UBFMSearcherGlobal::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    REP(i, UBFMSearcherGlobal::GPU_NUM) {
        this->worker.emplace_back(i);
        this->worker[i].init();
        this->worker[i].load_model();
    }
    this->clear_tree();
}

void UBFMSearcherGlobal::load_model() {
    REP(i, UBFMSearcherGlobal::GPU_NUM) {
        this->worker[i].load_model();
    }
}

void UBFMSearcherGlobal::clear_tree() {
    this->root_node.init();
    this->nodes = std::make_unique<Node>(); 
}

void UBFMSearcherGlobal::run() {
    REP(i, UBFMSearcherGlobal::GPU_NUM) {
        this->worker[i].run();
    }
}
void UBFMSearcherGlobal::join() {
    REP(i, UBFMSearcherGlobal::GPU_NUM) {
        this->worker[i].join();
    }
}

void UBFMSearcherGlobal::choice_best_move() {
    std::vector<int> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(1);
    }
    REP(i, this->root_node.child_len) {
        auto child = this->root_node.child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node.child_len) {
                    scores[i] = 0;
                }
                scores[i] = 1;
                break;
            } else if (child->is_draw()) {
                scores[i] = child->n + 100;
            } else if (child->is_win()) {
                scores[i] = 1;
            }
        } else {
            scores[i] = child->n - child->w + 100;
        }
    }
    auto index = -1;
    auto iter = std::max_element(scores.begin(), scores.end());
    index = std::distance(scores.begin(), iter);
    auto child = this->root_node.child(index);
    this->root_node.best_move = child->parent_move;
}

void UBFMSearcherGlobal::choice_best_move_e_greedy() {
    std::vector<NNScore> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(NNScore(0.0));
    }
    auto find_resolved_flag = false;
    REP(i, this->root_node.child_len) {
        auto child = this->root_node.child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node.child_len) {
                    scores[i] = NNScore(0.0);
                }
                scores[i] = score_win(0);
                find_resolved_flag = true;
                break;
            } else if (child->is_draw()) {
                scores[i] = NNScore(0.0) + (rand_double() / 1000);
            } else if (child->is_win()) {
                scores[i] = score_lose(0) + (rand_double() / 1000);
            }
        } else {
            scores[i] = -child->w;
        }
    }
    auto index = -1;
    if (!find_resolved_flag && rand_double() < this->TEMPERATURE) {
        index = my_rand(this->root_node.child_len);
    } else {
        auto iter = std::max_element(scores.begin(), scores.end());
        index = std::distance(scores.begin(), iter);
    }
    auto child = this->root_node.child(index);
    this->root_node.best_move = child->parent_move;
}

void UBFMSearcherGroup::init() {

    this->worker.clear();
    this->worker.shrink_to_fit();

    Tee<<"init group:"<<this->gpu_id<<" worker\n";
    REP(i, g_searcher_global.THREAD_NUM) {
        this->worker.emplace_back(this,i);
    }
}
void UBFMSearcherGroup::load_model() {
    this->lock_gpu.lock();
    REP(i, 10) {
        try {
            this->module = torch::jit::load("./model/best_single_jit.pt",this->device);
            this->module.eval();
            this->lock_gpu.unlock();
            return;
        } catch (const c10::Error& e) {
            Tee << "error loading the model\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));        //1秒間スリープします
            continue;
        }
    }
    this->lock_gpu.unlock();
    std::exit(EXIT_FAILURE);
}

void UBFMSearcherGroup::run() {
    REP(i, g_searcher_global.THREAD_NUM) {
        this->worker[i].run();
    }
}
void UBFMSearcherGroup::join() {
    REP(i, g_searcher_global.THREAD_NUM) {
        this->worker[i].join();
    }
}
void UBFMSearcherLocal::run() {
	this->thread = new std::thread([this]() {
        //Tee<<"thread:"<<this->thread_id<<" start"<<std::endl;
        if (g_searcher_global.IS_DESCENT) {
            assert(g_searcher_global.DESCENT_PO_NUM > 0);
            this->search<true>(g_searcher_global.root_node.pos, g_searcher_global.DESCENT_PO_NUM);
        } else {
            this->search<false>(g_searcher_global.root_node.pos, int(2000 / g_searcher_global.THREAD_NUM));
        }
    });
}
void UBFMSearcherLocal::join() {
    this->thread->join();
    delete this->thread;
}

template<bool is_descent>void UBFMSearcherLocal::search(const game::Position &pos, const int simulation_num) {
    
    const auto is_out = (this->thread_id == 0) && (this->group->gpu_id == 0);
    
    REP(i, simulation_num) {
        //Tee<<"start simulation:" << i <<"\n";
        g_searcher_global.root_node.n++;
        if (!is_descent && g_searcher_global.root_node.is_resolved()) {
            //Tee<<"root node is resolved\n";
            break;
        }
        if (is_descent) {
            this->evaluate_descent(&g_searcher_global.root_node);
        } else {
            this->evaluate(&g_searcher_global.root_node);
        }
        if (is_out) {
            //Tee<<this->str<false,true>(&this->root_node)<<std::endl;
        }
    }
    if (is_out) {
        //Tee<<g_searcher_global.root_node.str()<<std::endl;
    }
}

void UBFMSearcherLocal::evaluate(Node *node) {
    // if (node == nullptr) {
    //     Tee<<"???????"<<this->thread_id<<std::endl;
    //     return;
    // }
    ASSERT(node != nullptr);
    node->lock_node.lock();
    if (node->is_resolved()) {
        node->lock_node.unlock();
        return;
    }
    ASSERT2(!node->pos.is_done(),{
        Tee<<node->str()<<std::endl;
    });
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
        node->lock_node.unlock();
    } else {
        auto next_node = this->next_child(node);
        node->lock_node.unlock();
        this->evaluate(next_node);
    }
    node->lock_node.lock();
    this->update_node(node);
    
    node->lock_node.unlock();
}

void UBFMSearcherLocal::evaluate_descent(Node *node) {
    //ASSERT(!node->is_resolved());
    ASSERT2(!node->pos.is_done(),{
        Tee<<node->pos<<std::endl;
    });
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
        this->update_node(node);
    } 
    /*if (!node->is_resolved()) {
        auto next_node = this->next_child(node);
        this->evaluate_descent(next_node);
        this->update_node(node);
    }*/
    auto next_node = this->next_child(node);
    if (next_node != nullptr) {
        this->evaluate_descent(next_node);
        this->update_node(node);
    }
}

void UBFMSearcherLocal::expand(Node *node) {
    auto moveList = movelist::MoveList();
    node->pos.legal_moves(moveList);
    
    node->child_len = moveList.len();
    node->child_nodes = std::make_unique<std::unique_ptr<Node>[]>(node->child_len);
    REP(i, node->child_len) {
        auto next_pos = node->pos.next(moveList[i]);
        node->child_nodes[i] = std::make_unique<Node>();
        auto next_node = node->child_nodes[i].get();
        next_node->pos = next_pos;
        next_node->ply = node->ply+1;
        next_node->parent_move = moveList[i];
    }
}
void UBFMSearcherLocal::predict(Node *node) {
    
    std::vector<at::Tensor> tensor_list;
    REP(i, node->child_len) {
        auto child = node->child(i);
        auto &pos = child->pos;
        auto f = pos.feature();
        std::vector<at::Tensor> v;
        REP(j, FEAT_SIZE) {
            v.push_back(torch::tensor(torch::ArrayRef<int>(f[j])));
        }
        auto f_all = torch::stack(v).reshape({FEAT_SIZE, 3, 3});
        tensor_list.push_back(f_all);
    }
    auto feat_tensor = torch::stack(tensor_list);

    this->group->lock_gpu.lock();
    
    feat_tensor = feat_tensor.to(this->group->device,torch::kFloat);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(feat_tensor);
    auto output = this->group->module.forward(inputs).toTensor();
    
    this->group->lock_gpu.unlock();

    REP(i, node->child_len) {
        auto state = NodeUnknown;
        auto score = to_nnscore(output[i][0].item<float>());     
        if (score >= NNScore(1.0)) {
            score = NNScore(0.8999);
        } else if (score <= NNScore(-1.0)) {
            score = NNScore(-0.8999);
        }
        auto child = node->child(i);
        auto &pos = child->pos;

        if (pos.is_draw()) {
            score = NNScore(0.0);
            state = NodeDraw;
        } else if (pos.is_lose()) {
            score = score_lose(child->ply);
            state = NodeLose;
        } /*else if (pos.has_win() != -1) {
            score = score_win(child->ply);
            state = NodeWin;
        }*/
        child->n += 1;
        child->w = score;
        child->state = state;
    }
}
Node *UBFMSearcherLocal::next_child(const Node *node) const {
    ASSERT(node->child_len >= 0);
    Node *best_child = nullptr;
    NNScore best_score = NNScore(-1);
    auto min_num = UINT32_MAX;
    REP(i, node->child_len) {
        auto child = node->child(i);
        if (child->is_resolved()) { continue; }
        auto score = -child->w;
        if (score > best_score) {
            best_score = score;
            min_num = child->n;
            best_child = child;
        } else if (score == best_score) {
            if (child->n < min_num ) {
                min_num = child->n;
                best_child = child;
            }
        }
    }
    if (best_child != nullptr) {
        best_child->n++;
    }
    return best_child;
}
Node *UBFMSearcherLocal::next_child2(const Node *node) const {
    Node *best_child = nullptr;
    NNScore best_score = NNScore(-1);
    auto max_num = 0;
    REP(i, node->child_len) {
        auto child = node->child(i);
        if (child->is_resolved()) { continue; }
        if (-child->w > best_score) {
            best_score = -child->w;
            max_num = child->n;
            best_child = child;
        } else if (-child->w == best_score) {
            if (child->n > max_num ) {
                max_num = child->n;
                best_child = child;
            }
        }
    }
    ASSERT(best_child != nullptr);
    return best_child;
}
void UBFMSearcherLocal::update_node(Node *node) {

    Node *best_child = nullptr;
    auto max_value = NNScore(-1);
    auto max_num = -1;
    auto lose_num = 0;
    auto draw_num = 0;
    const auto child_len = node->child_len;
    ASSERT2(child_len > 0,{
        Tee<<node->pos<<std::endl;
        Tee<<child_len<<std::endl;
        Tee<<node->pos.is_done()<<std::endl;
        Tee<<node->str()<<std::endl;
    });

    REP(i, child_len) {
        auto child = node->child(i);

        if (child->is_resolved()) {

            //子供に負けを見つけた→つまり勝ちなので終わり
            if (child->is_lose()) {

                best_child = child;
                node->state = NodeWin;
                node->w = -child->w;
                node->best_move = child->parent_move;
                
                return;
            } else if (child->is_win()) {
                lose_num++;
            } else {
                ASSERT(child->is_draw());
                draw_num++;
            }
        }

        if (-child->w > max_value) {
            best_child = child;
            max_value = -child->w;
            max_num = child->n;
        } else if (-child->w == max_value) {
            if (child->n > max_num) {
                best_child = child;
                max_value = -child->w;
                max_num = child->n;
            }
        }
    }
    ASSERT(best_child != nullptr);
    
    if (child_len == draw_num) {
        node->state = NodeDraw;
        node->w = NNScore(0.0);
        return;
    } else if (child_len == lose_num) {
        node->state = NodeLose;
        node->w = -best_child->w;
        node->best_move = best_child->parent_move;
        return;
    } else if (child_len == (draw_num + lose_num)) {
        node->state = NodeDraw;
        node->w = NNScore(0.0);
        return;
    }
    node->w = -best_child->w;
    node->best_move = best_child->parent_move;
}


void UBFMSearcherGlobal::add_replay_buffer(const Node * node) const {
    selfplay::push_back(node->pos.hash_key(), node->w);
    REP(i, node->child_len) {
        auto child = node->child(i);
        if (child->is_terminal() && !child->is_resolved()) {
            continue;
        }
        this->add_replay_buffer(child);
    }
}
void test_ubfm() {
    g_searcher_global.init();
    game::Position pos;
    Tee<<pos<<std::endl;
    Timer timer;
    timer.start();
    Tee<<think_ubfm(pos)<<std::endl;
    timer.stop();
    Tee<<timer.elapsed()<<std::endl;
}
void test_ubfm2() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    //torch::Device device(torch::kCPU);
    auto module = torch::jit::load("./model/best_single_jit.pt",device);
    game::Position p;
    auto ff = [&](game::Position pos) {
        auto f = pos.feature();
        std::vector<at::Tensor> v;
        REP(i, FEAT_SIZE) {
            auto t = torch::tensor(torch::ArrayRef<int>((f[i])));
            v.push_back(t);
        }
        auto f_all = torch::stack(v,0);
        f_all = f_all.reshape({1, FEAT_SIZE, 3, 3});
        f_all = f_all.to(device,torch::kFloat);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(f_all);
        auto output = module.forward(inputs).toTensor();
        Tee<<pos<<std::endl;
        Tee<<output.item<float>()<<std::endl;
    };
    ff(p);
    ff(game::Position(8193));
    ff(game::Position(73728));
    ff(game::Position(74241));
    ff(game::Position(74304));
    ff(game::Position(336384));
    ff(game::Position(336000));
    ff(game::Position(163920));
    ff(game::Position(8208));
}


}
#endif