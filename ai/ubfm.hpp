#ifndef __UBFM_HPP__
#define __UBFM_HPP__

#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include "common.hpp"
#include "util.hpp"
#include "game.hpp"
#include "thread.hpp"
#include "nn.hpp"
#include "countreward.hpp"
#include "model.hpp"

namespace ubfm {

enum NodeState : int {
    NodeDraw = 0,
    NodeWin = 1,
    NodeLose = 2,
    NodeUnknown = -1
};

inline nn::NNScore score_win(const int ply) {
    return nn::NNScore(0.9999) - nn::NNScore(static_cast<double>(ply) / 2000.0);
}
inline nn::NNScore score_lose(const int ply) {
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
    Node() : child_nodes(nullptr),
             parent_move(MOVE_NONE),
             best_move(MOVE_NONE),
             state(NodeUnknown),
             w(0.0),
             init_w(0.0),
             n(0),
             child_len(-1),
             ply(-1){}
   Node( const Node & ) = delete ;
   Node & operator = ( const Node & ) = delete ;
   Node & operator = ( const Node && n) = delete;
    void init() {
        this->w = this->init_w = 0.0;
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
        str += padding + "init_w:" + to_string(init_w) + "\n";
        str += padding + "n:" + to_string(n) + "\n";
        str += padding + "child_len:" + to_string(child_len) + "\n";
        str += padding + "ply:" + to_string(ply) + "\n";
        str += padding + "parent_move:" + move_str(parent_move) + "\n";
        str += padding + "best_move:" + move_str(best_move) + "\n";
        str += padding + "state:" + to_string(state) + "\n";
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
        if (this->is_terminal()) {
            if (this->n != 1 && this->n != 0) {
                Tee<<"terminal error\n";
                return false;
            }
            return true;
        } else {
            const auto parent_n = this->n;
            auto child_n = 0;
            REP(i, this->child_len) {
                const auto child = this->child(i);
                child_n += child->n;
            }
            if (parent_n != (child_n+1)) {
                Tee<<"parent error\n";
                return false;
            }
            return true;
        }
    }
    bool is_ok2() const {
        if (this->is_terminal()) {
            // こないはず
            Tee<<"terminal error2\n";
            return false;
        }
        const auto parent_n = this->n;
        auto child_n = 0;
        REP(i, this->child_len) {
            const auto child = this->child(i);
            child_n += child->n;
        }
        if (parent_n != child_n) {
            Tee<<"parent error2\n";
            return false;
        }
        return true;
    }
    game::Position pos;
    std::unique_ptr<std::unique_ptr<Node>[]> child_nodes;
    Lockable lock_node;
    Move parent_move;
    Move best_move;
    NodeState state;
    nn::NNScore w;
    nn::NNScore init_w;
    int n;
    int child_len;
    int ply;
};
class UBFMSearcherGlobal;

class UBFMSearcherLocal {
public:
    UBFMSearcherLocal(int id, int gpu_id, UBFMSearcherGlobal * global) : 
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
    void expand(Node *node);
    template<bool is_descent> Node *next_child(const Node *node) const;
    void update_node(Node *node);
    void add_node(const game::Position &pos,const Move parent_move, int ply);
    bool interrupt(const uint32 current_num, const uint32 simulation_num) const;
    Node *root_node() const ;

    UBFMSearcherGlobal *global;
    std::thread *thread;
    int thread_id;
    int gpu_id;
};

class UBFMSearcherGlobal {
public:
    UBFMSearcherGlobal() :
                       THREAD_NUM(1){}
    UBFMSearcherGlobal(const int thread_num) : 
                       THREAD_NUM(thread_num){}
    Node root_node;
    void init();
    void clear_tree();
    void run();
    void join();
    void choice_best_move();

    int THREAD_NUM;
protected:
    std::vector<UBFMSearcherLocal> worker;
};

extern UBFMSearcherGlobal g_searcher_global;

Move think_ubfm(game::Position &pos) {
    ASSERT(g_searcher_global.root_node.n == 0);
    g_searcher_global.root_node.pos = pos;
    g_searcher_global.run();
    g_searcher_global.join();
    g_searcher_global.choice_best_move();
    return g_searcher_global.root_node.best_move;
}

Node *UBFMSearcherLocal::root_node() const {
    return &this->global->root_node;
}

void UBFMSearcherGlobal::init() {
    this->worker.clear();
    this->worker.shrink_to_fit();
    REP(i, UBFMSearcherGlobal::THREAD_NUM) {
        this->worker.emplace_back(i,0,this);
    }
    this->clear_tree();
}

void UBFMSearcherGlobal::clear_tree() {
    this->root_node.init();
}

void UBFMSearcherGlobal::run() {
    REP(i, UBFMSearcherGlobal::THREAD_NUM) {
        this->worker[i].run();
    }
}

void UBFMSearcherGlobal::join() {
    REP(i, UBFMSearcherGlobal::THREAD_NUM) {
        this->worker[i].join();
    }
}

void UBFMSearcherGlobal::choice_best_move() {
    std::vector<double> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(0.0);
    }
    REP(i, this->root_node.child_len) {
        ASSERT(i>=0);
        ASSERT(i<this->root_node.child_len);
        auto child = this->root_node.child(i);
        if (child->is_resolved()) {
            if (child->is_lose()) {
                REP(i, this->root_node.child_len) {
                    scores[i] = 0;
                }
                scores[i] = 1;
                break;
            } else if (child->is_draw()) {
                scores[i] = (child->n) + 1.0;
            } else if (child->is_win()) {
                scores[i] = 0.0;
            }
        } else {
            scores[i] = child->n + 1.0 - child->w;
        }
    }
    auto index = -1;
    auto iter = std::max_element(scores.begin(), scores.end());
    index = std::distance(scores.begin(), iter);
    ASSERT(index>=0);
    ASSERT(index<this->root_node.child_len);
    auto child = this->root_node.child(index);
    this->root_node.best_move = child->parent_move;
}

void UBFMSearcherLocal::run() {
	this->thread = new std::thread([this]() {
        this->search(int(2000 / this->global->THREAD_NUM));
    });
}
void UBFMSearcherLocal::join() {
    this->thread->join();
    delete this->thread;
}

bool UBFMSearcherLocal::interrupt(const uint32 current_num, const uint32 simulation_num) const {
    if (this->root_node()->is_resolved()) {
        return true;
    }
    if (current_num >= simulation_num) {
        return true;
    }
    return false;
}

void UBFMSearcherLocal::search(const uint32 simulation_num) {
    
    const auto is_out = (this->thread_id == 0) && (this->gpu_id == 0);
    for(auto i = 0u ;; ++i) {
        Tee<<"start simulation:" << i <<"/"<<simulation_num<<"\r";
        const auto interrupt = this->interrupt(i, simulation_num);
        if (interrupt) {
            break;
        }
        this->evaluate(this->root_node());
        if (is_out) {
            Tee<<this->root_node()->str()<<std::endl;
        }
    }
    if (is_out) {
        Tee<<this->root_node()->str()<<std::endl;
    }
}

void UBFMSearcherLocal::evaluate(Node *node) {
    // if (node == nullptr) {
    //     Tee<<"???????"<<this->thread_id<<std::endl;
    //     return;
    // }
    ASSERT(node != nullptr);
    ASSERT(std::fabs(node->w) <= 1);
    node->lock_node.lock();
    node->n++;

    if (node->pos.is_draw()) {
        node->w = nn::NNScore(0.0);
        node->state = NodeState::NodeDraw;
        node->lock_node.unlock();
        return;
    } 
    if (node->pos.is_lose()) {
        node->state = NodeState::NodeLose;
        node->lock_node.unlock();
        return;
    }
    if (node->is_resolved()) {
        node->lock_node.unlock();
        return;
    }
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
        node->lock_node.unlock();
    } else {
        auto next_node = this->next_child<false>(node);
        node->lock_node.unlock();
        this->evaluate(next_node);
    }
    node->lock_node.lock();
    this->update_node(node);

    node->lock_node.unlock();
}

void UBFMSearcherLocal::expand(Node *node) {
    auto moveList = movelist::MoveList();
    gen::legal_moves(node->pos, moveList);
    
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
    
    ASSERT2(node->child_len > 0,{
        Tee<<node->pos<<std::endl;
    });
    // Timer timer;
    // timer.start();
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> outputs_list;
    //Tee<<"  init:"<<timer.elapsed()<<std::endl;
    REP(i, node->child_len) {
        ASSERT(i>=0);
        ASSERT(i<node->child_len);
        auto child = node->child(i);
        auto &pos = child->pos;
        feat_list.push_back(nn::feature(pos));
    }
    //Tee<<"  push_back:"<<timer.elapsed()<<std::endl;

    model::predict(this->gpu_id, feat_list, outputs_list);
    
    //Tee<<"  predict:"<<timer.elapsed()<<std::endl;

    REP(i, node->child_len) {
        auto state = NodeUnknown;
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

        if (pos.is_draw()) {
            score = nn::NNScore(0.0);
            state = NodeDraw;
        } else if (pos.is_lose()) {
            score = score_lose(child->ply);
            state = NodeLose;
        } else if (pos.is_win()) {
            score = score_win(child->ply);
            state = NodeWin;
        } 
        ASSERT2(std::fabs(score)<=1,{
            Tee<<child->ply<<std::endl;
        });
        child->w = child->init_w = score;
        child->state = state;
    }
    //Tee<<"  end:"<<timer.elapsed()<<std::endl;
}
template<bool is_descent> Node *UBFMSearcherLocal::next_child(const Node *node) const {
    ASSERT(node->child_len >= 0);
    Node *best_child = nullptr;
    nn::NNScore best_score = nn::NNScore(-10000);
    auto min_num = INT32_MAX;
    REP(i, node->child_len) {

        ASSERT(i>=0);
        ASSERT(i<node->child_len);

        auto child = node->child(i);
        ASSERT(std::fabs(child->w)<=1); 
        if (child->is_resolved()) { continue; }
        auto score = -child->w;
        if (is_descent) {
            score += static_cast<nn::NNScore>(rand_gaussian(0.0,0.2));
        }
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
    return best_child;
}

void UBFMSearcherLocal::update_node(Node *node) {

    Node *best_child = nullptr;
    auto max_value = nn::NNScore(-1);
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
        ASSERT(i>=0);
        ASSERT(i<child_len);

        auto child = node->child(i);
        ASSERT(std::fabs(child->w)<=1);
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
    ASSERT2(best_child != nullptr,{
        Tee<<node->pos<<std::endl;
        Tee<<child_len<<std::endl;
        Tee<<node->pos.is_done()<<std::endl;
        Tee<<node->str()<<std::endl;
    });
    
    if (child_len == draw_num) {
        node->state = NodeDraw;
        node->w = nn::NNScore(0.0);
        return;
    } else if (child_len == lose_num) {
        node->state = NodeLose;
        node->w = -best_child->w;
        node->best_move = best_child->parent_move;
        return;
    } else if (child_len == (draw_num + lose_num)) {
        node->state = NodeDraw;
        node->w = nn::NNScore(0.0);
        return;
    }
    node->w = -best_child->w;
    node->best_move = best_child->parent_move;
}


}
#endif
