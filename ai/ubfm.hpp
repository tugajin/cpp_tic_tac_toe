#ifndef __UBFM_HPP__
#define __UBFM_HPP__

#include <array>
#include <algorithm>
#include <vector>
#include <chrono>
#include <thread>
#include <torch/torch.h>
#include <torch/script.h>
#include "nlohmann/json.hpp"
#include "common.hpp"
#include "util.hpp"
#include "game.hpp"
#include "selfplay.hpp"

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
             child_nodes(),
             ply(-1),
             state(NodeUnknown) {}
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
    std::string str() const {
        std::string str = "---------------------------\n";
       // str += pos.str();
        str += "w:" + to_string(w) + "\n";
        str += "n:" + to_string(n) + "\n";
        str += "child_len:" + to_string(child_len) + "\n";
        str += "ply:" + to_string(ply) + "\n";
        str += "parent_move:" + to_string(parent_move) + "\n";
        str += "best_move:" + to_string(best_move) + "\n";
        str += "state:" + to_string(state) + "\n";
        str += "---------------------------\n";
        return str;
    }
    game::Position pos;
    NNScore w;
    uint32 n;
    int child_len;
    int ply;
    std::array<uint32, POS_SIZE> child_nodes;
    Move parent_move;
    Move best_move;
    NodeState state;
};

class UBFMSearcher {
public:
    UBFMSearcher() : nodes(nullptr), 
                     root_node(Node()),
                     temperature(0),
                     device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
                     current_index(-1){
        if (torch::cuda::is_available()) {
            Tee<<"GPU\n";
        } else {
            Tee<<"CPU\n";
        }
    }
    ~UBFMSearcher() {
        this->free();
    }
    void allocate() {
        ASSERT(this->nodes == nullptr);
        nodes = new Node[NODE_SIZE];
        if (nodes == nullptr) {
            Tee<<"error allocate\n";
            std::exit(EXIT_FAILURE);;
        }
        this->current_index = 0;
    }
    void free() {
        if (this->nodes != nullptr) {
            delete[] this->nodes;
            this->nodes = nullptr;
        }
        this->current_index = -1;
    }
    Move best_move() const {
        return this->root_node.best_move;
    }
    template<bool is_descent = false>void search(const game::Position &pos, const int simulation_num);
    void load_model();
    bool is_ok();
private:
    void evaluate(Node *node);
    void evaluate_descent(Node *node);
    void expand(Node *node);
    void predict(Node *node);
    Node *next_child(const Node *node) const;
    Node *next_child2(const Node *node) const;
    void update_node(Node *node);
    void add_node(const game::Position &pos,const Move parent_move, int ply);
    void choice_best_move();
    void choice_best_move_e_greedy();
    void choice_best_move_ordinal();
    void add_replay_buffer() const ;
    template<bool disp_all = false, bool disp_child = true> std::string str(const Node *node) const;
    Node *child(const Node *node, const int index) const {
        ASSERT(node != nullptr);
        ASSERT(index >= 0);
        ASSERT2(index < node->child_len,{
            Tee<<index<<std::endl;
            Tee<<node->child_len<<std::endl;
        })
        ASSERT(node->child_nodes[index] >= 0);
        ASSERT(node->child_nodes[index] < NODE_SIZE);
        return &this->nodes[node->child_nodes[index]];
    }
    template<bool is_descent = false>void search_init(const game::Position &pos) {
        this->root_node = Node();
        this->root_node.pos = pos;
        this->current_index = 0;
        if (is_descent) {
            this->temperature = 0.2;
        } else {
            this->temperature = 0.0;
        }
    }
    static constexpr uint32 NODE_SIZE = 1 << 16;
    Node *nodes;
    Node root_node;
    uint32 current_index;
    double temperature;
    torch::Device device;
    torch::jit::script::Module module;
};

extern UBFMSearcher g_searcher;

void UBFMSearcher::choice_best_move() {
    std::vector<int> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(1);
    }
    REP(i, this->root_node.child_len) {
        auto child = this->child(&this->root_node,i);
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
    auto child = this->child(&this->root_node,index);
    this->root_node.best_move = child->parent_move;
}
void UBFMSearcher::choice_best_move_e_greedy() {
    ASSERT(this->temperature > 0.0);
    std::vector<NNScore> scores;
    REP(i, this->root_node.child_len) {
        scores.push_back(NNScore(0.0));
    }
    auto find_resolved_flag = false;
    REP(i, this->root_node.child_len) {
        auto child = this->child(&this->root_node,i);
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
    if (!find_resolved_flag && rand_double() < this->temperature) {
        index = my_rand(this->root_node.child_len);
    } else {
        auto iter = std::max_element(scores.begin(), scores.end());
        index = std::distance(scores.begin(), iter);
    }
    auto child = this->child(&this->root_node,index);
    this->root_node.best_move = child->parent_move;
}

template<bool is_descent>void UBFMSearcher::search(const game::Position &pos, const int simulation_num) {
    ASSERT(this->nodes != nullptr);
    this->search_init<is_descent>(pos);
    REP(i, simulation_num) {
        //Tee<<"start simulation:" << i <<"\n";
        if (!is_descent && this->root_node.is_resolved()) {
            //Tee<<"root node is resolved\n";
            break;
        }
        if (is_descent) {
            this->evaluate_descent(&this->root_node);
        } else {
            this->evaluate(&this->root_node);
        }
        //Tee<<this->str<false,true>(&this->root_node)<<std::endl;
    }
    //Tee<<this->str<false,true>(&this->root_node)<<std::endl;
    if (is_descent) {
        this->choice_best_move_e_greedy();
        this->add_replay_buffer();
    } else {
        this->choice_best_move();
    }
}

void UBFMSearcher::evaluate(Node *node) {
    ASSERT(!node->is_resolved());
    ASSERT2(!node->pos.is_done(),{
        Tee<<node->pos<<std::endl;
    });
    if (node->is_terminal()) {
        this->expand(node);
        this->predict(node);
    } else {
        auto next_node = this->next_child(node);
        this->evaluate(next_node);
    }
    this->update_node(node);
}

void UBFMSearcher::evaluate_descent(Node *node) {
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

void UBFMSearcher::add_node(const game::Position &pos, const Move parent_move, const int ply) {
    auto node = &this->nodes[this->current_index];
    node->pos = pos;
    node->w = NNScore(0.0);
    node->n = 0;
    node->child_len = -1;
    node->ply = ply;
    node->parent_move = parent_move;
    node->best_move = MOVE_NONE;
    node->state = NodeUnknown;
    this->current_index++;
}
void UBFMSearcher::expand(Node *node) {
    auto moveList = movelist::MoveList();
    node->pos.legal_moves(moveList);
    node->child_len = moveList.len();
    REP(i, node->child_len) {
        node->child_nodes[i] = this->current_index;
        auto next_pos = node->pos.next(moveList[i]);
        this->add_node(next_pos,moveList[i],node->ply+1);
    }
}
void UBFMSearcher::predict(Node *node) {
    std::vector<at::Tensor> tensor_list;
    REP(i, node->child_len) {
        auto child = this->child(node, i);
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
    feat_tensor = feat_tensor.to(this->device,torch::kFloat);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(feat_tensor);
    auto output = module.forward(inputs).toTensor();
    REP(i, node->child_len) {
        auto state = NodeUnknown;
        auto score = to_nnscore(output[i][0].item<float>());     
        if (score >= NNScore(1.0)) {
            score = NNScore(0.8999);
        } else if (score <= NNScore(-1.0)) {
            score = NNScore(-0.8999);
        }
        auto child = this->child(node, i);
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
Node *UBFMSearcher::next_child(const Node *node) const {
    ASSERT(node->child_len >= 0);
    Node *best_child = nullptr;
    NNScore best_score = NNScore(-1);
    auto min_num = UINT32_MAX;
    REP(i, node->child_len) {
        auto child = this->child(node,i);
        if (child->is_resolved()) { continue; }
        if (-child->w > best_score) {
            best_score = -child->w;
            min_num = child->n;
            best_child = child;
        } else if (-child->w == best_score) {
            if (child->n < min_num ) {
                min_num = child->n;
                best_child = child;
            }
        }
    }
    //ASSERT(best_child != nullptr);
    return best_child;
}
Node *UBFMSearcher::next_child2(const Node *node) const {
    Node *best_child = nullptr;
    NNScore best_score = NNScore(-1);
    auto max_num = 0;
    REP(i, node->child_len) {
        auto child = this->child(node,i);
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
void UBFMSearcher::update_node(Node *node) {
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
        auto child = this->child(node, i);
        if (child->is_resolved()) {
            //??????????????????????????????????????????????????????????????????
            if (child->is_lose()) {
                best_child = child;
                node->state = NodeWin;
                node->w = -child->w;
                node->best_move = child->parent_move;
                node->n++;
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
    node->n++;
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
template<bool disp_all, bool disp_child> std::string UBFMSearcher::str(const Node *node) const {
    std::string str = node->pos.str();
    str += node->str();
    if (disp_child) {
        str += "children\n";
        REP(i, node->child_len) {
            auto child = this->child(node,i);
            str += child->str();
        }
    }
    if (disp_all && !node->is_resolved() && !node->is_terminal()) {
        str += "next\n";
        auto child = this->next_child2(node);
        str += this->str<true,false>(child);
    }
    return str;
}
void UBFMSearcher::load_model() {
    REP(i, 10) {
        try {
            this->module = torch::jit::load("./model/best_single_jit.pt",this->device);
            this->module.eval();
            return;
        } catch (const c10::Error& e) {
            Tee << "error loading the model\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));        //1???????????????????????????
            continue;
        }
    }
    std::exit(EXIT_FAILURE);
}
void UBFMSearcher::add_replay_buffer() const {
    selfplay::push_back(root_node.pos.hash_key(),root_node.w);
    //Tee<<this->str(&this->root_node)<<std::endl;
    REP(i, this->current_index) {
        auto hash = this->nodes[i].pos.hash_key();
        auto w = this->nodes[i].w;
        if (this->nodes[i].is_terminal() && !this->nodes[i].is_resolved()) {
            //Tee<<"skip:"<<hash<<":"<<w<<std::endl;
            continue;
        }
        //Tee<<hash<<":"<<w<<std::endl;
        selfplay::push_back(hash, w);
    }
}
void test_ubfm() {
    g_searcher.allocate();
    g_searcher.load_model();
    //game::Position pos(74241);
    game::Position pos;
    pos = pos.next(Move(2));
    pos = pos.next(Move(1));
    Tee<<pos<<std::endl;
    g_searcher.search<true>(pos, 50);
    Move best_move = g_searcher.best_move();
    Tee<<best_move<<std::endl;
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