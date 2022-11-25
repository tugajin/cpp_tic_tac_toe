#include <vector>
#include <utility>
#include <thread>
#include <torch/torch.h>
#include <torch/script.h>
#include "nlohmann/json.hpp"
#include "util.hpp"
#include "common.hpp"
#include "game.hpp"
using json = nlohmann::json;

namespace selfplay {

struct ReplayInfo {
    uint32 hash;
    NNScore score;
    ReplayInfo(uint32 h, NNScore s) {
        this->hash = h;
        this->score = s;
    }
    ReplayInfo(const game::Position &pos, NNScore s) {
        this->hash = pos.hash_key();
        this->score = s;
    }
};

typedef std::vector<ReplayInfo> ReplayBuffer;

std::string dump(const ReplayBuffer & buf) {
    json info = {};
    for (auto &b : buf) {
        info.push_back({{"p", b.hash},{"s", b.score}});
    }
    return info.dump();
}
void test_nn() {
    // Position pos;
    // NNScore sc = 0.0;
    // ReplayInfo rp(pos,sc);
    // ReplayBuffer buf;
    
    // buf.push_back(rp);
    
    // pos = pos.next(Move(0));
    // rp = ReplayInfo(pos,sc);
    // buf.push_back(rp);
    // Tee<<dump(buf)<<std::endl;
    torch::jit::script::Module module;
    try {
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("../traced_resnet_model.pt",device);
        std::vector<torch::jit::IValue> inputs;
        int n = my_rand(10) + 1;
        inputs.push_back(torch::ones({n, 4, 3, 3},device));

        // Execute the model and turn its output into a tensor.
        Tee<<"forward\n";
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.sizes() << '\n';
        std::cout << output[0][0].item<float>() << '\n';
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit(1);
    }

  std::cout << "ok\n";
}
void test_selfplay() {
    std::thread th_a(test_nn);
    th_a.join();
}

}