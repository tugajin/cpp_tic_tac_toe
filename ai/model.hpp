#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <torch/torch.h>
#include <torch/script.h>
#include "util.hpp"
#include "thread.hpp"
#include "hash.hpp"
#include "game.hpp"

namespace model {

class GPUModel {
public:
    GPUModel():
        device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
        if (torch::cuda::is_available()) {
            Tee<<"GPU\n";
        } else {
            Tee<<"CPU\n";
        }
    }
    void load_model(const int id);
    void predict(std::vector<nn::Feature> &feat_list, std::vector<nn::NNScore> &outputs);
    static constexpr int GPU_NUM = 1;
private:
    torch::Device device;
    torch::jit::script::Module module;
    Lockable lock_gpu;
    int gpu_id;
};

extern GPUModel g_gpu_model[GPUModel::GPU_NUM];

void GPUModel::load_model(const int id) {
    this->gpu_id = id;
    this->lock_gpu.lock();
    Tee<<"load_model("<<id<<")...";
    REP(i, 10) {
        try {
            this->module = torch::jit::load("./model/best_single_jit.pt",this->device);
            this->module.eval();
            Tee<<"end\n";
            this->lock_gpu.unlock();
            return;
        } catch (const c10::Error& e) {
            Tee << "error loading the model\n";
            my_sleep(1000);
            continue;
        }
    }
    this->lock_gpu.unlock();
    std::exit(EXIT_FAILURE);
}
void GPUModel::predict(std::vector<nn::Feature> &feat_list, std::vector<nn::NNScore> &outputs) {
    
    // Timer timer;
    // timer.start();
    std::vector<at::Tensor> tensor_list;
    
    for (auto &feat : feat_list) {
        std::vector<at::Tensor> v;
        REP(i, nn::FEAT_SIZE) {
            v.push_back(torch::tensor(torch::ArrayRef<int>(feat[i])));
        }
        tensor_list.push_back(torch::stack(v).reshape({nn::FEAT_SIZE, FILE_SIZE, RANK_SIZE}));
    }

    auto feat_tensor = torch::stack(tensor_list);

    this->lock_gpu.lock();

    feat_tensor = feat_tensor.to(this->device,torch::kFloat);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(feat_tensor);
    auto output = this->module.forward(inputs).toTensor();
    
    this->lock_gpu.unlock();
    
    REP(i, static_cast<int>(feat_list.size())) {
        outputs.push_back(output[i][0].item<float>());
    }
    // timer.stop();
    // Tee<<"     elapsed:"<<timer.elapsed()<<std::endl;
}

void predict(const int gpu_id, std::vector<nn::Feature> &feat_list, std::vector<nn::NNScore> &outputs) {
    g_gpu_model[gpu_id].predict(feat_list, outputs);
}

void test_model() {
    g_gpu_model[0].load_model(0);
    game::Position pos;
    pos = hash::hirate();
    auto feat = nn::feature(pos);
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> output_list;
    feat_list.push_back(feat);
    feat_list.push_back(feat);
    feat_list.push_back(feat);
    predict(0, feat_list, output_list);
    Tee<<output_list[0]<<std::endl;
}
nn::NNScore predict_problem(const Key k) {
    auto pos = hash::from_hash(k);
    //Tee<<pos<<std::endl;
    if (pos.is_win()) {
        return nn::NNScore(0.99);
    } else if (pos.is_lose()) {
        return nn::NNScore(-0.99);
    }
    auto feat = nn::feature(pos);
    std::vector<nn::Feature> feat_list;
    std::vector<nn::NNScore> output_list;
    feat_list.push_back(feat);
    predict(0, feat_list, output_list);
    return output_list[0];
}

}

#endif