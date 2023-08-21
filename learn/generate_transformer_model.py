#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from game import *
import numpy as np

DN_INPUT_SHAPE = (3, 3, 2) # 入力シェイプ

class Conv2DwithBatchNorm(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(Conv2DwithBatchNorm, self).__init__()
        self.conv_ = nn.Conv2d(input_ch, output_ch, kernel_size, bias=False, padding=kernel_size // 2)
        self.norm_ = nn.BatchNorm2d(output_ch)

    def forward(self, x):
        t = self.conv_.forward(x)
        t = self.norm_.forward(t)
        return t

class ValueHead(nn.Module):
    def __init__(self, channel_num, unit_num, hidden_size=256):
        super(ValueHead, self).__init__()
        self.value_conv_and_norm_ = Conv2DwithBatchNorm(channel_num, channel_num, 1)
        self.value_linear0_ = nn.Linear(channel_num, hidden_size)
        self.value_linear1_ = nn.Linear(hidden_size, unit_num)

    def forward(self, x):
        value = self.value_conv_and_norm_.forward(x)
        value = F.relu(value)
        value = F.avg_pool2d(value, [value.shape[2], value.shape[3]])
        value = value.view([-1, value.shape[1]])
        value = self.value_linear0_.forward(value)
        value = F.relu(value)
        value = self.value_linear1_.forward(value)
        value = torch.tanh(value)
        return value

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        input_channel_num = 2
        channel_num = 256
        square_num = 9
        layer_num = 5
        self.channel_num = channel_num
        self.first_encoding_ = torch.nn.Linear(input_channel_num, channel_num)
        encoder_layer = torch.nn.TransformerEncoderLayer(channel_num, nhead=8,dim_feedforward=channel_num*4)
        self.encoder_ = torch.nn.TransformerEncoder(encoder_layer,layer_num)
        self.value_head_ = ValueHead(channel_num, 1)
        #self.positional_encoding_ = torch.nn.Parameter(torch.zeros([square_num, 1, channel_num]), requires_grad=True)
        self.positional_encoding_ = torch.nn.Parameter(torch.zeros([1, square_num, channel_num]), requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view([batch_size, 2, 9])
        #x = x.permute([2, 0, 1])
        x = x.permute([0, 2, 1])
        x = self.first_encoding_(x)
        x = F.relu(x)
        x = x + self.positional_encoding_
        x = self.encoder_(x)
        #x = x.permute([1, 2, 0])
        x = x.permute([0, 2, 1])
        x = x.view([batch_size, self.channel_num, 3, 3])
        value = self.value_head_.forward(x)
        return value

def conv_jit():
    model = TransformerModel()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model.eval()
    sm = torch.jit.script(model)
    sm.save("./model/best_single_jit.pt")

def main():

    model = TransformerModel()
    model.eval()
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    torch.save(model.state_dict(), './model/best_single.h5')
    conv_jit()
    
    state = State()
    f = np.array(state.feature())
    print(type(f))
    f = f.reshape(2,3,3)
    input_data = torch.Tensor([f for i in range(10)])
    print(input_data.shape)
    loaded_model = torch.jit.load('./model/best_single_jit.pt')
    out = loaded_model(input_data)
    print(out.shape)
    print(out)

if __name__ == "__main__":
    main()
