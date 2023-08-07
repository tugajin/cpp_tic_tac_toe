#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from game import *
import numpy as np

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute([0, 2, 1, 3])
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute([0, 2, 1, 3])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        """ [input]
            - dim (int) : パッチのベクトルが変換されたベクトルの長さ
            - hidden_dim (int) : 隠れ層のノード数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        """[input]
            - x (torch.Tensor)
                - x.shape = torch.Size([batch_size, n_patches + 1, dim])
        """
        x = self.net(x)
        return x

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.pool = Pooling()
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim)

    def forward(self, x):
        x = x + self.pool(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

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

class PoolformerModel(nn.Module):
    def __init__(self):
        super(PoolformerModel, self).__init__()
        input_channel_num = 2
        channel_num = 256
        nhead = 12
        block_num = 5
        square_num = 9
        self.first_encoding_ = torch.nn.Linear(input_channel_num, channel_num)
        self.encoder_ = nn.Sequential(*[Block(dim=channel_num, num_heads=nhead) for _ in range(block_num)])
        self.channel_num = channel_num
        self.value_head_ = ValueHead(channel_num, 1)
        self.positional_encoding_ = torch.nn.Parameter(torch.zeros([square_num, 1, channel_num]), requires_grad=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view([batch_size, 2, 9])
        x = x.permute([2, 0, 1])
        x = self.first_encoding_(x)
        x = F.relu(x)
        x = x + self.positional_encoding_
        x = self.encoder_(x)
        x = x.permute([1, 2, 0])
        x = x.view([batch_size, self.channel_num, 3, 3])

        value = self.value_head_.forward(x)
        return value

def conv_jit():
    model = PoolformerModel()
    model.load_state_dict(torch.load('./model/best_single.h5'))
    model.eval()
    sm = torch.jit.script(model)
    sm.save("./model/best_single_jit.pt")

def main():

    model = PoolformerModel()
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

if __name__ == "__main__":
    main()
