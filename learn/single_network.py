# ====================
# シングルネットワークの作成
# ====================

# パッケージのインポート
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from game import *
# パラメータの準備
DN_FILTERS  = 256 # 畳み込み層のカーネル数（本家は256）
DN_RESIDUAL_NUM =  16 # 残差ブロックの数（本家は19）
DN_INPUT_SHAPE = (3, 4, 14) # 入力シェイプ
DN_OUTPUT_SIZE = 132 # 行動数(駒の移動先(12)*駒の移動元(11))

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return F.relu(out + x)
class SingleNet(nn.Module):
    def __init__(self, blocks=15, channels=192, fcl=256):
        super(SingleNet, self).__init__()
        self.convl1 = nn.Conv2d(in_channels=14, out_channels=channels, kernel_size=3, padding=1, bias=False)
        
        self.norm1 = nn.BatchNorm2d(channels)

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=DN_OUTPUT_SIZE, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(DN_OUTPUT_SIZE)
        self.value_fc1 = nn.Linear(1584, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, feature1):
        x1_1 = self.convl1(feature1)
        x = F.relu(self.norm1(x1_1))

        # resnet blocks
        x = self.blocks(x)

        # value head
        value = F.relu(self.value_norm1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))
        value = torch.tanh(self.value_fc2(value))
        return value
        
# デュアルネットワークの作成
def single_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best_single.h5'):
        return
    
    model = SingleNet()

    # モデルの保存    
    torch.save(model.state_dict(), './model/best_single.h5')# ベストプレイヤーのモデル

def print_network():
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    model = SingleNet()
    model.load_state_dict(torch.load('./model/best_single.h5',device))
    model = model.to(device)
    model.eval()
    state = State()

    # 推論のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    x = np.array(state.pieces_array())
    x = x.reshape(channel, file, rank)
    x = np.array([x])
    x = torch.tensor(x,dtype=torch.float32)
       
    x = x.to(device)
    with torch.no_grad():
        # 推論
        y = model(x)

    # 価値の取得
    value = y[0].item()
    print(value)

# 動作確認
if __name__ == '__main__':
    single_network()
    print_network()
