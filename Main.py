import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

print("開始執行")

# 超參數
learning_rate = 0.001
gamma = 0.99
epsilon = 0.01
min_epsilon = 0.01
epsilon_decay = 0.995
batch_size = 256
memory_size = 10000
target_update_freq = 50
num_episodes = 2000
grid_size = 10
cell_size = 1

# 初始化統計數據
stats_file = 'training_stats.json'
stats = {
    "total_episodes": 0,
    "wall_hits": 0,
    "self_hits": 0,
    "avg_score": 0,
    "max_score": 0
}

# 定義 DQN 網絡
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化神經網絡與優化器
input_dim = 9  # 狀態特徵的維度
output_dim = 4  # 動作的數量 (上、下、左、右)
device = torch.device("cpu")

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)




