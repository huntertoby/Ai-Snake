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


