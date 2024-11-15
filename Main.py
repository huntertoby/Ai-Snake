import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F

# 超參數
learning_rate = 0.001
gamma = 0.99
epsilon = 1
min_epsilon = 0.01
epsilon_decay = 0.999
batch_size = 1024
memory_size = 100000
target_update_freq = 50
num_episodes = 5000
grid_size = 10

# 初始化統計數據
stats_file = 'training_stats.json'
stats = {
    "total_episodes": 0,
    "wall_hits": 0,
    "self_hits": 0,
    "avg_score": 0,
    "max_score": 0
}

# 嘗試加載現有統計數據
if os.path.exists(stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# 嘗試加載已保存的模型
model_path = 'dqn_snake_model2.pth'
if os.path.exists(model_path):
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    print("已加載先前保存的模型。")
else:
    print("未找到先前保存的模型，將從頭開始訓練。")


# 選擇動作
def choose_action(state):
    global epsilon, snake_direction
    # 定義合法動作的集合，過濾掉「回頭」的方向
    valid_actions = []
    if snake_direction == 'UP':
        valid_actions = [0, 2, 3]  # 上、左、右
    elif snake_direction == 'DOWN':
        valid_actions = [1, 2, 3]  # 下、左、右
    elif snake_direction == 'LEFT':
        valid_actions = [0, 1, 2]  # 上、下、左
    elif snake_direction == 'RIGHT':
        valid_actions = [0, 1, 3]  # 上、下、右

    if np.random.rand() < epsilon:
        # 隨機選擇合法動作
        return random.choice(valid_actions)
    else:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
            q_values = q_values.cpu().numpy()
            valid_q_values = [(i, q_values[i]) for i in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action


# 儲存經驗到回放記憶庫
def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# 更新 Q 網絡
def update_network():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # 計算當前 Q 值
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 計算目標 Q 值
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # 計算損失
    loss = nn.MSELoss()(q_values, target_q_values)

    # 反向傳播更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 模擬遊戲環境
def reset_game():
    snake = [[5, 5]]
    food = [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]
    direction = 'RIGHT'
    return snake, food, direction


def step_game(snake, food, direction):
    head = list(snake[0])
    if direction == 'UP':
        head[1] -= 1
    elif direction == 'DOWN':
        head[1] += 1
    elif direction == 'LEFT':
        head[0] -= 1
    elif direction == 'RIGHT':
        head[0] += 1

    # 判斷撞牆或撞到自己
    if head in snake or head[0] < 0 or head[1] < 0 or head[0] >= grid_size or head[1] >= grid_size:
        return snake, food, direction, -30, True

    # 吃到食物
    if head == food:
        snake.insert(0, head)
        food = [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]
        reward = 10
    else:
        snake.insert(0, head)
        snake.pop()
        reward = -1

    return snake, food, direction, reward, False


# 訓練過程
for episode in range(num_episodes):
    snake, food, snake_direction = reset_game()
    state = (snake[0][0], snake[0][1], food[0], food[1], snake_direction)
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)
        snake, food, snake_direction, reward, done = step_game(snake, food, snake_direction)
        next_state = (snake[0][0], snake[0][1], food[0], food[1], snake_direction)

        store_transition(state, action, reward, next_state, done)
        update_network()

        state = next_state
        total_reward += reward

    # 記錄數據
    stats["total_episodes"] += 1
    if any(part[0] < 0 or part[1] < 0 or part[0] >= grid_size or part[1] >= grid_size for part in snake):
        stats["wall_hits"] += 1
    if len(snake) != len(set(tuple(part) for part in snake)):
        stats["self_hits"] += 1
    stats["avg_score"] = (stats["avg_score"] * (stats["total_episodes"] - 1) + total_reward) / stats["total_episodes"]
    stats["max_score"] = max(stats["max_score"], total_reward)

    # 衰減 epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    # 更新目標網絡
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 保存模型
    if (episode + 1) % 100 == 0:
        torch.save(policy_net.state_dict(), model_path)

# 儲存統計數據
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=4)

print("訓練完成！")
