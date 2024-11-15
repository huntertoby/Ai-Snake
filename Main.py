import os
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
device = torch.device("cpu")

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
    global epsilon
    direction = state[0]  # 獲取當前方向（數值）

    # 定義合法動作，過濾掉「回頭」的方向
    valid_actions = []
    if direction == 0:  # UP
        valid_actions = [0, 2, 3]
    elif direction == 1:  # DOWN
        valid_actions = [1, 2, 3]
    elif direction == 2:  # LEFT
        valid_actions = [0, 1, 2]
    elif direction == 3:  # RIGHT
        valid_actions = [0, 1, 3]

    if np.random.rand() < epsilon:
        return random.choice(valid_actions)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            q_values = q_values.cpu().numpy()
            valid_q_values = [(i, q_values[i]) for i in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action


# 檢查危險
def is_danger(point, snake_body):
    snake_body_set = set(tuple(body_part) for body_part in snake_body[1:])
    boundary_danger = (point[0] < 0 or point[0] >= grid_size or
                       point[1] < 0 or point[1] >= grid_size)
    body_danger = tuple(point) in snake_body_set
    return boundary_danger or body_danger


# 將遊戲狀態轉換為 DQN 狀態表示
def convert_to_state(game_state):
    snake_head = game_state["snake_pos"][0]
    food_pos = game_state["food_pos"]
    snake_body = game_state["snake_pos"]
    direction_code = game_state["snake_direction"]

    food_left = int(food_pos[0] < snake_head[0])
    food_right = int(food_pos[0] > snake_head[0])
    food_up = int(food_pos[1] < snake_head[1])
    food_down = int(food_pos[1] > snake_head[1])

    danger_left = is_danger([snake_head[0] - cell_size, snake_head[1]], snake_body)
    danger_right = is_danger([snake_head[0] + cell_size, snake_head[1]], snake_body)
    danger_up = is_danger([snake_head[0], snake_head[1] - cell_size], snake_body)
    danger_down = is_danger([snake_head[0], snake_head[1] + cell_size], snake_body)

    state = (direction_code, food_left, food_right, food_up, food_down,
             danger_left, danger_right, danger_up, danger_down)
    return state


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

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 獲取距離
def get_distance(snake_head, food_pos):
    return abs(snake_head[0] - food_pos[0]) + abs(snake_head[1] - food_pos[1])

def generate_food(snake_body):
    while True:
        food = [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]
        if food not in snake_body:
            return food


# 模擬遊戲環境
def reset_game():
    snake = [[5, 5]]
    food = generate_food(snake)
    direction = 3
    return {"snake_pos": snake, "food_pos": food, "snake_direction": direction}

def step_game(state, action):
    head = list(state["snake_pos"][0])
    if action == 0:
        head[1] -= 1
    elif action == 1:
        head[1] += 1
    elif action == 2:
        head[0] -= 1
    elif action == 3:
        head[0] += 1

    if head in state["snake_pos"] or head[0] < 0 or head[1] < 0 or head[0] >= grid_size or head[1] >= grid_size:
        return state, -30, True

    reward = 0
    new_distance = get_distance(head, state["food_pos"])
    if new_distance < get_distance(state["snake_pos"][0], state["food_pos"]):
        reward += 1
    elif new_distance > get_distance(state["snake_pos"][0], state["food_pos"]):
        reward -= 1

    if head == state["food_pos"]:
        reward += 10
        state["food_pos"] = generate_food(state["snake_pos"])  # 使用新的生成邏輯
        state["snake_pos"].insert(0, head)
    else:
        state["snake_pos"].insert(0, head)
        state["snake_pos"].pop()

    return state, reward, False



# 訓練過程
for episode in range(num_episodes):

    
    state = reset_game()
    state_representation = convert_to_state(state)
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state_representation)
        next_state, reward, done = step_game(state, action)
        next_state_representation = convert_to_state(next_state)

        store_transition(state_representation, action, reward, next_state_representation, done)
        update_network()

        state_representation = next_state_representation
        total_reward += reward
    
    stats["total_episodes"] += 1
    if total_reward < -10:
        stats["wall_hits"] += 1
    stats["avg_score"] = (stats["avg_score"] * (stats["total_episodes"] - 1) + total_reward) / stats["total_episodes"]
    stats["max_score"] = max(stats["max_score"], total_reward)

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 100 == 0:
        torch.save(policy_net.state_dict(), model_path)
        
    print(total_reward)


with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=4)

print("訓練完成！")

他連開始執行都沒顯示ㄟ
