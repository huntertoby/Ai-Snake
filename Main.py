
import random
import os
import numpy as np

import torch
import torch.nn as nn 
import torch.optim as optim
from collections import deque



# 遊戲參數
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
CELL_SIZE = 20

# 計算網格大小
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE  # 10
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE  # 10

# 顏色設置
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

snake_speed = 50

# 超參數
learning_rate = 0.001
gamma = 0.99
epsilon = 1
min_epsilon = 0.01
epsilon_decay = 0.995
batch_size = 512
memory_size = 10000
target_update_freq = 50
num_episodes = 1000

# 定義 DQN 網絡
import torch
import torch.nn as nn
import torch.nn.functional as F


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
model_path = 'dqn_snake_model1.pth'
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
        # 使用 DQN 模型進行預測
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
            # 過濾掉「回頭」的動作，只選擇合法的動作
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


# 執行動作
def perform_action(action):
    global snake_direction
    if action == 0 and snake_direction != 'DOWN':
        snake_direction = 'UP'
    elif action == 1 and snake_direction != 'UP':
        snake_direction = 'DOWN'
    elif action == 2 and snake_direction != 'RIGHT':
        snake_direction = 'LEFT'
    elif action == 3 and snake_direction != 'LEFT':
        snake_direction = 'RIGHT'

def get_game_state():
    return {
        "snake_pos": snake_pos,
        "food_pos": food_pos,
        "snake_direction": snake_direction
    }

# 檢查正左、正右、正上、正下的危險
def is_danger(point, snake_body):
    # 使用集合來存儲蛇的身體部分（不包括頭）
    snake_body_set = set(tuple(body_part) for body_part in snake_body[1:])

    # 判斷是否超出邊界
    boundary_danger = (point[0] < 0 or point[0] >= SCREEN_WIDTH or
                       point[1] < 0 or point[1] >= SCREEN_HEIGHT)

    # 判斷是否撞到自己的身體
    body_danger = tuple(point) in snake_body_set

    return boundary_danger or body_danger


# 將遊戲狀態轉換為 DQN 可使用的狀態表示
def convert_to_state(game_state):
    snake_head = game_state["snake_pos"][0]
    food_pos = game_state["food_pos"]
    snake_body = game_state["snake_pos"]
    snake_direction = game_state["snake_direction"]

    # 目前方向编碼
    direction_mapping = {'UP': 3, 'DOWN': 4, 'LEFT': 1, 'RIGHT': 2}
    direction_code = direction_mapping[snake_direction]

    # 食物位置
    food_left = int(food_pos[0] < snake_head[0])
    food_right = int(food_pos[0] > snake_head[0])
    food_up = int(food_pos[1] < snake_head[1])
    food_down = int(food_pos[1] > snake_head[1])

    # 檢查正左、正右、正上、正下的危險
    danger_left = [snake_head[0] - CELL_SIZE, snake_head[1]]
    danger_right = [snake_head[0] + CELL_SIZE, snake_head[1]]
    danger_up = [snake_head[0], snake_head[1] - CELL_SIZE]
    danger_down = [snake_head[0], snake_head[1] + CELL_SIZE]

    danger_left_flag = int(is_danger(danger_left, snake_body))
    danger_right_flag = int(is_danger(danger_right, snake_body))
    danger_up_flag = int(is_danger(danger_up, snake_body))
    danger_down_flag = int(is_danger(danger_down, snake_body))

    # 狀態元組
    state = (direction_code, food_left, food_right, food_up, food_down,
             danger_left_flag, danger_right_flag, danger_up_flag, danger_down_flag)

    return state


# 獲取距離
def get_distance(snake_head, food_pos):
    return abs(snake_head[0] - food_pos[0]) + abs(snake_head[1] - food_pos[1])

# 訓練循環
for episode in range(num_episodes):
    snake_pos = [[100, 60], [80, 60], [60, 60]]
    snake_direction = 'RIGHT'
    food_pos = [random.randrange(0, GRID_WIDTH) * CELL_SIZE,
                random.randrange(0, GRID_HEIGHT) * CELL_SIZE]
    food_spawn = True
    game_over = False

    game_state = get_game_state()
    state = convert_to_state(game_state)

    total_reward = 0
    steps = 0
    food_eaten = 0

    while not game_over:

        # 選擇並執行動作
        game_state = get_game_state()
        state = convert_to_state(game_state)
        action = choose_action(state)
        perform_action(action)

        # 更新蛇的位置
        if snake_direction == 'UP':
            new_head = [snake_pos[0][0], snake_pos[0][1] - CELL_SIZE]
        elif snake_direction == 'DOWN':
            new_head = [snake_pos[0][0], snake_pos[0][1] + CELL_SIZE]
        elif snake_direction == 'LEFT':
            new_head = [snake_pos[0][0] - CELL_SIZE, snake_pos[0][1]]
        elif snake_direction == 'RIGHT':
            new_head = [snake_pos[0][0] + CELL_SIZE, snake_pos[0][1]]

        snake_pos.insert(0, new_head)

        reward = 0
        new_distance = get_distance(new_head, food_pos)

        # 根據距離變化給予獎勵/懲罰
        if new_distance < get_distance(snake_pos[0], food_pos):
            reward += 1  # 如果接近果實，給予獎勵
        elif new_distance > get_distance(snake_pos[0], food_pos):
            reward -= 1  # 如果遠離果實，給予懲罰

        if snake_pos[0] == food_pos:
            reward += 5 * len(snake_pos)  # 吃到食物的獎勵
            food_spawn = False
            food_eaten += 1
        else:
            snake_pos.pop()

        if (snake_pos[0][0] < 0 or snake_pos[0][0] >= SCREEN_WIDTH or
                snake_pos[0][1] < 0 or snake_pos[0][1] >= SCREEN_HEIGHT or
                snake_pos[0] in snake_pos[1:]):
            reward -= 30
            game_over = True
            if snake_pos[0][0] < 0 or snake_pos[0][0] >= SCREEN_WIDTH or snake_pos[0][1] < 0 or snake_pos[0][
                1] >= SCREEN_HEIGHT:
                print("Game over: Snake hit the wall.")
            else:
                print("Game over: Snake hit itself.")

        total_reward += reward
        steps += 1

        # 食物重新生成
        if not food_spawn:
            while True:
                food_pos = [random.randrange(0, GRID_WIDTH) * CELL_SIZE,
                            random.randrange(0, GRID_HEIGHT) * CELL_SIZE]
                if food_pos not in snake_pos:
                    break
            food_spawn = True

        next_game_state = get_game_state()
        next_state = convert_to_state(next_game_state)

        store_transition(state, action, reward, next_state, game_over)

        # 每隔4步更新一次 Q 網絡
        if steps % 4 == 0:
            update_network()

        state = next_state

    # 衰減 epsilon（探索率）
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    # 每隔一定回合更新目標網絡
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 每隔 100 回合保存模型
    if (episode + 1) % 100 == 0:
        torch.save(policy_net.state_dict(), model_path)
        print(f"模型在第 {episode + 1} 回合已保存。")

    # 打印回合信息
    print(f"回合 {episode + 1}/{num_episodes} 完成。")
    print(f"總獎勵: {total_reward}")
    print(f"步數: {steps}")
    print(f"吃到食物次數: {food_eaten}")
    print(f"Epsilon: {epsilon:.4f}")
    print("-" * 30)

    # a = input()

