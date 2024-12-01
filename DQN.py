import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch import optim


# 更新 DQN 类，使用卷积神经网络
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),  # 输出维度: 32 x H x W
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 输出维度: 64 x H x W
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 输出维度: 64 x H x W
            nn.ReLU(),
        )
        # 计算卷积层的输出维度
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# 定义经验回放缓冲区
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN 代理
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = ReplayMemory(50000)
        self.batch_size = 32
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.update_target_freq = 100  # 目标网络更新频率

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.steps_done = 0

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # 增加批量和通道维度
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_values_filtered = q_values[0, valid_actions]
                max_action_index = q_values_filtered.argmax().item()
                return valid_actions[max_action_index]

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from memory
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).unsqueeze(1).to(self.device)
        actions = torch.tensor(batch[1]).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))


        loss = self.loss_fn(q_values, target_q_values)

        if loss.item()>10:
            print(f"Loss: {loss.item():.4f}")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.update_target_freq == 0:
            self.update_target_network()



        self.steps_done += 1
