import torch
import DQN
import SnakeGameEnv

env = SnakeGameEnv.SnakeGame()
agent = DQN.DQNAgent((1, 10, 10), 4)

# 尝试加载模型
model_path = "snake_dqn_latest.pth"
try:
    agent.policy_net.load_state_dict(torch.load(model_path,weights_only=True))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print(f"模型已加载：{model_path}")
except FileNotFoundError:
    print(f"未找到模型文件：{model_path}，将从头开始训练。")

num_episodes = 100000

best_round = [0,0,0]

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    if episode % 1000 == 0: agent.epsilon = 1

    agent.epsilon = agent.epsilon*agent.epsilon_decay

    # agent.epsilon = 0.01

    while not done:
        # 环境渲染（可选）

        # env.render(false)

        valid_actions = env.get_valid_actions()

        action = agent.select_action(state, valid_actions)

        # 执行动作
        next_state, reward, done = env.step(action)

        # 存储经验
        agent.store_transition(state, action, reward, next_state, done)

        # 训练代理
        agent.train()

        state = next_state
        total_reward += reward

    if total_reward > best_round[1]:

        best_round = [num_episodes,total_reward,agent.epsilon]

    # 每50个回合保存一次模型
    if (episode + 1) % 50 == 0:
        # 也可以保存最新的模型，以便下次加载
        torch.save(agent.policy_net.state_dict(), "snake_dqn_latest.pth")
        print(f"模型已保存")

    print(f"回合 {episode + 1}/{num_episodes} 完成，得分: {total_reward}, Epsilon: {agent.epsilon:.4f}")

