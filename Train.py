import torch
import DQN
import SnakeGameEnv

# 初始化環境和代理
env = SnakeGameEnv.SnakeGame()
agent = DQN.DQNAgent((1, 10, 10), 4)

# 嘗試加載模型
model_path = "snake_dqn_latest.pth"
try:
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print(f"模型已加載：{model_path}")
except FileNotFoundError:
    print(f"未找到模型文件：{model_path}，將從頭開始訓練。")

# 訓練相關參數
num_episodes = 1
highest_score = 0
results = []  # 保存訓練結果

for episode in range(1, num_episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False

    # 每1000局重置 epsilon（探索率）
    if episode % 1000 == 0:
        agent.epsilon = 1.0

    # 遞減 epsilon
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)

        # 執行動作
        next_state, reward, done = env.step(action)

        # 存儲經驗
        agent.store_transition(state, action, reward, next_state, done)

        # 訓練代理
        agent.train()

        state = next_state
        total_reward += reward

    # 更新最高分
    highest_score = max(highest_score, total_reward)

    # 保存當前結果
    results.append({
        "episode": episode,
        "score": total_reward,
        "highest_score": highest_score,
        "epsilon": round(agent.epsilon, 4)
    })

    print(f"回合 {episode}/{num_episodes} 完成，得分: {total_reward}, 最高分: {highest_score}, Epsilon: {agent.epsilon:.4f}")

    # 每50局保存一次模型
    if episode % 50 == 0:
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"模型已保存：{model_path}")

# 保存結果到文件
with open("train_results.txt", "w") as file:
    for result in results:
        file.write(f"Episode {result['episode']} | "
                   f"Score: {result['score']} | "
                   f"Highest Score: {result['highest_score']} | "
                   f"Epsilon: {result['epsilon']}\n")
