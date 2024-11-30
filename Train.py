import torch
import DQN
import SnakeGameEnv

env = SnakeGameEnv.SnakeGame()
agent = DQN.DQNAgent((1, 10, 10), 4)


model_path = "snake_dqn_latest.pth"
try:
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print(f"模型已加載：{model_path}")
except FileNotFoundError:
    print(f"未找到模型文件：{model_path}，將從頭開始訓練。")


num_episodes = 1
highest_score = 0
results = []

highest_score_data = {
    "score": 0,
    "snake_pos": None,
    "food_pos": None,
    "snake_direction": None
}

for episode in range(1, num_episodes + 1):


    state = env.reset()
    total_reward = 0
    done = False

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)

        next_state, reward, done = env.step(action)

        agent.store_transition(state, action, reward, next_state, done)

        agent.train()

        state = next_state
        total_reward += reward

        print(str(action) + " ",end="")

    total_reward += 40

    if total_reward > highest_score_data["score"]:
        highest_score_data["score"] = total_reward
        highest_score_data["snake_pos"] = env.last_snake_pos
        highest_score_data["food_pos"] = env.food_pos
        highest_score_data["snake_direction"] = env.snake_direction

    results.append({
        "episode": episode,
        "score": total_reward,
        "highest_score": highest_score,
        "epsilon": round(agent.epsilon, 4)
    })

    print(f"回合 {episode}/{num_episodes} 完成，長度: {int(total_reward/10)}, 最長長度: {int(highest_score_data['score']/10)}, Epsilon: {agent.epsilon:.4f}")


torch.save(agent.policy_net.state_dict(), model_path)
print(f"模型已保存：{model_path}")

image_path = env.plot_game_state(highest_score_data, episode, is_game_over=True)
print(f"最高分的遊戲畫面已儲存：{image_path}")

with open("train_results.txt", "w") as file:
    for result in results:
        file.write(f"Episode {result['episode']} | "
                   f"Score: {result['score']} | "
                   f"Highest Score: {result['highest_score']} | "
                   f"Epsilon: {result['epsilon']}\n")
