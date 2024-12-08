import torch
import DQN
import SnakeGameEnv

env = SnakeGameEnv.SnakeGame()
agent = DQN.DQNAgent((1, 10, 10), 4)

model_path = "snake_dqn_latest.pth"
try:
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print(f"Model loaded: {model_path}")
except FileNotFoundError:
    print(f"Model file not found: {model_path}. Starting training from scratch.")

num_episodes = 100
highest_score = 0
results = []

highest_score_data = {
    "Len": 0,
    "snake_pos": None,
    "food_pos": None,
    "snake_direction": None,
    "episode": None
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

    if len(env.snake_pos) > highest_score_data["Len"]:
        highest_score_data["Len"] = len(env.snake_pos)
        highest_score_data["snake_pos"] = env.last_snake_pos
        highest_score_data["food_pos"] = env.food_pos
        highest_score_data["snake_direction"] = env.snake_direction
        highest_score_data["episode"] = episode

    results.append({
        "episode": episode,
        "Len": len(env.snake_pos),
        "LongestLen": highest_score_data["Len"],
        "epsilon": round(agent.epsilon, 4)
    })

    print(f"Episode {episode}/{num_episodes} Finished, Length: {len(env.snake_pos)}, Longest Length: {highest_score_data['Len']}, Epsilon: {agent.epsilon:.4f}")

torch.save(agent.policy_net.state_dict(), model_path)
print(f"Model saved: {model_path}")

image_path = env.plot_game_state(highest_score_data, highest_score_data["episode"])
print(f"Highest score game state image saved: {image_path}")

with open("train_results.txt", "w") as file:
    for result in results:
        file.write(f"Episode {result['episode']} | "
                   f"Len: {result['Len']} | "
                   f"Longest Len: {result['LongestLen']} | "
                   f"Epsilon: {result['epsilon']}\n")
