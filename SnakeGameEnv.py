import random

import matplotlib.pyplot as plt
import numpy as np


class SnakeGame:

    def __init__(self):
        self.screen_width = 10
        self.screen_height = 10
        self.cell_size = 1
        self.ACTION_SPACE = ['up', 'right', 'down', 'left']
        self.snake_pos = []
        self.snake_direction = ''
        self.food_pos = []
        self.last_snake_pos = []

    def reset(self):
        self.snake_pos = [[5 * self.cell_size, 3 * self.cell_size],
                          [4 * self.cell_size, 3 * self.cell_size],
                          [3 * self.cell_size, 3 * self.cell_size]]
        self.snake_direction = 'right'
        self.food_pos = self.spawn_food()
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.screen_width // self.cell_size, self.screen_height // self.cell_size), dtype=np.float32)
        for pos in self.snake_pos[1:]:
            x, y = pos[0] // self.cell_size, pos[1] // self.cell_size
            if 0 <= x < self.screen_width // self.cell_size and 0 <= y < self.screen_height // self.cell_size:
                state[y][x] = 0.5
        head_x, head_y = self.snake_pos[0][0] // self.cell_size, self.snake_pos[0][1] // self.cell_size
        if 0 <= head_x < self.screen_width // self.cell_size and 0 <= head_y < self.screen_height // self.cell_size:
            state[head_y][head_x] = 1.0
        food_x, food_y = self.food_pos[0] // self.cell_size, self.food_pos[1] // self.cell_size
        if 0 <= food_x < self.screen_width // self.cell_size and 0 <= food_y < self.screen_height // self.cell_size:
            state[food_y][food_x] = -1.0
        return state

    def spawn_food(self):
        while True:
            pos = [random.randrange(0, self.screen_width // self.cell_size) * self.cell_size,
                   random.randrange(0, self.screen_height // self.cell_size) * self.cell_size]
            if pos not in self.snake_pos:
                return pos

    def change_direction(self, direction):
        if direction == 0 and self.snake_direction != 'down':
            self.snake_direction = 'up'
        elif direction == 1 and self.snake_direction != 'left':
            self.snake_direction = 'right'
        elif direction == 2 and self.snake_direction != 'up':
            self.snake_direction = 'down'
        elif direction == 3 and self.snake_direction != 'right':
            self.snake_direction = 'left'

    def update_snake_position(self):
        x, y = self.snake_pos[0]
        if self.snake_direction == 'right':
            x += self.cell_size
        elif self.snake_direction == 'left':
            x -= self.cell_size
        elif self.snake_direction == 'up':
            y -= self.cell_size
        elif self.snake_direction == 'down':
            y += self.cell_size
        self.snake_pos.insert(0, [x, y])

    def is_collision(self):
        print(self.snake_direction)
        head = self.snake_pos[0]
        if (head[0] < 0 or head[0] >= self.screen_width or
                head[1] < 0 or head[1] >= self.screen_height or
                head in self.snake_pos[1:]):
            return True
        return False

    def get_valid_actions(self):
        opposite_directions = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        invalid_action = opposite_directions[self.snake_direction]
        valid_actions = [i for i, action in enumerate(self.ACTION_SPACE) if action != invalid_action]
        return valid_actions

    def step(self, action):
        reward = 0

        self.change_direction(action)

        self.last_snake_pos = list(self.snake_pos)

        self.update_snake_position()

        done = False
        if self.snake_pos[0] == self.food_pos:
            self.food_pos = self.spawn_food()
            reward = 10
        else:
            self.snake_pos.pop()

        # 檢查蛇是否碰撞
        if self.is_collision():
            done = True
            reward = -10

        return self.get_state(), reward, done

    def plot_game_state(self, highest_score_data, episode):
        snakePos = highest_score_data["snake_pos"]
        foodPos = highest_score_data["food_pos"]
        snake_direction = highest_score_data["snake_direction"]
        score = highest_score_data["score"]

        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # 畫蛇身（淺綠色）
        if len(snakePos) > 1:
            body_x, body_y = zip(*snakePos[1:])
            body_x = [x + self.cell_size / 2 for x in body_x]
            body_y = [y + self.cell_size / 2 for y in body_y]
            ax.scatter(body_x, body_y, color="lightgreen", label="Snake Body", s=75, marker="o")

            for i in range(len(snakePos) - 1):
                x1, y1 = snakePos[i]
                x2, y2 = snakePos[i + 1]
                x1, y1 = x1 + self.cell_size / 2, y1 + self.cell_size / 2
                x2, y2 = x2 + self.cell_size / 2, y2 + self.cell_size / 2
                ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1, fc='lightgreen', ec='lightgreen')

        # 畫蛇頭（深綠色）
        head_x, head_y = snakePos[0]
        head_x += self.cell_size / 2
        head_y += self.cell_size / 2
        ax.scatter(head_x, head_y, color="darkgreen", label="Snake Head", s=75, marker="o")

        # 畫蛇頭的方向箭頭
        if snake_direction == 'up':
            ax.arrow(head_x, head_y, 0, -self.cell_size / 2, head_width=0.3, head_length=0.3, fc='black', ec='black')
        elif snake_direction == 'right':
            ax.arrow(head_x, head_y, self.cell_size / 2, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
        elif snake_direction == 'down':
            ax.arrow(head_x, head_y, 0, self.cell_size / 2, head_width=0.3, head_length=0.3, fc='black', ec='black')
        elif snake_direction == 'left':
            ax.arrow(head_x, head_y, -self.cell_size / 2, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')

        # 畫食物（紅色）
        food_x, food_y = foodPos
        food_x += self.cell_size / 2
        food_y += self.cell_size / 2
        ax.scatter(food_x, food_y, color="red", label="Food", s=75, marker="x")

        ax.set_title(f"Episode {episode} | Len: {score/10}")
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(0, self.screen_height)
        ax.set_xticks(range(0, self.screen_width + 1, self.cell_size))
        ax.set_yticks(range(0, self.screen_height + 1, self.cell_size))
        ax.grid(True)

        ax.legend()

        # 儲存圖像
        image_path = f"highest_score_game_state_episode_{episode}_len_{score/10}.png"
        plt.savefig(image_path)
        plt.close()
        return image_path

