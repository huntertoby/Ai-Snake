import random
import numpy as np



class SnakeGame:

    def __init__(self):
        self.screen_width = 400
        self.screen_height = 400
        self.cell_size = 40
        self.ACTION_SPACE = ['up', 'right', 'down', 'left']

        snake_Speed = 50
        self.max_steps_without_food = 50

    def reset(self):
        self.snake_pos = [[5 * self.cell_size, 3 * self.cell_size],
                          [4 * self.cell_size, 3 * self.cell_size],
                          [3 * self.cell_size, 3 * self.cell_size]]
        self.snake_direction = 'right'
        self.food_pos = self.spawn_food()
        self.steps_since_last_food = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.screen_width // self.cell_size, self.screen_height // self.cell_size), dtype=np.float32)
        for pos in self.snake_pos[1:]:
            x, y = pos[0] // self.cell_size, pos[1] // self.cell_size
            if 0 <= x < self.screen_width // self.cell_size and 0 <= y < self.screen_height // self.cell_size:
                state[y][x] = 0.5  # 身体标记为 0.5
        head_x, head_y = self.snake_pos[0][0] // self.cell_size, self.snake_pos[0][1] // self.cell_size
        if 0 <= head_x < self.screen_width // self.cell_size and 0 <= head_y < self.screen_height // self.cell_size:
            state[head_y][head_x] = 1.0  # 头部标记为 1.0
        food_x, food_y = self.food_pos[0] // self.cell_size, self.food_pos[1] // self.cell_size
        if 0 <= food_x < self.screen_width // self.cell_size and 0 <= food_y < self.screen_height // self.cell_size:
            state[food_y][food_x] = -1.0  # 食物标记为 -1.0
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
            y -= self.cell_size  # 修正为减少 y 值
        elif self.snake_direction == 'down':
            y += self.cell_size  # 修正为增加 y 值
        self.snake_pos.insert(0, [x, y])

    def is_collision(self):
        head = self.snake_pos[0]
        if (head[0] < 0 or head[0] >= self.screen_width or
                head[1] < 0 or head[1] >= self.screen_height or
                head in self.snake_pos[1:]):
            return True
        return False

    # 在 SnakeGameEnv 类中，添加一个函数来获取当前的允许动作
    def get_valid_actions(self):
        opposite_directions = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        invalid_action = opposite_directions[self.snake_direction]
        valid_actions = [i for i, action in enumerate(self.ACTION_SPACE) if action != invalid_action]
        return valid_actions

    def get_distance(self, pos1, pos2):
        # 计算两个位置之间的欧几里得距离
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx ** 2 + dy ** 2) ** 0.5
        
    def step(self, action):

        # 在移动蛇之前计算蛇头和食物之间的距离
        prev_distance = self.get_distance(self.snake_pos[0], self.food_pos)

        reward = 0
        # 执行动作
        self.change_direction(action)
        self.update_snake_position()
        
        done = False

        if self.snake_pos[0] == self.food_pos:
            self.food_pos = self.spawn_food()
            self.steps_since_last_food = 0  # 重置计数器
            reward = 10
        else:
            self.snake_pos.pop()

        # 检查游戏是否结束
        if self.is_collision() or self.is_dead_end() :
            done = True
            reward = -10  # 碰撞的惩罚

        return self.get_state(), reward, done
