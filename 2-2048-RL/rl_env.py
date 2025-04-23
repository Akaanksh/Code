import numpy as np
import random

from game_logic import move  # updated move returns (grid, reward)

class Game2048Env:
    def __init__(self):
        self.grid = None
        self.score = 0
        self.reset()

    def reset(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.add_tile()
        self.add_tile()
        self.score = 0
        return self.get_state()

    def get_state(self):
        # Normalized log2 state for better neural network learning
        return np.log2(self.grid + 1).flatten() / 16

    def add_tile(self):
        empty = list(zip(*np.where(self.grid == 0)))
        if empty:
            x, y = random.choice(empty)
            self.grid[x, y] = 2 if random.random() < 0.9 else 4

    def step(self, action):  # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
        old_grid = self.grid.copy()
        new_grid, reward = move(self.grid.copy(), action)  # updated to unpack reward

        # Only update if the move changes the grid
        if not np.array_equal(old_grid, new_grid):
            self.grid = new_grid
            self.add_tile()
        else:
            reward = -1  # Optional: penalize no-op moves

        self.score += reward
        done = self.check_game_over()
        return self.get_state(), reward, done, {}

    def check_game_over(self):
        from game_logic import move
        for action in range(4):
            test_grid, _ = move(self.grid.copy(), action)  # move now returns (grid, reward)
            if not np.array_equal(self.grid, test_grid):
                return False
        return True

    def render(self):
        print(self.grid)
