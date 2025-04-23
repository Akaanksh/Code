import pygame
import random
import numpy as np
from game_logic import move, LEFT, RIGHT, UP, DOWN

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = 400
GRID_SIZE = 4
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
BACKGROUND_COLOR = (187, 173, 160)
CELL_COLOR = (204, 192, 179)
FONT_COLOR = (119, 110, 101)
FONT = pygame.font.SysFont("Arial", 40)

# Directions for the moves
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Colors for the tiles
COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}

# Function to initialize the grid
def init_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    add_new_tile(grid)
    add_new_tile(grid)
    return grid

# Function to add a new tile
def add_new_tile(grid):
    empty_cells = list(zip(*np.where(grid == 0)))
    if empty_cells:
        x, y = random.choice(empty_cells)
        grid[x, y] = random.choice([2, 4])

# Function to draw the grid
def draw_grid(screen, grid):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = j * CELL_SIZE
            y = i * CELL_SIZE
            pygame.draw.rect(screen, COLORS.get(grid[i, j], CELL_COLOR), (x, y, CELL_SIZE, CELL_SIZE))
            if grid[i, j] != 0:
                text = FONT.render(str(grid[i, j]), True, FONT_COLOR)
                text_rect = text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                screen.blit(text, text_rect)

# Function to check if the game is over
def check_game_over(grid):
    if np.any(grid == 0):
        return False
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if i + 1 < GRID_SIZE and grid[i, j] == grid[i + 1, j]:
                return False
            if j + 1 < GRID_SIZE and grid[i, j] == grid[i, j + 1]:
                return False
    return True

# Main game loop
def game_loop():
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("2048")

    grid = init_grid()
    running = True
    while running:
        screen.fill(BACKGROUND_COLOR)
        draw_grid(screen, grid)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    grid = move(grid, LEFT)
                elif event.key == pygame.K_RIGHT:
                    grid = move(grid, RIGHT)
                elif event.key == pygame.K_UP:
                    grid = move(grid, UP)
                elif event.key == pygame.K_DOWN:
                    grid = move(grid, DOWN)

                add_new_tile(grid)
                if check_game_over(grid):
                    print("Game Over!")
                    running = False

        pygame.time.wait(100)

    pygame.quit()

# Run the game
game_loop()