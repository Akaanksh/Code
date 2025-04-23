import numpy as np

LEFT, RIGHT, UP, DOWN = 0, 1, 2, 3

def move_left(grid):
    score = 0
    for row in grid:
        tiles = row[row != 0]
        new_row = []
        skip = False
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged_value = tiles[i] * 2
                new_row.append(merged_value)
                score += merged_value  # accumulate reward
                i += 2  # skip the next tile
            else:
                new_row.append(tiles[i])
                i += 1
        new_row += [0] * (4 - len(new_row))
        row[:] = new_row
    return score

def move(grid, direction):
    original_grid = grid.copy()
    if direction == LEFT:
        reward = move_left(grid)
    elif direction == RIGHT:
        grid[:] = np.fliplr(grid)
        reward = move_left(grid)
        grid[:] = np.fliplr(grid)
    elif direction == UP:
        grid[:] = np.rot90(grid)
        reward = move_left(grid)
        grid[:] = np.rot90(grid, k=-1)
    elif direction == DOWN:
        grid[:] = np.rot90(grid, k=-1)
        reward = move_left(grid)
        grid[:] = np.rot90(grid)

    # Penalize no-op moves (grid unchanged)
    if np.array_equal(original_grid, grid):
        reward = -1

    return grid, reward