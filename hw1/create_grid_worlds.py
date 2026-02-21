"""
gen_test_json.py — Generate N random 101x101 mazes and save as mazes.json. Uses same algorithm as maze_generator.py.

Usage:
    python gen_test_json.py [--num_mazes N] [--seed S] [--output FILE]
"""
import json
import random
import argparse
import random
from constants import ROWS
from tqdm import tqdm
import argparse

# set random seed for reproducibility
random.seed(42)

def create_maze() -> list:
    # TODO: Implement this function to generate and return a random maze as a 2D list of 0s and 1s.
    size = ROWS
    grid = [[-1] * size for _ in range(size)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(r, c):
        return 0 <= r < size and 0 <= c < size

    start_r = random.randint(0, size - 1)
    start_c = random.randint(0, size - 1)
    stack = [(start_r, start_c)]
    grid[start_r][start_c] = 0  # FREE

    while stack:
        r, c = stack[-1]

        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr][nc] == -1:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = random.choice(neighbors)
            if random.random() < 0.30:
                grid[nr][nc] = 1  # BLOCKED
            else:
                grid[nr][nc] = 0  # FREE
                stack.append((nr, nc))
        else:
            stack.pop()

    # Any unvisited cell defaults to FREE
    for r in range(size):
        for c in range(size):
            if grid[r][c] == -1:
                grid[r][c] = 0 # FREE

    return grid

def main():
    parser = argparse.ArgumentParser(description="Generate random mazes as JSON")
    parser.add_argument("--num_mazes", type=int, default=50,
                        help="Number of mazes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mazes.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    random.seed(args.seed)
    
    mazes = []
    for _ in tqdm(range(args.num_mazes), desc="Generating mazes"):  
        mazes.append(create_maze())

    with open(args.output, "w") as fp:
        json.dump(mazes, fp)
    print(f"Generated {args.num_mazes} mazes (seed={args.seed}) -> {args.output}")

if __name__ == "__main__":
    main()
