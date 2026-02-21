"""
q5.py — Adaptive A* with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G Adaptive A* on the current maze
- 2 : run MIN-G Adaptive A* on the current maze
- ESC or close window : quit

Maze file format helper:
- readFile(fname) reads 0/1 space-separated tokens, 1=blocked, 0=free, one row per line.

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path
YELLOW = start + agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
import time
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import pygame
from q2 import repeated_forward_astar
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
from custom_pq import CustomPQ_maxG


# ---------------- FILE LOADER ----------------
def readMazes(fname: str) -> List[List[List[int]]]:
    """
    Reads a JSON file containing a list of mazes.
    Each maze is a list of ROWS lists, each with ROWS int values (0=free, 1=blocked).
    Returns a list of maze[r][c] grids.
    """
    with open(fname, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    mazes: List[List[List[int]]] = []
    for idx, grid in enumerate(data):
        if len(grid) != ROWS or any(len(row) != ROWS for row in grid):
            raise ValueError(f"Maze {idx}: expected {ROWS}x{ROWS}, got {len(grid)}x{len(grid[0]) if grid else 0}")
        maze = [[int(v) for v in row] for row in grid]
        maze[START_NODE[0]][START_NODE[1]] = 0
        maze[END_NODE[0]][END_NODE[1]] = 0
        mazes.append(maze)
    return mazes

def adaptive_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    
    # TODO: Implement Adaptive A* with max_g tie-braking strategy.
    # Use heapq for standard priority queue implementation and name your max_g heap class as `CustomPQ_maxG` and use it. 
    
    # ── Setup ────────────────────────────────────────────────────────────────
    actual_grid = [[BLACK if actual_maze[r][c] == 1 else WHITE for c in range(ROWS)] for r in range(ROWS)]
    agent_grid  = [[GREY] * ROWS for _ in range(ROWS)]

    # h[r][c] starts as Manhattan distance to goal, updated after each search
    gr, gc = goal
    h = [[abs(gr - r) + abs(gc - c) for c in range(ROWS)] for r in range(ROWS)]

    # g-values and search counters for lazy reset
    g      = [[float("inf")] * ROWS for _ in range(ROWS)]
    search = [[0] * ROWS for _ in range(ROWS)]
    counter = 0

    # Extract visualization callbacks
    on_frontier = visualize_callbacks.get("on_frontier") if visualize_callbacks else None
    on_expanded = visualize_callbacks.get("on_expanded") if visualize_callbacks else None
    on_move     = visualize_callbacks.get("on_move")     if visualize_callbacks else None

    # ── Sense: reveal neighbors of pos into agent_grid ───────────────────────
    def sense(pos: Tuple[int, int]):
        pr, pc = pos
        agent_grid[pr][pc] = actual_grid[pr][pc]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < ROWS and 0 <= nc < ROWS:
                agent_grid[nr][nc] = actual_grid[nr][nc]

    # ── Neighbors visible to agent (not BLACK in agent_grid) ─────────────────
    def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        out = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < ROWS and agent_grid[nr][nc] != BLACK:
                out.append((nr, nc))
        return out

    # ── Path reconstruction ───────────────────────────────────────────────────
    def reconstruct(camefrom: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        while current in camefrom:
            path.append(current)
            current = camefrom[current]
        path.append(current)
        path.reverse()
        return path

    # ── Single Adaptive A* search ─────────────────────────────────────────────
    # Returns (path, closed_set, g_values_at_goal) so we can update h afterward
    def astar_adaptive(start_pos: Tuple[int, int]) -> Tuple[Optional[List], set, float]:
        nonlocal counter
        counter += 1

        sr, sc = start_pos
        g[sr][sc] = 0
        search[sr][sc] = counter

        goalr, goalc = goal
        g[goalr][goalc] = float("inf")
        search[goalr][goalc] = counter

        camefrom: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: set = set()

        pq = CustomPQ_maxG()
        pq.put(start_pos, h[sr][sc], 0.0)

        while not pq.is_empty():
            current, _, _ = pq.get()
            if current is None:
                break
            if current in closed:
                continue

            closed.add(current)
            if on_expanded:
                on_expanded(current)

            if current == goal:
                return reconstruct(camefrom, current), closed, g[goalr][goalc]

            cr, cc = current
            for nb in neighbors(current):
                nr, nc = nb

                if search[nr][nc] != counter:
                    g[nr][nc] = float("inf")
                    search[nr][nc] = counter

                new_g = g[cr][cc] + 1
                if new_g < g[nr][nc]:
                    g[nr][nc] = new_g
                    camefrom[nb] = current
                    f = new_g + h[nr][nc]
                    if pq.contains_node(nb):
                        pq.remove(nb)
                    pq.put(nb, f, new_g)
                    if on_frontier:
                        on_frontier(nb)

        return None, closed, float("inf")

    # ── Main agent loop ───────────────────────────────────────────────────────
    current  = start
    executed = [current]
    total_expanded = 0
    replans  = 0

    sense(current)

    while current != goal:
        path, closed_set, goal_g = astar_adaptive(current)
        replans += 1

        if path is None:
            return False, executed, total_expanded, replans

        total_expanded += len(closed_set)

        # ── Adaptive A* h-value update ────────────────────────────────────────
        # For every expanded state s: h_new(s) = g(goal) - g(s)
        # This is admissible and >= previous h, so future searches expand fewer nodes
        if goal_g < float("inf"):
            for (er, ec) in closed_set:
                if search[er][ec] == counter and g[er][ec] < float("inf"):
                    h[er][ec] = goal_g - g[er][ec]

        # Execute path step by step
        for step in path[1:]:
            nr, nc = step

            sense(current)

            # Check if next step is actually blocked
            if actual_grid[nr][nc] == BLACK:
                agent_grid[nr][nc] = BLACK
                break  # replan

            # Move
            current = step
            executed.append(current)
            agent_grid[nr][nc] = PATH

            if on_move:
                on_move(current)

            sense(current)

            if current == goal:
                return True, executed, total_expanded, replans

    return True, executed, total_expanded, replans

def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    # This function should display the maze used, the agent's knowledge, and the search process as the agent plans and executes.
    # As a reference, this function takes pygame Surface 'win' to draw on, the actual maze grid, the algorithm name for labeling, 
    # and optional parameters for controlling the visualization speed and saving a screenshot.
    # You are free to use other visualization libraries other than pygame. 
    # You can call repeated_backward_astar with visualize_callbacks that update the Pygame display as the agent plans and executes.
    # In the end it should store the visualization as a PNG file if save_path is provided, or default to "vis_{algo}.png".
    # print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")

    if save_path is None:
        save_path = f"vis_{algo}.png"

    n = NODE_LENGTH
    clock = pygame.time.Clock()

    agent_grid = [[GREY] * ROWS for _ in range(ROWS)]
    agent_pos  = [START_NODE]

    def draw_both_panes():
        for r in range(ROWS):
            for c in range(ROWS):
                true_color = BLACK if actual_maze[r][c] == 1 else WHITE
                pygame.draw.rect(win, true_color, (c * n, r * n, n, n))
                pygame.draw.rect(win, agent_grid[r][c], (GRID_LENGTH + GAP + c * n, r * n, n, n))
        sr, sc = START_NODE
        pygame.draw.rect(win, YELLOW, (sc * n, sr * n, n, n))
        pygame.draw.rect(win, YELLOW, (GRID_LENGTH + GAP + sc * n, sr * n, n, n))
        gr, gc = END_NODE
        pygame.draw.rect(win, BLUE, (gc * n, gr * n, n, n))
        pygame.draw.rect(win, BLUE, (GRID_LENGTH + GAP + gc * n, gr * n, n, n))
        ar, ac = agent_pos[0]
        pygame.draw.rect(win, YELLOW, (GRID_LENGTH + GAP + ac * n, ar * n, n, n))

    def refresh():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        draw_both_panes()
        pygame.display.flip()
        clock.tick(fps)
        if step_delay_ms > 0:
            pygame.time.delay(step_delay_ms)

    def on_frontier(node: Tuple[int, int]):
        r, c = node
        if agent_grid[r][c] not in (PATH, YELLOW, BLUE):
            agent_grid[r][c] = GREY
        refresh()

    def on_expanded(node: Tuple[int, int]):
        r, c = node
        if agent_grid[r][c] not in (PATH, YELLOW, BLUE):
            agent_grid[r][c] = GREY
        refresh()

    def on_move(node: Tuple[int, int]):
        agent_pos[0] = node
        r, c = node
        agent_grid[r][c] = PATH
        refresh()

    # Run the appropriate algorithm based on algo label
    if algo == "adaptive":
        found, executed, expanded, replans = adaptive_astar(
            actual_maze=actual_maze,
            start=START_NODE,
            goal=END_NODE,
            visualize_callbacks={"on_frontier": on_frontier, "on_expanded": on_expanded, "on_move": on_move},
        )
    else:
        found, executed, expanded, replans = repeated_forward_astar(
            actual_maze=actual_maze,
            start=START_NODE,
            goal=END_NODE,
            tie_breaking="max_g",
            visualize_callbacks={"on_frontier": on_frontier, "on_expanded": on_expanded, "on_move": on_move},
        )

    draw_both_panes()
    pygame.display.flip()

    print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")

    # If 'win' is the display surface (it is), this works:
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Q5: Adaptive A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q5.json",
                        help="Path to output JSON results file")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q5-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        t0 = time.perf_counter()
        found, executed, expanded, replans = adaptive_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
        )
        t1 = time.perf_counter()

        entry["adaptive"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }

        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_forward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
            tie_breaking="max_g",
        )
        t1 = time.perf_counter()

        entry["fwd"] = {
            "found": found,
            "path_length": len(executed) - 1 if found else -1,
            "expanded": expanded,
            "replans": replans,
            "runtime_ms": (t1 - t0) * 1000,
        }

        results.append(entry)

    if args.show_vis:
        # In case, PyGame is used for visualization, this code initializes a window and runs the visualization for the selected maze and algorithm.
        # Feel free to modify this code if you use a different visualization library or approach.
        pygame.init()
        win = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Adaptive A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "adaptive"
        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_algo = "adaptive"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "adaptive"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "fwd"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()