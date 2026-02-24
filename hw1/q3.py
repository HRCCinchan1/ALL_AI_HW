"""
q3.py — Repeated BACKWARD A* (Backward Replanning) with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G on the current maze
- 2 : run MIN-G on the current maze
- ESC or close window : quit

Maze file loader (optional helper): readFile(fname) reads 0/1 tokens (space-separated), 1=blocked, 0=free.

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path (agent actually walked)
YELLOW = start + current agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import pygame
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
from custom_pq import CustomPQ_maxG, CustomPQ_minG
from q2 import repeated_forward_astar


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

def repeated_backward_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    tie_breaking = "max_g", 
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    
    # # TODO: Implement Backward A* with max_g tie-braking strategy. 

    actual_grid = [[BLACK if actual_maze[r][c] == 1 else WHITE for c in range(ROWS)] for r in range(ROWS)]
    agent_grid = [[GREY] * ROWS for _ in range(ROWS)]

    g = [[float("inf")] * ROWS for _ in range(ROWS)]
    search = [[0] * ROWS for _ in range(ROWS)]
    counter = 0

    on_move = visualize_callbacks.get("on_move") if visualize_callbacks else None

    def sense(pos):
        r, c = pos
        agent_grid[r][c] = actual_grid[r][c]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < ROWS:
                agent_grid[nr][nc] = actual_grid[nr][nc]

    def neighbors(pos):
        r, c = pos
        out = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < ROWS:
                if agent_grid[nr][nc] != BLACK:
                    out.append((nr, nc))
        return out

    def reconstruct(came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        path.reverse()
        return path

    def backward_astar(agent_pos):
        nonlocal counter
        counter += 1

        ar, ac = agent_pos
        gr, gc = goal

        pq = CustomPQ_maxG()
        came_from = {}
        closed = set()

        g[gr][gc] = 0
        search[gr][gc] = counter

        h = lambda r, c: abs(r - ar) + abs(c - ac)
        pq.put(goal, h(gr, gc), 0)

        while not pq.is_empty():
            current, _, g_cur = pq.get()
            if current is None or current in closed:
                continue

            closed.add(current)

            if current == agent_pos:
                path = reconstruct(came_from, current)
                path.reverse()  
                return path, len(closed)

            cr, cc = current
            for nb in neighbors(current):
                nr, nc = nb
                if search[nr][nc] != counter:
                    g[nr][nc] = float("inf")
                    search[nr][nc] = counter
                new_g = g[cr][cc] + 1
                if new_g < g[nr][nc]:
                    g[nr][nc] = new_g
                    came_from[nb] = current
                    f = new_g + h(nr, nc)
                    if pq.contains_node(nb):
                        pq.remove(nb)
                    pq.put(nb, f, new_g)

        return None, len(closed)

    current = start #Start 
    executed = [current]
    total_expanded = 0
    replans = 0

    sense(current)

    while current != goal:
        path, exp = backward_astar(current)
        replans += 1
        total_expanded += exp

        if path is None:
            return False, executed, total_expanded, replans

        for step in path[1:]:
            r, c = step
            sense(current)

            if actual_grid[r][c] == BLACK:
                agent_grid[r][c] = BLACK
                break #replan 

            current = step
            executed.append(current)
            agent_grid[r][c] = PATH #Move 
            if on_move:
                on_move(current)

            sense(current)
            if current == goal: #Reached goal 
                return True, executed, total_expanded, replans

    return True, executed, total_expanded, replans

def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    
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
        
        # Start, Goal, and Current Agent position 
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

    found, executed, expanded, replans = repeated_backward_astar(
        actual_maze=actual_maze,
        start=START_NODE,
        goal=END_NODE,
        visualize_callbacks={"on_frontier": on_frontier, "on_expanded": on_expanded, "on_move": on_move},
    )

    draw_both_panes()
    pygame.display.flip()

    print(f"[{algo}] found={found}  executed_steps={len(executed)-1}  expanded={expanded}  replans={replans}")
    # Win
    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}") #Saved Paths 

def main() -> None:
    parser = argparse.ArgumentParser(description="Q3: Repeated Backward A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q3.json",
                        help="Path to output JSON results file")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q3-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        t0 = time.perf_counter()
        found, executed, expanded, replans = repeated_backward_astar(
            actual_maze=mazes[maze_id],
            start=START_NODE,
            goal=END_NODE,
            tie_breaking="max_g",
        )
        t1 = time.perf_counter()

        entry["bwd"] = {
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
        pygame.display.set_caption("Repeated Backward A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "max_g"
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
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "min_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()