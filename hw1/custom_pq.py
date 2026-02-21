"""
custom_pq.py — Custom Priority Queue implementations for A* tie-breaking.

CustomPQ_maxG : breaks ties in favor of LARGER g-values (more focused search)
CustomPQ_minG : breaks ties in favor of SMALLER g-values
"""

import heapq
from typing import Tuple


class CustomPQ_maxG:
    """
    Min-heap priority queue that breaks f-value ties in favor of LARGER g-values.
    Heap key: (f, -g, counter)
    """

    def __init__(self):
        self._heap = []
        self._counter = 0
        self._node_set = set()

    def put(self, node: Tuple[int, int], f: float, g: float):
        self._counter += 1
        heapq.heappush(self._heap, (f, -g, self._counter, node))
        self._node_set.add(node)

    def get(self) -> Tuple[Tuple[int, int], float, float]:
        """Returns (node, f, g)"""
        while self._heap:
            f, neg_g, _, node = heapq.heappop(self._heap)
            if node in self._node_set:
                self._node_set.discard(node)
                return node, f, -neg_g
        return None, None, None

    def contains_node(self, node: Tuple[int, int]) -> bool:
        return node in self._node_set

    def remove(self, node: Tuple[int, int]):
        """Lazy removal — mark as removed from set, will be skipped on get()."""
        self._node_set.discard(node)

    def is_empty(self) -> bool:
        return len(self._node_set) == 0


class CustomPQ_minG:
    """
    Min-heap priority queue that breaks f-value ties in favor of SMALLER g-values.
    Heap key: (f, +g, counter)
    """

    def __init__(self):
        self._heap = []
        self._counter = 0
        self._node_set = set()

    def put(self, node: Tuple[int, int], f: float, g: float):
        self._counter += 1
        heapq.heappush(self._heap, (f, g, self._counter, node))
        self._node_set.add(node)

    def get(self) -> Tuple[Tuple[int, int], float, float]:
        """Returns (node, f, g)"""
        while self._heap:
            f, g, _, node = heapq.heappop(self._heap)
            if node in self._node_set:
                self._node_set.discard(node)
                return node, f, g
        return None, None, None

    def contains_node(self, node: Tuple[int, int]) -> bool:
        return node in self._node_set

    def remove(self, node: Tuple[int, int]):
        """Lazy removal — mark as removed from set, will be skipped on get()."""
        self._node_set.discard(node)

    def is_empty(self) -> bool:
        return len(self._node_set) == 0