"""Occupancy grid inflated by the robot radius, and geodesic distance fields on it.

A cell is free when the robot centre placed at the cell centre does not touch any wall (clearance >= r).
The geodesic field from B (8-connected Dijkstra, no corner cutting) drives the progress reward and
answers "is B reachable from A?" (the demo's "no path" warning uses the same grid, ported to TS).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from . import geometry

SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
_DIAGONALS = [(-1, -1), (1, -1), (-1, 1), (1, 1)]


@dataclass
class Grid:
    width: float
    height: float
    cell: float
    free: np.ndarray  # (ny, nx) bool
    clear: np.ndarray  # (ny, nx) clearance at cell centres

    @property
    def shape(self):
        return self.free.shape

    def centre(self, i: int, j: int):
        return (i + 0.5) * self.cell, (j + 0.5) * self.cell

    def index(self, x: float, y: float):
        ny, nx = self.free.shape
        i = min(max(int(x / self.cell), 0), nx - 1)
        j = min(max(int(y / self.cell), 0), ny - 1)
        return i, j


def build_grid(width: float, height: float, caps: np.ndarray, robot_radius: float, cell: float | None = None):
    cell = cell if cell is not None else robot_radius / 2.0
    nx = max(1, math.ceil(width / cell))
    ny = max(1, math.ceil(height / cell))
    xs = (np.arange(nx) + 0.5) * cell
    ys = (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    clear = geometry.clearance_grid(gx.ravel(), gy.ravel(), caps).reshape(ny, nx)
    inside = (gx < width) & (gy < height)
    free = (clear >= robot_radius) & inside
    return Grid(width, height, cell, free, clear)


def geodesic_field(grid: Grid, gx: float, gy: float, seed_radius: float | None = None) -> np.ndarray:
    """Geodesic distance (world units) from point (gx, gy) to every free cell; inf where unreachable.

    Seeds every free cell within `seed_radius` (default: 1.5 cells) of the point with its straight-line
    distance, so the field is ~0 at the goal even if the goal's own cell is blocked.
    """
    ny, nx = grid.free.shape
    c = grid.cell
    dist = np.full((ny, nx), np.inf)
    seed_radius = seed_radius if seed_radius is not None else 1.5 * c
    heap = []
    i0, j0 = grid.index(gx, gy)
    k = math.ceil(seed_radius / c) + 1
    for j in range(max(0, j0 - k), min(ny, j0 + k + 1)):
        for i in range(max(0, i0 - k), min(nx, i0 + k + 1)):
            if not grid.free[j, i]:
                continue
            cx, cy = (i + 0.5) * c, (j + 0.5) * c
            d = math.sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
            if d <= seed_radius and d < dist[j, i]:
                dist[j, i] = d
                heapq.heappush(heap, (d, i, j))

    free = grid.free
    diag = SQRT2 * c
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[j, i]:
            continue
        for di, dj, w in _NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[b, a]:
                nd = d + w * c
                if nd < dist[b, a]:
                    dist[b, a] = nd
                    heapq.heappush(heap, (nd, a, b))
        for di, dj in _DIAGONALS:
            a, b = i + di, j + dj
            # no corner cutting: both orthogonal neighbours must be free
            if 0 <= a < nx and 0 <= b < ny and free[b, a] and free[j, a] and free[b, i]:
                nd = d + diag
                if nd < dist[b, a]:
                    dist[b, a] = nd
                    heapq.heappush(heap, (nd, a, b))
    return dist


def lookup(grid: Grid, dist: np.ndarray, x: float, y: float, k: int = 2) -> float:
    """Continuous geodesic distance at (x, y): min over nearby reached cells of field + straight line.

    Works when the robot centre sits in a cell marked blocked (touching a wall). inf if nothing nearby
    is reachable.
    """
    ny, nx = dist.shape
    i0, j0 = grid.index(x, y)
    c = grid.cell
    best = math.inf
    for j in range(max(0, j0 - k), min(ny, j0 + k + 1)):
        for i in range(max(0, i0 - k), min(nx, i0 + k + 1)):
            d = dist[j, i]
            if d < math.inf:
                ex = (i + 0.5) * c - x
                ey = (j + 0.5) * c - y
                v = d + math.sqrt(ex * ex + ey * ey)
                if v < best:
                    best = v
    return best


def reachable(grid: Grid, ax: float, ay: float, bx: float, by: float) -> bool:
    return lookup(grid, geodesic_field(grid, bx, by), ax, ay) < math.inf
