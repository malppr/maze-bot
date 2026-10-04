"""Map generator. Every wall is a capsule; maps come in categories (see PLAN.md §3).

`sample_map(rng, category)` returns a `Map` with walls, a start pose and a goal such that the goal is
reachable for the robot (on the inflated grid) and at least `min_geo` away along the path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import distance_field as dfield
from . import geometry

# New categories go at the END: evaluation seeds are keyed by the category index.
CATEGORIES = ("open", "obstacles", "scribbles", "mazes", "mixed", "traps", "presets", "mazes_easy")
TRAIN_CATEGORIES = ("open", "obstacles", "scribbles", "mazes", "mixed")
PRESETS_PATH = Path(__file__).resolve().parent.parent / "presets" / "presets.json"

ROBOT_RADIUS = 0.3  # must match SimParams.radius
BORDER_RAD = 0.05


@dataclass
class Map:
    width: float
    height: float
    walls: np.ndarray  # interior capsules (N, 5)
    start: tuple[float, float, float]  # x, y, heading
    goal: tuple[float, float]
    category: str = ""
    name: str = ""
    grid: dfield.Grid | None = field(default=None, repr=False)
    field: np.ndarray | None = field(default=None, repr=False)  # geodesic field from goal

    @property
    def caps(self) -> np.ndarray:
        """All capsules including the arena border."""
        return np.vstack([geometry.border(self.width, self.height, BORDER_RAD), self.walls])

    def geo(self, x: float, y: float) -> float:
        return dfield.lookup(self.grid, self.field, x, y)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "width": self.width,
            "height": self.height,
            "walls": self.walls.round(4).tolist(),
            "start": list(self.start),
            "goal": list(self.goal),
        }


# ---------------------------------------------------------------- arena + wall generators


def arena_size(rng: np.random.Generator, lo: float = 6.0, hi: float = 12.0, max_aspect: float = 1.8):
    while True:
        w, h = rng.uniform(lo, hi, size=2)
        if max(w / h, h / w) <= max_aspect:
            return float(w), float(h)


def _rand_point(rng, w, h, margin):
    return rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)


def gen_obstacles(rng, w, h, density: float = 1.0):
    """Posts (circles) and bars (straight capsules) at random positions and orientations."""
    n = int(round(rng.uniform(0.05, 0.11) * w * h * density))
    rows = []
    for _ in range(max(n, 1)):
        x, y = _rand_point(rng, w, h, 0.5)
        if rng.random() < 0.4:
            rows.append([x, y, x, y, rng.uniform(0.15, 0.55)])
        else:
            length = rng.uniform(0.8, 3.0)
            a = rng.uniform(0, math.pi)
            dx, dy = 0.5 * length * math.cos(a), 0.5 * length * math.sin(a)
            rows.append([x - dx, y - dy, x + dx, y + dy, rng.uniform(0.06, 0.25)])
    return geometry.capsules(rows)


def _wall_stroke(rng, w, h, step):
    """A long, gently curving stroke starting on the border and heading inwards — how people draw walls."""
    side = int(rng.integers(4))
    t = rng.uniform(0.15, 0.85)
    x, y, a = [(t * w, 0.0, math.pi / 2), (w, t * h, math.pi), (t * w, h, -math.pi / 2), (0.0, t * h, 0.0)][
        side
    ]
    a += rng.uniform(-0.5, 0.5)
    length = rng.uniform(0.35, 0.8) * (h if side in (0, 2) else w)
    turn = 0.0
    bend = rng.uniform(0.0, 0.08)
    pts = [(x, y)]
    for _ in range(int(length / step)):
        turn = 0.9 * turn + rng.normal(0, bend)
        a += turn
        x += step * math.cos(a)
        y += step * math.sin(a)
        if not (0.05 < x < w - 0.05 and 0.05 < y < h - 0.05):
            break
        pts.append((x, y))
    return np.array(pts)


def _scribble_points(rng, w, h):
    """A wall-like stroke from the border, a smoothed random walk, or a wobbly closed loop."""
    step = 0.15
    kind = rng.random()
    if kind < 0.4:
        return _wall_stroke(rng, w, h, step)
    if kind < 0.6:  # closed loop
        cx, cy = _rand_point(rng, w, h, 1.2)
        r0 = rng.uniform(0.4, 1.3)
        n = int(2 * math.pi * r0 / step) + 1
        phases = rng.uniform(0, 2 * math.pi, 3)
        amps = rng.uniform(0, 0.25, 3) * r0
        ts = np.linspace(0, 2 * math.pi, n)
        rr = r0 + sum(
            a * np.sin((k + 2) * ts + ph) for k, (a, ph) in enumerate(zip(amps, phases, strict=True))
        )
        pts = np.column_stack([cx + rr * np.cos(ts), cy + rr * np.sin(ts)])
    else:
        length = rng.uniform(2.0, 8.0)
        n = int(length / step)
        x, y = _rand_point(rng, w, h, 0.3)
        a = rng.uniform(0, 2 * math.pi)
        turn = 0.0
        wiggle = rng.uniform(0.02, 0.2)
        pts = [(x, y)]
        for _ in range(n):
            turn = 0.85 * turn + rng.normal(0, wiggle)
            a += turn
            x += step * math.cos(a)
            y += step * math.sin(a)
            if not (0.1 < x < w - 0.1 and 0.1 < y < h - 0.1):
                break
            pts.append((x, y))
        pts = np.array(pts)
    return pts


def gen_scribbles(rng, w, h, count: int | None = None):
    """Freehand-looking strokes: random walks and loops, RDP-simplified, varying thickness."""
    count = count if count is not None else int(rng.integers(3, 5 + int(w * h / 20)))
    parts = []
    for _ in range(count):
        pts = _scribble_points(rng, w, h)
        if len(pts) < 2:
            continue
        pts = geometry.rdp(pts, 0.03)
        parts.append(geometry.polyline_capsules(pts, rng.uniform(0.04, 0.2)))
    return np.vstack(parts) if parts else geometry.EMPTY.copy()


def maze_walls(rng, nx: int, ny: int, cell: float, ox: float = 0.0, oy: float = 0.0, braid: float = 0.1):
    """Recursive-backtracker maze on an nx x ny cell grid; returns interior wall segments (M, 4).

    `braid` = probability of knocking out an extra interior wall (creates loops).
    Outer walls are omitted (the arena border, or the caller, closes the maze).
    """
    # walls: v[j, i] = wall between cell (i, j) and (i+1, j); hz[j, i] = wall between (i, j) and (i, j+1)
    v = np.ones((ny, nx - 1), dtype=bool)
    hz = np.ones((ny - 1, nx), dtype=bool)
    seen = np.zeros((ny, nx), dtype=bool)
    stack = [(int(rng.integers(nx)), int(rng.integers(ny)))]
    seen[stack[0][1], stack[0][0]] = True
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbrs = [(a, b) for a, b in nbrs if 0 <= a < nx and 0 <= b < ny and not seen[b, a]]
        if not nbrs:
            stack.pop()
            continue
        a, b = nbrs[int(rng.integers(len(nbrs)))]
        if a != i:
            v[j, min(a, i)] = False
        else:
            hz[min(b, j), i] = False
        seen[b, a] = True
        stack.append((a, b))
    v &= rng.random(v.shape) >= braid
    hz &= rng.random(hz.shape) >= braid
    segs = []
    for j, i in zip(*np.nonzero(v), strict=True):
        x = ox + (i + 1) * cell
        segs.append([x, oy + j * cell, x, oy + (j + 1) * cell])
    for j, i in zip(*np.nonzero(hz), strict=True):
        y = oy + (j + 1) * cell
        segs.append([ox + i * cell, y, ox + (i + 1) * cell, y])
    return np.array(segs).reshape(-1, 4)


def wobble(rng, segs: np.ndarray, rad_lo: float, rad_hi: float, amp: float, pieces: int = 3):
    """Turn straight segments into slightly wobbly capsule chains so they look hand-drawn."""
    parts = []
    for ax, ay, bx, by in segs:
        ts = np.linspace(0, 1, pieces + 1)
        dx, dy = bx - ax, by - ay
        ln = math.sqrt(dx * dx + dy * dy) or 1.0
        nx_, ny_ = -dy / ln, dx / ln
        off = rng.normal(0, amp, pieces + 1)
        off[[0, -1]] *= 0.3
        pts = np.column_stack([ax + ts * dx + off * nx_, ay + ts * dy + off * ny_])
        parts.append(geometry.polyline_capsules(pts, rng.uniform(rad_lo, rad_hi)))
    return np.vstack(parts) if parts else geometry.EMPTY.copy()


def gen_maze(rng, w, h, braid: float | None = None, cell_range=(1.1, 1.8), braid_range=(0.0, 0.2)):
    cell = rng.uniform(*cell_range)
    nx, ny = max(2, int(w // cell)), max(2, int(h // cell))
    # fit the grid to the arena exactly so outer cells close against the border
    cw, ch = w / nx, h / ny
    segs = maze_walls(rng, nx, ny, 1.0, braid=braid if braid is not None else rng.uniform(*braid_range))
    segs = segs * np.array([cw, ch, cw, ch])
    return wobble(rng, segs, 0.05, 0.12, amp=rng.uniform(0.0, 0.06))


def gen_mixed(rng, w, h):
    parts = [gen_obstacles(rng, w, h, density=rng.uniform(0.3, 0.7))]
    parts.append(gen_scribbles(rng, w, h, count=int(rng.integers(1, 4))))
    if rng.random() < 0.5:  # a sparse maze fragment
        segs = gen_maze(rng, w, h, braid=0.0)
        keep = rng.random(len(segs)) < rng.uniform(0.2, 0.5)
        parts.append(segs[keep])
    return np.vstack(parts)


def gen_trap(rng, w, h):
    """A U-shaped pocket whose opening faces the start, with the goal behind it.

    Returns (walls, start_xy, goal_xy): a memoryless goal-seeker drives straight into the pocket.
    """
    cx, cy = w / 2 + rng.uniform(-0.5, 0.5), h / 2 + rng.uniform(-0.5, 0.5)
    a = rng.uniform(0, 2 * math.pi)  # direction from start to goal
    ux, uy = math.cos(a), math.sin(a)
    px, py = -uy, ux
    width = rng.uniform(1.6, 3.2)
    depth = rng.uniform(1.0, 2.5)
    rad = rng.uniform(0.06, 0.15)
    hw = width / 2
    u = np.array([ux, uy])
    p = np.array([px, py])
    c = np.array([cx, cy])
    # (cx, cy) is the centre of the back of the U; the arms extend `depth` towards the start (-u).
    if rng.random() < 0.5:  # rounded back: semicircle bulging towards the goal
        ts = np.linspace(-math.pi / 2, math.pi / 2, 9)
        back = c + hw * (np.cos(ts)[:, None] * u + np.sin(ts)[:, None] * p)
        bulge = hw
    else:  # square back
        back = np.array([c - hw * p, c + hw * p])
        bulge = 0.0
    pts = np.vstack([back[:1] - depth * u, back, back[-1:] - depth * u])
    walls = geometry.polyline_capsules(pts, rad)
    start = c - (depth + rng.uniform(1.0, 2.0)) * u
    goal = c + (bulge + rng.uniform(0.7, 1.5)) * u
    return walls, (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))


# ---------------------------------------------------------------- presets


def load_presets(path: Path = PRESETS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["presets"]


def preset_map(preset: dict, portrait: bool = False) -> Map:
    """A preset layout (defined landscape). Portrait = transpose x/y (mirror across the diagonal)."""
    w, h = preset["width"], preset["height"]
    walls = geometry.capsules(preset["walls"])
    (sx, sy, sth), (gx, gy) = preset["start"], preset["goal"]
    if portrait:
        w, h = h, w
        walls = walls[:, [1, 0, 3, 2, 4]]
        sx, sy, gx, gy = sy, sx, gy, gx
        sth = math.pi / 2 - sth
    return Map(w, h, walls, (sx, sy, sth), (gx, gy), category="presets", name=preset["name"])


# ---------------------------------------------------------------- sampling


def attach_field(m: Map) -> Map:
    m.grid = dfield.build_grid(m.width, m.height, m.caps, ROBOT_RADIUS)
    m.field = dfield.geodesic_field(m.grid, *m.goal)
    return m


def _place_ab(rng, m_w, m_h, walls, grid, min_geo, start_xy=None, goal_xy=None, tries=20):
    """Pick goal, then a start reachable from it with geodesic distance >= min_geo. None on failure."""
    caps_all = np.vstack([geometry.border(m_w, m_h, BORDER_RAD), walls])
    margin = ROBOT_RADIUS + 0.1
    ok = grid.free & (grid.clear >= margin)
    cells = np.argwhere(ok)
    if len(cells) < 2:
        return None
    for _ in range(tries):
        if goal_xy is None:
            j, i = cells[int(rng.integers(len(cells)))]
            gx, gy = grid.centre(i, j)
        else:
            gx, gy = goal_xy
        dist = dfield.geodesic_field(grid, gx, gy)
        if start_xy is None:
            cand = np.argwhere(ok & (dist >= min_geo) & np.isfinite(dist))
            if len(cand) == 0:
                continue
            j, i = cand[int(rng.integers(len(cand)))]
            sx, sy = grid.centre(i, j)
            # jitter inside the cell, keep clearance
            jx, jy = sx + rng.uniform(-0.4, 0.4) * grid.cell, sy + rng.uniform(-0.4, 0.4) * grid.cell
            if geometry.clearance(jx, jy, caps_all) >= margin:
                sx, sy = jx, jy
        else:
            sx, sy = start_xy
            if dfield.lookup(grid, dist, sx, sy) == math.inf:
                return None
        return (sx, sy), (gx, gy), dist
    return None


def sample_map(rng: np.random.Generator, category: str, min_geo: float = 3.0) -> Map:
    """Sample a solvable map of the given category (resamples walls until A/B placement succeeds)."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    for _ in range(100):
        start_xy = goal_xy = None
        name = ""
        if category == "presets":
            presets = load_presets()
            pr = presets[int(rng.integers(len(presets)))]
            m0 = preset_map(pr, portrait=bool(rng.random() < 0.5))
            w, h, walls, name = m0.width, m0.height, m0.walls, m0.name
        else:
            w, h = arena_size(rng)
            if category == "open":
                walls = geometry.EMPTY.copy()
            elif category == "obstacles":
                walls = gen_obstacles(rng, w, h)
            elif category == "scribbles":
                walls = gen_scribbles(rng, w, h)
            elif category == "mazes":
                walls = gen_maze(rng, w, h)
            elif category == "mazes_easy":  # curriculum only: wide corridors, more loops
                walls = gen_maze(rng, w, h, cell_range=(1.8, 2.6), braid_range=(0.2, 0.4))
            elif category == "mixed":
                walls = gen_mixed(rng, w, h)
            else:  # traps
                walls, start_xy, goal_xy = gen_trap(rng, w, h)
                walls = np.vstack([walls, gen_obstacles(rng, w, h, density=0.2)])
        caps_all = np.vstack([geometry.border(w, h, BORDER_RAD), walls])
        grid = dfield.build_grid(w, h, caps_all, ROBOT_RADIUS)
        if start_xy is not None:  # trap: start/goal must be inside the arena and clear
            margin = ROBOT_RADIUS + 0.1
            if not all(
                margin < p[0] < w - margin
                and margin < p[1] < h - margin
                and geometry.clearance(*p, caps_all) >= margin
                for p in (start_xy, goal_xy)
            ):
                continue
        placed = _place_ab(rng, w, h, walls, grid, min_geo, start_xy, goal_xy)
        if placed is None:
            continue
        (sx, sy), (gx, gy), dist = placed
        m = Map(w, h, walls, (sx, sy, float(rng.uniform(-math.pi, math.pi))), (gx, gy), category, name)
        m.grid, m.field = grid, dist
        return m
    raise RuntimeError(f"could not sample a solvable {category} map")
