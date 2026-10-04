"""Capsule geometry: point-segment distance, ray casting, circle push-out.

A wall is a capsule: segment (ax, ay)-(bx, by) swept by radius `rad`. Capsules are stored as an
(N, 5) float64 array of rows [ax, ay, bx, by, rad]. A zero-length segment is a circle (post).

Everything here is ported 1:1 to `web/src/sim/geometry.ts`. To keep the two bit-compatible we only use
+ - * / and sqrt (correctly rounded in IEEE 754 in both languages), never hypot/atan2.
"""

from __future__ import annotations

import numpy as np

EMPTY = np.zeros((0, 5))


def capsules(rows) -> np.ndarray:
    """Coerce a list of [ax, ay, bx, by, rad] rows into an (N, 5) float64 array."""
    arr = np.asarray(rows, dtype=np.float64)
    return arr.reshape(-1, 5)


def border(width: float, height: float, rad: float = 0.05) -> np.ndarray:
    """Four capsules on the arena edges (their inner surface is `rad` inside the arena)."""
    w, h = width, height
    return capsules([[0, 0, w, 0, rad], [w, 0, w, h, rad], [w, h, 0, h, rad], [0, h, 0, 0, rad]])


def closest_points(px: float, py: float, caps: np.ndarray):
    """Closest point on each capsule's core segment to (px, py), and the distance to it.

    Returns (cx, cy, dist) arrays of shape (N,).
    """
    ax, ay, bx, by = caps[:, 0], caps[:, 1], caps[:, 2], caps[:, 3]
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0.0, l2, 1.0)
    t = np.where(l2 > 0.0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    ex = px - cx
    ey = py - cy
    return cx, cy, np.sqrt(ex * ex + ey * ey)


def clearance(px: float, py: float, caps: np.ndarray) -> float:
    """Distance from a point to the nearest capsule surface (negative inside a wall)."""
    if len(caps) == 0:
        return np.inf
    _, _, d = closest_points(px, py, caps)
    return float(np.min(d - caps[:, 4]))


def clearance_grid(xs: np.ndarray, ys: np.ndarray, caps: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Clearance at many points (flattened arrays xs, ys). Vectorized over points x capsules."""
    out = np.full(xs.shape, np.inf)
    if len(caps) == 0:
        return out
    ax, ay, bx, by, rad = (caps[:, i][None, :] for i in range(5))
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0.0, l2, 1.0)
    for s in range(0, len(xs), chunk):
        px = xs[s : s + chunk, None]
        py = ys[s : s + chunk, None]
        t = np.where(l2 > 0.0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
        t = np.clip(t, 0.0, 1.0)
        ex = px - (ax + t * dx)
        ey = py - (ay + t * dy)
        out[s : s + chunk] = np.min(np.sqrt(ex * ex + ey * ey) - rad, axis=1)
    return out


def _ray_circle(ox, oy, dx, dy, cx, cy, r):
    """First positive hit distance of rays vs circles (broadcast), inf on miss."""
    qx = ox - cx
    qy = oy - cy
    b = dx * qx + dy * qy
    c = qx * qx + qy * qy - r * r
    h = b * b - c
    t = -b - np.sqrt(np.maximum(h, 0.0))
    return np.where((h >= 0.0) & (t > 0.0), t, np.inf)


def cast_rays(ox: float, oy: float, dirs: np.ndarray, caps: np.ndarray, max_range: float) -> np.ndarray:
    """Distance along each unit direction (K, 2) to the first capsule hit, capped at max_range.

    A capsule is the union of a "body" (the segment swept sideways by rad) and two end circles, so the
    first hit is the minimum of the first hits on those three parts. Rays start outside all walls.
    """
    if len(caps) == 0:
        return np.full(len(dirs), max_range)
    dx = dirs[:, 0][:, None]
    dy = dirs[:, 1][:, None]
    ax, ay, bx, by, rad = (caps[:, i][None, :] for i in range(5))

    # End circles.
    t = np.minimum(_ray_circle(ox, oy, dx, dy, ax, ay, rad), _ray_circle(ox, oy, dx, dy, bx, by, rad))

    # Body: infinite cylinder around the segment line, clipped to the segment's extent (iq's capsule test).
    sx = bx - ax
    sy = by - ay
    oax = ox - ax
    oay = oy - ay
    baba = sx * sx + sy * sy
    bard = sx * dx + sy * dy
    baoa = sx * oax + sy * oay
    rdoa = dx * oax + dy * oay
    oaoa = oax * oax + oay * oay
    a = baba - bard * bard
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa * baoa - rad * rad * baba
    h = b * b - a * c
    ok = (a > 1e-12) & (h >= 0.0)
    tb = (-b - np.sqrt(np.maximum(h, 0.0))) / np.where(ok, a, 1.0)
    y = baoa + tb * bard
    ok &= (tb > 0.0) & (y > 0.0) & (y < baba)
    t = np.minimum(t, np.where(ok, tb, np.inf))

    return np.minimum(np.min(t, axis=1), max_range)


def push_out(px: float, py: float, r: float, caps: np.ndarray, iters: int = 4):
    """Move a circle of radius r at (px, py) out of any capsule it overlaps.

    Each iteration resolves the deepest overlap along its contact normal, which makes the robot slide
    along walls. Deterministic (argmax takes the first index on ties). Returns (px, py, contact).
    """
    contact = False
    if len(caps) == 0:
        return px, py, contact
    for _ in range(iters):
        cx, cy, d = closest_points(px, py, caps)
        pen = r + caps[:, 4] - d
        i = int(np.argmax(pen))
        if pen[i] <= 0.0:
            break
        contact = True
        if d[i] > 1e-12:
            nx = (px - cx[i]) / d[i]
            ny = (py - cy[i]) / d[i]
        else:  # centre exactly on the core segment: push perpendicular to it
            sx = caps[i, 2] - caps[i, 0]
            sy = caps[i, 3] - caps[i, 1]
            ln = np.sqrt(sx * sx + sy * sy)
            nx, ny = (-sy / ln, sx / ln) if ln > 0.0 else (1.0, 0.0)
        px = px + nx * pen[i]
        py = py + ny * pen[i]
    return px, py, contact


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification (iterative). points: (M, 2)."""
    n = len(points)
    if n < 3:
        return points.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a = points[i]
        b = points[j]
        seg = np.concatenate([a, b, [0.0]])[None, :]
        _, _, d = closest_points(points[i + 1 : j, 0], points[i + 1 : j, 1], np.repeat(seg, j - i - 1, 0))
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return points[keep]


def polyline_capsules(points: np.ndarray, rad: float) -> np.ndarray:
    """Chain of capsules along a polyline (consecutive points)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 1:
        return capsules([[p[0, 0], p[0, 1], p[0, 0], p[0, 1], rad]])
    rows = np.column_stack([p[:-1], p[1:], np.full(len(p) - 1, rad)])
    return capsules(rows)
