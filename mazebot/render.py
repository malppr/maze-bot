"""Matplotlib rendering of maps, distance fields, rays and trajectories (y down, like the browser)."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

WALL = "#1d1c1a"
ACCENT = "#c94436"
GOAL = "#2f7d4f"


def capsule_polygon(cap, n: int = 10) -> np.ndarray:
    ax, ay, bx, by, r = cap
    a = math.atan2(by - ay, bx - ax) if (bx != ax or by != ay) else 0.0
    t1 = np.linspace(a + math.pi / 2, a + 3 * math.pi / 2, n)
    t2 = np.linspace(a - math.pi / 2, a + math.pi / 2, n)
    return np.vstack(
        [
            np.column_stack([ax + r * np.cos(t1), ay + r * np.sin(t1)]),
            np.column_stack([bx + r * np.cos(t2), by + r * np.sin(t2)]),
        ]
    )


def draw_map(ax, m, show_field: bool = True, show_free: bool = True, title: str | None = None):
    """Walls, inflated (blocked) cells, geodesic field contours, A and B."""
    if show_field and m.grid is not None and m.field is not None:
        g = m.grid
        ext = (0, g.free.shape[1] * g.cell, g.free.shape[0] * g.cell, 0)
        f = np.where(np.isfinite(m.field), m.field, np.nan)
        ax.imshow(f, extent=ext, cmap="Blues_r", alpha=0.35, interpolation="nearest", origin="upper")
        if show_free:
            blocked = np.where(g.free, np.nan, 1.0)
            ax.imshow(blocked, extent=ext, cmap="Greys", vmin=0, vmax=4, alpha=0.5, interpolation="nearest")
        unreachable = np.where(g.free & ~np.isfinite(m.field), 1.0, np.nan)
        ax.imshow(unreachable, extent=ext, cmap="Oranges", vmin=0, vmax=2, alpha=0.5, interpolation="nearest")
    polys = [Polygon(capsule_polygon(c), closed=True) for c in m.caps]
    ax.add_collection(PatchCollection(polys, facecolor=WALL, edgecolor="none"))
    sx, sy, sth = m.start
    ax.add_patch(Circle((sx, sy), 0.3, facecolor=ACCENT, edgecolor="white", lw=1, zorder=5))
    ax.plot([sx, sx + 0.45 * math.cos(sth)], [sy, sy + 0.45 * math.sin(sth)], color="white", lw=1.5, zorder=6)
    ax.add_patch(Circle(m.goal, 0.35, facecolor="none", edgecolor=GOAL, lw=2, zorder=5))
    ax.text(*m.goal, "B", ha="center", va="center", color=GOAL, fontsize=8, weight="bold", zorder=6)
    ax.set_xlim(-0.1, m.width + 0.1)
    ax.set_ylim(m.height + 0.1, -0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def draw_trajectory(ax, xs, ys, color=ACCENT, **kw):
    ax.plot(xs, ys, color=color, lw=1.2, alpha=0.9, **kw)


def draw_rays(ax, x, y, th, rays, angles):
    for a, d in zip(angles, rays, strict=True):
        ax.plot([x, x + d * math.cos(th + a)], [y, y + d * math.sin(th + a)], color=ACCENT, lw=0.8, ls="--")


def render_rgb(env) -> np.ndarray:
    m = env.map
    fig = plt.figure(figsize=(m.width / 2, m.height / 2), dpi=60)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_map(ax, m, show_free=False)
    draw_rays(ax, env.x, env.y, env.th, env.rays, env.sim.ray_angles)
    ax.add_patch(Circle((env.x, env.y), env.sim.radius, facecolor=ACCENT, zorder=7))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
