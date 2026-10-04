import math

import numpy as np
import pytest

from mazebot import distance_field as df
from mazebot import geometry as g

R = 0.3


def wall_with_gap(gap):
    """Arena 10x6 split by a wall at x=5 with a gap of `gap` (free width between capsule surfaces)."""
    rad = 0.1
    y0 = 3 - gap / 2 - rad
    y1 = 3 + gap / 2 + rad
    return np.vstack([g.border(10, 6), g.capsules([[5, 0, 5, y0, rad], [5, y1, 5, 6, rad]])])


def test_open_field_matches_euclid():
    caps = g.border(10, 6)
    grid = df.build_grid(10, 6, caps, R)
    f = df.geodesic_field(grid, 8, 3)
    assert df.lookup(grid, f, 2, 3) == pytest.approx(6.0, rel=0.02)
    # diagonal: the octile metric overestimates euclid by at most ~8%
    d = df.lookup(grid, f, 2, 0.8)
    e = math.hypot(6, 2.2)
    assert e - 0.05 <= d <= 1.09 * e + 0.05


def test_gap_narrower_than_robot_is_blocked():
    caps = wall_with_gap(0.5)  # robot diameter 0.6
    grid = df.build_grid(10, 6, caps, R)
    assert not df.reachable(grid, 2, 3, 8, 3)


def test_gap_wider_than_robot_is_open_and_detours():
    caps = wall_with_gap(1.0)
    grid = df.build_grid(10, 6, caps, R)
    assert df.reachable(grid, 2, 1, 8, 1)
    f = df.geodesic_field(grid, 8, 1)
    assert df.lookup(grid, f, 2, 1) > 2 * math.hypot(3, 2) - 0.3  # must go through the gap at y=3


def test_lookup_near_wall_is_finite():
    caps = wall_with_gap(1.0)
    grid = df.build_grid(10, 6, caps, R)
    f = df.geodesic_field(grid, 8, 3)
    # robot touching the wall: its own cell may be blocked, lookup still finds a value
    assert math.isfinite(df.lookup(grid, f, 5 - 0.1 - R, 1.0))


def test_closed_box_unreachable():
    box = g.polyline_capsules(np.array([[3, 2], [7, 2], [7, 4], [3, 4], [3, 2]], dtype=float), 0.1)
    caps = np.vstack([g.border(10, 6), box])
    grid = df.build_grid(10, 6, caps, R)
    assert not df.reachable(grid, 1, 1, 5, 3)
    assert df.reachable(grid, 1, 1, 9, 5)
