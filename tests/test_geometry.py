import math

import numpy as np
import pytest

from mazebot import geometry as g


def dirs(*angles):
    return np.array([[math.cos(a), math.sin(a)] for a in angles])


def test_ray_hits_vertical_wall_body():
    caps = g.capsules([[5, -2, 5, 2, 0.1]])
    d = g.cast_rays(0, 0, dirs(0.0), caps, 10.0)
    assert d[0] == pytest.approx(4.9, abs=1e-12)


def test_ray_hits_end_cap():
    # ray along the segment's axis hits the rounded end first
    caps = g.capsules([[3, 0, 6, 0, 0.5]])
    d = g.cast_rays(0, 0, dirs(0.0), caps, 10.0)
    assert d[0] == pytest.approx(2.5, abs=1e-12)


def test_ray_hits_circle_post_at_angle():
    caps = g.capsules([[3, 3, 3, 3, 1.0]])
    d = g.cast_rays(0, 0, dirs(math.pi / 4), caps, 10.0)
    assert d[0] == pytest.approx(math.sqrt(18) - 1.0, abs=1e-12)


def test_ray_misses_and_is_capped():
    caps = g.capsules([[5, 1, 5, 2, 0.1]])
    d = g.cast_rays(0, 0, dirs(0.0, math.pi), caps, 3.0)
    assert np.all(d == 3.0)


def test_ray_behind_origin_ignored():
    caps = g.capsules([[-2, -1, -2, 1, 0.1]])
    assert g.cast_rays(0, 0, dirs(0.0), caps, 10.0)[0] == 10.0


def test_ray_takes_nearest_of_many():
    caps = g.capsules([[8, -1, 8, 1, 0.1], [4, -1, 4, 1, 0.1], [6, -1, 6, 1, 0.1]])
    assert g.cast_rays(0, 0, dirs(0.0), caps, 10.0)[0] == pytest.approx(3.9)


def test_ray_matches_brute_force_march():
    rng = np.random.default_rng(0)
    caps = g.capsules(np.column_stack([rng.uniform(1, 9, (30, 4)), rng.uniform(0.02, 0.4, 30)]))
    ts = np.arange(0, 6.0, 1e-3)
    checked = 0
    for _ in range(60):
        o = rng.uniform(0, 10, 2)
        if g.clearance(*o, caps) < 0.05:
            continue
        a = rng.uniform(-math.pi, math.pi)
        d = g.cast_rays(o[0], o[1], dirs(a), caps, 6.0)[0]
        clear = g.clearance_grid(o[0] + ts * math.cos(a), o[1] + ts * math.sin(a), caps)
        inside = np.nonzero(clear <= 0)[0]
        hit = ts[inside[0]] if len(inside) else 6.0
        assert d == pytest.approx(hit, abs=2e-3)
        checked += 1
    assert checked > 20


def test_push_out_resolves_overlap_along_normal():
    caps = g.capsules([[0, 0, 4, 0, 0.1]])
    x, y, c = g.push_out(2.0, 0.3, 0.3, caps)
    assert c
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(0.4)
    assert g.clearance(x, y, caps) >= 0.3 - 1e-12


def test_push_out_no_contact():
    caps = g.capsules([[0, 0, 4, 0, 0.1]])
    assert g.push_out(2.0, 1.0, 0.3, caps) == (2.0, 1.0, False)


def test_push_out_corner_resolves_both_walls():
    caps = g.capsules([[0, 0, 4, 0, 0.1], [0, 0, 0, 4, 0.1]])
    x, y, _ = g.push_out(0.2, 0.25, 0.3, caps)
    assert g.clearance(x, y, caps) >= 0.3 - 1e-9


def test_rdp_keeps_endpoints_and_corners():
    pts = np.array([[0, 0], [1, 0.001], [2, 0], [2, 1], [2, 2]], dtype=float)
    out = g.rdp(pts, 0.01)
    assert out.tolist() == [[0, 0], [2, 0], [2, 2]]


def test_clearance_grid_matches_scalar():
    rng = np.random.default_rng(1)
    caps = g.capsules(np.column_stack([rng.uniform(0, 5, (10, 4)), rng.uniform(0, 0.3, 10)]))
    xs, ys = rng.uniform(0, 5, 200), rng.uniform(0, 5, 200)
    grid = g.clearance_grid(xs, ys, caps, chunk=37)
    assert np.allclose(grid, [g.clearance(x, y, caps) for x, y in zip(xs, ys, strict=True)])
