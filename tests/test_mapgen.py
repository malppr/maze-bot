import math

import numpy as np
import pytest

from mazebot import geometry as g
from mazebot import mapgen

N = 15


@pytest.mark.parametrize("cat", mapgen.CATEGORIES)
def test_maps_are_solvable_and_valid(cat):
    rng = np.random.default_rng(123)
    for _ in range(N):
        m = mapgen.sample_map(rng, cat)
        sx, sy, sth = m.start
        assert min(m.width, m.height) >= 6 - 1e-9 and max(m.width, m.height) <= 12 + 1e-9
        assert m.walls.shape[1] == 5
        assert -math.pi <= sth <= math.pi
        for x, y in [(sx, sy), m.goal]:
            assert 0 < x < m.width and 0 < y < m.height
        assert g.clearance(sx, sy, m.caps) >= mapgen.ROBOT_RADIUS
        geo = m.geo(sx, sy)
        assert math.isfinite(geo)
        if cat != "traps":
            assert geo >= 3.0 - 0.15  # start is jittered within its grid cell


@pytest.mark.parametrize("cat", mapgen.CATEGORIES)
def test_same_seed_same_map(cat):
    a = mapgen.sample_map(np.random.default_rng(7), cat)
    b = mapgen.sample_map(np.random.default_rng(7), cat)
    assert np.array_equal(a.walls, b.walls)
    assert a.start == b.start and a.goal == b.goal


def test_different_seeds_differ():
    a = mapgen.sample_map(np.random.default_rng(1), "scribbles")
    b = mapgen.sample_map(np.random.default_rng(2), "scribbles")
    assert a.walls.shape != b.walls.shape or not np.allclose(a.walls, b.walls)


def test_presets_load_and_default_ab_reachable():
    presets = mapgen.load_presets()
    assert len(presets) >= 5
    for p in presets:
        for portrait in (False, True):
            m = mapgen.attach_field(mapgen.preset_map(p, portrait))
            assert math.isfinite(m.geo(*m.start[:2])), (p["name"], portrait)
            assert g.clearance(*m.start[:2], m.caps) >= mapgen.ROBOT_RADIUS


def test_portrait_preset_is_transposed():
    p = mapgen.load_presets()[1]
    land, port = mapgen.preset_map(p), mapgen.preset_map(p, portrait=True)
    assert (port.width, port.height) == (land.height, land.width)
    assert port.walls[0, 0] == land.walls[0, 1]
