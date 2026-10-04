import math

import numpy as np
import pytest

from mazebot import geometry as g
from mazebot.sim import SimParams, observe, physics_step, wrap_angle

P = SimParams()
NONE = g.EMPTY


def run(x, y, th, ul, ur, caps=NONE, steps=30):
    contact = False
    for _ in range(steps):
        x, y, th, c = physics_step(x, y, th, ul, ur, caps, P)
        contact |= c
    return x, y, th, contact


def test_equal_wheels_drive_straight():
    x, y, th, _ = run(0, 0, 0, 1, 1)  # 1 s at full speed
    assert (x, y, th) == pytest.approx((P.v_max, 0.0, 0.0))


def test_reverse():
    x, y, _, _ = run(0, 0, 0, -1, -1)
    assert (x, y) == pytest.approx((-P.v_max, 0.0))


def test_spin_in_place_and_direction_convention():
    # left wheel forward, right wheel back => clockwise on screen (y down) => heading increases
    x, y, th, _ = run(0, 0, 0, 0.5, -0.5, steps=1)
    assert (x, y) == pytest.approx((0, 0))
    assert th == pytest.approx(P.v_max * 1.0 / P.wheel_base * P.dt)


def test_turning_right_moves_towards_positive_y():
    _, y, _, _ = run(0, 0, 0, 1.0, 0.6)
    assert y > 0.05


def test_no_tunnelling_through_thin_wall_at_full_speed():
    caps = g.capsules([[1, -5, 1, 5, 0.02]])
    x, y, _, contact = run(0, 0, 0, 1, 1, caps, steps=120)
    assert contact
    assert x <= 1 - 0.02 - P.radius + 1e-9


def test_slides_along_wall():
    caps = g.capsules([[1, -5, 1, 5, 0.05]])
    th = math.radians(30)  # hit the wall at an angle
    x, y, _, contact = run(0, 0, th, 1, 1, caps, steps=90)
    assert contact
    assert x == pytest.approx(1 - 0.05 - P.radius, abs=1e-6)
    assert y > 1.0  # kept moving along the wall


def test_wrap_angle():
    for a in [-7.0, -math.pi, 0.0, math.pi, 4.0, 13.0]:
        w = wrap_angle(a)
        assert -math.pi < w <= math.pi
        assert math.isclose(math.sin(w), math.sin(a), abs_tol=1e-12)
        assert math.isclose(math.cos(w), math.cos(a), abs_tol=1e-12)


def test_observation_layout_and_ranges():
    caps = g.border(10, 10)
    obs, rays = observe(5, 5, 0.0, 5, 8, caps, P)  # goal straight "below" = clockwise 90 deg
    assert obs.shape == (8,)
    assert np.all((obs[:5] >= 0) & (obs[:5] <= 1))
    assert obs[5] == pytest.approx(1.0)  # sin: goal to the right
    assert obs[6] == pytest.approx(0.0, abs=1e-12)
    assert obs[7] == pytest.approx(0.3)
    assert rays[2] == pytest.approx(min(4.95, P.ray_range))


def test_goal_distance_obs_clipped():
    obs, _ = observe(0.5, 0.5, 0.0, 30, 30, NONE, P)
    assert obs[7] == 1.0
    assert np.all(obs[:5] == 1.0)
